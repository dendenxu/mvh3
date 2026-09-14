#!/usr/bin/env python3
"""Check CPU-offloaded EMA consistency across real FSDP replica groups."""

import argparse
import hashlib
import json
from pathlib import Path

import runtime_env
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from h3.distributed.fsdp import fsdp_options
from utils import distributed as groups
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.ema import ShardedEMA


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fs-size", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config("configs/worldviews_grouped.yaml", ["sp_size=1", f"fs_size={args.fs_size}"])
    groups.launch_distributed_job(sp_size_arg=1, fs_size_arg=args.fs_size)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world > args.fs_size and world % args.fs_size == 0, "Use at least two replica groups"
    assert cfg.generator_cpu_offload and cfg.ema_cpu_offload
    torch.set_num_threads(1)
    torch.manual_seed(310)
    device = torch.device("cuda", torch.cuda.current_device())
    module = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8))
    module[0].bias.requires_grad_(False)
    options = fsdp_options(cfg)
    for index in (0, 2):
        module[index] = FSDP(module[index], **options)
    model = FSDP(module, **options)
    for parameter in model.parameters():
        parameter.grad_dtype = None
    ema = ShardedEMA.from_config(model, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, fused=True)
    torch.manual_seed(420 + rank)
    inputs = torch.randn(3 + rank, 16, device=device)

    def update():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(inputs).float().square().mean()
        loss.backward()
        norm = model.clip_grad_norm_(1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        ema.update(model)

    def check_replicas():
        state = dict(raw=tensor_digest(model.named_parameters()),
                     ema=tensor_digest(ema.weights.items()), updates=ema.num_updates)
        states = [None] * world
        dist.all_gather_object(states, state)
        for start in range(args.fs_size):
            assert all(state == states[start] for state in states[start::args.fs_size]), states

    @torch.no_grad()
    def predict():
        with torch.autocast("cuda", torch.bfloat16):
            return model(inputs).detach().clone()

    for _ in range(8):
        update()
        check_replicas()
    raw = predict()
    for _ in range(16):
        with ema.average_parameters(model):
            averaged = predict()
        assert torch.equal(predict(), raw)
    assert not torch.equal(averaged, raw)
    checkpoint = save_checkpoint(model, optimizer, cfg, 8, 1, {}, args.output / "ckpt", ema=ema)
    for _ in range(2):
        update()
    state = load_checkpoint(model, optimizer, cfg, checkpoint, ema=ema)
    torch.testing.assert_close(optimizer.state_dict(), state["optimizer"], rtol=0, atol=0)
    assert ema.num_updates == 8 and torch.equal(predict(), raw)
    with ema.average_parameters(model):
        assert torch.equal(predict(), averaged)
    check_replicas()
    for _ in range(2):
        update()
        check_replicas()
    report = dict(status="passed", world_size=world, fs_size=args.fs_size,
                  replica_groups=world // args.fs_size, input_rows_by_rank=list(range(3, 3 + world)),
                  cpu_offload=True, ema_cpu_offload=True, ema_decay_cap=ema.decay,
                  ema_warmup=ema.warmup, ema_current_decay=ema.current_decay,
                  ema_updates=ema.num_updates, raw_and_ema_replica_shards_exact=True,
                  optimizer_resume_exact=True, raw_and_ema_inference_restored_exact=True,
                  swap_cycles=16,
                  scope="tiny nested FSDP with unequal rank inputs; excludes H3 and multi-host transport")
    groups.shutdown_distributed()
    if rank == 0:
        (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
