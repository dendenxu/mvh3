#!/usr/bin/env python3
"""Bounded full-H3 compile reuse check on the existing native feature bank."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import runtime_env
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from h3.checkpoint import load_original_transformer
from h3.distributed.fsdp import wrap_model, compile_blocks
from h3.utils.training import parameter_groups
from model.diffusion import WorldViewsObjective
from utils import distributed as groups
from utils.camera import prepare_camera_geometry
from utils.config import load_config, validate_config
from utils.ema import ShardedEMA
from utils.tracking import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_compile_buckets.yaml")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 8:
        parser.error("At least eight shape-changing updates are required")
    cfg = validate_config(load_config(args.config))
    cfg.h3.logdir = str(args.output)
    cfg.h3.compile_cache = str(args.output / "compile")
    cfg.max_iters = args.steps
    torch.set_num_threads(1)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    if dist.get_world_size() != 8 or groups.get_sp_size() != 8:
        raise ValueError("This bounded check requires one existing eight-GPU host")
    rank = groups.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        OmegaConf.save(cfg, args.output / "resolved.yaml")
    documents = [prepare_camera_geometry(document, cfg) for document in
                 torch.load(args.features / "documents.pt", map_location="cpu", weights_only=True)]
    torch.manual_seed(cfg.seed)
    model = load_original_transformer(cfg.h3.checkpoint, progress=print if rank == 0 else None)
    model.configure_attention(cfg)
    assert sum(p.numel() for p in model.parameters()) == 33122992896
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3853523200
    model = wrap_model(model, cfg)
    compile_blocks(model.module, cfg)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg), betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay, fused=True)
    ema = ShardedEMA.from_config(model, cfg)
    objective = WorldViewsObjective(cfg)
    frozen = {name: p.detach().flatten()[:8].cpu().clone()
              for name, p in model.named_parameters() if not p.requires_grad}
    tracker = Tracker(cfg, args.output)
    report = dict(status="running", scope="full pretrained H3 on existing native Qwen/VAE features; one SP8/FSDP8 node",
                  multihost=False, production=False, config=args.config, steps=args.steps,
                  features=str(args.features), optimizer_updates=0)

    def persist():
        if rank == 0:
            (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")

    persist()
    success = False
    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(42000 + step)
            document = objective.prepare_document(deepcopy(documents[(step - 1) % len(documents)]), device)
            torch.cuda.synchronize()
            started = time.monotonic()
            previous_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            loss, row = objective(model, document, device, step - 1)
            loss.backward()
            norm = model.clip_grad_norm_(cfg.clip_grad_norm)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite update at {step}")
            optimizer.step()
            ema.update(model)
            torch.cuda.synchronize()
            row.pop("x0", None)
            row.pop("rf", None)
            new_graphs = torch.tensor(torch._dynamo.utils.counters["stats"]["unique_graphs"] - previous_graphs,
                                      device=device)
            dist.all_reduce(new_graphs, op=dist.ReduceOp.MAX)
            row.update(step=step, loss=float(loss), grad_norm=float(norm), seconds=time.monotonic() - started,
                       new_compile_graphs=int(new_graphs), ema_updates=ema.num_updates,
                       ema_decay=ema.current_decay, stage=1, sf_depth=0, self=0)
            tracker.training(row, optimizer, document)
            if rank == 0:
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            report["optimizer_updates"] = step
            persist()
        unchanged = all(torch.equal(p.detach().flatten()[:8].cpu(), frozen[name])
                        for name, p in model.named_parameters() if name in frozen)
        check = torch.tensor(int(unchanged and ema.num_updates == args.steps), device=device)
        dist.all_reduce(check, op=dist.ReduceOp.MIN)
        assert check.item(), "Frozen samples or EMA count changed unexpectedly"
        report.update(status="passed", frozen_samples_unchanged=True, ema_updates=ema.num_updates)
        success = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
        tracker.finish(success=success)
    groups.shutdown_distributed()


if __name__ == "__main__":
    main()
