#!/usr/bin/env python3
"""Full Qwen FSDP encoding, reference-feature parity, and unequal-rank cache misses."""

import argparse
from functools import partial
import json
from pathlib import Path

import runtime_env
import torch
import torch.distributed as dist

from h3.distributed.fsdp import wrap_text
from utils import distributed as groups
from utils.config import load_config
from utils.h3_wrapper import TextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("local/real_probe"))
    parser.add_argument("--output", type=Path, default=Path("local/text_encoder_verification"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    world = int(__import__("os").environ["WORLD_SIZE"])
    groups.launch_distributed_job(sp_size_arg=world, fs_size_arg=world)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = load_config("configs/worldviews.yaml")
    cfg.h3.checkpoint = str(args.checkpoint)
    cfg.h3.text_cache = str(args.output / f"cache_rank{dist.get_rank()}")
    cfg.sp_size = cfg.fs_size = world
    text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
    caption = json.loads((args.features / "features.json").read_text())["caption"]
    expected = torch.load(args.features / "short_mono.pt", map_location="cpu", weights_only=True)["prompt_embeds"]
    actual = text([caption])[0]
    relative = float((actual.float() - expected.float()).norm() / expected.float().norm())
    assert relative < .01, relative
    original = actual.clone()
    # Rank 0 has a hit while the other rank has a miss. Every FSDP rank must
    # still run the same module collectives, even for unrelated caption lengths.
    prompt = caption if dist.get_rank() == 0 else "A camera moves along the street."
    text([prompt])
    cached = text([caption])[0]
    torch.testing.assert_close(cached, original, rtol=0, atol=0)
    report = dict(status="passed",
                  world_size=world,
                  layer=50,
                  width=5120,
                  relative_error=relative,
                  shape=list(actual.shape),
                  unequal_rank_cache_misses=True,
                  cached_features_exact=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    groups.shutdown_distributed()
    if rank == 0:
        (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
