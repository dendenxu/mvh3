#!/usr/bin/env python3
"""Encode paired Ours documents through the production native feature stream."""

import argparse
from collections import deque
from functools import partial
import json
from pathlib import Path

import runtime_env
import torch

from h3.distributed.fsdp import wrap_text
from trainer.diffusion import SourceStream
from utils import distributed as groups
from utils.config import load_config
from utils.h3_wrapper import TextEncoder, VideoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/overfit_paired.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    torch.set_num_threads(1)
    device = torch.device("cuda", torch.cuda.current_device())
    video = VideoEncoder(cfg.h3.vae, device, cfg.vae_compile)
    text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    args.output.mkdir(parents=True, exist_ok=True)
    encoded, policies = [], []
    for index, case in enumerate(manifest["cases"]):
        raw = torch.load(args.inputs / f"row{case['row']:06d}" / "raw.pt", map_location="cpu", weights_only=True)
        torch.manual_seed(cfg.seed + index)
        for view in raw["views"]:
            pixels = view["pixels"]
            assert pixels.dtype == torch.uint8
            if groups.get_rank() == 0:
                torch.save(pixels, args.output / f"pixels_{index}.pt")
            view["pixels"] = pixels.float() / 255
        stream = SourceStream.__new__(SourceStream)
        stream.cfg, stream.video, stream.text = cfg, video, text
        stream.validation, stream.stage, stream.negative = False, 2, None
        stream.pending, stream.mixed = deque([raw]), deque()
        document = stream.next()
        encoded.append(document)
        policies.append("chunk" if raw["views"][0].get("chunk_prompts") else "global")
        if groups.get_rank() == 0:
            print(json.dumps(dict(sample=index, row=case["row"], policy=policies[-1],
                                  text_shapes=[list(value.shape) for _, value in document["views"][0]["texts"]])), flush=True)
    groups.shutdown_distributed()
    if groups.get_rank() == 0:
        torch.save(encoded, args.output / "documents.pt")
        metadata = dict(status="complete", text_conditioning="fl2va", text_source="native Qwen3-VL image+text layer50",
                        views=list(range(len(encoded))), cases=manifest["cases"], caption_policies=policies,
                        source=manifest["source"], fps=cfg.dataset.model_fps, frames=cfg.overfit.frames)
        (args.output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
