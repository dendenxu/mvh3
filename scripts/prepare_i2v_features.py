#!/usr/bin/env python3
"""Refresh fixed documents with the released FL2VA image/text conditioner."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import runtime_env
import torch

from h3.distributed.fsdp import wrap_text
from utils import distributed as groups
from utils.config import load_config
from utils.h3_wrapper import TextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config("configs/h3_native.yaml")
    torch.set_num_threads(1)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    docs = torch.load(args.source / "documents.pt", map_location="cpu", weights_only=True)
    encoder = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, doc in enumerate(docs):
        pixels = torch.load(args.source / f"pixels_{index}.pt", map_location="cpu", weights_only=True)
        view = doc["views"][0]
        encoded = encoder.i2v([(view["prompt"], [pixels[0]])])[0]
        view["text"], view["text_tags"] = encoded["features"], encoded["tags"]
        view["condition_image"] = pixels[0]
        if groups.get_rank() == 0:
            target = args.output / f"pixels_{index}.pt"
            if not target.exists():
                target.symlink_to((args.source / f"pixels_{index}.pt").resolve())
            print(json.dumps(dict(sample=index, shape=list(view["text"].shape),
                                  visual_tokens=int((view["text_tags"] == 0).sum()))), flush=True)
    if groups.get_rank() == 0:
        torch.save(docs, args.output / "documents.pt")
        metadata = json.loads((args.source / "features.json").read_text())
        metadata.update(text_conditioning="fl2va", text_shapes=[list(d["views"][0]["text"].shape) for d in docs],
                        text_source="Qwen3-VL-32B full model, original processor, Picture labels + vision + caption",
                        base_features_sha256=hashlib.sha256((args.source / "documents.pt").read_bytes()).hexdigest())
        (args.output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n")
    groups.shutdown_distributed()


if __name__ == "__main__":
    main()
