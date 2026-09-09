#!/usr/bin/env python3
"""Decode actual samples through every full WorldViews source-family adapter."""

import argparse
import gc
import json
from pathlib import Path
import time

import runtime_env
import torch
from omegaconf import OmegaConf

from utils.config import load_config
from utils.h3_wrapper import extract_views, source_documents
from dataset import DATASET_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", default="")
    parser.add_argument("--rows", type=int, default=8, help="Bound preload per source; production uses all rows")
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = load_config(args.config)
    shared = OmegaConf.to_container(cfg.dataset, resolve=True)
    sources = shared.pop("datasets")
    shared.pop("type")
    shared["num_workers"] = 0
    shared["seq_sample"] = [0, args.rows, 1]
    selection = [int(x) for x in args.sources.split(",")] if args.sources else range(len(sources))
    report = dict(status="running", preload_rows_per_source=args.rows, sources=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in selection:
        started = time.monotonic()
        options = {**shared, **sources[index]}
        kind = options.pop("type")
        dataset = DATASET_REGISTRY[kind](config=cfg, **options)
        sample = dataset[0]
        views = extract_views(sample)
        mono = source_documents(sample, 1, cfg.h3.short_frames)
        assert sum(len(x["pixels"]) for x in views) == sum(len(d["views"][0]["pixels"]) for d in mono)
        assert all(len(d["views"]) == 1 and d["isolated"] for d in mono)
        for view in views:
            assert torch.isfinite(view["pixels"]).all() and torch.isfinite(view["pose"]).all()
            assert len(view["pixels"]) == len(view["pose"]) == len(view["projection"])
        entry = dict(index=index,
                     type=kind,
                     source=Path(options["data_path"]).name,
                     shapes=[list(v["pixels"].shape) for v in views],
                     mono_clips=len(mono),
                     seconds=time.monotonic() - started,
                     status="passed")
        report["sources"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        del dataset, sample, views, mono
        gc.collect()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
