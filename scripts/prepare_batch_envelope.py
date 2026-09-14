#!/usr/bin/env python3
"""Decode every canonical view/duration bucket without reducing the source batch."""

import argparse
import gc
from functools import partial
import hashlib
import json
from pathlib import Path

import runtime_env
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset import DATASET_REGISTRY
from utils.config import load_config
from utils.h3_wrapper import source_documents
from verify_worldviews_data import source_captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cross-source", type=int, default=6)
    parser.add_argument("--isolated-source", type=int, default=14)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = load_config(args.config)
    shared = OmegaConf.to_container(cfg.dataset, resolve=True)
    sources = shared.pop("datasets")
    shared.pop("type")
    shared.update(num_workers=0, seq_sample=[0, args.rows, 1])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    cases = json.loads(manifest.read_text())["cases"] if manifest.is_file() else []
    buckets = [(False, list(pair)) for pair in cfg.dataset.shape_pool]
    buckets += [(True, list(pair)) for pair in cfg.dataset.short_paths]
    for isolated, (views, latents) in buckets:
        name = f"{'isolated' if isolated else 'cross'}_{views}x{latents}"
        if any(case["name"] == name and (args.output / case["path"]).is_file() for case in cases):
            continue
        index = args.isolated_source if isolated else args.cross_source
        options = {**shared, **sources[index]}
        kind = options.pop("type")
        # Select an existing canonical bucket for coverage, preserving all its
        # views, pixels and source-caption policies. No smaller rescue bucket.
        options.update(long_gen_size=0, short_paths=[], shape_pool=[] if isolated else [[views, latents]],
                       shape_pool_weights=None, mv_size=views, gen_size=latents,
                       dynamic_mv_size=views, dynamic_gen_size=latents)
        dataset = DATASET_REGISTRY[kind](config=cfg, **options)
        if isolated:
            # The canonical short_paths route explicitly uses a full-resolution
            # strip, unlike the legacy fixed-mv pack used by the main fallback.
            dataset.try_view_iso = partial(dataset.try_view_iso, force_strip=True)
        sample = dataset[0]
        document = source_documents(sample, 2)[0]
        actual_isolated = document["isolated"]
        assert len(document["views"]) == views, (name, len(document["views"]))
        assert actual_isolated == isolated, (name, actual_isolated)
        expected_frames = latents * 4 - 3
        assert all(len(view["pixels"]) == expected_frames for view in document["views"]), name
        assert all(tuple(view["pixels"].shape[-2:]) == (cfg.dataset.height, cfg.dataset.width)
                   for view in document["views"]), name
        source_rows = np.asarray(sample["cpu"]["rows"]).reshape(-1).tolist()
        captions = source_captions(options["data_path"], source_rows)
        for i, view in enumerate(document["views"]):
            assert view["prompt"] == captions[source_rows[i if len(source_rows) > 1 else 0]], (name, i, source_rows)
            assert not view.get("chunk_prompts")
            view["pixels"] = view["pixels"].mul(255).round().clamp(0, 255).byte()
        path = args.output / f"{name}.pt"
        torch.save(document, path)
        row = dict(name=name, path=path.name, source_index=index, source=options["data_path"],
                   rows=source_rows, views=views, reference_latents=latents, isolated=actual_isolated,
                   frames=expected_frames, source_batch_size=cfg.dataset.batch_size,
                   logical_batch_size=views if actual_isolated else 1,
                   shapes=[list(v["pixels"].shape) for v in document["views"]],
                   fps=[v["fps"] for v in document["views"]],
                   caption_sha256=[hashlib.sha256(v["prompt"].encode()).hexdigest() for v in document["views"]])
        cases.append(row)
        (args.output / "manifest.json").write_text(json.dumps(dict(status="running", cases=cases), indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del dataset, sample, document
        gc.collect()
    report = dict(status="passed", source_batch_size=cfg.dataset.batch_size,
                  gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                  sp_size=cfg.sp_size, fs_size=cfg.fs_size, cases=cases,
                  scope="all configured shape_pool and short_paths decoded at original view count and resolution")
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
