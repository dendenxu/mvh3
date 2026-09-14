#!/usr/bin/env python3
"""Decode actual samples through every full WorldViews source-family adapter."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import runtime_env
import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from omegaconf import OmegaConf

from utils.config import load_config
from utils.h3_wrapper import extract_views, source_documents
from dataset import DATASET_REGISTRY


def source_captions(path, rows):
    wanted, found, offset = set(map(int, rows)), {}, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500, columns=["caption"]):
        for index in wanted:
            if offset <= index < offset + len(batch):
                found[index] = batch.column(0)[index - offset].as_py()
        offset += len(batch)
        if wanted == found.keys():
            return found
    raise ValueError(f"Missing source caption rows: {wanted - found.keys()}")


def export_case(directory, index, view, source, row, frames):
    from utils.video import write_video
    if view["fps"] != 24:
        raise ValueError("Select an actual 24 FPS sample for native H3 case export")
    directory = directory / f"source{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stop = min(frames, len(view["pixels"]))
    pixels = view["pixels"][:stop].mul(255).round().clamp(0, 255).byte()
    Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(directory / "input.png")
    np.save(directory / "recorded.npy", view["pose"][:stop].numpy().astype(np.float32))
    np.save(directory / "static.npy", view["pose"][:1].expand(stop, -1).numpy().astype(np.float32))
    provenance = dict(parquet=str(source), row=int(row), caption_policy="source_parquet_caption",
                      caption_sha256=hashlib.sha256(view["prompt"].encode()).hexdigest())
    for mode in ("static", "recorded"):
        request = dict(prompt=view["prompt"], fps=view["fps"], provenance=provenance,
                       views=[dict(image="input.png", camera=f"{mode}.npy", scale=view["scale"])])
        (directory / f"{mode}.json").write_text(json.dumps(request, indent=2) + "\n")
    raw_view = {**view, **{key: view[key][:stop] for key in ("pixels", "pose", "projection", "inverse")}}
    torch.save(dict(views=[raw_view], isolated=True, source=str(source)), directory / "raw.pt")
    write_video(str(directory / "source.mp4"), pixels.permute(0, 2, 3, 1).numpy(), fps=view["fps"])
    return str(directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", default="")
    parser.add_argument("--rows", type=int, default=8, help="Bound preload per source; production uses all rows")
    parser.add_argument("--case-output", type=Path, help="Export separate raw image/camera requests for inference")
    parser.add_argument("--case-sources", default="0,3,6,14")
    parser.add_argument("--case-frames", type=int, default=39)
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
        rows = np.asarray(sample["cpu"]["rows"]).reshape(-1).tolist()
        captions = source_captions(options["data_path"], rows)
        for view_index, view in enumerate(views):
            row = rows[view_index if len(rows) > 1 else 0]
            assert view["prompt"] == captions[row], f"Source caption changed for {kind} row {row}"
            assert not view.get("chunk_prompts"), "Large-model sources must retain their original global caption"
        mono = source_documents(sample, 1, cfg.h3.short_frames)
        assert sum(len(x["pixels"]) for x in views) == sum(len(v["pixels"]) for d in mono for v in d["views"])
        assert all(d["isolated"] for d in mono)
        assert len(mono[0]["views"]) == len(views)
        for view in views:
            assert torch.isfinite(view["pixels"]).all() and torch.isfinite(view["pose"]).all()
            assert len(view["pixels"]) == len(view["pose"]) == len(view["projection"])
        entry = dict(index=index,
                     type=kind,
                     source=Path(options["data_path"]).name,
                     shapes=[list(v["pixels"].shape) for v in views],
                     fps=[v["fps"] for v in views],
                     rows=rows,
                     caption_policy="source_parquet_caption",
                     caption_sha256=[hashlib.sha256(v["prompt"].encode()).hexdigest() for v in views],
                     mono_clips=sum(len(d["views"]) for d in mono),
                     short_batches=len(mono),
                     seconds=time.monotonic() - started,
                     status="passed")
        if args.case_output and index in {int(x) for x in args.case_sources.split(",")}:
            entry["case"] = export_case(args.case_output, index, views[0], options["data_path"], rows[0], args.case_frames)
        report["sources"].append(entry)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        del dataset, sample, views, mono
        gc.collect()
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
