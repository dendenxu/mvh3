#!/usr/bin/env python3
"""Decode every actual SHORT/FULL Ours bucket using its paired captions/config."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import runtime_env
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from dataset.presampled import PresampledDataset, resolve_runtime_shape
from prepare_presampled_comparison import read_rows
from utils.config import load_config, stage_dataset_config
from utils.h3_wrapper import source_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", cases=[], dataset_type="paired_presampled",
                  source_batch_size=cfg.dataset.batch_size,
                  gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                  sp_size=cfg.sp_size, fs_size=cfg.fs_size)
    for stage in (1, 2):
        dc = stage_dataset_config(cfg, stage)
        original = pq.ParquetFile(dc.spec)
        ranges = json.loads(original.schema_arrow.metadata[b"presampled_shape_ranges"])
        ids = [bounds[0] for bounds in ranges.values()]
        table = read_rows(dc.spec, ids)
        rows = table.to_pylist()
        metadata = {**table.schema.metadata, b"presampled_shape_ranges": json.dumps(
            {key: [i, i + 1] for i, key in enumerate(ranges)}).encode()}
        subset = args.output / f"stage{stage}.parquet"
        pq.write_table(table.replace_schema_metadata(metadata), subset, row_group_size=500)
        options = OmegaConf.to_container(dc, resolve=True)
        options.pop("type")
        options.update(spec=str(subset), num_workers=0)
        decode_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        decode_cfg.sp_size = 1
        dataset = PresampledDataset(config=decode_cfg, **options)
        assert dataset.n_samples == len(ids), "An envelope row is excluded by the real loader"
        for local, (name, row_id, row) in enumerate(zip(ranges, ids, rows)):
            sample = dataset[local]
            document = source_documents(sample, stage, cfg.h3.short_frames)[0]
            views, gen = resolve_runtime_shape(row["mv"], row["gen"], row["view_isolated"], dict(dc.shape_remap))
            assert len(document["views"]) == views
            assert all(len(v["pixels"]) == 4 * gen - 3 for v in document["views"])
            assert all(tuple(v["pixels"].shape[-2:]) == (dc.height, dc.width) for v in document["views"])
            assert document["isolated"] == bool(row["view_isolated"])
            policy = "chunk" if sample["cpu"].get("per_chunk_text") else "global"
            if cfg.h3.get("single_sequence", False):
                policy = "bd_overlap"
            for index, view in enumerate(document["views"]):
                text_view = index if row["view_isolated"] else 0
                if policy == "bd_overlap":
                    assert view["caption_scene"] == str(row["scene"][text_view] or "")
                    assert view["caption_motions"] == list(row["chunks"][text_view] or [])
                    assert view["caption_source_frames"] == row["gen"] * 4 - 3
                else:
                    assert view["prompt"] == row["caption"][text_view]
                if policy == "chunk":
                    chunks = row["chunks"][text_view]
                    count = max(1, len(chunks) * gen // row["gen"])
                    assert view["chunk_prompts"] == [row["scene"][text_view] + "\n[CHUNK]" + c for c in chunks[:count]]
                view["pixels"] = view["pixels"].mul(255).round().clamp(0, 255).byte()
            case = f"stage{stage}_{name}"
            document["source"] = dc.spec
            document["probe_case"] = case
            path = args.output / f"{case}.pt"
            torch.save(document, path)
            entry = dict(name=case, path=path.name, stage=stage, source=dc.spec, parquet_row=row_id,
                         original_bucket=name, views=views, reference_latents=gen, frames=4 * gen - 3,
                         isolated=document["isolated"], logical_batch_size=views if document["isolated"] else 1,
                         caption_policy=policy, source_rows=row["row"], source_parquets=row["data_parquet"],
                         frame_start=row["frame_start"], frame_end=row["frame_end"],
                         shapes=[list(v["pixels"].shape) for v in document["views"]],
                         fps=[v["fps"] for v in document["views"]],
                         caption_sha256=[hashlib.sha256(v["prompt"].encode()).hexdigest() for v in document["views"]])
            report["cases"].append(entry)
            (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: entry[k] for k in ("name", "parquet_row", "views", "frames", "logical_batch_size", "caption_policy")}), flush=True)
            del sample, document
            gc.collect()
        del dataset, table, rows
        gc.collect()
    report["status"] = "passed"
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
