#!/usr/bin/env python3
"""Export old/new caption pairs from the exact paired Ours training Parquet."""

import argparse
import hashlib
import json
from pathlib import Path

import runtime_env
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from omegaconf import OmegaConf
import torch

from dataset.presampled import PresampledDataset
from utils.config import load_config
from utils.h3_wrapper import source_documents


def read_rows(path, indices):
    parquet = pq.ParquetFile(path)
    found, offset = {}, 0
    for group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        selected = [i for i in indices if offset <= i < offset + count]
        if selected:
            table = parquet.read_row_group(group)
            for i in selected:
                found[i] = table.slice(i - offset, 1)
        offset += count
    return pa.concat_tables([found[i] for i in indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--rows", default="97752,97753")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    cfg = load_config(args.config)
    ids = [int(value) for value in args.rows.split(",")]
    table = read_rows(cfg.dataset.spec, ids)
    rows = table.to_pylist()
    assert all(row["mv"] == 1 and row["gen"] == 20 and row["view_isolated"] for row in rows)
    metadata = {**table.schema.metadata, b"presampled_shape_ranges": json.dumps({"1x20_iso": [0, len(rows)]}).encode()}
    args.output.mkdir(parents=True, exist_ok=True)
    subset = args.output / "paired_rows.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), subset, row_group_size=500)
    options = OmegaConf.to_container(cfg.dataset, resolve=True)
    options.pop("type")
    options.update(spec=str(subset), num_workers=0)
    decode_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    decode_cfg.sp_size = 1  # Standalone CPU replay has no distributed shard group.
    dataset = PresampledDataset(config=decode_cfg, **options)
    report = dict(status="running", source=str(cfg.dataset.spec), cases=[])
    for local, (row_id, row) in enumerate(zip(ids, rows)):
        document = source_documents(dataset[local], 2)[0]
        view = document["views"][0]
        assert view["prompt"] == row["caption"][0]
        if view.get("chunk_prompts"):
            assert view["chunk_prompts"] == [row["scene"][0] + "\n[CHUNK]" + chunk for chunk in row["chunks"][0]]
        directory = args.output / f"row{row_id:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        pixels = view["pixels"].mul(255).round().clamp(0, 255).byte()
        Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(directory / "input.png")
        output_frames = round((len(pixels) - 1) / view["fps"] * 24) + 1
        indices = np.round(np.arange(output_frames) * view["fps"] / 24).astype(np.int64)
        camera = view["pose"].numpy().astype(np.float32)
        np.save(directory / "recorded.npy", camera[indices])
        np.save(directory / "static.npy", np.repeat(camera[:1], output_frames, axis=0))
        provenance = dict(parquet=str(cfg.dataset.spec), parquet_row=row_id, source_parquets=row["data_parquet"],
                          source_rows=row["row"], source_views=row["view_ids"], frame_start=row["frame_start"],
                          frame_end=row["frame_end"], fps_ratio=row["fps_ratio"], sample_fps=view["fps"],
                          output_to_sample_indices=indices.tolist(), input_sha256=hashlib.sha256(pixels[0].numpy().tobytes()).hexdigest())
        paths = []
        for policy, caption in (("source", row["simple_caption"][0]), ("scene_motion", row["caption"][0])):
            assert caption
            request = dict(prompt=caption, fps=24, provenance={**provenance, "caption_policy": policy,
                           "caption_sha256": hashlib.sha256(caption.encode()).hexdigest()},
                           views=[dict(image="input.png", camera="static.npy", scale=view["scale"])])
            path = directory / f"{policy}.json"
            path.write_text(json.dumps(request, indent=2) + "\n")
            paths.append(str(path.resolve()))
        torch.save({**document, "views": [{**view, "pixels": pixels}]}, directory / "raw.pt")
        report["cases"].append(dict(row=row_id, requests=paths, source_frames=len(pixels), output_frames=output_frames,
                                   shapes=list(pixels.shape), provenance=provenance))
        print(json.dumps({"row": row_id, "requests": paths, "output_frames": output_frames}), flush=True)
    report.update(status="passed", scope="same-case base-H3 caption A/B; static camera; complete paired source window")
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
