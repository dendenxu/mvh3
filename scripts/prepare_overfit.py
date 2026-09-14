#!/usr/bin/env python3
"""Reuse exact video features and encode all contiguous caption combinations."""

import argparse
import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

# Resolve the existing environment before importing Torch or repository modules.
import runtime_env  # noqa: F401; isort: skip

# isort: split
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from h3.distributed.fsdp import wrap_text
from h3.encoders import TextEncoder
from utils import distributed as groups
from utils.config import load_config


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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/overfit_diffusion_forcing.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(1)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    documents = torch.load(args.features / "documents.pt", map_location="cpu", weights_only=True)
    assert len(documents) == 1 and len(documents[0]["views"]) == 1
    metadata = json.loads((args.features / "features.json").read_text())
    assert metadata["cases"][0]["row"] == args.row
    row = read_rows(str(args.parquet), [args.row]).to_pylist()[0]
    view = documents[0]["views"][0]
    assert view["source_frames"] == 4 * row["gen"] - 3
    scene, motions = str(row["scene"][0] or ""), list(row["chunks"][0] or [])
    view.update(
        caption_scene=scene,
        caption_motions=motions,
        caption_source_frames=view["source_frames"],
        prompt=scene,
    )
    # Any temporal majority selection is a contiguous subset of source windows.
    captions = [scene or cfg.negative_prompt]
    for start in range(len(motions)):
        for stop in range(start + 1, len(motions) + 1):
            captions.append(
                "\n".join(str(x).strip() for x in [scene, *motions[start:stop]] if str(x).strip())
                or cfg.negative_prompt
            )
    captions = list(dict.fromkeys(captions))
    pixels = torch.load(args.features / "pixels_0.pt", map_location="cpu", weights_only=True)
    assert pixels.dtype == torch.uint8
    encoder = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
    bank = encoder.i2v([(caption, [pixels[0]]) for caption in captions])
    view["caption_feature_bank"] = dict(zip(captions, bank))
    for name in (
        "generation_chunks",
        "clean_prefix_chunks",
        "texts_by_bd",
        "texts",
        "text_tag_specs",
        "caption_specs",
    ):
        view.pop(name, None)
    view["text"], view["text_tags"] = bank[0]["features"], bank[0]["tags"]
    if groups.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(documents, args.output / "documents.pt")
        shutil.copy2(args.features / "pixels_0.pt", args.output / "pixels_0.pt")
        metadata.update(
            caption_policy="bd_overlap",
            caption_bank=list(captions),
            feature_parent_sha256=hashlib.sha256((args.features / "documents.pt").read_bytes()).hexdigest(),
            partition="resampled before each clean-prefix cut; contiguous caption feature bank",
        )
        (args.output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(dict(status="complete", captions=len(bank), output=str(args.output))), flush=True)
    groups.shutdown_distributed()


if __name__ == "__main__":
    main()
