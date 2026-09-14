#!/usr/bin/env python3
"""Prepare fixed real monocular documents using the production H3 encoder."""

import argparse
import json
from pathlib import Path

import runtime_env
import torch
from omegaconf import OmegaConf

from dataset.mvgame import select_pose_stable_factor
from h3.data import read_parquet_clip
from h3.modules.camera import camera_projection
from utils.config import load_config, validate_config
from utils.h3_wrapper import VideoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overfit.yaml")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--text-from", type=Path, required=True,
                        help="Verified full-Qwen features for the identical source caption")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = validate_config(load_config(args.config))
    args.output.mkdir(parents=True, exist_ok=True)
    clip = read_parquet_clip(args.parquet, cfg.overfit.row, list(cfg.overfit.views),
                             cfg.overfit.frames, fps=cfg.dataset.model_fps,
                             height=cfg.dataset.height, width=cfg.dataset.width)
    previous = json.loads((args.text_from / "features.json").read_text())
    if (previous["status"] != "complete" or previous["caption"] != clip["caption"]
            or Path(previous["checkpoint"]).resolve() != Path(cfg.h3.checkpoint).resolve()
            or previous["text_layer"] != 50):
        raise ValueError("Text reuse requires the same full encoder and exact source caption")
    text = torch.load(args.text_from / "short_mono.pt", map_location="cpu", weights_only=True)["prompt_embeds"]
    assert text.shape[-1] == 5120 and torch.isfinite(text).all()
    poses = clip["pose"].float().clone()
    poses[..., 2:4] -= .5
    scale, diameter = select_pose_stable_factor(poses[..., 7:10].reshape(-1, 3), cfg.dataset.pose_stable_factors)
    poses[..., 7:10] /= scale
    video = VideoEncoder(cfg.h3.vae, args.device)
    documents = []
    for i, (pixels, pose) in enumerate(zip(clip["pixels"], poses)):
        pixels = pixels[0].permute(1, 0, 2, 3)
        matrix = camera_projection(pose[None])
        raw = dict(views=[dict(pixels=pixels.float() / 255, pose=pose,
                              projection=matrix.projection[0], inverse=matrix.inverse[0],
                              prompt=clip["caption"], fps=clip["fps"], scale=scale,
                              source_view=int(cfg.overfit.views[i]), source_start=0)],
                   isolated=True, source=str(args.parquet))
        torch.manual_seed(cfg.seed + i)
        document = video.prepare(raw, cfg, validation=True)
        document["views"][0]["text"] = text
        documents.append(document)
        torch.save(pixels, args.output / f"pixels_{i}.pt")
        view = document["views"][0]
        print(json.dumps(dict(sample=i, fps=view["fps"], source_frames=view["source_frames"],
                              latent_shape=list(view["latent"].shape),
                              condition_shape=list(view["condition"]["latent"].shape))), flush=True)
    torch.save(documents, args.output / "documents.pt")
    metadata = dict(status="complete", checkpoint=str(Path(cfg.h3.checkpoint).resolve()),
                    parquet=str(args.parquet), row=int(cfg.overfit.row), views=list(cfg.overfit.views),
                    frames=int(cfg.overfit.frames), fps=clip["fps"],
                    height=int(cfg.dataset.height), width=int(cfg.dataset.width),
                    source_paths=clip["source_paths"], source_indices=clip["sample_indices"],
                    caption=clip["caption"], text_layer=50, text_shape=list(text.shape),
                    text_source=str(args.text_from), pose_stable_factor=scale, pose_diameter=diameter,
                    condition_encode_seed=int(cfg.h3.condition_encode_seed))
    (args.output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n")
    OmegaConf.save(cfg, args.output / "resolved.yaml")


if __name__ == "__main__":
    main()
