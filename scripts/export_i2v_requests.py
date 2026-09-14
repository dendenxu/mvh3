#!/usr/bin/env python3
"""Export first-frame images and calibrated static/recorded camera requests."""

import argparse
import json
from pathlib import Path

import runtime_env
import numpy as np
from PIL import Image
import torch

from h3.data import read_parquet_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.features / "features.json").read_text())
    clip = read_parquet_clip(metadata["parquet"], metadata["row"], metadata["views"], metadata["frames"],
                            metadata["fps"], metadata["height"], metadata["width"])
    poses = clip["pose"].float().clone()
    poses[..., 2:4] -= .5
    poses[..., 7:10] /= metadata["pose_stable_factor"]
    args.output.mkdir(parents=True, exist_ok=True)
    views = []
    for index, pose in enumerate(poses):
        pixels = torch.load(args.features / f"pixels_{index}.pt", map_location="cpu", weights_only=True)
        image = f"input{index}.png"
        Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(args.output / image)
        for mode, trajectory in (("recorded", pose), ("static", pose[:1].expand_as(pose))):
            camera = f"{mode}{index}.npy"
            np.save(args.output / camera, trajectory.numpy().astype(np.float32))
            view = dict(image=image, camera=camera, scale=metadata["pose_stable_factor"])
            request = dict(prompt=metadata["caption"], fps=metadata["fps"], views=[view])
            (args.output / f"{mode}{index}.json").write_text(json.dumps(request, indent=2) + "\n")
            if mode == "recorded":
                views.append(view)
    (args.output / "multiview.json").write_text(json.dumps(dict(prompt=metadata["caption"], fps=metadata["fps"], views=views), indent=2) + "\n")
    print(json.dumps(dict(status="complete", directory=str(args.output), views=len(views))))


if __name__ == "__main__":
    main()
