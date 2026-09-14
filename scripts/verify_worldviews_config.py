#!/usr/bin/env python3
"""Compare every resolved reference setting with a live WorldViews checkout."""

import argparse
import json
from pathlib import Path

import runtime_env
from omegaconf import OmegaConf
from utils.config import load_config, stage_dataset_config


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result.extend(differences(a.get(key), b.get(key), f"{prefix}.{key}".lstrip(".")))
        return result
    return [] if a == b else [prefix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local/config_verification.json"))
    args = parser.parse_args()
    original = OmegaConf.to_container(load_config(args.reference), resolve=True)
    snapshot = OmegaConf.to_container(load_config("configs/worldviews_reference.yaml"), resolve=True)
    reference_delta = differences(original, snapshot)
    config = load_config("configs/worldviews.yaml")
    port = OmegaConf.to_container(config, resolve=True)
    port.pop("h3")
    adaptations = {"model.timestep_shift": 12.0, "timestep_shift": 12.0, "sampling_solver": "h3_euler",
                   "sampling_steps": 50, "vis_sampling_steps": 50, "guidance_scale": 1.0,
                   "cfg_rescale_factor": 0.0,
                   "model.prope_unwrapped": False, "model.prope_mode": "decomposed",
                   "model.scale_cond": False, "dataset.pose_stable_factors": [1.0],
                   "val_dataset.pose_stable_factors": [1.0]}
    paired = load_config(args.reference.parent / "pre200.yaml")
    short = load_config(args.reference.parent / "pre200_short.yaml")
    expected = OmegaConf.merge(OmegaConf.create(original), paired)
    changes = {}
    for key, value in adaptations.items():
        previous = OmegaConf.select(expected, key)
        if OmegaConf.is_config(previous):
            previous = OmegaConf.to_container(previous, resolve=True)
        changes[key] = dict(worldviews=previous, h3=value)
        OmegaConf.update(expected, key, value)
    delta = differences(OmegaConf.to_container(expected, resolve=True), port)
    for stage, override in ((1, short), (2, paired)):
        for validation, key in ((False, "dataset"), (True, "val_dataset")):
            target = OmegaConf.merge(OmegaConf.create(original[key]), override[key])
            target.shape_remap = override[key].shape_remap
            target.pose_stable_factors = [1.0]
            delta.extend(differences(OmegaConf.to_container(target, resolve=True),
                                    OmegaConf.to_container(stage_dataset_config(config, stage, validation), resolve=True),
                                    f"stage{stage}.{key}"))
    report = dict(status="failed" if delta or reference_delta else "passed",
                  compared_top_level_keys=len(original),
                  reference_differences=reference_delta, unexpected_differences=delta,
                  native_h3_adaptations=changes,
                  paired_data_reference=str(args.reference.parent / "pre200.yaml"),
                  short_data_reference=str(args.reference.parent / "pre200_short.yaml"),
                  full_training_parquet=port["dataset"]["spec"],
                  short_training_parquet=stage_dataset_config(config, 1).spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if delta or reference_delta:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
