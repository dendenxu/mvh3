#!/usr/bin/env python3
"""Compare every resolved reference setting with a live WorldViews checkout."""

import argparse
import json
from pathlib import Path

import runtime_env
from omegaconf import OmegaConf
from utils.config import load_config


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
    port = OmegaConf.to_container(load_config("configs/worldviews.yaml"), resolve=True)
    port.pop("h3")
    delta = differences(original, port)
    report = dict(status="failed" if delta else "passed",
                  compared_top_level_keys=len(original),
                  differences=delta,
                  source_count=len(port["dataset"]["datasets"]),
                  validation_source_count=len(port["val_dataset"]["datasets"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if delta:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
