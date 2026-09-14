#!/usr/bin/env python3
"""Standalone i2v: torchrun scripts/infer.py --request request.json --config training.yaml."""

import argparse

# Resolve the existing environment before importing Torch or repository modules.
import runtime_env  # noqa: F401; isort: skip

# isort: split

from pipeline.inference import run_inference
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Exact training resolved.yaml when loading a checkpoint"
    )
    parser.add_argument(
        "--request",
        required=True,
        action="append",
        help="Repeat to reuse the encoders and backbone across cases",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument(
        "--protocol",
        choices=["auto", "ar", "joint"],
        default="auto",
        help="Native joint for base weights; AR after loading a training checkpoint",
    )
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--verify-init", action="store_true", help="Compare every static-camera prediction with base weights"
    )
    args = parser.parse_args()
    if args.weights == "ema" and not args.checkpoint:
        parser.error("--weights ema requires a training checkpoint containing EMA")
    cfg = load_config(args.config)
    cfg.inference_weights = args.weights
    run_inference(
        cfg,
        request_path=args.request,
        output=args.output,
        checkpoint=args.checkpoint,
        protocol=args.protocol,
        seed=args.seed,
        steps=args.steps,
        camera=not args.no_camera,
        verify_init=args.verify_init,
    )


if __name__ == "__main__":
    main()
