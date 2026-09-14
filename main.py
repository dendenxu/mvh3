#!/usr/bin/env python3
"""torchrun entry for the complete WorldViews H3 adaptation."""

import argparse
import os
import sys
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

for candidate in (Path(__file__).resolve().parent.parent / "python_deps", ):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from utils.config import load_config, validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", default="configs/diffusion_forcing.yaml")
    parser.add_argument("--print-config", action="store_true")
    args, overrides = parser.parse_known_args()
    cfg = validate_config(load_config(args.config_path, overrides))
    if args.print_config:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if "RANK" not in os.environ:
        raise SystemExit("Launch with torchrun --nproc_per_node=8 main.py -c configs/diffusion_forcing.yaml")
    import torch
    # Match WorldViews' main/autograd thread pinning. Decoder worker threads
    # are configured separately; mixing the two breaks checkpoint compile guards.
    torch.set_num_threads(int(os.environ.get("WORLDGEN_TORCH_NUM_THREADS", "1")))
    if cfg.task == "inference" and cfg.get("inference_request"):
        sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
        from scripts.infer import run
        run(cfg, cfg.inference_request, cfg.h3.logdir, cfg.resume_ckpt,
            cfg.get("inference_protocol", "auto"), cfg.seed)
        return
    from trainer.diffusion import Trainer
    if cfg.task == "inference":
        cfg.validation_weights = cfg.get("inference_weights", "auto")
    trainer = Trainer(cfg)
    if cfg.task == "train":
        trainer.train()
    elif cfg.task == "inference":
        trainer.validate(cfg.inference_num_samples)
    else:
        raise ValueError(f"Unknown task: {cfg.task}")
    from utils.distributed import shutdown_distributed
    shutdown_distributed()


if __name__ == "__main__":
    main()
