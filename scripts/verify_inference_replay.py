#!/usr/bin/env python3
"""Replay saved native inputs without per-call tracing or retained GPU tensors."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import runtime_env
import torch
from omegaconf import OmegaConf

from h3.checkpoint import load_original_transformer
from h3.distributed.fsdp import wrap_model, compile_blocks
from pipeline.ar_inference import generate
from utils import distributed as groups
from utils.checkpoint import load_checkpoint
from utils.config import load_config, recipe_digest
from utils.tracking import Tracker


def persist(output, report):
    if groups.get_rank() == 0:
        temporary = output / "verification.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "verification.json")


def errors(actual, expected, device):
    values = torch.tensor([float((a.float() - b.float()).abs().max())
                           for a, b in zip(actual, expected)], device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    return values.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["raw", "ema"], nargs="+", default=["raw", "ema"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7891)
    parser.add_argument("--atol", type=float, default=.02)
    args = parser.parse_args()
    if args.repeats < 2 or args.steps < 2 or args.atol < 0:
        parser.error("Use at least two repeats and sigma points, and a nonnegative tolerance")
    cfg = load_config(args.config, [f"h3.logdir={args.output}"])
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if groups.get_rank() == 0:
        OmegaConf.save(cfg, args.output / "resolved.yaml")
    tracker = Tracker(cfg, args.output)
    report = dict(status="running", phase="loading", recipe=recipe_digest(cfg),
                  checkpoint=str(args.checkpoint), document=str(args.document),
                  document_sha256=hashlib.sha256(args.document.read_bytes()).hexdigest(),
                  seed=args.seed, sigma_points=args.steps, atol=args.atol,
                  scope="Uninstrumented full-model cached/recomputed rollout from saved encoded inputs",
                  runs=[])
    persist(args.output, report)
    success = False
    try:
        document = torch.load(args.document, map_location="cpu", weights_only=False)
        report["chunk_sizes"] = [torch.bincount(v["generation_chunks"]).tolist() for v in document["views"]]
        model = load_original_transformer(cfg.h3.checkpoint, progress=print if groups.get_rank() == 0 else None)
        model.configure_attention(cfg)
        model = wrap_model(model, cfg)
        compiled = False
        local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        local.kv_gpu_budget_gb = 0
        report["inference_overrides"] = {"kv_gpu_budget_gb": 0}
        for weight_kind in args.weights:
            # CPU-offloaded shards may still be in use by asynchronous H2D copies.
            # Weight transitions occur only between complete rollouts.
            torch.cuda.synchronize()
            state = load_checkpoint(model, None, cfg, args.checkpoint, restore_random=False, weights=weight_kind)
            report["checkpoint_step"] = state["step"]
            del state
            if not compiled:
                compile_blocks(model.module, cfg)
                compiled = True
            reference = {}
            for repeat in range(args.repeats):
                outputs = {}
                for mode, enabled in (("cached", True), ("recomputed", False)):
                    report.update(phase="inference", active=dict(weights=weight_kind, repeat=repeat, mode=mode))
                    persist(args.output, report)
                    torch.manual_seed(args.seed)
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    outputs[mode] = generate(model, document, None, local, device, steps=args.steps, use_cache=enabled)
                    torch.cuda.synchronize()
                    row = dict(weights=weight_kind, repeat=repeat, mode=mode,
                               seconds=time.monotonic() - started,
                               peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                               finite=all(bool(torch.isfinite(x).all()) for x in outputs[mode]))
                    if mode in reference:
                        row["repeat_max_abs_error"] = errors(outputs[mode], reference[mode], device)
                    else:
                        reference[mode] = outputs[mode]
                    if mode == "recomputed":
                        row["cached_vs_recomputed_max_abs_error"] = errors(outputs[mode], outputs["cached"], device)
                    report["runs"].append(row)
                    persist(args.output, report)
                    if groups.get_rank() == 0:
                        torch.save(outputs[mode], args.output / f"{weight_kind}_{repeat}_{mode}.pt")
                        print(json.dumps(row), flush=True)
                    metrics = {f"replay/{key}": row[key] for key in ("seconds", "peak_gpu_allocated_gib", "finite")}
                    for key in ("repeat_max_abs_error", "cached_vs_recomputed_max_abs_error"):
                        if key in row:
                            metrics[f"replay/{key}"] = max(row[key])
                    tracker.log(metrics, len(report["runs"]))
        failed = [row for row in report["runs"] if not row["finite"] or any(
            max(row[key]) > args.atol for key in ("repeat_max_abs_error", "cached_vs_recomputed_max_abs_error") if key in row)]
        if failed:
            raise AssertionError(f"Inference repeat/cache comparison failed: {failed}")
        groups.shutdown_distributed()
        report.update(status="passed", phase="complete")
        report.pop("active", None)
        persist(args.output, report)
        success = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist(args.output, report)
        raise
    finally:
        if tracker.run:
            tracker.run.summary.update({"verification/status": report["status"],
                                        "verification/path": str(args.output / "verification.json")})
        tracker.finish(success=success)


if __name__ == "__main__":
    main()
