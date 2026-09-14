#!/usr/bin/env python3
"""Run original WorldViews buckets through native encoders and optimizer updates."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time

import runtime_env
import torch
import torch.distributed as dist

import trainer.diffusion as training
from utils import distributed as groups
from utils.config import load_config


def trace_nonfinite_gradients(trainer, output):
    original_pack, original_clip = trainer.objective.pack, trainer.model.clip_grad_norm_
    captured = {}

    def pack(document, *args, **kwargs):
        result = original_pack(document, *args, **kwargs)
        captured.update(document=document, packed=result)
        return result

    def to_cpu(value):
        from h3.modules.masking import TokenLayout
        if isinstance(value, (torch.Tensor, TokenLayout)):
            return value.to("cpu")
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(to_cpu(item) for item in value)
        return value

    def clip(*args, **kwargs):
        # Inspect before clipping: a NaN norm would contaminate every gradient.
        bad = []
        for name, parameter in trainer.model.named_parameters():
            if parameter.grad is not None:
                finite = torch.isfinite(parameter.grad)
                if not finite.all():
                    bad.append(dict(name=name, elements=parameter.grad.numel(),
                                    nonfinite=int((~finite).sum())))
        norm = original_clip(*args, **kwargs)
        if not torch.isfinite(norm):
            all_bad = [None] * dist.get_world_size()
            dist.all_gather_object(all_bad, bad)
            if groups.get_rank() == 0:
                step = trainer.step + 1
                torch.save(to_cpu(captured), output / f"nonfinite_step{step:09d}.pt")
                (output / f"nonfinite_step{step:09d}.json").write_text(json.dumps(
                    dict(step=step, case=captured["document"].get("probe_case"),
                         norm=str(float(norm)), bad_gradients_by_rank=all_bad), indent=2) + "\n")
        return norm

    trainer.objective.pack, trainer.model.clip_grad_norm_ = pack, clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("local/batch_envelope"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--cases", nargs="+", help="Bounded diagnostic subset; never reported as full-envelope coverage")
    parser.add_argument("--trace-nonfinite", action="store_true", help="Save failing packed inputs and pre-clip gradient diagnostics")
    args = parser.parse_args()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    cases = manifest["cases"]
    # Exercise the largest independent batch and longest mono input first.
    cases.sort(key=lambda c: (c["logical_batch_size"], c["frames"]), reverse=True)
    cfg = load_config(args.config, [f"h3.logdir={args.output}",
        f"h3.text_cache={args.output / 'text_cache'}", "auto_resume=false", "resume_ckpt=null",
        "save_interval=0", "vis_interval=0", "vis_init=false", "vis_list=[]"])
    if manifest.get("dataset_type") == "paired_presampled":
        assert {c["stage"] for c in cases} == {1, 2}
        assert cfg.dataset.type == "presampled"
    else:
        expected = {(False, *pair) for pair in cfg.dataset.shape_pool}
        expected.update((True, *pair) for pair in cfg.dataset.short_paths)
        assert {(c["isolated"], c["views"], c["reference_latents"]) for c in cases} == expected
    assert cfg.dataset.batch_size == manifest["source_batch_size"] == 1
    assert cfg.gradient_accumulation_steps == manifest["gradient_accumulation_steps"] == 1
    canonical_count = len(cases)
    if args.cases:
        assert set(args.cases) <= {case["name"] for case in cases}, "Unknown diagnostic case"
        cases = [case for case in cases if case["name"] in args.cases]
    cfg.max_iters = math.ceil(len(cases) / cfg.sp_size) * cfg.sp_size
    assert cfg.max_iters < cfg.resampling_forcing_warmup_steps

    class EnvelopeStream(training.SourceStream):
        def __init__(self, cfg, video, text, step=0, validation=False):
            self.cfg, self.video, self.text, self.validation = cfg, video, text, validation
            self.pending, self.mixed = deque(), deque()
            self.sampler_generator = torch.Generator().manual_seed(cfg.seed + groups.get_rank())
            self.stage, self.negative, self.round = 2, None, 0

        def next(self):
            if not self.pending and not self.mixed:
                case = cases[(self.round * cfg.sp_size + groups.get_sp_rank()) % len(cases)]
                raw = torch.load(args.inputs / case["path"], map_location="cpu", weights_only=True)
                if cfg.h3.get("single_sequence", False) and manifest.get("dataset_type") == "paired_presampled":
                    from paired_caption_metadata import attach_captions
                    raw = attach_captions(raw, case, args.inputs)
                for view in raw["views"]:
                    assert view["pixels"].dtype == torch.uint8
                    assert tuple(view["pixels"].shape) == (case["frames"], 3, cfg.dataset.height, cfg.dataset.width)
                    view["pixels"] = view["pixels"].float() / 255
                raw["probe_case"] = case["name"]
                self.pending.append(raw)
                self.round += 1
            return super().next()

    training.SourceStream = EnvelopeStream
    started = time.monotonic()
    trainer = training.Trainer(cfg)
    # This is a memory/update probe of full buckets from base weights. The
    # separate continuation check owns stage transitions and checkpoint I/O.
    trainer.stage = 2
    trainer.stream.set_stage(2)
    trainer.save = lambda: None
    if args.trace_nonfinite:
        trace_nonfinite_gradients(trainer, args.output)
    observed = []
    original_tracking = trainer.tracker.training

    def track(row, optimizer, document):
        scales = [float(view["scale"]) for view in document["views"]]
        if list(cfg.dataset.pose_stable_factors) == [1.0]:
            assert all(scale == 1.0 for scale in scales), scales
        peak = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                            device=trainer.device, dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        row.update(batch_case=document["probe_case"], peak_reserved_mib=int(peak[1]) // 1024**2)
        original_tracking(row, optimizer, document)
        observed.append(dict(case=document["probe_case"], step=row["step"], loss=row["loss"],
                             grad_norm=row["grad_norm"], seconds=row["seconds"], tokens=row["tokens"],
                             source_batch_size=row["source_batch_size"], logical_batch_size=row["bs"],
                             views=row["views"], peak_allocated_mib=int(peak[0]) // 1024**2,
                             peak_reserved_mib=int(peak[1]) // 1024**2,
                             text_encoded_requests=row.get("text_encoded_requests", 0),
                             pose_stable_factors=scales,
                             source_pose_stable_factors=[float(v.get("source_pose_stable_factor", v["scale"]))
                                                        for v in document["views"]],
                             source_shapes=[[v["source_frames"], v["height"], v["width"]]
                                            for v in document["views"]]))
        if groups.get_rank() == 0:
            temporary = args.output / "progress.tmp"
            temporary.write_text(json.dumps(dict(status="running", expected_cases=len(cases), cases=observed,
                                                 seconds=time.monotonic() - started), indent=2) + "\n")
            temporary.replace(args.output / "progress.json")

    trainer.tracker.training = track
    if trainer.tracker.run:
        trainer.tracker.run.summary.update({"operation": "original-batch-envelope",
                                           "full_source_training": False,
                                           "source_manifest": str((args.inputs / "manifest.json").resolve())})
    success = False
    try:
        trainer.train_loop()
        assert {row["case"] for row in observed} == {case["name"] for case in cases}
        assert sum(row["text_encoded_requests"] for row in observed) > 0
        if trainer.ema is not None:
            assert trainer.ema.num_updates == trainer.step == cfg.max_iters
            assert all(value.dtype == torch.float32 and torch.isfinite(value).all()
                       for value in trainer.ema.weights.values())
            if cfg.get("ema_cpu_offload", True):
                assert all(value.device.type == "cpu" for value in trainer.ema.weights.values())
        report = dict(status="passed", scope="live native Qwen/VAE and full 33B optimizer updates",
                      canonical_cases=canonical_count, tested_cases=len(cases), full_envelope=len(cases) == canonical_count,
                      source_batch_size=cfg.dataset.batch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                      sp_size=cfg.sp_size, fs_size=cfg.fs_size, cpu_offload=cfg.generator_cpu_offload,
                      text_cpu_offload=cfg.text_encoder_cpu_offload, checkpointing=cfg.gradient_checkpointing,
                      compiled=cfg.attn_block_compile, shape_or_batch_reduction=False,
                      ema_enabled=trainer.ema is not None,
                      ema_decay_cap=trainer.ema.decay if trainer.ema else 0.,
                      ema_updates=trainer.ema.num_updates if trainer.ema else 0,
                      ema_cpu_offload=bool(cfg.get("ema_cpu_offload", True)),
                      max_allocated_mib=max(row["peak_allocated_mib"] for row in observed),
                      max_reserved_mib=max(row["peak_reserved_mib"] for row in observed),
                      cases=observed, seconds=time.monotonic() - started)
        groups.shutdown_distributed()
        if groups.get_rank() == 0:
            (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
            if trainer.tracker.run:
                trainer.tracker.run.summary.update({"verification/status": "passed",
                                                    "verification/path": str(args.output / "verification.json")})
        success = True
    except BaseException as error:
        if groups.get_rank() == 0:
            report = dict(status="failed", error=f"{type(error).__name__}: {error}", cases=observed,
                          expected_cases=len(cases), shape_or_batch_reduction=False)
            (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        trainer.tracker.finish(success=success)


if __name__ == "__main__":
    main()
