#!/usr/bin/env python3
"""Exercise the real trainer/encoders/RF state machine on bounded native i2v inputs."""

import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import runtime_env
import numpy as np
import torch

from h3.modules.camera import camera_projection
import trainer.diffusion as training
from utils import distributed as groups
from utils.config import load_config


def persist_report(output, report):
    if groups.get_rank() == 0:
        temporary = output / "verification.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "verification.json")


def verify_restored_state(trainer, directory):
    from h3.distributed.fsdp import canonical_name

    path = Path(directory)
    if path.name == "latest.json":
        path = path.parent / json.loads(path.read_text())["path"]
    saved = torch.load(path / f"rank{groups.get_rank():05d}.pt", map_location="cpu", weights_only=False)
    error = None
    try:
        actual = {canonical_name(name): p for name, p in trainer.model.named_parameters() if p.requires_grad}
        assert actual.keys() == saved["weights"].keys()
        for name, value in saved["weights"].items():
            torch.testing.assert_close(actual[name].detach().cpu(), value, rtol=0, atol=0)
        optimizer = trainer.optimizer.state_dict()
        torch.testing.assert_close(optimizer, saved["optimizer"], rtol=0, atol=0)
        if trainer.ema is not None:
            assert trainer.ema.num_updates == saved["ema"]["num_updates"] == saved["step"]
            torch.testing.assert_close(trainer.ema.state_dict(), saved["ema"], rtol=0, atol=0)
    except AssertionError as failure:
        error = f"rank {groups.get_rank()}: {failure}"
    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error)
    if any(errors):
        raise AssertionError("Restored state mismatch: " + "; ".join(item for item in errors if item))
    count = sum(value.numel() for value in saved["weights"].values())
    count = torch.tensor(count, device=trainer.device, dtype=torch.long)
    torch.distributed.all_reduce(count)
    return dict(status="passed", weights_and_adamw_exact=True, trainable_elements=int(count),
                ema_exact=True if trainer.ema is not None else None)


def raw_documents(features, requests):
    metadata = json.loads((features / "features.json").read_text())
    request = json.loads((requests / "multiview.json").read_text())
    assert request["prompt"] == metadata["caption"]
    views = []
    for index, spec in enumerate(request["views"]):
        pixels = torch.load(features / f"pixels_{index}.pt", map_location="cpu", weights_only=True)
        pose = torch.from_numpy(np.load(requests / spec["camera"], allow_pickle=False)).float()
        matrix = camera_projection(pose[None])
        views.append(dict(pixels=pixels.float() / 255, pose=pose, projection=matrix.projection[0],
                          inverse=matrix.inverse[0], prompt=request["prompt"], fps=request["fps"],
                          scale=spec["scale"], source_view=index, source_start=0))
    return [dict(views=views[:1], isolated=True, source=metadata["parquet"]),
            dict(views=views, isolated=False, source=metadata["parquet"])]


def paired_documents(directory, names, cfg=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "passed" and manifest["dataset_type"] == "paired_presampled"
    cases = {case["name"]: case for case in manifest["cases"]}
    documents = []
    for stage, name in enumerate(names, 1):
        case = cases[name]
        assert case["stage"] == stage
        raw = torch.load(directory / case["path"], map_location="cpu", weights_only=True)
        if cfg is not None and cfg.h3.get("single_sequence", False):
            from paired_caption_metadata import attach_captions
            raw = attach_captions(raw, case, directory)
        assert len(raw["views"]) == case["views"]
        for view in raw["views"]:
            assert view["pixels"].dtype == torch.uint8 and len(view["pixels"]) == case["frames"]
            view["pixels"] = view["pixels"].float() / 255
        raw["probe_case"] = name
        documents.append(raw)
    return documents


def input_fingerprint(value):
    from dataclasses import fields, is_dataclass

    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous()
        digest = hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        return dict(shape=list(data.shape), dtype=str(data.dtype), sha256=digest)
    if isinstance(value, dict):
        return {key: input_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_fingerprint(item) for item in value]
    if is_dataclass(value):
        return {field.name: input_fingerprint(getattr(value, field.name)) for field in fields(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported input trace type: {type(value).__name__}")


class InferenceTrace:
    """Record full-model call boundaries without changing attention kernels."""

    def __init__(self, model, directory, deferred=False):
        self.model, self.directory, self.calls = model, directory, 0
        self.deferred, self.pending = deferred, []
        if groups.get_rank() == 0:
            directory.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, **inputs):
        if self.deferred:
            saved = None
            if groups.get_rank() == 0:
                saved = {key: value for key, value in inputs.items() if key != "kv_caches"}
                lengths = [sum(segment[0].shape[1] for segment in cache.segments)
                           for cache in inputs.get("kv_caches", [])]
            result = self.model(**inputs)
            if saved is not None:
                # Keep tensors on device until rollout completes. Immediate CPU
                # fingerprints synchronize calls and can hide ordering defects.
                self.pending.append((self.calls, saved, lengths, result.sample.detach().clone()))
            self.calls += 1
            return result
        row = None
        if groups.get_rank() == 0:
            from h3.modules.masking import CONDITION
            layout = inputs["attention_mask"]
            selected = (layout.kind != CONDITION) & layout.active
            row = dict(call=self.calls, update=bool(inputs.get("update_cache", False)),
                       chunk=int(layout.chunk[selected].max()),
                       inputs=input_fingerprint({key: value for key, value in inputs.items() if key != "kv_caches"}),
                       cache_lengths=[sum(segment[0].shape[1] for segment in cache.segments)
                                      for cache in inputs.get("kv_caches", [])])
        result = self.model(**inputs)
        if row is not None:
            prediction = result.sample.detach().cpu()
            row["prediction"] = input_fingerprint(prediction)
            torch.save(prediction, self.directory / f"call_{self.calls:04d}.pt")
            with (self.directory / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        self.calls += 1
        return result

    def flush(self):
        from h3.modules.masking import CONDITION

        for call, inputs, lengths, prediction in self.pending:
            layout = inputs["attention_mask"]
            selected = (layout.kind != CONDITION) & layout.active
            row = dict(call=call, update=bool(inputs.get("update_cache", False)),
                       chunk=int(layout.chunk[selected].max()), inputs=input_fingerprint(inputs),
                       cache_lengths=lengths, prediction=input_fingerprint(prediction))
            torch.save(prediction.cpu(), self.directory / f"call_{call:04d}.pt")
            with (self.directory / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        self.pending.clear()


def check_repeated_backward(trainer, target_step, output):
    """Measure backward variation before an update, retaining its original gradient/RNG."""
    from utils.checkpoint import rng_state, restore_rng

    original = trainer.objective
    target_steps = {target_step} if isinstance(target_step, int) else set(target_step)

    class RememberInputs:
        repeating = False

        def __getattr__(self, name):
            return getattr(original, name)

        def __call__(self, *arguments, **keywords):
            if trainer.step + 1 in target_steps:
                self.random = rng_state()
                self.arguments, self.keywords = arguments, keywords
            result = original(*arguments, **keywords)
            self.loss = float(result[0].detach())
            return result

    remembered = RememberInputs()
    trainer.objective = remembered
    original_clip = trainer.model.clip_grad_norm_

    def clip(*arguments, **keywords):
        norm = original_clip(*arguments, **keywords)
        if trainer.step + 1 not in target_steps:
            return norm
        after = rng_state()
        gradients = [(parameter, parameter.grad.detach().clone()) for parameter in trainer.model.parameters()
                     if parameter.grad is not None]
        try:
            trainer.optimizer.zero_grad(set_to_none=True)
            restore_rng(remembered.random)
            # Invoke the original objective so the saved primary inputs/RNG stay intact.
            remembered.repeating = True
            repeated_loss, _ = original(*remembered.arguments, **remembered.keywords)
            repeated_loss.backward()
            repeated_norm = original_clip(*arguments, **keywords)
            assert torch.isfinite(repeated_loss) and torch.isfinite(repeated_norm)
            error_sum = reference_sum = 0.
            max_error = 0.
            for parameter, before in gradients:
                assert parameter.grad is not None and parameter.grad.shape == before.shape
                difference = parameter.grad.detach().float() - before.float()
                error_sum += float(difference.double().square().sum())
                reference_sum += float(before.double().square().sum())
                if difference.numel():
                    max_error = max(max_error, float(difference.abs().max()))
            totals = torch.tensor([error_sum, reference_sum], device=trainer.device, dtype=torch.float64)
            maximum = torch.tensor(max_error, device=trainer.device, dtype=torch.float64)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
                torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
            result = dict(step=trainer.step + 1, primary_loss=remembered.loss, repeated_loss=float(repeated_loss),
                          primary_grad_norm=float(norm), repeated_grad_norm=float(repeated_norm),
                          clipped_gradient_relative_l2=float((totals[0] / totals[1].clamp_min(1e-30)).sqrt()),
                          clipped_gradient_max_abs=float(maximum),
                          scope="same weights, document and restored RNG, before any optimizer update")
            if groups.get_rank() == 0:
                (output / "repeated_backward.json").write_text(json.dumps(result, indent=2) + "\n")
                (output / f"repeated_backward_step{trainer.step + 1:09d}.json").write_text(json.dumps(result, indent=2) + "\n")
            assert result["clipped_gradient_relative_l2"] <= .01, result
        finally:
            remembered.repeating = False
            for parameter, before in gradients:
                if parameter.grad is None:
                    parameter.grad = before
                else:
                    parameter.grad.copy_(before)
            restore_rng(after)
        return norm

    trainer.model.clip_grad_norm_ = clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/worldviews.yaml")
    parser.add_argument("--features", type=Path, default=Path("local/overfit_i2v/features"))
    parser.add_argument("--requests", type=Path, default=Path("local/i2v_requests"))
    parser.add_argument("--inputs", type=Path, help="Use unchanged paired SHORT/FULL batch-envelope cases")
    parser.add_argument("--short-case", default="stage1_2x10_iso")
    parser.add_argument("--full-case", default="stage2_4x20")
    parser.add_argument("--trace-inputs", action="store_true", help="Hash packed tensors to diagnose fresh-process drift")
    parser.add_argument("--repeat-backward-step", type=int, default=17,
                        help="Repeat a backward before this optimizer step to measure nondeterminism; 0 disables")
    parser.add_argument("--repeat-backward-steps", type=int, nargs="+",
                        help="Repeat each listed update, overriding --repeat-backward-step")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--inference", action="store_true", help="Check native cached/recomputed rollout after training")
    parser.add_argument("--trace-inference", action="store_true",
                        help="Repeat a failing cache comparison with input/output traces; diagnostics synchronize CPU copies")
    parser.add_argument("--defer-inference-trace", action="store_true",
                        help="Retain call tensors on device and fingerprint only after each complete rollout")
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("The runtime probe requires multiple optimizer updates")
    cfg = load_config(args.config, [
        f"h3.logdir={args.output}", "h3.stage1_steps=2", "h3.short_frames=39",
        f"h3.text_cache={args.output / 'text_cache'}",
        "resampling_forcing_warmup_steps=2", "resampling_forcing_max_depth=2",
        f"max_iters={args.steps}", "save_interval=8", "max_checkpoints=8", "vis_interval=0", "vis_init=false",
        "auto_resume=false", f"generator_cpu_offload={str(not args.resident).lower()}"])
    if args.resume:
        cfg.resume_ckpt = str(args.resume)
    documents = (paired_documents(args.inputs, (args.short_case, args.full_case), cfg) if args.inputs
                 else raw_documents(args.features, args.requests))

    class ProbeStream(training.SourceStream):
        def __init__(self, cfg, video, text, step=0, validation=False):
            self.cfg, self.video, self.text, self.validation = cfg, video, text, validation
            self.pending, self.mixed = deque(), deque()
            self.sampler_generator = torch.Generator().manual_seed(cfg.seed + groups.get_rank())
            self.stage, self.negative = cfg.h3.stage, None

        def next(self):
            if not self.pending and not self.mixed:
                self.pending.append(copy.deepcopy(documents[self.stage - 1]))
            return super().next()

    # Keep the production trainer, feature preparation, SP gather and resume
    # code. Only the raw input iterator is fixed for this bounded runtime check.
    training.SourceStream = ProbeStream
    started = time.monotonic()
    trainer = training.Trainer(cfg)
    if args.trace_inputs:
        original_pack = trainer.objective.pack

        def traced_pack(*arguments, **keywords):
            packed = original_pack(*arguments, **keywords)
            if (groups.get_rank() == 0 and keywords.get("inference") is None
                    and not getattr(trainer.objective, "repeating", False)):
                row = dict(step=trainer.step + 1, stage=trainer.stage,
                           inputs=input_fingerprint(packed[:3]))
                with (args.output / "input_traces.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            return packed

        trainer.objective.pack = traced_pack
    repeated_steps = args.repeat_backward_steps or args.repeat_backward_step
    if repeated_steps:
        check_repeated_backward(trainer, repeated_steps, args.output)
    if trainer.tracker.run:
        trainer.tracker.run.summary.update({"operation": "native-runtime-verification", "full_source_training": False,
                                           "rf_warmup_scope": "two-step functional probe; H3 training warmup undecided"})
    success = False
    report = dict(status="running", phase="initialization", initial_step=trainer.step,
                  scope="full native trainer on fixed source images/video, not full-mixture training")
    persist_report(args.output, report)
    try:
        initial_step = trainer.step
        if args.resume:
            report["restored_state"] = verify_restored_state(trainer, args.resume)
        report["phase"] = "training"
        persist_report(args.output, report)
        trainer.train_loop()
        report.update(phase="inference" if args.inference else "comparison", final_step=trainer.step)
        persist_report(args.output, report)
        inference = None
        if args.inference:
            from pipeline.ar_inference import generate
            from omegaconf import OmegaConf
            local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            local.kv_gpu_budget_gb = 0
            trainer.stream.mixed.clear()
            trainer.stream.pending.clear()
            trainer.stream.set_stage(2)
            document = trainer.stream.next()
            if groups.get_rank() == 0:
                torch.save(document, args.output / "inference_document.pt")

            def compare_cache(weights):
                results = []
                for name, enabled in (("cached", True), ("recomputed", False)):
                    torch.manual_seed(7891)
                    traced = (InferenceTrace(trainer.model, args.output / "inference_deferred" / weights / name,
                                             deferred=True) if args.defer_inference_trace else trainer.model)
                    result = generate(traced, document, None, local, trainer.device, steps=4, use_cache=enabled)
                    if args.defer_inference_trace:
                        traced.flush()
                    results.append(result)
                cached, recomputed = results
                errors = [float((a - b).abs().max()) for a, b in zip(cached, recomputed)]
                report.setdefault("inference_attempts", {})[weights] = dict(
                    cached_vs_recomputed_max_abs_error=errors, atol=.02)
                persist_report(args.output, report)
                if max(errors) > .02 and args.trace_inference:
                    report["inference_original_errors"] = errors
                    report["phase"] = "inference_trace"
                    persist_report(args.output, report)
                    for name, enabled in (("cached", True), ("recomputed", False)):
                        torch.manual_seed(7891)
                        traced = InferenceTrace(trainer.model, args.output / "inference_trace" / weights / name)
                        result = generate(traced, document, None, local, trainer.device, steps=4, use_cache=enabled)
                        if groups.get_rank() == 0:
                            torch.save(result, args.output / "inference_trace" / weights / f"{name}.pt")
                assert max(errors) <= .02, errors
                assert all(torch.isfinite(value).all() for value in (*cached, *recomputed))
                return dict(cpu_history_offload=True, sigma_points=4, views=len(document["views"]),
                            cached_vs_recomputed_max_abs_error=errors, atol=.02)

            inference = compare_cache("raw")
            if trainer.ema is not None:
                with trainer.ema.average_parameters(trainer.model):
                    inference["ema"] = compare_cache("ema")
            report["inference"] = inference
            persist_report(args.output, report)
        groups.shutdown_distributed()
        if groups.get_rank() == 0:
            rows = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
            repeated = args.output / "repeated_backward.json"
            if repeated.exists():
                report["repeated_backward"] = json.loads(repeated.read_text())
            assert rows[-1]["step"] == args.steps
            assert all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in rows)
            summaries = []
            for views in sorted({row["views"] for row in rows}):
                selected = [row for row in rows if row["views"] == views and row["step"] > initial_step + 2]
                if selected:
                    summaries.append(dict(views=views, count=len(selected),
                                          median_seconds=statistics.median(row["seconds"] for row in selected),
                                          max_peak_mib=max(row["vram_max"] for row in selected),
                                          resampling_steps=sum(row["self"] for row in selected)))
            report.update(phase="comparison",
                          full_pretrained=True, sp_size=cfg.sp_size, fs_size=cfg.fs_size,
                          cpu_offload=cfg.generator_cpu_offload, text_cpu_offload=cfg.text_encoder_cpu_offload,
                          checkpointing=cfg.gradient_checkpointing, compiled=cfg.attn_block_compile,
                          ema_enabled=trainer.ema is not None,
                          ema_decay_cap=trainer.ema.decay if trainer.ema else 0.,
                          ema_updates=trainer.ema.num_updates if trainer.ema else 0,
                          input_caption_policy=("paired_ours_bd_overlap" if cfg.h3.get("single_sequence", False)
                                                else "paired_ours_global_chunk" if args.inputs else "source_parquet_caption"),
                          paired_cases=[doc["probe_case"] for doc in documents] if args.inputs else None,
                          rf_warmup_test_steps=2,
                          text_encoded_requests=sum(row.get("text_encoded_requests", 0) for row in rows),
                          source_geometry=[dict(views=len(doc["views"]),
                                                shapes=[list(view["pixels"].shape) for view in doc["views"]])
                                           for doc in documents],
                          initial_step=initial_step, final_step=trainer.step,
                          inference=inference,
                          rf_steps=sum(row["self"] for row in rows), summaries=summaries,
                          seconds=time.monotonic() - started)
            assert report["rf_steps"] > 0, "No RF continuation was exercised"
            if args.compare_to:
                expected = {row["step"]: row for row in map(json.loads, args.compare_to.read_text().splitlines())}
                expected_traces = args.compare_to.parent / "input_traces.jsonl"
                if args.trace_inputs:
                    reference = {row["step"]: row["inputs"] for row in map(json.loads, expected_traces.read_text().splitlines())}
                    actual = [json.loads(line) for line in (args.output / "input_traces.jsonl").read_text().splitlines()]
                    report["packed_input_comparisons"] = [dict(step=row["step"], exact=row["inputs"] == reference[row["step"]])
                                                          for row in actual]
                comparisons = []
                for row in rows:
                    prior = expected[row["step"]]
                    matching_control = all(row[key] == prior[key] for key in ("stage", "self", "sf_depth", "views", "high", "tokens"))
                    errors = {key: abs(row[key] - prior[key]) / max(abs(prior[key]), 1e-12) for key in ("loss", "grad_norm")}
                    comparisons.append(dict(step=row["step"], matching_control=matching_control, **errors))
                report["fresh_process_comparisons"] = comparisons
                persist_report(args.output, report)
                assert all(row["matching_control"] and max(row["loss"], row["grad_norm"]) <= 1e-3
                           for row in comparisons), comparisons
            report.update(status="passed", phase="complete")
            persist_report(args.output, report)
            trainer.tracker.log({"checks/rf_steps": report["rf_steps"],
                                 "checks/text_encoded_requests": report["text_encoded_requests"],
                                 "checks/final_step": trainer.step}, trainer.step)
            if trainer.tracker.run:
                trainer.tracker.run.summary.update({"verification/status": "passed",
                                                    "verification/path": str(args.output / "verification.json")})
        success = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist_report(args.output, report)
        raise
    finally:
        trainer.tracker.finish(success=success)


if __name__ == "__main__":
    main()
