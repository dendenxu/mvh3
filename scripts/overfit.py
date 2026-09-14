#!/usr/bin/env python3
"""Overfit the full released H3 on fixed real documents with held-out noise."""

import os
import json
import math
import time
import shutil
import hashlib
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.ema import ShardedEMA
from utils.random import set_seed
from utils.tracking import Tracker
from utils import distributed as groups
from utils.distributed import canonical_name
from model.diffusion import DiffusionObjective
from trainer.diffusion import parameter_groups
from pipeline.chunked_inference import generate
from h3.modules.model import MiniMaxH3Transformer3DModel
from h3.distributed.fsdp import wrap_model, compile_blocks
from utils.config import load_config, recipe_digest, validate_config
from utils.checkpoint import rng_state, restore_rng, load_checkpoint, save_checkpoint


def sampled_weights(model, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable and parameter.numel():
            flat = parameter.detach().flatten()
            count = min(32, flat.numel())
            indices = torch.arange(count, device=flat.device) * (flat.numel() - 1) // max(count - 1, 1)
            digest.update(name.encode())
            digest.update(flat[indices].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def exact_state(actual, saved):
    if isinstance(saved, torch.Tensor):
        return (
            isinstance(actual, torch.Tensor)
            and actual.dtype == saved.dtype
            and actual.shape == saved.shape
            and torch.equal(actual.detach().cpu(), saved.cpu())
        )
    if isinstance(saved, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == saved.keys()
            and all(exact_state(actual[key], saved[key]) for key in saved)
        )
    if isinstance(saved, (tuple, list)):
        return (
            type(actual) is type(saved)
            and len(actual) == len(saved)
            and all(exact_state(a, b) for a, b in zip(actual, saved))
        )
    return type(actual) is type(saved) and actual == saved


def restored_state_checks(model, optimizer, ema, saved):
    parameters = {
        canonical_name(name): value for name, value in model.named_parameters() if value.requires_grad
    }
    return dict(
        raw=exact_state(parameters, saved["weights"]),
        adamw=exact_state(optimizer.state_dict(), saved["optimizer"]),
        ema=exact_state(ema.state_dict() if ema is not None else None, saved["ema"]),
    )


def resume_evaluation_decision(raw_error, ema_error, reason=None, exact_state_by_rank=None):
    errors = [raw_error] + ([] if ema_error is None else [ema_error])
    finite = all(math.isfinite(value) and value >= 0 for value in errors)
    passed = finite and all(value <= 1e-3 for value in errors)
    diagnostic = isinstance(reason, str) and bool(reason.strip())
    state_exact = bool(exact_state_by_rank) and all(
        all(row.get(key) is True for key in ("raw", "adamw", "ema")) for row in exact_state_by_rank
    )
    return dict(
        status="passed" if passed else "failed",
        relative_tolerance=1e-3,
        continue_training=(passed if reason is None else finite and diagnostic and state_exact),
        diagnostic_only=diagnostic,
        diagnostic_reason=reason,
    )


def measure(model, objective, document, device, sigma):
    inputs, target, weights, _, _ = objective.pack(document, device, evaluation_sigma=sigma)
    prediction = model(**inputs).sample
    loss = ((prediction.float() - target.float()).square().mean(-1)[0] * weights).sum() / weights.sum()
    return float(loss)


@torch.no_grad()
def evaluate(model, objective, documents, cfg, device):
    rng, training = rng_state(), model.training
    model.eval()
    rows = []
    try:
        for index, document in enumerate(documents):
            for j, sigma in enumerate(cfg.overfit.eval_sigmas):
                set_seed(cfg.overfit.eval_seed + index * 100 + j)
                loss = measure(model, objective, document, device, sigma)
                if not torch.isfinite(torch.tensor(loss)):
                    raise FloatingPointError("Nonfinite fixed-noise evaluation")
                rows.append(dict(sample=index, sigma=float(sigma), loss=loss))
    finally:
        restore_rng(rng)
        model.train(training)
    return dict(mean_loss=sum(row["loss"] for row in rows) / len(rows), entries=rows)


def generate_samples(model, documents, cfg, device, directory, label, tracker, step):
    state = rng_state()
    try:
        planned = []
        for index, document in enumerate(documents):
            set_seed(cfg.overfit.generation_seed + index)
            document = DiffusionObjective(cfg).prepare_document(document, device, training=False)
            planned.append(document)
            if groups.get_rank() == 0:
                print(
                    f"Generating {label} sample {index}: {cfg.sampling_steps} native sigma points per chunk",
                    flush=True,
                )
            outputs = generate(model, document, None, cfg, device)
            if not all(torch.isfinite(value).all() for value in outputs):
                raise FloatingPointError("Nonfinite generated latents")
            if groups.get_rank() == 0:
                target = directory / f"{label}_{index}.pt"
                temporary = target.with_suffix(".tmp")
                torch.save(outputs, temporary)
                temporary.replace(target)
                write_json(
                    directory / f"{label}_{index}.json",
                    dict(
                        step=step,
                        seed=cfg.overfit.generation_seed + index,
                        views=[
                            dict(
                                fps=view["fps"],
                                frames=view["source_frames"],
                                generation_chunks=(
                                    view["generation_chunks"].tolist()
                                    if "generation_chunks" in view
                                    else None
                                ),
                                captions=view.get("caption_specs"),
                            )
                            for view in document["views"]
                        ],
                    ),
                )
                print(f"Saved {target}", flush=True)
        failed = torch.zeros((), dtype=torch.int32, device=device)
        if groups.get_rank() == 0:
            try:
                from utils.visualization import write_overfit_generation

                media = write_overfit_generation(cfg, planned, directory, label, str(device))
                tracker.media(media, step, "overfit/generation")
            except Exception as error:
                print(f"Scheduled generation output failed: {error}", flush=True)
                failed.fill_(1)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError("Scheduled generation decoding or tracking failed")
    finally:
        restore_rng(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overfit_diffusion_forcing.yaml")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stop-after", type=int, help="Save and exit at this step for a fresh-process resume check"
    )
    parser.add_argument(
        "--ema-schedule-change",
        metavar="REASON",
        help="Explicitly change only the EMA schedule while retaining its saved weights and count",
    )
    parser.add_argument(
        "--diagnostic-resume-after-eval-drift",
        metavar="REASON",
        help="Continue bounded overfit after exact state checks; retain failed evaluation acceptance",
    )
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    cfg = validate_config(load_config(args.config, args.opts))
    if cfg.resampling_forcing and cfg.max_iters > cfg.resampling_forcing_warmup_steps:
        raise ValueError(
            "Use the production Trainer for resampling forcing continuation; this fixed-source probe ends during warmup"
        )
    cfg.h3.logdir = str(args.output)
    resume = cfg.resume_ckpt
    latest = args.output / "ckpt" / "latest.json"
    if not resume and cfg.auto_resume and latest.is_file():
        resume = str(latest)
    diagnostic_reason = args.diagnostic_resume_after_eval_drift
    if diagnostic_reason is not None and (
        not diagnostic_reason.strip()
        or not resume
        or args.stop_after is None
        or args.stop_after > cfg.max_iters
        or args.ema_schedule_change is not None
    ):
        parser.error(
            "Diagnostic resume requires a reason, checkpoint, explicit bounded --stop-after, and unchanged EMA"
        )
    if resume and args.stop_after is not None:
        checkpoint = Path(resume)
        if checkpoint.name == "latest.json" and checkpoint.is_file():
            checkpoint = checkpoint.parent / json.loads(checkpoint.read_text())["path"]
        committed_step = json.loads((checkpoint / "manifest.json").read_text())["step"]
        if committed_step >= args.stop_after:
            # The launcher runs a bounded phase before its normal resume.
            # Do not repeat full training when that phase is already saved.
            print(
                f"Committed step {committed_step} already reaches --stop-after {args.stop_after}", flush=True
            )
            return
    torch.set_num_threads(int(os.environ.get("WORLDGEN_TORCH_NUM_THREADS", "1")))
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    if groups.get_sp_size() != cfg.sp_size or groups.fs_size != cfg.fs_size:
        raise ValueError("Launch the requested SP/FSDP topology")
    device = torch.device("cuda", torch.cuda.current_device())
    rank = groups.get_rank()
    set_seed(cfg.seed)
    from utils.camera import prepare_camera_geometry

    documents = [
        prepare_camera_geometry(doc, cfg)
        for doc in torch.load(args.features / "documents.pt", map_location="cpu", weights_only=True)
    ]
    metadata = json.loads((args.features / "features.json").read_text())
    feature_sha256 = hashlib.sha256((args.features / "documents.pt").read_bytes()).hexdigest()
    assert metadata["status"] == "complete"
    for document in documents:
        assert document["isolated"] and len(document["views"]) == 1
        view = document["views"][0]
        assert view["fps"] == cfg.dataset.model_fps and view["source_frames"] == cfg.overfit.frames
        assert (view["height"], view["width"]) == (cfg.dataset.height, cfg.dataset.width)
    if cfg.h3.get("text_conditioning", "text_only") != metadata.get("text_conditioning", "text_only"):
        raise ValueError("Feature text/image conditioning differs from the configured encoder")
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        OmegaConf.save(cfg, args.output / "resolved.yaml")
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        cfg.h3.checkpoint, progress=print if rank == 0 else None
    )
    model.configure_attention(cfg)
    report = dict(
        status="running",
        recipe=recipe_digest(cfg),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        world_size=dist.get_world_size(),
        source=metadata,
        feature_sha256=feature_sha256,
        evaluations=[],
        training=[],
    )
    assert report["total_parameters"] == 33122992896
    assert report["trainable_parameters"] == 3853523200
    model = wrap_model(model, cfg)
    compile_blocks(model.module, cfg)
    ema = ShardedEMA.from_config(model, cfg)
    report["ema_enabled"] = ema is not None
    optimizer = torch.optim.AdamW(
        parameter_groups(model, cfg), betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, fused=True
    )
    if cfg.optim_compile:
        optimizer.step = torch.compile(optimizer.step)
    objective = DiffusionObjective(cfg)
    frozen, initial = sampled_weights(model, False), sampled_weights(model, True)
    first_step = 0
    exact_checks = None
    if resume:
        restored = load_checkpoint(
            model, optimizer, cfg, resume, ema=ema, ema_schedule_change=args.ema_schedule_change
        )
        first_step = restored["step"]
        report = restored["runtime"]["overfit_report"]
        if report["feature_sha256"] != feature_sha256 or report["source"] != metadata:
            raise ValueError("Overfit resume requires the identical fixed source features")
        if diagnostic_reason is not None:
            if restored["recipe"] != recipe_digest(cfg):
                raise ValueError("Diagnostic resume requires the identical checkpoint recipe")
            checks = restored_state_checks(model, optimizer, ema, restored)
            exact_checks = [None] * dist.get_world_size()
            dist.all_gather_object(exact_checks, checks)
            passed = all(all(row.values()) for row in exact_checks)
            if rank == 0:
                write_json(
                    args.output / f"diagnostic_state_{first_step:09d}.json",
                    dict(
                        status="passed" if passed else "failed",
                        step=first_step,
                        exact_state_by_rank=exact_checks,
                    ),
                )
            if not passed:
                raise AssertionError(f"Diagnostic resume state is not exact: {exact_checks}")
            if not report["evaluations"] or report["evaluations"][-1]["step"] != first_step:
                raise ValueError("Diagnostic resume requires the checkpoint's fixed-noise evaluation")
            checkpoint = Path(resume)
            if checkpoint.name == "latest.json":
                checkpoint = checkpoint.parent / json.loads(checkpoint.read_text())["path"]
            report.setdefault("diagnostic_origin_checkpoints", []).append(str(checkpoint.resolve()))
            report.update(diagnostic_only=True, production_acceptance=False)
            previous_receipt = args.output / f"resume_{first_step:09d}.json"
            if previous_receipt.is_file():
                previous_check = json.loads(previous_receipt.read_text())
                if previous_check["status"] == "failed" and previous_check not in report.get(
                    "resume_checks", []
                ):
                    report.setdefault("resume_checks", []).append(previous_check)
        if "ema_decay_change" in restored:
            report.setdefault("initial_recipe", report["recipe"])
            report.setdefault("ema_decay_changes", []).append(restored["ema_decay_change"])
            report["recipe"] = recipe_digest(cfg)
        if "ema_schedule_change" in restored:
            report.setdefault("initial_recipe", report["recipe"])
            report.setdefault("ema_schedule_changes", []).append(restored["ema_schedule_change"])
            report["recipe"] = recipe_digest(cfg)
        del restored
        if rank == 0:
            print(f"Resumed model and AdamW at step {first_step} from {resume}", flush=True)
    if first_step > cfg.max_iters:
        raise ValueError("The checkpoint exceeds the requested final step")
    tracker = Tracker(cfg, args.output, first_step)
    if tracker.run and report.get("ema_schedule_changes"):
        tracker.run.summary.update({"ema/schedule_changes": report["ema_schedule_changes"]})

    success = False
    try:

        def record_evaluation(step):
            value = evaluate(model, objective, documents, cfg, device)
            if ema is not None:
                with ema.average_parameters(model):
                    averaged = evaluate(model, objective, documents, cfg, device)
                value.update(ema_mean_loss=averaged["mean_loss"], ema_entries=averaged["entries"])
            value["step"] = step
            report["evaluations"].append(value)
            tracker.evaluation(value)
            if rank == 0:
                print(json.dumps(dict(evaluation=value)), flush=True)
                write_json(args.output / "progress.json", report)

        def save_progress(step):
            checkpoint = save_checkpoint(
                model, optimizer, cfg, step, 1, dict(overfit_report=report), args.output / "ckpt", ema=ema
            )
            report["checkpoint"] = str(checkpoint)
            tracker.checkpoint(checkpoint, step)
            if rank == 0:
                print(f"Checkpoint committed at step {step}: {checkpoint}", flush=True)
                write_json(args.output / "progress.json", report)
                complete = sorted(p.parent for p in (args.output / "ckpt").glob("step_*/manifest.json"))
                for old in complete[: -int(cfg.max_checkpoints)]:
                    if old != checkpoint and str(old.resolve()) not in report.get(
                        "diagnostic_origin_checkpoints", []
                    ):
                        shutil.rmtree(old)
            return checkpoint

        if first_step == 0:
            record_evaluation(0)
            generate_samples(model, documents, cfg, device, args.output, "before", tracker, 0)
        elif report["evaluations"][-1]["step"] == first_step:
            repeated = evaluate(model, objective, documents, cfg, device)
            previous = report["evaluations"][-1]["mean_loss"]
            relative_error = abs(repeated["mean_loss"] - previous) / max(previous, 1e-12)
            ema_error, repeated_ema = None, None
            if ema is not None:
                with ema.average_parameters(model):
                    repeated_ema = evaluate(model, objective, documents, cfg, device)
                previous_ema = report["evaluations"][-1]["ema_mean_loss"]
                ema_error = abs(repeated_ema["mean_loss"] - previous_ema) / max(previous_ema, 1e-12)
            decision = resume_evaluation_decision(relative_error, ema_error, diagnostic_reason, exact_checks)

            # Preserve strict acceptance separately from this bounded convergence experiment.
            continuation = torch.tensor(int(decision["continue_training"]), device=device)
            dist.all_reduce(continuation, op=dist.ReduceOp.MIN)
            decision["continue_training"] = bool(continuation.item())
            receipt = dict(
                **decision,
                step=first_step,
                previous_loss=previous,
                restored_loss=repeated["mean_loss"],
                relative_error=relative_error,
                ema_relative_error=ema_error,
                ema_updates=ema.num_updates if ema else 0,
                ema_decay_cap=ema.decay if ema else 0.0,
                ema_warmup=ema.warmup if ema else False,
                previous_evaluation=report["evaluations"][-1],
                restored_raw_evaluation=repeated,
                restored_ema_evaluation=repeated_ema,
            )
            report.setdefault("resume_checks", []).append(receipt)
            if rank == 0:
                name = (
                    f"resume_{first_step:09d}"
                    if diagnostic_reason is None
                    else f"diagnostic_resume_{first_step:09d}"
                )
                write_json(args.output / f"{name}.json", receipt)
                if diagnostic_reason is not None:
                    print(json.dumps(dict(diagnostic_resume=receipt)), flush=True)
                    if tracker.run:
                        tracker.run.summary.update(
                            {
                                "diagnostic_only": True,
                                "production_acceptance": False,
                                "resume/evaluation_passed": all(
                                    row["status"] == "passed" for row in report["resume_checks"]
                                ),
                                "resume/diagnostic_reason": diagnostic_reason,
                                "resume/exact_state_by_rank": exact_checks,
                            }
                        )
            if not decision["continue_training"]:
                raise AssertionError(
                    f"Fresh-process restored evaluation differs: raw={relative_error}, "
                    f"ema={ema_error}, tolerance=0.001"
                )
            if rank == 0:
                report["status"] = "running"
                write_json(args.output / "progress.json", report)
        model.train()
        for step in range(first_step, cfg.max_iters):
            started = time.monotonic()

            # A disjoint seed stream makes evaluation repeatable without ever
            # training on its Gaussian noise or changing future training draws.
            set_seed(cfg.seed + step)
            document = objective.prepare_document(documents[step % len(documents)], device)
            optimizer.zero_grad(set_to_none=False)
            loss, log = objective.compute_loss(model, document, device, step)
            fwd_mem = torch.cuda.memory_allocated() // 1024**2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            norm = model.clip_grad_norm_(cfg.clip_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradients at step {step}")
            if cfg.warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * min(1.0, (step + 1) / cfg.warmup_steps)
            optimizer.step()
            ema_started = time.monotonic()
            if ema is not None:
                ema.update(model)
            ema_seconds = time.monotonic() - ema_started
            row = dict(
                step=step + 1,
                sample=step % len(documents),
                loss=float(loss.detach()),
                grad_norm=float(norm),
                sigma=log["sigma"],
                tokens=log["tokens"],
                sigma_min=log["sigma_min"],
                sigma_max=log["sigma_max"],
                fwd_mem=fwd_mem,
                ema_seconds=ema_seconds,
                ema_updates=ema.num_updates if ema else 0,
                ema_decay=ema.current_decay if ema else 0.0,
                seconds=time.monotonic() - started,
            )
            row.update(
                {
                    key: log[key]
                    for key in (
                        "chunk_size_min",
                        "chunk_size_max",
                        "chunk_count",
                        "clean_prefix_chunks",
                        "clean_video_tokens",
                        "supervised_video_tokens",
                    )
                    if key in log
                }
            )
            report["training"].append(row)
            tracker.training(row, optimizer, document)
            if rank == 0:
                print(json.dumps(row), flush=True)
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            del loss, log
            if (step + 1) % cfg.overfit.eval_interval == 0 or step + 1 == cfg.max_iters:
                record_evaluation(step + 1)
            if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0 and step + 1 < cfg.max_iters:
                save_progress(step + 1)
            if (
                cfg.overfit.get("generation_interval", 0)
                and (step + 1) % cfg.overfit.generation_interval == 0
                and step + 1 < cfg.max_iters
            ):
                generate_samples(
                    model, documents, cfg, device, args.output, f"step{step + 1:09d}", tracker, step + 1
                )
            if args.stop_after == step + 1 and step + 1 < cfg.max_iters:
                if not cfg.save_interval or (step + 1) % cfg.save_interval:
                    save_progress(step + 1)
                success = True
                groups.shutdown_distributed()
                return
        assert sampled_weights(model, False) == frozen
        assert sampled_weights(model, True) != initial
        report["frozen_samples_unchanged"] = True
        save_progress(cfg.max_iters)
        generate_samples(model, documents, cfg, device, args.output, "after", tracker, cfg.max_iters)
        if ema is not None:
            with ema.average_parameters(model):
                generate_samples(
                    model, documents, cfg, device, args.output, "after_ema", tracker, cfg.max_iters
                )
        baseline, final = report["evaluations"][0], report["evaluations"][-1]
        improvement = 1 - final["mean_loss"] / baseline["mean_loss"]
        report["relative_improvement"] = improvement
        report["loss_criterion_passed"] = improvement >= cfg.overfit.minimum_relative_improvement
        if ema is not None:
            report["ema_relative_improvement"] = 1 - final["ema_mean_loss"] / baseline["ema_mean_loss"]
            report["ema_loss_criterion_passed"] = (
                report["ema_relative_improvement"] >= cfg.overfit.minimum_relative_improvement
            )
        report["peak_allocated_gib"] = (
            max(tracker.peak_allocated, torch.cuda.max_memory_allocated()) / 1024**3
        )
        report["status"] = "training_and_generation_complete"
        if rank == 0:
            write_json(args.output / "training_report.json", report)
            from utils.visualization import write_overfit_comparison

            review = write_overfit_comparison(cfg, args.features, args.output, str(device))
            tracker.media(args.output / "review", cfg.max_iters, "overfit")
            if tracker.run:
                tracker.run.summary.update(
                    {
                        "relative_improvement": improvement,
                        "loss_criterion_passed": report["loss_criterion_passed"],
                        "frozen_samples_unchanged": True,
                        "review/path": str((args.output / "review").resolve()),
                    }
                )
            write_json(
                args.output / "completion.json",
                dict(
                    status="complete",
                    training=report["status"],
                    review=review["status"],
                    diagnostic_only=report.get("diagnostic_only", False),
                    resume_evaluation_passed=all(
                        row["status"] == "passed" for row in report.get("resume_checks", [])
                    ),
                ),
            )
            write_json(args.output / "progress.json", report)
            print(json.dumps(dict(status=report["status"], relative_improvement=improvement)), flush=True)
            success = True
        groups.shutdown_distributed()
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            completed_training_step=report["training"][-1]["step"] if report["training"] else first_step,
        )
        if rank == 0:
            write_json(args.output / "progress.json", report)
        raise
    finally:
        tracker.finish(success=success)


if __name__ == "__main__":
    main()
