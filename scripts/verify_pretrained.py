#!/usr/bin/env python3
"""Full 33B original-weight training verification on real H3-encoded samples.

Uses one process with intact layers on multiple local GPUs, not cluster jobs.
"""

import runtime_env
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch._dynamo.config

from mvh3.checkpoint import load_original_transformer
from mvh3.execution import place_transformer
from mvh3.optim import MasterAdamW
from mvh3.packing import teacher_forcing_batch
from mvh3.training import attention_parameters, flow_matching_loss, parameter_signature


def log(message):
    print(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), message, flush=True)


def sample_digest(named_tensors):
    digest = hashlib.sha256()
    for name, value in named_tensors:
        flat = value.detach().flatten()
        count = min(512, flat.numel())
        indices = torch.arange(count, device=flat.device) * (flat.numel() - 1) // max(count - 1, 1)
        digest.update(name.encode())
        digest.update(flat[indices].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def prefetch_checkpoint(path, workers):
    """Bound memory while avoiding serial read latency on shared checkpoint storage."""
    if workers == 0:
        return
    size = path.stat().st_size
    chunk = 16 * 1024**2
    descriptor = os.open(path, os.O_RDONLY)

    def read(offset):
        expected = min(chunk, size - offset)
        if len(os.pread(descriptor, expected, offset)) != expected:
            raise IOError("Short checkpoint read")
        return expected

    log(f"Reading {size / 1024**3:.2f} GiB stage checkpoint with {workers} bounded read workers")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            count = 0
            for length in pool.map(read, range(0, size, chunk)):
                count += length
                if count % (16 * 1024**3) == 0:
                    log(f"Checkpoint read: {count / 1024**3:.0f} / {size / 1024**3:.2f} GiB")
    finally:
        os.close(descriptor)


def restore_checkpoint(path, model, optimizer, read_workers=8):
    """Read a real checkpoint and compare every restored tensor, including moments."""
    prefetch_checkpoint(path, read_workers)
    loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if loaded["base_revision"] != "42ed227ee7df40d41602854ae760620d6eb651fe":
        raise ValueError("Stage checkpoint belongs to a different pretrained base")
    if loaded.get("flow_convention") != "h3_t1_clean_data_minus_noise":
        raise ValueError("Stage checkpoint must use the native H3 time and velocity convention")
    if set(loaded["model"]) != set(optimizer.names):
        raise ValueError("Stage checkpoint has a different trainable parameter set")
    optimizer.load_state_dict(loaded["optimizer"])
    for name, parameter, master, saved_master in zip(
        optimizer.names, optimizer.parameters, optimizer.masters, loaded["optimizer"]["masters"], strict=True
    ):
        if not torch.equal(parameter, loaded["model"][name].to(parameter.device)):
            raise AssertionError(f"Saved model/master disagree: {name}")
        if not torch.equal(master, saved_master.to(master.device)):
            raise AssertionError(f"Master restore mismatch: {name}")
    saved_adamw = loaded["optimizer"]["adamw"]
    current_adamw = optimizer.optimizer.state_dict()
    if current_adamw["param_groups"] != saved_adamw["param_groups"]:
        raise AssertionError("Optimizer hyperparameters did not restore")
    for index, saved_state in saved_adamw["state"].items():
        for key, saved in saved_state.items():
            actual = current_adamw["state"][index][key]
            equal = torch.equal(actual, saved.to(actual.device)) if isinstance(saved, torch.Tensor) else actual == saved
            if not equal:
                raise AssertionError(f"Optimizer state restore mismatch: {index}/{key}")
    global_step = loaded["global_step"]
    steps = loaded.get("steps", [])
    if not steps and path.with_name("progress.json").is_file():
        steps = json.loads(path.with_name("progress.json").read_text())
        steps = [step for step in steps if step["global_step"] <= global_step]
    if steps and (steps[-1]["global_step"] != global_step or
                  steps[-1]["master_digest_after"] != sample_digest(zip(optimizer.names, optimizer.masters))):
        raise AssertionError("Checkpoint and recorded stage progress disagree")
    del loaded, saved_adamw, current_adamw
    gc.collect()
    return global_step, steps


@torch.no_grad()
def verify_native_blocks(model):
    """Compare loaded first/middle/last blocks with pinned upstream computation."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    results = []
    for index in (0, len(model.transformer_blocks) // 2, len(model.transformer_blocks) - 1):
        block = model.transformer_blocks[index]
        config = model.config
        with torch.device("meta"):
            upstream = MiniMaxH3TransformerBlock(
                config.hidden_size, config.num_attention_heads, config.attention_head_dim,
                config.ffn_dim, config.time_embed_dim, config.norm_eps, config.qk_norm_eps,
            )
        upstream.load_state_dict(block.state_dict(), strict=True, assign=True)
        device, dtype = block.attn.to_q.weight.device, block.attn.to_q.weight.dtype
        generator = torch.Generator(device=device).manual_seed(13)
        hidden = torch.randn(1, 256, config.hidden_size, device=device, dtype=dtype, generator=generator)
        temb = torch.randn(2, config.time_embed_dim, device=device, dtype=torch.float32, generator=generator)
        indices = torch.arange(256, device=device) % 6
        rope = (torch.ones(256, 96, device=device), torch.zeros(256, 96, device=device))
        original = upstream(hidden, temb, indices, rope)
        adapted = block(hidden, temb, indices, rope)
        if not torch.equal(original, adapted):
            raise AssertionError(f"Loaded block {index} differs from upstream without camera/mask")
        results.append({"block": index, "exact_upstream_parity": True, "device": str(device)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime_env.ROOT.parent / "ckpts/MiniMax-H3/FL2VA")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--short-steps", type=int, default=2)
    parser.add_argument("--long-steps", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="Resume stage 2 in a fresh process from stage1_resume.pt")
    parser.add_argument("--extra-features", type=Path, action="append", default=[], help="Additional real stage-2 shape feature files, one update each")
    parser.add_argument("--checkpoint-read-workers", type=int, default=8, help="Bounded shared-storage readers; 0 disables read-ahead")
    args = parser.parse_args()
    if args.short_steps < 1 or args.long_steps < 1:
        parser.error("Both stages need at least one optimizer update")
    if not 0 <= args.checkpoint_read_workers <= 16:
        parser.error("Checkpoint read workers must be between 0 and 16")
    if not (args.features / "features.json").is_file():
        parser.error("Real video/text feature preparation must finish before verification")
    torch.set_num_threads(8)
    devices = [f"cuda:{int(value)}" for value in args.devices.split(",")]
    # Each placement device and sequence shape specializes both mask and attention.
    variants = 2 * len(devices) * (2 + len(args.extra_features)) + 8
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, variants)
    args.output.mkdir(parents=True, exist_ok=True)
    log("Loading every tensor from the full original 33B checkpoint on CPU")
    model = load_original_transformer(args.checkpoint, progress=log)
    assert sum(p.numel() for p in model.parameters()) == 33_122_992_896
    signature = parameter_signature(model)
    log("Full checkpoint loaded; no missing/meta parameters")
    place_transformer(model, devices)
    native_parity = verify_native_blocks(model)
    log("Loaded first/middle/last blocks exactly match upstream without camera or masking")
    model.train()
    model.enable_gradient_checkpointing()
    attention_parameters(model)
    log("Original layers placed; creating FP32-master AdamW for all 7.71B attention parameters")
    optimizer = MasterAdamW(model.named_parameters(), lr=1e-6)
    frozen_before = sample_digest((name, value) for name, value in model.named_parameters() if not value.requires_grad)
    stages = []
    global_step = 0
    if args.resume:
        log("Restoring stage 1 into the freshly loaded original model and new optimizer")
        global_step, stages = restore_checkpoint(args.resume, model, optimizer, args.checkpoint_read_workers)
        log(f"Every trainable weight, FP32 master, AdamW moment and step restored; global step={global_step}")
    schedule = [("long_multiview", args.long_steps)] if args.resume else [("short_mono", args.short_steps), ("long_multiview", args.long_steps)]
    schedule = [(stage, steps, args.features / f"{stage}.pt") for stage, steps in schedule]
    schedule.extend((path.parent.name, 1, path) for path in args.extra_features)
    for stage, steps, feature_path in schedule:
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        inputs, target, mask = teacher_forcing_batch(features, devices[0], cross_view=stage != "short_mono")
        log(f"{stage}: real latent shape={list(features['latents'].shape)}, packed tokens={len(inputs['token_tags'])}, loss tokens={mask.sum().item()}")
        for local_step in range(steps):
            started = time.monotonic()
            optimizer.zero_grad()
            before = sample_digest(zip(optimizer.names, optimizer.masters))
            model_before = sample_digest(zip(optimizer.names, optimizer.parameters))
            prediction = model(**inputs).sample
            loss = flow_matching_loss(prediction, target.to(prediction.device), mask.to(prediction.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretrained-model loss")
            log(f"{stage} forward step {global_step + 1}: loss={loss.item():.8f}")
            loss.backward()
            log(f"{stage} backward step {global_step + 1}: complete")
            norm = optimizer.step()
            after = sample_digest(zip(optimizer.names, optimizer.masters))
            model_after = sample_digest(zip(optimizer.names, optimizer.parameters))
            if before == after:
                raise AssertionError("The original attention master weights did not update")
            if model_before == model_after:
                raise AssertionError("Updates did not reach the original model weights")
            global_step += 1
            result = {"stage": stage, "global_step": global_step, "loss": loss.item(), "gradient_norm": norm,
                      "seconds": time.monotonic() - started, "master_digest_before": before, "master_digest_after": after,
                      "model_digest_before": model_before, "model_digest_after": model_after,
                      "latent_shape": list(features["latents"].shape), "source_frames": features["source_frames"],
                      "packed_tokens": len(inputs["token_tags"]), "loss_tokens": int(mask.sum()),
                      "gpu_peak_gib": [torch.cuda.max_memory_allocated(device) / 1024**3 for device in devices]}
            stages.append(result)
            (args.output / "progress.json").write_text(json.dumps(stages, indent=2) + "\n")
            log(f"{stage} optimizer step {global_step}: grad_norm={norm:.6f}, {result['seconds']:.1f}s")
            del prediction, loss
        if stage == "short_mono":
            optimizer.zero_grad()
            log("Saving actual trainable weights and complete FP32-master/AdamW state for the stage transition")
            path = args.output / "stage1_resume.pt"
            checkpoint = {
                "model": {name: value.detach() for name, value in model.named_parameters() if value.requires_grad},
                "optimizer": optimizer.state_dict(), "global_step": global_step, "stage": stage,
                "base_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
                "flow_convention": "h3_t1_clean_data_minus_noise",
                "steps": stages,
            }
            temporary = path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(path)
            del checkpoint
            log(f"Saved {path.stat().st_size / 1024**3:.2f} GiB; loading stage transition state")
            restored_step, _ = restore_checkpoint(path, model, optimizer, args.checkpoint_read_workers)
            assert restored_step == global_step
            assert parameter_signature(model) == signature
            assert sample_digest(zip(optimizer.names, optimizer.masters)) == stages[-1]["master_digest_after"]
            log("Stage transition restored all trainable weights, optimizer tensors and global step exactly")
        del features, inputs, target, mask
        optimizer.zero_grad()
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    frozen_after = sample_digest((name, value) for name, value in model.named_parameters() if not value.requires_grad)
    assert frozen_after == frozen_before
    assert parameter_signature(model) == signature
    report = {
        "status": "passed", "scope": "Full original 33B H3; real 448x832 video and Qwen3-VL layer-50 features; single-process layer placement",
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "added_parameters": 0, "devices": devices, "steps": stages, "global_step": global_step,
        "frozen_sample_digest": frozen_after, "optimizer_resume": "passed", "cluster_job_launched": False,
        "fresh_process_resume": args.resume is not None,
        "pretrained_native_block_parity": native_parity,
        "flow_convention": "h3_t1_clean_data_minus_noise",
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    log("Full original-weight real-data training verification passed")


if __name__ == "__main__":
    main()
