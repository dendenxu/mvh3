#!/usr/bin/env python3
"""Bounded torchrun validation of actual SP/FSDP, gradients and stage resume."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import faulthandler
import hashlib

import runtime_env
import torch
import torch.distributed as dist

from h3.modules.camera import camera_projection
from h3.checkpoint import load_original_transformer
from utils.config import load_config
from h3.distributed.fsdp import configure_model, wrap_model, compile_blocks, parameter_groups
from model.diffusion import WorldViewsObjective
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils import distributed as groups


def sampled_weights(model, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable and parameter.numel():
            flat = parameter.detach().flatten()
            count = min(32, flat.numel())
            indices = torch.arange(count, device=flat.device) * (flat.numel()-1) // max(count-1, 1)
            digest.update(name.encode())
            digest.update(flat[indices].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_same_state(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_same_state(first, second)
    else:
        assert a == b


def real_document(path, cfg, mono=False):
    from dataset.mvgame import select_pose_stable_factor

    data = torch.load(path, map_location="cpu", weights_only=True)
    views = []
    count = 1 if mono else data["latents"].shape[0]
    poses = data["camera_pose"][:count].float().clone()
    # The older encoder probes store normalized [0,1] principal points and
    # unscaled, world-locked centers. Match the actual WorldViews sampler before
    # exercising matrix PRoPE; otherwise this tests a different camera recipe.
    poses[..., 2:4] -= .5
    scale, diameter = select_pose_stable_factor(poses[..., 7:10].reshape(-1, 3), cfg.dataset.pose_stable_factors)
    poses[..., 7:10] /= scale
    for i in range(count):
        pose = poses[i]
        matrix = camera_projection(pose[None])
        latent = data["latents"][i:i + 1].float()
        frames = data["rotary_frames"]
        condition = None
        if i == 0:
            condition = dict(latent=latent[:, :, :1],
                             frames=frames[:1],
                             valid=data["valid_frames"][:1],
                             pose=pose[:1],
                             projection=matrix.projection[0, :1],
                             inverse=matrix.inverse[0, :1])
        views.append(
            dict(latent=latent,
                 pose=pose,
                 projection=matrix.projection[0],
                 inverse=matrix.inverse[0],
                 condition=condition,
                 frames=frames,
                 valid=data["valid_frames"],
                 spatial_weights=torch.ones(latent.shape[-2] // 2, latent.shape[-1] // 2),
                 fps=float(data["fps"]),
                 scale=scale,
                 prompt="real cached caption",
                 text=data["prompt_embeds"],
                 source_frames=data["source_frames"],
                 height=448,
                 width=832))
    return dict(views=views, isolated=mono, source=str(path), pose_diameter=diameter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--features", type=Path, default=Path("local/real_probe"))
    parser.add_argument("--extra-features", type=Path, action="append", default=[])
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume the stage-1 checkpoint in a fresh process")
    parser.add_argument("--compare-to", type=Path, help="Compare resumed loss exactly and BF16 gradient norm at rtol=1e-3")
    args = parser.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True)
    torch.set_num_threads(int(os.environ.get("WORLDGEN_TORCH_NUM_THREADS", "1")))
    world = int(os.environ["WORLD_SIZE"])
    groups.launch_distributed_job(sp_size_arg=world, fs_size_arg=world)
    device = torch.device("cuda", torch.cuda.current_device())
    cfg = load_config("configs/worldviews.yaml")
    cfg.h3.checkpoint, cfg.h3.vae = str(args.checkpoint or "/unused/h3"), "/unused/vae"
    cfg.fs_size = cfg.sp_size = world
    cfg.attn_block_compile = args.compile
    cfg.model.fa4 = args.full
    cfg.mixed_precision = args.full
    cfg.resampling_forcing_warmup_steps = 0
    cfg.h3.logdir = str(args.output)
    torch.manual_seed(42)
    if args.full:
        model = load_original_transformer(args.checkpoint, progress=print if dist.get_rank() == 0 else None)
        documents = [
            real_document(args.features / "short_mono.pt", cfg, True),
            real_document(args.features / "long_multiview.pt", cfg)
        ]
        documents.extend(real_document(path, cfg) for path in args.extra_features)
    else:
        sys.path.insert(0, str(runtime_env.ROOT / "tests"))
        from fixtures_h3 import feature_document
        from h3.modules.model import MiniMaxH3Transformer3DModel
        model = MiniMaxH3Transformer3DModel(num_attention_heads=world,
                                            attention_head_dim=128,
                                            hidden_size=32,
                                            num_layers=3,
                                            num_refiner_layers=1,
                                            ffn_dim=64,
                                            in_channels=24,
                                            audio_in_channels=32,
                                            patch_size=(1, 2, 2),
                                            text_dim=32,
                                            freq_dim=32,
                                            time_embed_hidden_dim=32,
                                            time_embed_dim=16,
                                            rope_freq_dim=16)
        documents = [feature_document(), feature_document(views=2)]
    signature = configure_model(model, cfg)
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reference = copy.deepcopy(model).to(device) if not args.full else None
    model = wrap_model(model, cfg)
    compile_blocks(model.module, cfg)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg),
                                  betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay,
                                  fused=True)
    objective = WorldViewsObjective(cfg)
    report = dict(status="running",
                  world_size=world,
                  sp_size=world,
                  fs_size=world,
                  full_pretrained=args.full,
                  compiled=args.compile,
                  zero_added_parameters=True,
                  steps=[],
                  total_parameters=total_parameters,
                  trainable_parameters=trainable_parameters)
    first_step = 0
    if args.resume:
        state = load_checkpoint(model, optimizer, cfg, args.resume)
        assert state["step"] == 1 and state["stage"] == 1
        assert_same_state(state["optimizer"], optimizer.state_dict())
        from h3.distributed.fsdp import canonical_name
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert torch.equal(parameter.cpu(), state["weights"][canonical_name(name)])
        first_step, documents = state["step"], documents[1:]
        report["fresh_process_resume"] = True
        report["optimizer_resume_exact"] = True
        del state
    expected_report = json.loads(args.compare_to.read_text()) if args.compare_to else None
    frozen_digest = sampled_weights(model, False)
    args.output.mkdir(parents=True, exist_ok=True)
    for step, document in enumerate(documents, first_step):
        start = time.monotonic()
        before_digest = sampled_weights(model, True)
        inputs, target, weights, records, high = objective.pack(document, device, step)
        optimizer.zero_grad(set_to_none=True)
        if reference is not None and step == 0:
            expected = reference(**inputs).sample
            ref_loss = ((expected - target).square().mean(-1)[0] * weights).sum() / weights.sum()
            ref_loss.backward()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg.clip_grad_norm)
        actual = model(**inputs).sample
        loss = ((actual - target).square().mean(-1)[0] * weights).sum() / weights.sum()
        zero_loss = (target.float().square().mean(-1)[0] * weights).sum() / weights.sum()
        # Finite values alone miss catastrophic conditioning amplitudes.
        # Allow substantial adaptation error, but fail above 10x target RMS.
        assert loss.detach() <= 100 * zero_loss, f"Prediction error exceeds the signal scale: {float(loss.detach())} / {float(zero_loss)}"
        loss.backward()
        norm = model.clip_grad_norm_(cfg.clip_grad_norm)
        if reference is not None and step == 0:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(norm.cpu(), expected_norm.cpu(), rtol=2e-4, atol=2e-4)
        assert torch.isfinite(loss) and torch.isfinite(norm) and norm > 0
        optimizer.step()
        assert sampled_weights(model, True) != before_digest
        assert sampled_weights(model, False) == frozen_digest
        row = dict(step=step + 1,
                   views=len(document["views"]),
                   loss=float(loss.detach()),
                   zero_prediction_loss=float(zero_loss),
                   grad_norm=float(norm),
                   high_noise=high,
                   sigma_min=float((1 - inputs["timestep"]).min()),
                   sigma_max=float((1 - inputs["timestep"]).max()),
                   pose_stable_factor=document["views"][0]["scale"],
                   tokens=len(inputs["token_tags"]),
                   seconds=time.monotonic() - start)
        report["steps"].append(row)
        if expected_report is not None:
            expected_row = next(item for item in expected_report["steps"] if item["step"] == row["step"])
            for key in ("loss", "zero_prediction_loss", "high_noise", "tokens"):
                assert row[key] == expected_row[key], (key, row[key], expected_row[key])
            # BF16 attention backward is not promised to be bitwise deterministic
            # across fresh compiled processes. Keep the measured error visible.
            error = abs(row["grad_norm"] - expected_row["grad_norm"]) / max(abs(expected_row["grad_norm"]), 1e-12)
            assert error <= 1e-3, ("resume gradient relative error", error)
            report["continuous_resume_loss_exact"] = True
            report["resume_gradient_relative_error"] = error
            report["resume_gradient_rtol"] = 1e-3
        if dist.get_rank() == 0:
            print(json.dumps(row), flush=True)
            (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        if step == 0:
            path = save_checkpoint(model, optimizer, cfg, 1, 1, {}, args.output / "ckpt")
            saved_optimizer = copy.deepcopy(optimizer.state_dict())
            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            for p in model.parameters():
                if p.requires_grad:
                    with torch.no_grad():
                        p.zero_()
            state = load_checkpoint(model, optimizer, cfg, path)
            assert_same_state(saved_optimizer, optimizer.state_dict())
            assert state["step"] == 1 and state["stage"] == 1
            for n, p in model.named_parameters():
                if p.requires_grad:
                    assert torch.equal(before[n], p)
            report["stage_resume_exact"] = True
            report["optimizer_resume_exact"] = True
            del before, saved_optimizer, state
        del actual, loss, inputs, target, weights
    if args.inference:
        from pipeline.ar_inference import generate
        document = documents[-1]
        negative = document["views"][0]["text"] * .5
        torch.manual_seed(77)
        cached = generate(model, document, negative, cfg, device, steps=3, use_cache=True)
        torch.manual_seed(77)
        uncached = generate(model, document, negative, cfg, device, steps=3, use_cache=False)
        error = max(float((a - b).abs().max()) for a, b in zip(cached, uncached))
        for a, b in zip(cached, uncached):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)
        report["kv_recompute_max_error"] = error
    report["frozen_weight_samples_unchanged"] = True
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    rank = dist.get_rank()
    print(f"Rank {dist.get_rank()}: verification complete, synchronizing", flush=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    print(f"Rank {dist.get_rank()}: destroying process groups", flush=True)
    groups.shutdown_distributed()
    faulthandler.cancel_dump_traceback_later()
    report["status"] = "passed"
    if rank == 0:
        (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
