#!/usr/bin/env python3
"""Full pretrained local VAE parity and the actual WorldViews encode/decode adapter."""

import argparse
import gc
import json
from pathlib import Path
import sys

import runtime_env
import torch

from h3.data import read_parquet_clip, temporal_layout, pad_video
from utils.config import load_config
from utils.h3_wrapper import VideoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, default=Path("local/real_probe/vae"))
    parser.add_argument("--features", type=Path, default=Path("local/real_probe/features.json"))
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, default=Path("local/local_vae_verification.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    meta = json.loads(args.features.read_text())
    clip = read_parquet_clip(Path(meta["parquet"]), meta["row"], [0], 22)
    pixels = clip["pixels"][0].float().div(255).to(args.device)
    layout = temporal_layout(22)
    mean = torch.tensor([.485, .456, .406], device=args.device).reshape(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225], device=args.device).reshape(1, 3, 1, 1, 1)
    sys.path.insert(0, str(runtime_env.ROOT.parent / "diffusers/src"))
    from diffusers import AutoencoderKLMiniMaxH3 as Reference
    reference = Reference.from_pretrained(args.vae, torch_dtype=torch.float32, local_files_only=True)
    signature = {n: tuple(p.shape) for n, p in reference.named_parameters()}
    reference.to(args.device).eval().requires_grad_(False)
    with torch.no_grad():
        torch.manual_seed(42)
        posterior = reference.encode((pad_video(pixels, layout) - mean) / std).latent_dist
        sampled = posterior.sample().half().float()
        expected_moments = posterior.parameters.cpu()
        expected_decode = (reference.decode(sampled).sample * std + mean).clamp(0, 1).cpu()
        del posterior
    latent_mean = torch.tensor(reference.config.latents_mean).reshape(1, 24, 1, 1, 1)
    latent_std = torch.tensor(reference.config.latents_std).reshape(1, 24, 1, 1, 1)
    expected_latent = (sampled.cpu() - latent_mean) / latent_std
    del reference, sampled
    gc.collect()
    torch.cuda.empty_cache()
    adapter = VideoEncoder(args.vae, args.device)
    assert signature == {n: tuple(p.shape) for n, p in adapter.model.named_parameters()}
    with torch.no_grad():
        moments = adapter.model.encode((pad_video(pixels, layout) - mean) / std).latent_dist.parameters.cpu()
    torch.testing.assert_close(moments, expected_moments, rtol=0, atol=0)
    torch.manual_seed(42)
    latent, timeline, weights = adapter.encode(pixels[0].permute(1, 0, 2, 3).cpu())
    torch.testing.assert_close(latent, expected_latent, rtol=0, atol=0)
    decoded = adapter.decode(latent, 448, 832, 22)
    torch.testing.assert_close(decoded, expected_decode[0].permute(1, 0, 2, 3), rtol=0, atol=0)
    short, _, _ = adapter.encode(pixels[0, :, :1].permute(1, 0, 2, 3).cpu())
    assert adapter.decode(short, 448, 832, 1).shape == (1, 3, 448, 832)
    cfg = load_config("configs/worldviews.yaml")
    pose = clip["pose"][0]
    projection = torch.eye(4).expand(22, 4, 4).clone()
    raw = dict(views=[
        dict(pixels=pixels[0].permute(1, 0, 2, 3).cpu(),
             pose=pose,
             projection=projection,
             inverse=projection,
             fps=16.,
             scale=1.,
             prompt=meta["caption"],
             source_view=0,
             source_start=0)
    ],
               isolated=True,
               source=meta["parquet"])
    prepared = adapter.prepare(raw, cfg)
    assert prepared["views"][0]["projection"].shape == (7, 4, 4, 4)
    assert prepared["views"][0]["condition"] is not None
    report = dict(status="passed",
                  parameters=sum(p.numel() for p in adapter.model.parameters()),
                  source_frames=22,
                  shape=list(latent.shape),
                  moments_max_error=float((moments - expected_moments).abs().max()),
                  adapter_encode_exact=True,
                  adapter_decode_exact=True,
                  single_frame_tail_decode=True,
                  per_latent_subframe_cameras=4,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 1024**3)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
