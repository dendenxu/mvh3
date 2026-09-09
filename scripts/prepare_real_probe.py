#!/usr/bin/env python3
"""Encode real source videos with the full released H3 VAE and Qwen3-VL weights."""

import runtime_env
import argparse
from datetime import datetime, timezone, timedelta
import gc
import json
from pathlib import Path

import torch

from mvh3.data import pad_video, read_parquet_clip, temporal_layout


def log(message):
    print(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), message, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=runtime_env.ROOT.parent / "ckpts/MiniMax-H3/FL2VA")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--views", default="0,1")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--short-frames", type=int, default=77)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-text-from", type=Path, help="Directory of prior full-encoder features for the same source caption")
    parser.add_argument("--vae-cache", type=Path, help="Reuse the already converted video VAE for this checkpoint")
    args = parser.parse_args()
    if not 1 <= args.short_frames <= min(args.frames, 77):
        parser.error("Short frames must be between 1 and min(frames, 77)")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    views = [int(value) for value in args.views.split(",")]
    clip = read_parquet_clip(args.parquet, args.row, views, args.frames)
    log(f"Decoded {len(views)} real views: {[list(p.shape) for p in clip['pixels']]}")

    from diffusers import AutoencoderKLMiniMaxH3, __version__ as diffusers_version
    from mvh3.vendor.convert_minimax_h3 import MINIMAX_H3_VIDEO_VAE_CONFIG, convert_video_vae

    converted = args.vae_cache or args.output / "vae"
    if not (converted / "config.json").exists():
        log("Converting the original full video VAE")
        convert_video_vae(str(args.checkpoint), str(converted), MINIMAX_H3_VIDEO_VAE_CONFIG, diffusers_version, 2 * 1024**3)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(converted, torch_dtype=torch.float32, local_files_only=True).to(args.device).eval()
    vae.requires_grad_(False)
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 24, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).reshape(1, 24, 1, 1, 1)
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).reshape(1, 3, 1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], device=args.device).reshape(1, 3, 1, 1, 1)
    encoded = {}
    for stage, frames, selected in (("short_mono", args.short_frames, [0]), ("long_multiview", args.frames, list(range(len(views))))):
        layout = temporal_layout(frames)
        log(f"Encoding {stage}: real={frames}, padded={layout.padded_frames}, latents={len(layout.valid)}, valid={layout.valid.sum().item()}")
        latents = []
        for view in selected:
            pixels = pad_video(clip["pixels"][view][:, :, :frames], layout).to(args.device)
            pixels = (pixels.float() / 255 - pixel_mean) / pixel_std
            with torch.no_grad():
                posterior = vae.encode(pixels).latent_dist
                latent = posterior.sample(generator=torch.Generator(device=args.device).manual_seed(42))
            latent = (latent.half().float().cpu() - mean) / std
            if latent.shape[2] != len(layout.valid) or not torch.isfinite(latent).all():
                raise ValueError("VAE output does not match its actual temporal layout")
            latents.append(latent)
            log(f"Encoded view {views[view]}: {tuple(latent.shape)}")
        encoded[stage] = {
            "latents": torch.cat(latents, dim=0), "camera_pose": clip["pose"][selected][:, layout.camera_frames],
            "rotary_frames": layout.rotary_frames, "valid_frames": layout.valid,
            "source_frames": frames, "padded_frames": layout.padded_frames, "fps": clip["fps"],
        }
        torch.save(encoded[stage], args.output / f"{stage}.pt")
    del vae, posterior, pixels
    gc.collect()
    torch.cuda.empty_cache()
    if args.reuse_text_from:
        report = json.loads((args.reuse_text_from / "features.json").read_text())
        if (report["status"] != "complete" or report["caption"] != clip["caption"] or
                Path(report["checkpoint"]).resolve() != args.checkpoint.resolve() or report["text_layer"] != 50):
            raise ValueError("Cached text must come from the same full encoder and exact source caption")
        prompt = torch.load(args.reuse_text_from / "short_mono.pt", map_location="cpu", weights_only=True)["prompt_embeds"]
        log("Reused full-encoder features for the identical source caption")
    else:
        log("Loading the full Qwen3-VL-32B text encoder")
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration

        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.checkpoint / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True,
            device_map={"": args.device}, attn_implementation="sdpa",
        ).eval()
        encoder.requires_grad_(False)
        tokenizer = Qwen2TokenizerFast.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
        token_ids = tokenizer(clip["caption"], add_special_tokens=False)["input_ids"]
        if any(value in token_ids for value in (encoder.config.image_token_id, encoder.config.video_token_id)):
            raise ValueError("This text-only verification entry does not accept vision placeholder tokens")
        input_ids = torch.tensor([token_ids], device=args.device)
        with torch.no_grad():
            output = encoder.model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=torch.zeros_like(input_ids), use_cache=False, output_hidden_states=True,
            )
            prompt = output.hidden_states[50].cpu()
    assert prompt.shape[-1] == 5120 and torch.isfinite(prompt).all()
    log(f"Encoded actual source caption: {tuple(prompt.shape)}")
    for stage, data in encoded.items():
        data["prompt_embeds"] = prompt
        torch.save(data, args.output / f"{stage}.pt")
    report = {
        "status": "complete", "checkpoint": str(args.checkpoint), "parquet": str(args.parquet), "row": args.row,
        "source_paths": clip["source_paths"], "source_view_count": clip["source_view_count"],
        "selected_views": views, "sample_indices": clip["sample_indices"], "caption": clip["caption"],
        "stages": {stage: {"shape": list(data["latents"].shape), "source_frames": data["source_frames"], "padded_frames": data["padded_frames"], "valid_latents": int(data["valid_frames"].sum())} for stage, data in encoded.items()},
        "text_shape": list(prompt.shape), "text_layer": 50,
        "reused_text_from": str(args.reuse_text_from) if args.reuse_text_from else None,
    }
    (args.output / "features.json").write_text(json.dumps(report, indent=2) + "\n")
    log("Real H3 video/text features saved")


if __name__ == "__main__":
    main()
