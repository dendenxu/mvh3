#!/usr/bin/env python3
"""Convert released video-VAE weights to the names used by h3/modules/vae.py.

This is an offline preparation step. Training reads the converted checkpoint
from MVH3_VAE and never imports this script or any audio/publishing converter.
"""

import os
import json
import argparse
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3.checkpoint import split_fused_qkv, reorder_interleaved_qkv

SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"

MINIMAX_H3_VIDEO_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,  # out_ch
    "latent_channels": 24,  # z_channels == embed_dim
    "block_out_channels": [128, 256, 256, 512, 512, 1024],  # ch * ch_mult
    "layers_per_block": 2,  # num_res_blocks
    "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],  # space_down
    "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],  # time_down
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "spatial_padding_mode": "reflect",  # padding_mode
    "decoder_num_layers": 36,  # vit_decoder_kwargs.num_layers
    "decoder_num_attention_heads": 32,  # vit_decoder_kwargs.heads
    "decoder_attention_head_dim": 64,  # vit_decoder_kwargs.dim_head
    "decoder_num_register_tokens": 4,  # ViT3DDecoder default
    "decoder_ffn_mult": 4,  # FeedForward default
    "decoder_rope_theta": 100.0,  # vit_decoder_kwargs.rope_theta
    "decoder_rope_dim_ratio": 0.75,  # vit_decoder_kwargs.rope_dim_ratio
    "decoder_norm_eps": 1e-05,  # ViT3DDecoder eps
    "clip_length": 17,  # video_vae/config.json vae_clip_length
    "token_drop": 3,  # video_vae/config.json vae_token_drop
}

MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS = ("decoder.mask_token",)


def convert_video_vae_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original video-VAE key/tensor pair into the diffusers key/tensor pair(s) it maps to.

    `quant_conv` / `post_quant_conv`, the encoder's `conv_in` / `norm_out` / `conv_out` and the ViT decoder's
    `register_tokens` / `norm_out` / `proj_out` / `norm{1,2}` / `scale{1,2}` are pure pass-throughs. What moves:

    * the encoder's CNN levels are renamed from the original CompVis spelling onto the diffusers autoencoder idiom:
      `down.{i}.block.{j}` -> `down_blocks.{i}.resnets.{j}`, `nin_shortcut` -> `conv_shortcut`, and
      `down.{i}.downsample` -> `down_blocks.{i}.downsamplers.0`,
    * the ViT decoder's `x_embedder` becomes `proj_in`, the counterpart of the `proj_out` it already ships,
    * the fused per-head-interleaved `attn.to_qkv` is split into `attn.to_q` / `to_k` / `to_v`,
    * `attn.to_out` becomes `attn.to_out.0` (diffusers wraps the output projection in an `nn.ModuleList`),
    * the gated FFN's `w1` / `w2` become `ff.net.0.proj` / `ff.net.2`, and the two halves of `w1` are swapped because
      diffusers' `SwiGLU` reads `[up; gate]` where the reference stores `[gate; up]`.
    """
    if source_key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        return []

    if ".attn.to_qkv." in source_key:
        # Same per-head interleave as the DiT: `[head0: q k v, head1: q k v, ...]`.
        reordered = reorder_interleaved_qkv(
            tensor, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        query, key, value = split_fused_qkv(
            reordered, config["decoder_num_attention_heads"], config["decoder_attention_head_dim"]
        )
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [
            (f"{prefix}.attn.to_q.{suffix}", query),
            (f"{prefix}.attn.to_k.{suffix}", key),
            (f"{prefix}.attn.to_v.{suffix}", value),
        ]

    target_key = rename_video_vae_key(source_key)

    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return [(target_key, torch.cat([up, gate], dim=0).contiguous())]

    return [(target_key, tensor)]


def rename_video_vae_key(source_key: str) -> str:
    """Rename one original video-VAE key onto its diffusers module path (no tensor transform)."""
    target_key = source_key
    if target_key.startswith("encoder.down."):
        level, rest = target_key.removeprefix("encoder.down.").split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target_key = f"encoder.down_blocks.{level}.{rest}"
    target_key = target_key.replace("decoder.x_embedder.", "decoder.proj_in.")
    target_key = target_key.replace(".attn.to_out.", ".attn.to_out.0.")
    target_key = target_key.replace(".ff.w1.", ".ff.net.0.proj.")
    target_key = target_key.replace(".ff.w2.", ".ff.net.2.")
    return target_key


def get_video_vae_key_plan(config: dict[str, Any]) -> dict[str, list[str]]:
    """Map every original video-VAE key to the diffusers key(s) it produces, derived from the config alone."""
    block_out_channels = config["block_out_channels"]
    block_in_channels = [block_out_channels[0]] + block_out_channels[:-1]
    plan: dict[str, list[str]] = {}

    def renamed(*keys: str) -> None:
        """Register keys whose diffusers name is `rename_video_vae_key(key)` and whose tensor is unchanged."""
        for key in keys:
            plan[key] = [rename_video_vae_key(key)]

    renamed("quant_conv.weight", "quant_conv.bias", "post_quant_conv.weight", "post_quant_conv.bias")
    renamed("encoder.conv_in.weight", "encoder.conv_in.bias")
    for level, (in_channels, out_channels) in enumerate(zip(block_in_channels, block_out_channels)):
        for i in range(config["layers_per_block"]):
            prefix = f"encoder.down.{level}.block.{i}"
            for name in ("norm1", "conv1", "norm2", "conv2"):
                renamed(f"{prefix}.{name}.weight", f"{prefix}.{name}.bias")
            if (in_channels if i == 0 else out_channels) != out_channels:
                renamed(f"{prefix}.nin_shortcut.weight", f"{prefix}.nin_shortcut.bias")
        if config["spatial_downsample_factors"][level] * config["temporal_downsample_factors"][level] > 1:
            renamed(
                f"encoder.down.{level}.downsample.conv.weight", f"encoder.down.{level}.downsample.conv.bias"
            )
    renamed(
        "encoder.norm_out.weight", "encoder.norm_out.bias", "encoder.conv_out.weight", "encoder.conv_out.bias"
    )

    renamed("decoder.x_embedder.weight", "decoder.x_embedder.bias", "decoder.register_tokens")
    renamed(
        "decoder.norm_out.weight", "decoder.norm_out.bias", "decoder.proj_out.weight", "decoder.proj_out.bias"
    )
    for i in range(config["decoder_num_layers"]):
        prefix = f"decoder.transformer_blocks.{i}"
        renamed(f"{prefix}.norm1.weight", f"{prefix}.norm2.weight", f"{prefix}.scale1", f"{prefix}.scale2")
        for suffix in ("weight", "bias"):
            plan[f"{prefix}.attn.to_qkv.{suffix}"] = [
                f"{prefix}.attn.to_q.{suffix}",
                f"{prefix}.attn.to_k.{suffix}",
                f"{prefix}.attn.to_v.{suffix}",
            ]
            plan[f"{prefix}.attn.to_out.{suffix}"] = [f"{prefix}.attn.to_out.0.{suffix}"]
            plan[f"{prefix}.ff.w1.{suffix}"] = [f"{prefix}.ff.net.0.proj.{suffix}"]
            plan[f"{prefix}.ff.w2.{suffix}"] = [f"{prefix}.ff.net.2.{suffix}"]
    for key in MINIMAX_H3_VIDEO_VAE_DROPPED_KEYS:
        plan[key] = []
    return plan


def convert_video_vae(
    checkpoint_path: str,
    output_path: str,
    config: dict[str, Any],
    diffusers_version: str,
    max_shard_size: int,
) -> None:
    """Convert the video VAE and emit its config.

    The original weights live one level deeper than the rest of the checkpoint
    (`video_vae/source/model.safetensors`, resolved by a hook in the reference); the diffusers layout flattens that to
    `vae/`. `latents_mean` / `latents_std` and the tiling geometry come from `video_vae/config.json` — MiniMax-H3
    normalizes latents per channel instead of with a `scaling_factor`.
    """
    source_dir = os.path.join(checkpoint_path, "video_vae")
    with open(os.path.join(source_dir, "config.json")) as f:
        wrapper_config = json.load(f)
    for key in ("latents_mean", "latents_std"):
        if key not in wrapper_config:
            raise KeyError(f"{source_dir}/config.json does not carry `{key}`.")

    weights_path = os.path.join(
        source_dir, wrapper_config["source_path"], wrapper_config["source_safetensors_path"]
    )
    plan = get_video_vae_key_plan(config)

    os.makedirs(output_path, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0
    written: list[str] = []
    buffer: dict[str, torch.Tensor] = {}
    buffer_size = 0
    seen_source_keys: set[str] = set()

    def flush() -> None:
        nonlocal buffer, buffer_size
        if not buffer:
            return
        path = os.path.join(output_path, f".tmp-shard-{len(written):05d}.safetensors")
        save_file(buffer, path, metadata={"format": "pt"})
        for key in buffer:
            weight_map[key] = path
        written.append(path)
        buffer = {}
        buffer_size = 0

    # `safe_open` memory-maps the file, so only the tensor being read is materialized.
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for source_key in f.keys():
            if source_key not in plan:
                raise KeyError(f"Unexpected key in {weights_path}: {source_key}")
            seen_source_keys.add(source_key)
            for target_key, tensor in convert_video_vae_key(source_key, f.get_tensor(source_key), config):
                if tensor.dtype != torch.float32:
                    raise ValueError(f"{source_key}: expected torch.float32, got {tensor.dtype}.")
                buffer[target_key] = tensor
                buffer_size += tensor.numel() * tensor.element_size()
                total_size += tensor.numel() * tensor.element_size()
            if buffer_size >= max_shard_size:
                flush()
    flush()

    missing = sorted(set(plan) - seen_source_keys)
    if missing:
        raise KeyError(f"{len(missing)} planned key(s) missing from {weights_path}, e.g. {missing[:5]}.")

    renames = {
        path: os.path.join(
            output_path, f"diffusion_pytorch_model-{i + 1:05d}-of-{len(written):05d}.safetensors"
        )
        for i, path in enumerate(written)
    }
    for old, new in renames.items():
        os.rename(old, new)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: os.path.basename(renames[path]) for key, path in weight_map.items()},
    }
    with open(os.path.join(output_path, SAFE_WEIGHTS_INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(
            {
                "_class_name": "AutoencoderKLMiniMaxH3",
                "_diffusers_version": diffusers_version,
                **config,
                "latents_mean": wrapper_config["latents_mean"],
                "latents_std": wrapper_config["latents_std"],
            },
            f,
            indent=2,
        )

    print(
        f"video_vae: {len(seen_source_keys)} original keys -> {len(weight_map)} diffusers keys in "
        f"{len(written)} shard(s), {total_size / 1024**3:.2f} GiB "
        f"(latents_mean/latents_std: {len(wrapper_config['latents_mean'])}/"
        f"{len(wrapper_config['latents_std'])} channels; tiling {wrapper_config['vae_tile_size']}px / "
        f"{wrapper_config['vae_tile_overlap_min']}px min overlap)."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Original FL2VA checkpoint directory")
    parser.add_argument("--output", required=True, help="Destination for converted video-VAE weights")
    args = parser.parse_args()
    convert_video_vae(args.checkpoint, args.output, MINIMAX_H3_VIDEO_VAE_CONFIG, "local-d30c748", 2 * 1024**3)


if __name__ == "__main__":
    main()
