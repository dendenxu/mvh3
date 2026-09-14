"""Strict streaming loads from the original, complete H3 checkpoint."""

import json
import time
import inspect
from typing import Any
from pathlib import Path

import torch
from safetensors import safe_open

MINIMAX_H3_TRANSFORMER_CONFIG = {
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "hidden_size": 5376,
    "num_layers": 50,
    "num_refiner_layers": 2,  # token_refiner_num_layers
    "ffn_dim": 14336,  # ffn_hidden_size
    "in_channels": 24,  # latents_dim
    "audio_in_channels": 32,  # audio_latents_dim
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "freq_dim": 256,  # timestep_input_dim
    "time_embed_hidden_dim": 5376,  # time_embed_hidden_size
    "time_embed_dim": 2688,
    "rope_freq_dim": 16,  # rope_inv_freq_len
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "qk_norm_eps": 1e-05,
    "final_norm_eps": 1e-05,
}

MINIMAX_H3_FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

MINIMAX_H3_TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)


def original_config(checkpoint):
    raw = json.loads((Path(checkpoint) / "transformer/config.json").read_text())
    config = dict(MINIMAX_H3_TRANSFORMER_CONFIG)
    names = {
        "token_refiner_num_layers": "num_refiner_layers",
        "ffn_hidden_size": "ffn_dim",
        "latents_dim": "in_channels",
        "audio_latents_dim": "audio_in_channels",
        "timestep_input_dim": "freq_dim",
        "time_embed_hidden_size": "time_embed_hidden_dim",
        "rope_inv_freq_len": "rope_freq_dim",
    }
    for key, value in raw.items():
        target = names.get(key, key)
        if target in config:
            config[target] = value
    return config


def original_tensors(checkpoint, config, progress=print):
    directory = Path(checkpoint) / "transformer"
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    plan = get_transformer_key_plan(config)
    if set(index) != set(plan):
        raise ValueError(
            f"Source index differs from the architecture: missing={set(plan)-set(index)}, extra={set(index)-set(plan)}"
        )
    seen = set()
    for filename in sorted(set(index.values())):
        started = time.monotonic()
        with safe_open(directory / filename, framework="pt", device="cpu") as reader:
            for source in reader.keys():
                if source not in plan or source in seen or index[source] != filename:
                    raise ValueError(f"Unexpected/duplicate/misindexed source tensor: {source}")
                seen.add(source)
                tensor = reader.get_tensor(source)
                if source.endswith(".attn.qkv_proj.weight"):
                    tensor = reorder_interleaved_qkv(
                        tensor, config["num_attention_heads"], config["attention_head_dim"]
                    )
                converted = convert_transformer_key(source, tensor, config)
                if [(name, list(value.shape)) for name, value in converted] != plan[source]:
                    raise ValueError(f"Converted tensor shape differs from the architecture: {source}")
                for name, value in converted:
                    expected_dtype = (
                        torch.float32
                        if source.startswith(MINIMAX_H3_FP32_SOURCE_PREFIXES)
                        else torch.bfloat16
                    )
                    if value.dtype != expected_dtype:
                        raise ValueError(f"Incorrect checkpoint dtype for {source}: {value.dtype}")
                    yield name, value
        if progress is not None:
            progress(
                f"Loaded {filename}: {len(seen)}/{len(plan)} source tensors, {time.monotonic()-started:.1f}s"
            )
    if seen != set(plan):
        raise ValueError("Incomplete original checkpoint")


def reorder_interleaved_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> torch.Tensor:
    """Reorder a *raw-checkpoint* per-head-interleaved fused QKV weight into `[q_all; k_all; v_all]`.

    The original checkpoint shards store rows as `[head0: q(head_dim) k(head_dim) v(head_dim), head1: q, k, v, ...]`.
    The reference implementation applies exactly this reorder at load time (`_reorder_grouped_qkv_to_qkv` with one head
    per query group), so `[q_all; k_all; v_all]` is the reference's in-memory / state-dict layout. There is no
    transpose.
    """
    expected_rows = num_attention_heads * 3 * attention_head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{expected_rows} = {num_attention_heads} heads * 3 * {attention_head_dim}."
        )
    grouped = weight.reshape(num_attention_heads, 3 * attention_head_dim, *weight.shape[1:])
    query, key, value = grouped.split(attention_head_dim, dim=1)
    return torch.cat(
        [
            tensor.reshape(num_attention_heads * attention_head_dim, *weight.shape[1:])
            for tensor in (query, key, value)
        ],
        dim=0,
    )


def split_fused_qkv(
    weight: torch.Tensor, num_attention_heads: int, attention_head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Split a fused `[q_all; k_all; v_all]` QKV weight into separate `to_q` / `to_k` / `to_v` weights.

    The input is the *reference model* layout — what `MiniMaxH3DiTModel.state_dict()` holds after the reference's
    load-time reorder — i.e. the three logical projection matrices stacked contiguously, NOT the raw checkpoint's
    per-head interleave (see `reorder_interleaved_qkv`, which the shard streamer applies first).
    """
    inner_dim = num_attention_heads * attention_head_dim
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv weight has {weight.shape[0]} rows, expected "
            f"{3 * inner_dim} = 3 * {num_attention_heads} heads * {attention_head_dim}."
        )
    query, key, value = weight.split(inner_dim, dim=0)
    return tuple(tensor.contiguous() for tensor in (query, key, value))


def get_transformer_key_plan(config: dict[str, Any]) -> dict[str, list[tuple[str, list[int]]]]:
    """Map every original transformer key to the local parameter key(s) it produces, with the resulting shapes.

    The plan is derived from the config alone, so it can be printed and checked without any weights present.
    """
    hidden_size = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    inner_dim = heads * head_dim
    ffn_dim = config["ffn_dim"]
    time_embed_dim = config["time_embed_dim"]
    video_patch_dim = (
        config["in_channels"] * config["patch_size"][0] * config["patch_size"][1] * config["patch_size"][2]
    )

    plan: dict[str, list[tuple[str, list[int]]]] = {
        "video_patch_proj.weight": [("proj_in.weight", [hidden_size, video_patch_dim])],
        "video_patch_proj.bias": [("proj_in.bias", [hidden_size])],
        "audio_patch_proj.weight": [("audio_proj_in.weight", [hidden_size, config["audio_in_channels"]])],
        "audio_patch_proj.bias": [("audio_proj_in.bias", [hidden_size])],
        "condition_proj.weight": [("context_embedder.weight", [hidden_size, config["text_dim"]])],
        "condition_proj.bias": [("context_embedder.bias", [hidden_size])],
        # `Timesteps` + `TimestepEmbedding` reproduce the reference sinusoid and MLP exactly, so the timestep MLP is
        # renamed onto `TimestepEmbedding`'s `linear_1` / `linear_2`.
        "time_embedder.proj_in.weight": [
            ("time_embedder.linear_1.weight", [config["time_embed_hidden_dim"], config["freq_dim"]])
        ],
        "time_embedder.proj_in.bias": [("time_embedder.linear_1.bias", [config["time_embed_hidden_dim"]])],
        "time_embedder.proj_out.weight": [
            ("time_embedder.linear_2.weight", [time_embed_dim, config["time_embed_hidden_dim"]])
        ],
        "time_embedder.proj_out.bias": [("time_embedder.linear_2.bias", [time_embed_dim])],
        "token_refiner.final_norm.weight": [("token_refiner.final_norm.weight", [hidden_size])],
        "final_layer.norm.weight": [("norm_out.norm.weight", [hidden_size])],
        "final_layer.adaln_proj.linear.weight": [
            ("norm_out.linear.weight", [2 * hidden_size, time_embed_dim])
        ],
        "final_layer.adaln_proj.linear.bias": [("norm_out.linear.bias", [2 * hidden_size])],
        "final_layer.video_out.weight": [("proj_out.weight", [video_patch_dim, hidden_size])],
        "final_layer.video_out.bias": [("proj_out.bias", [video_patch_dim])],
        "final_layer.audio_out.weight": [
            ("audio_proj_out.weight", [config["audio_in_channels"], hidden_size])
        ],
        "final_layer.audio_out.bias": [("audio_proj_out.bias", [config["audio_in_channels"]])],
    }
    for key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        plan[key] = []

    block_specs = [
        ("blocks", "transformer_blocks", config["num_layers"], True),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", config["num_refiner_layers"], False),
    ]
    for source_prefix, target_prefix, num_layers, has_adaln in block_specs:
        for i in range(num_layers):
            source = f"{source_prefix}.{i}"
            target = f"{target_prefix}.{i}"
            plan[f"{source}.norm1.weight"] = [(f"{target}.norm1.weight", [hidden_size])]
            plan[f"{source}.norm2.weight"] = [(f"{target}.norm2.weight", [hidden_size])]
            plan[f"{source}.attn.qkv_proj.weight"] = [
                (f"{target}.attn.to_q.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_k.weight", [inner_dim, hidden_size]),
                (f"{target}.attn.to_v.weight", [inner_dim, hidden_size]),
            ]
            plan[f"{source}.attn.q_norm.weight"] = [(f"{target}.attn.norm_q.weight", [head_dim])]
            plan[f"{source}.attn.k_norm.weight"] = [(f"{target}.attn.norm_k.weight", [head_dim])]
            plan[f"{source}.attn.out_proj.weight"] = [
                (f"{target}.attn.to_out.0.weight", [hidden_size, inner_dim])
            ]

            # `fc1` stays fused, as diffusers' `SwiGLU` also fuses its two projections, but the halves are swapped
            # from `[gate; value]` to `[value; gate]` (see `convert_transformer_key`).
            plan[f"{source}.mlp.fc1.weight"] = [
                (f"{target}.ff.net.0.proj.weight", [2 * ffn_dim, hidden_size])
            ]
            plan[f"{source}.mlp.fc2.weight"] = [(f"{target}.ff.net.2.weight", [hidden_size, ffn_dim])]
            if has_adaln:
                plan[f"{source}.adaln_proj.linear.weight"] = [
                    (f"{target}.adaln_proj.linear.weight", [6 * 3 * hidden_size, time_embed_dim])
                ]
                plan[f"{source}.adaln_proj.linear.bias"] = [
                    (f"{target}.adaln_proj.linear.bias", [6 * 3 * hidden_size])
                ]

    return plan


def convert_transformer_key(
    source_key: str, tensor: torch.Tensor, config: dict[str, Any]
) -> list[tuple[str, torch.Tensor]]:
    """Convert one original key/tensor pair into the diffusers key/tensor pair(s) it maps to."""
    if source_key in MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return []

    target_key = source_key
    if target_key.startswith("token_refiner.blocks."):
        target_key = target_key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target_key.startswith("blocks."):
        target_key = target_key.replace("blocks.", "transformer_blocks.", 1)
    target_key = target_key.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target_key = target_key.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target_key = target_key.replace("video_patch_proj.", "proj_in.")
    target_key = target_key.replace("audio_patch_proj.", "audio_proj_in.")
    target_key = target_key.replace("condition_proj.", "context_embedder.")
    target_key = target_key.replace("final_layer.norm.", "norm_out.norm.")
    target_key = target_key.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target_key = target_key.replace("final_layer.video_out.", "proj_out.")
    target_key = target_key.replace("final_layer.audio_out.", "audio_proj_out.")
    target_key = target_key.replace(".attn.q_norm.", ".attn.norm_q.")
    target_key = target_key.replace(".attn.k_norm.", ".attn.norm_k.")
    target_key = target_key.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target_key.endswith(".attn.qkv_proj.weight"):
        # `convert_transformer_key` consumes tensors in the reference model's state-dict layout, where the fused QKV
        # rows are already `[q_all; k_all; v_all]`. Raw checkpoint shards are per-head interleaved instead; the shard
        # streamer (`original_tensors`) normalizes them with `reorder_interleaved_qkv` before calling this.
        query, key, value = split_fused_qkv(
            tensor, config["num_attention_heads"], config["attention_head_dim"]
        )
        prefix = target_key.removesuffix("qkv_proj.weight")
        return [
            (f"{prefix}to_q.weight", query),
            (f"{prefix}to_k.weight", key),
            (f"{prefix}to_v.weight", value),
        ]

    if target_key.endswith(".mlp.fc1.weight"):
        # The reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]`; diffusers' `SwiGLU` computes
        # `value * silu(gate)` from a fused `[value; gate]`, so the two halves swap places. Identical transform to the
        # video VAE's `ff.w1` (see `convert_video_vae_key`).
        gate, value = tensor.chunk(2, dim=0)
        target_key = target_key.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target_key, torch.cat([value, gate], dim=0).contiguous())]

    target_key = target_key.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target_key, tensor)]


def load_local_model(cls, path, *, torch_dtype=None, local_files_only=True):
    """Strictly load a local converted checkpoint; never download or infer a model."""
    from safetensors import safe_open

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    keys = inspect.signature(cls.__init__).parameters
    unknown = {key for key in config if not key.startswith("_") and key not in keys}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} configuration: {sorted(unknown)}")
    with torch.device("meta"):
        model = cls(**{key: value for key, value in config.items() if key in keys})

    # Materialize nonpersistent RoPE buffers from a small empty constructor.
    buffers = dict(model.named_buffers())
    model.to_empty(device="cpu")
    for name in buffers:
        owner_name, leaf = name.rsplit(".", 1)
        owner = model.get_submodule(owner_name)
        if leaf == "inv_freq" and hasattr(owner, "reset_parameters"):
            owner.reset_parameters()
        else:
            raise ValueError(f"No initializer for nonpersistent buffer {name}")
    expected = model.state_dict()
    index_file = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(path / filename, framework="pt") as reader:
            index = {key: filename for key in reader.keys()}
    if set(index) != set(expected):
        raise ValueError(
            f"Checkpoint keys differ: missing={set(expected)-set(index)}, extra={set(index)-set(expected)}"
        )
    seen = set()
    for filename in sorted(set(index.values())):
        with safe_open(path / filename, framework="pt") as reader:
            state = {}
            for key in reader.keys():
                if key in seen or index.get(key) != filename:
                    raise ValueError(f"Duplicate or misindexed tensor: {key}")
                tensor = reader.get_tensor(key)
                if tensor.shape != expected[key].shape:
                    raise ValueError(f"Wrong checkpoint shape for {key}")
                state[key] = tensor.to(torch_dtype) if torch_dtype is not None else tensor
                seen.add(key)
            model.load_state_dict(state, strict=False, assign=True)
    if seen != set(expected):
        raise ValueError("Incomplete checkpoint")
    return model
