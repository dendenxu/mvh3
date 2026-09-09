"""Strict streaming loads from the original, complete H3 checkpoint."""

import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from h3.modules.model import MiniMaxH3RotaryPosEmbed, MiniMaxH3Transformer3DModel
from h3.vendor.convert_minimax_h3 import MINIMAX_H3_TRANSFORMER_CONFIG, MINIMAX_H3_FP32_SOURCE_PREFIXES, convert_transformer_key, get_transformer_key_plan, reorder_interleaved_qkv


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
                    tensor = reorder_interleaved_qkv(tensor, config["num_attention_heads"],
                                                     config["attention_head_dim"])
                converted = convert_transformer_key(source, tensor, config)
                if [(name, list(value.shape)) for name, value in converted] != plan[source]:
                    raise ValueError(f"Converted tensor shape differs from the architecture: {source}")
                for name, value in converted:
                    expected_dtype = torch.float32 if source.startswith(
                        MINIMAX_H3_FP32_SOURCE_PREFIXES) else torch.bfloat16
                    if value.dtype != expected_dtype:
                        raise ValueError(f"Incorrect checkpoint dtype for {source}: {value.dtype}")
                    yield name, value
        if progress is not None:
            progress(f"Loaded {filename}: {len(seen)}/{len(plan)} source tensors, {time.monotonic()-started:.1f}s")
    if seen != set(plan):
        raise ValueError("Incomplete original checkpoint")


def load_original_transformer(checkpoint,
                              device_for_name=None,
                              model_class=MiniMaxH3Transformer3DModel,
                              progress=print):
    """Load every released tensor, preserving mixed precision and all AdaLN weights.

    device_for_name optionally places complete blocks on different devices for
    bounded single-process verification. It does not implement FSDP or SP.
    """
    config = original_config(checkpoint)
    with torch.device("meta"):
        model = model_class(**config)
    expected = dict(model.named_parameters())
    loaded = set()
    for name, tensor in original_tensors(checkpoint, config, progress):
        if name not in expected or name in loaded or tensor.shape != expected[name].shape:
            raise ValueError(f"Checkpoint cannot be assigned to this model: {name}")
        device = device_for_name(name) if device_for_name else "cpu"
        owner, leaf = name.rsplit(".", 1)
        model.get_submodule(owner).register_parameter(leaf, torch.nn.Parameter(tensor.to(device), requires_grad=False))
        loaded.add(name)
    if loaded != set(expected):
        raise ValueError(f"Uninitialized model parameters: {set(expected)-loaded}")
    model.rope = MiniMaxH3RotaryPosEmbed(config["rope_freq_dim"], config["rope_theta"])
    device = device_for_name("rope.inv_freq") if device_for_name else "cpu"
    model.rope.to(device)
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError("Checkpoint loading left meta tensors")
    return model
