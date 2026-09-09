"""Single-process layer placement for bounded pretrained-model validation."""

import torch


def place_transformer(model, devices):
    """Place intact layers on GPUs; this is not distributed data/sequence parallelism."""
    devices = [torch.device(value) for value in devices]
    if not devices:
        raise ValueError("Provide at least one execution device")
    for name in ("proj_in", "audio_proj_in", "context_embedder", "token_refiner", "time_proj", "time_embedder", "rope"):
        getattr(model, name).to(devices[0])
    for index, block in enumerate(model.transformer_blocks):
        block.to(devices[min(index * len(devices) // len(model.transformer_blocks), len(devices) - 1)])
        block.attn.processor._attention_backend = "flex"
    for name in ("norm_out", "proj_out", "audio_proj_out"):
        getattr(model, name).to(devices[-1])
    return model
