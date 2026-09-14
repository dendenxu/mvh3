"""Original-attention parameter selection and flow-matching validation helpers."""

import torch

from h3.utils.model import canonical_name


def parameter_groups(model, cfg):
    """Group the selected original attention weights by their configured learning rate."""
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = canonical_name(name)
        index = int(name.split("transformer_blocks.")[1].split(".")[0])
        lr = float(cfg.ar_lr if index % cfg.model.ar_interval == 0 else cfg.sa_lr)
        buckets.setdefault(lr, []).append(parameter)
    return [dict(params=params, lr=lr, initial_lr=lr) for lr, params in buckets.items()]


def attention_parameters(model: torch.nn.Module):
    """Select existing main-block Q/K/V/O projections and Q/K RMS norms."""
    parameters = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("transformer_blocks.") and ".attn." in name
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    if not parameters:
        raise ValueError("No original H3 main-block attention parameters were found")
    return parameters


def parameter_signature(model: torch.nn.Module):
    return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}


def flow_matching_loss(prediction: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor):
    """FP32 video loss over target tokens only, excluding clean context/padding."""
    if prediction.shape != target.shape or loss_mask.shape != prediction.shape[:-1]:
        raise ValueError("Loss target/mask must match the predicted video token layout")
    if loss_mask.dtype != torch.bool or not loss_mask.any():
        raise ValueError("loss_mask must select at least one target token")
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    return error[loss_mask].mean()
