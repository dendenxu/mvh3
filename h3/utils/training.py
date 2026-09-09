"""Historical all-attention probe helpers; see h3.distributed.fsdp for the recipe."""

import torch


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
