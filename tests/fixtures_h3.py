"""Small native models/documents shared by CPU and torchrun verification."""

import torch

from h3 import MVH3Transformer3DModel
from h3.data import temporal_layout
from h3.modules.camera import camera_projection


def tiny_model(cls=MVH3Transformer3DModel):
    return cls(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=32,
        num_layers=3,
        num_refiner_layers=2,
        ffn_dim=64,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
        text_dim=32,
        freq_dim=32,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=16,
    )


def feature_document(views=1, frames=22, text_dim=32):
    timeline = temporal_layout(frames)
    result = []
    for i in range(views):
        t = len(timeline.valid)
        pose = torch.zeros(t, 10)
        pose[:, :2] = 1
        pose[:, 7] = i * 0.2 + torch.arange(t) * 0.01
        projection = camera_projection(pose[None])
        result.append(
            dict(
                latent=torch.randn(1, 24, t, 2, 2),
                pose=pose,
                projection=projection.projection[0],
                inverse=projection.inverse[0],
                condition=None,
                frames=timeline.rotary_frames,
                valid=timeline.valid,
                spatial_weights=torch.ones(1, 1),
                fps=16.0,
                scale=1.0,
                prompt="test",
                text=torch.randn(1, 3, text_dim),
                source_frames=frames,
                height=32,
                width=32,
            )
        )
    return dict(views=result, isolated=views == 1, source="fixture")


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
