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
        pose[:, 7] = i * .2 + torch.arange(t) * .01
        projection = camera_projection(pose[None])
        result.append(
            dict(latent=torch.randn(1, 24, t, 2, 2),
                 pose=pose,
                 projection=projection.projection[0],
                 inverse=projection.inverse[0],
                 condition=None,
                 frames=timeline.rotary_frames,
                 valid=timeline.valid,
                 spatial_weights=torch.ones(1, 1),
                 fps=16.,
                 scale=10.,
                 prompt="test",
                 text=torch.randn(1, 3, text_dim),
                 source_frames=frames,
                 height=32,
                 width=32))
    return dict(views=result, isolated=views == 1, source="fixture")
