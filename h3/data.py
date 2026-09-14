"""The released H3 VAE temporal geometry and native rotary clock."""

import math
from dataclasses import dataclass

import torch
import numpy as np


def spatial_rotary_grid(height, width):
    area = (height * width) ** 0.5
    yy = torch.from_numpy(
        np.linspace((1 - height / area) / 2, (1 + height / area) / 2, height // 2, endpoint=False) * 32
    )
    xx = torch.from_numpy(
        np.linspace((1 - width / area) / 2, (1 + width / area) / 2, width // 2, endpoint=False) * 32
    )
    return torch.stack(torch.meshgrid(yy, xx, indexing="ij"), -1).flatten(0, 1)


def temporal_rotary_clock(latents, fps):
    spans = torch.tensor([(40 / fps) * (1, 4, 4, 4, 4)[i % 5] for i in range(latents)], dtype=torch.float64)
    return torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


@dataclass(frozen=True)
class TemporalLayout:
    """`valid` marks requested intervals, not VAE/attention padding to discard.

    Every latent is needed by the non-causal decoder, including the encoded
    repeated-frame tail. Generate and supervise the complete aligned sequence.
    """

    source_frames: int
    padded_frames: int
    camera_frames: torch.Tensor
    rotary_frames: torch.Tensor
    valid: torch.Tensor


def temporal_layout(num_frames: int) -> TemporalLayout:
    """Keep all source frames despite H3 dropping its final three encoder latents.

    Encoder chunks produce anchors [0,4,8,12,16] every 17 pixel frames. The
    native rotary clock uses interval starts [0,1,5,9,13]; camera anchors use
    causal interval ends. Arbitrary lengths are padded to 17*n+5 first.
    """
    if num_frames < 1:
        raise ValueError("A clip needs at least one frame")
    if num_frames == 1:
        return TemporalLayout(
            1, 1, torch.tensor([0]), torch.tensor([0.0], dtype=torch.float64), torch.tensor([True])
        )
    chunks = max(0, math.ceil((num_frames - 5) / 17))
    padded_frames, latents = 17 * chunks + 5, 5 * chunks + 2
    ids = torch.arange(latents)
    anchors = 17 * (ids // 5) + 4 * (ids % 5)
    starts = 17 * (ids // 5) + torch.tensor([0, 1, 5, 9, 13])[ids % 5]

    # Use this only for requested duration/chunk boundaries, never attention/loss.
    valid = starts < num_frames
    return TemporalLayout(
        num_frames, padded_frames, anchors.clamp_max(num_frames - 1), starts.double(), valid
    )


def pad_video(pixels: torch.Tensor, layout: TemporalLayout):
    if pixels.shape[2] != layout.source_frames:
        raise ValueError("Video and temporal layout have different frame counts")
    count = layout.padded_frames - layout.source_frames
    return torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, count, -1, -1)), dim=2) if count else pixels
