# Adapted from HuggingFace Diffusers (Apache-2.0); see licenses/DIFFUSERS-APACHE-2.0.txt.
"""The H3 feed-forward and time embedding layers, with original state-dict names."""

import math

import torch
from torch import nn


class SwiGLU(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2 * dim_out, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * self.activation(gate)


class FeedForward(nn.Module):

    def __init__(self,
                 dim,
                 dim_out=None,
                 mult=4,
                 dropout=0.,
                 activation_fn="swiglu",
                 final_dropout=False,
                 inner_dim=None,
                 bias=True):
        super().__init__()
        if activation_fn != "swiglu":
            raise ValueError("H3 uses SwiGLU")
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        self.net = nn.ModuleList([
            SwiGLU(dim, inner_dim, bias),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim if dim_out is None else dim_out, bias=bias)
        ])
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class Timesteps(nn.Module):

    def __init__(self, num_channels, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        half = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half - self.downscale_freq_shift)
        angles = self.scale * timesteps[:, None].float() * exponent.exp()[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.flip_sin_to_cos:
            embedding = torch.cat((embedding[:, half:], embedding[:, :half]), dim=-1)
        if self.num_channels % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels, time_embed_dim, out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim if out_dim is None else out_dim)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))
