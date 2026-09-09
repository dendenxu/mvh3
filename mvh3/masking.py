"""Teacher-forcing visibility for every modality in H3's packed sequence."""

from dataclasses import dataclass

import torch


CONDITION = 0
CLEAN = 1
NOISY = 2


@dataclass(frozen=True)
class TokenLayout:
    """One shared layout per batch; scope=-1 is global, chunk=-1 is always available.

    Text is CONDITION. Audio follows the same CLEAN/NOISY rules as video.
    CONDITION queries never read generated media, preventing a relay through
    text or conditioning audio in subsequent transformer blocks.
    """

    kind: torch.Tensor
    chunk: torch.Tensor
    scope: torch.Tensor
    cross_view: bool = True
    active: torch.Tensor | None = None

    def __post_init__(self):
        if self.kind.ndim != 1 or self.kind.numel() == 0:
            raise ValueError("Token layout must contain at least one token")
        for value in (self.kind, self.chunk, self.scope):
            if value.shape != self.kind.shape or value.dtype != torch.long or value.device != self.kind.device:
                raise ValueError("Layout fields must be int64 vectors with matching shapes and devices")
        if ((self.kind < CONDITION) | (self.kind > NOISY)).any():
            raise ValueError("Token kinds must be CONDITION, CLEAN or NOISY")
        if (self.scope < -1).any() or (self.chunk < -1).any():
            raise ValueError("Scope/chunk indices must be >= -1")
        if ((self.kind != CONDITION) & ((self.chunk < 0) | (self.scope < 0))).any():
            raise ValueError("Clean/noisy media need explicit nonnegative chunk and scope indices")
        if self.active is not None and (self.active.shape != self.kind.shape or self.active.dtype != torch.bool or self.active.device != self.kind.device):
            raise ValueError("active must be a boolean vector on the layout device")

    def to(self, device):
        return type(self)(self.kind.to(device), self.chunk.to(device), self.scope.to(device), self.cross_view,
                          self.active.to(device) if self.active is not None else None)

    def mask_mod(self, batch, head, query, key):
        # Padded flex-attention blocks may evaluate indices beyond the real sequence.
        size = self.kind.shape[0]
        valid = (query < size) & (key < size)
        q, k = query.clamp_max(size - 1), key.clamp_max(size - 1)
        qkind, kkind = self.kind[q], self.kind[k]
        qchunk, kchunk = self.chunk[q], self.chunk[k]
        condition = (kkind == CONDITION) & (kchunk <= qchunk)
        clean = (qkind == CLEAN) & (kkind == CLEAN) & (kchunk <= qchunk)
        previous = (qkind == NOISY) & (kkind == CLEAN) & (kchunk < qchunk)
        current = (qkind == NOISY) & (kkind == NOISY) & (kchunk == qchunk)
        qscope, kscope = self.scope[q], self.scope[k]
        if self.cross_view:
            # Global conditions cannot absorb local conditions; that would relay
            # view/chunk-local captions into unrelated global text tokens.
            scope = (qscope >= 0) | (kscope < 0)
        else:
            scope = (kscope < 0) | (qscope == kscope)
        visible = scope & (condition | clean | previous | current)
        if self.active is not None:
            visible = (self.active[q] & self.active[k] & visible) | (~self.active[q] & (q == k))
        return valid & visible

    def dense(self, indices: torch.Tensor | None = None, max_tokens: int = 4096) -> torch.Tensor:
        """Small-layout reference mask. Use block_mask for training-sized sequences."""
        if indices is None:
            indices = torch.arange(self.kind.numel(), device=self.kind.device)
        if indices.numel() > max_tokens:
            raise ValueError("Dense mask exceeds the bounded reference size; use block_mask")
        return self.mask_mod(0, 0, indices[:, None], indices[None, :])

    def block_mask(self, indices: torch.Tensor | None = None):
        from torch.nn.attention.flex_attention import create_block_mask

        if indices is None:
            size = self.kind.numel()

            def mask_mod(batch, head, query, key):
                return self.mask_mod(batch, head, query, key)
        else:
            size = indices.numel()

            def mask_mod(batch, head, query, key):
                valid = (query < size) & (key < size)
                q, k = indices[query.clamp_max(size - 1)], indices[key.clamp_max(size - 1)]
                return valid & self.mask_mod(batch, head, q, k)

        return create_block_mask(mask_mod, None, None, size, size, device=self.kind.device, _compile=True)
