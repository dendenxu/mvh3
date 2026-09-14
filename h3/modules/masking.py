"""Teacher-forcing visibility for every modality in H3's packed sequence."""

from dataclasses import dataclass

import torch

CONDITION = 0
CLEAN = 1
NOISY = 2


def block_counts(mask_mod, query_length, key_length, device, block_size=128):
    from torch.nn.attention.flex_attention import create_mask

    if device.type == "cuda" and not torch.compiler.is_compiling():
        raise RuntimeError("GPU block-mask reduction requires compilation; eager execution is quadratic")
    q_size, k_size = (block_size, block_size) if isinstance(block_size, int) else block_size
    q_blocks, k_blocks = (query_length + q_size - 1) // q_size, (key_length + k_size - 1) // k_size
    mask = create_mask(mask_mod, None, None, q_blocks * q_size, k_blocks * k_size, device=device)
    return mask.view(1, 1, q_blocks, q_size, k_blocks, k_size).sum(dim=(3, 5))


_compiled_block_counts = torch.compile(block_counts, dynamic=False, fullgraph=True)


def ordered_blocks(visible):
    values = visible.to(torch.int32)
    return values.sum(-1).to(torch.int32), values.argsort(dim=-1, descending=True, stable=True).to(
        torch.int32
    )


@torch.compiler.disable(recursive=False)
def build_block_mask(mask_mod, query_length, key_length, device, block_size=128):
    """Compile visibility reduction without fusing the small block-grid sorts."""
    from torch.nn.attention.flex_attention import BlockMask

    block_size = (block_size, block_size) if isinstance(block_size, int) else tuple(block_size)
    if len(block_size) != 2 or any(not isinstance(value, int) or value <= 0 for value in block_size):
        raise ValueError("Block size must contain two positive integer dimensions")
    device = torch.device(device)
    counts_fn = _compiled_block_counts if device.type == "cuda" else block_counts
    counts = counts_fn(mask_mod, query_length, key_length, device, block_size)
    block_area = block_size[0] * block_size[1]
    partial = ordered_blocks((counts > 0) & (counts < block_area))
    full = ordered_blocks(counts == block_area)
    return BlockMask.from_kv_blocks(
        *partial, *full, BLOCK_SIZE=block_size, mask_mod=mask_mod, seq_lengths=(query_length, key_length)
    )


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
    history_dropout: torch.Tensor | None = None
    history: bool = True
    joint: bool = False
    single_sequence: bool = False

    def __post_init__(self):
        if self.kind.ndim != 1 or self.kind.numel() == 0:
            raise ValueError("Token layout must contain at least one token")
        for value in (self.kind, self.chunk, self.scope):
            if (
                value.shape != self.kind.shape
                or value.dtype != torch.long
                or value.device != self.kind.device
            ):
                raise ValueError("Layout fields must be int64 vectors with matching shapes and devices")
        if ((self.kind < CONDITION) | (self.kind > NOISY)).any():
            raise ValueError("Token kinds must be CONDITION, CLEAN or NOISY")
        if (self.scope < -1).any() or (self.chunk < -1).any():
            raise ValueError("Scope/chunk indices must be >= -1")
        if ((self.kind != CONDITION) & ((self.chunk < 0) | (self.scope < 0))).any():
            raise ValueError("Clean/noisy media need explicit nonnegative chunk and scope indices")
        if self.active is not None and (
            self.active.shape != self.kind.shape
            or self.active.dtype != torch.bool
            or self.active.device != self.kind.device
        ):
            raise ValueError("active must be a boolean vector on the layout device")
        if self.history_dropout is not None:
            if (
                self.history_dropout.ndim != 2
                or not self.history_dropout.numel()
                or self.history_dropout.dtype != torch.bool
                or self.history_dropout.device != self.kind.device
            ):
                raise ValueError("history_dropout must be a nonempty boolean matrix on the layout device")
            # A 1x1 CuTe auxiliary buffer has ambiguous strides; its flat view
            # preserves the same mask with a unique contiguous dimension.
            object.__setattr__(self, "_history_dropout_flat", self.history_dropout.reshape(-1))

    def to(self, device):
        return type(self)(
            self.kind.to(device),
            self.chunk.to(device),
            self.scope.to(device),
            self.cross_view,
            self.active.to(device) if self.active is not None else None,
            self.history_dropout.to(device) if self.history_dropout is not None else None,
            self.history,
            self.joint,
            self.single_sequence,
        )

    def mask_mod(self, batch, head, query, key):
        # Padded flex-attention blocks may evaluate indices beyond the real sequence.
        size = self.kind.shape[0]
        valid = (query < size) & (key < size)
        q, k = query.clamp_max(size - 1).to(torch.int32), key.clamp_max(size - 1).to(torch.int32)
        qkind, kkind = self.kind[q], self.kind[k]
        qchunk, kchunk = self.chunk[q], self.chunk[k]
        condition = (kkind == CONDITION) & (kchunk <= qchunk)
        clean = (qkind == CLEAN) & (kkind == CLEAN) & (kchunk <= qchunk)
        previous = (qkind == NOISY) & (kkind == CLEAN) & (kchunk < qchunk)
        if not self.history:
            clean = clean & (kchunk == qchunk)
            previous = previous & False
        if self.history_dropout is not None:
            n, m = self.history_dropout.shape
            # FA4/CuTe indirect buffer indices must be Int32, including values
            # loaded from the canonical int64 layout tensors.
            drop_index = qchunk.clamp(0, n - 1) * m + kchunk.clamp(0, m - 1)
            drop = self._history_dropout_flat[drop_index.to(torch.int32)]
            previous = previous & ~drop
        current = (qkind == NOISY) & (kkind == NOISY) & (kchunk == qchunk)
        qscope, kscope = self.scope[q], self.scope[k]
        if self.cross_view:
            # Global conditions cannot absorb local conditions; that would relay
            # view/chunk-local captions into unrelated global text tokens.
            scope = (qscope >= 0) | (kscope < 0)
        else:
            scope = (kscope < 0) | (qscope == kscope)
        visible = scope & (condition | clean | previous | current)
        if self.single_sequence:
            allowed_history = (kchunk < qchunk) & (self.history | (kchunk < 0))
            if self.history_dropout is not None:
                allowed_history = allowed_history & (~drop | (kchunk < 0))
            scope = ((q >= 0) & (k >= 0)) if self.cross_view else ((kscope < 0) | (qscope == kscope))
            visible = scope & ((kchunk == qchunk) | allowed_history)
        if self.joint:
            visible = (
                ((q >= 0) & (k >= 0))
                if self.cross_view
                else ((kscope < 0) | (qscope < 0) | (qscope == kscope))
            )
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

    def block_mask(self, indices: torch.Tensor | None = None, block_size=128):
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

        # Keep sorting on the small block grid outside the compiled reduction.
        # Fusing both sorts and their transpose scatters creates huge ptxas kernels.
        return build_block_mask(mask_mod, size, size, self.kind.device, block_size)
