"""Per-layer head-sharded history cache with a bounded GPU budget."""

import torch

from h3.modules.masking import CLEAN, TokenLayout


class HistoryCache:

    def __init__(self, offload=True, budget_bytes=0, sink_chunks=0, window_chunks=0, sink_view0_only=False):
        self.offload, self.budget_bytes = offload, budget_bytes
        self.sink_chunks, self.window_chunks, self.sink_view0_only = sink_chunks, window_chunks, sink_view0_only
        self.segments = []

    def clear(self):
        self.segments.clear()

    def read_and_append(self, key, value, layout, update):
        from torch.nn.attention.flex_attention import create_block_mask
        if torch.is_grad_enabled():
            raise RuntimeError("History caches are inference-only")
        old_count = sum(x[0].shape[1] for x in self.segments)
        layouts = [segment[2].to(key.device) for segment in self.segments] + [layout]
        combined = TokenLayout(
            *(torch.cat([getattr(x, name) for x in layouts]) for name in ("kind", "chunk", "scope")),
            layout.cross_view, torch.cat([x.active for x in layouts]))
        qn, kn = key.shape[1], old_count + key.shape[1]

        def mask_mod(b, h, q, k):
            return (q < qn) & (k < kn) & combined.mask_mod(b, h,
                                                           (q + old_count).clamp_max(kn - 1), k.clamp_max(kn - 1))

        mask = create_block_mask(mask_mod, None, None, qn, kn, device=key.device, _compile=True)
        keys = [s[0].to(key.device, non_blocking=True) for s in self.segments] + [key]
        values = [s[1].to(value.device, non_blocking=True) for s in self.segments] + [value]
        result = torch.cat(keys, 1), torch.cat(values, 1), mask
        if update:
            select = (layout.kind == CLEAN) & layout.active
            if select.any():
                cache_layout = TokenLayout(layout.kind[select], layout.chunk[select], layout.scope[select],
                                           layout.cross_view, layout.active[select])
                self.segments.append((key[:, select].detach(), value[:, select].detach(), cache_layout))
                self._trim(int(layout.chunk[select].max()))
                self._place()
        return result

    def _trim(self, newest):
        if self.window_chunks <= 0:
            return
        segments = []
        for key, value, layout in self.segments:
            sink = layout.chunk < self.sink_chunks
            if self.sink_view0_only:
                sink &= layout.scope == 0
            keep = sink | (layout.chunk > newest - self.window_chunks)
            if keep.any():
                segments.append((key[:, keep.to(key.device)], value[:, keep.to(value.device)],
                                 TokenLayout(layout.kind[keep], layout.chunk[keep], layout.scope[keep],
                                             layout.cross_view, layout.active[keep])))
        self.segments = segments

    def _place(self):
        if not self.offload:
            return
        used, result = 0, []
        for key, value, layout in reversed(self.segments):
            size = key.numel() * key.element_size() + value.numel() * value.element_size()
            keep = used + size <= self.budget_bytes
            if keep:
                used += size
            else:
                key, value, layout = key.cpu(), value.cpu(), layout.to("cpu")
                if torch.cuda.is_available():
                    key, value = key.pin_memory(), value.pin_memory()
            result.append((key, value, layout))
        self.segments = list(reversed(result))


def make_caches(model, cfg, enabled=True):
    if not enabled:
        return None
    count = len(model.transformer_blocks)
    # Positive and negative streams share the configured per-rank GPU budget.
    budget = int(cfg.kv_gpu_budget_gb * 1024**3 / (2 * count))
    return [
        HistoryCache(cfg.kv_offload, budget, math_ceil_div(cfg.kv_sink_size, cfg.chunk_size),
                     math_ceil_div(cfg.kv_window_size, cfg.chunk_size), cfg.kv_sink_view0_only) for _ in range(count)
    ]


def math_ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)
