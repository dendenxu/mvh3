"""Per-layer head-sharded history cache with a bounded GPU budget."""

import torch
from dataclasses import replace

from h3.modules.masking import CLEAN, CONDITION, TokenLayout, build_block_mask


class HistoryCache:

    def __init__(self, offload=True, budget_bytes=0, sink_chunks=0, window_chunks=0, sink_view0_only=False,
                 intervals=None, sink_frames=0, window_frames=0):
        self.offload, self.budget_bytes = offload, budget_bytes
        self.sink_chunks, self.window_chunks, self.sink_view0_only = sink_chunks, window_chunks, sink_view0_only
        self.intervals, self.sink_frames, self.window_frames = intervals, sink_frames, window_frames
        self.segments = []

    def clear(self):
        self.segments.clear()

    def read_and_append(self, key, value, layout, update):
        if torch.is_grad_enabled():
            raise RuntimeError("History caches are inference-only")
        old_count = sum(x[0].shape[1] for x in self.segments)
        layouts = [segment[2].to(key.device) for segment in self.segments] + [layout]
        combined = replace(layout, **{name: torch.cat([getattr(x, name) for x in layouts])
                                      for name in ("kind", "chunk", "scope", "active")})
        qn, kn = key.shape[1], old_count + key.shape[1]

        def mask_mod(b, h, q, k):
            return (q < qn) & (k < kn) & combined.mask_mod(b, h,
                                                           (q + old_count).clamp_max(kn - 1), k.clamp_max(kn - 1))

        mask = build_block_mask(mask_mod, qn, kn, key.device)
        # Copy CPU history directly into its final attention buffer. Moving
        # every segment before cat keeps two full GPU copies alive at once.
        joined_key = key.new_empty(key.shape[0], kn, *key.shape[2:])
        joined_value = value.new_empty(value.shape[0], kn, *value.shape[2:])
        offset = 0
        for old_key, old_value, _ in [*self.segments, (key, value, layout)]:
            count = old_key.shape[1]
            joined_key[:, offset:offset + count].copy_(old_key, non_blocking=True)
            joined_value[:, offset:offset + count].copy_(old_value, non_blocking=True)
            offset += count
        result = joined_key, joined_value, mask
        if update:
            select = (layout.kind == CLEAN) & layout.active
            if layout.single_sequence:
                select |= (layout.kind == CONDITION) & (layout.chunk >= 0) & layout.active
            if select.any():
                cache_layout = self._select(layout, select)
                self.segments.append((key[:, select].detach(), value[:, select].detach(), cache_layout))
                self._trim(int(layout.chunk[select].max()))
                self._place()
        return result

    @staticmethod
    def _select(layout, select):
        return replace(layout, **{name: getattr(layout, name)[select]
                                  for name in ("kind", "chunk", "scope", "active")})

    def _trim(self, newest):
        window = self.window_frames if self.intervals is not None else self.window_chunks
        if window <= 0:
            return
        segments = []
        for key, value, layout in self.segments:
            if self.intervals is None:
                sink = layout.chunk < self.sink_chunks
                recent = layout.chunk > newest - self.window_chunks
            else:
                sink, recent = torch.zeros_like(layout.active), torch.zeros_like(layout.active)
                for scope, times in enumerate(self.intervals):
                    times = times.to(layout.chunk.device)
                    selected = (layout.scope == scope) | (layout.scope < 0)
                    ids = layout.chunk.clamp(0, len(times) - 1)
                    end = times[min(newest, len(times) - 1), 1]
                    sink |= selected & (times[ids, 0] < self.sink_frames)
                    recent |= selected & (times[ids, 1] > end - self.window_frames)
            if self.sink_view0_only:
                sink &= layout.scope == 0
            keep = sink | recent
            if keep.any():
                segments.append((key[:, keep.to(key.device)], value[:, keep.to(value.device)],
                                 self._select(layout, keep)))
        self.segments = segments

    def _place(self):
        if not self.offload:
            return
        used, result = 0, []
        for key, value, layout in reversed(self.segments):
            count = key.shape[1]
            token_bytes = (key.numel() * key.element_size() + value.numel() * value.element_size()) // count
            keep = min(count, max(0, self.budget_bytes - used) // token_bytes)
            cut = count - keep
            if keep:
                resident_layout = self._select(layout, slice(cut, None))
                # Clone a split suffix so its storage cannot retain an offloaded prefix.
                resident_key = key[:, cut:].contiguous().clone() if cut else key
                resident_value = value[:, cut:].contiguous().clone() if cut else value
                result.append((resident_key, resident_value, resident_layout))
                used += keep * token_bytes
            if cut:
                host_key, host_value = key[:, :cut].cpu(), value[:, :cut].cpu()
                host_layout = self._select(layout, slice(None, cut)).to("cpu")
                if torch.cuda.is_available():
                    host_key = host_key if host_key.is_pinned() else host_key.pin_memory()
                    host_value = host_value if host_value.is_pinned() else host_value.pin_memory()
                result.append((host_key, host_value, host_layout))
        self.segments = list(reversed(result))


def make_caches(model, cfg, enabled=True, document=None):
    if not enabled:
        return None
    count = len(model.transformer_blocks)
    # The distilled CFG=1 path has only one stream and can use the whole budget.
    streams = 1 if cfg.guidance_scale == 1 else 2
    budget = int(cfg.kv_gpu_budget_gb * 1024**3 / (streams * count))
    options = {}
    if document is not None and any("generation_chunks" in view for view in document["views"]):
        from model.chunks import chunk_intervals
        # Existing sink/window knobs retain their WorldViews source-time units.
        options = dict(intervals=[chunk_intervals(view, cfg.chunk_size, cfg.h3.get("chunk_size_range") is not None)
                                  for view in document["views"]],
                       sink_frames=max(0, 4 * cfg.kv_sink_size - 3), window_frames=4 * cfg.kv_window_size)
    return [
        HistoryCache(cfg.kv_offload, budget, math_ceil_div(cfg.kv_sink_size, cfg.chunk_size),
                     math_ceil_div(cfg.kv_window_size, cfg.chunk_size), cfg.kv_sink_view0_only,
                     **options) for _ in range(count)
    ]


def math_ceil_div(value, divisor):
    return (int(value) + int(divisor) - 1) // int(divisor)
