"""KV-cache offload helpers.

The self/ar KV cache is offloaded to pinned CPU during the AR rollout; a per-GPU memory
budget decides how many of the newest tokens stay resident on GPU (the write path in
``wan/modules/ar_model.py:handle_kv_cache_python`` does the actual placement / split).
This module only turns that GB budget into a token count so the policy math lives in one
place -- not scattered through the model or the inference pipeline.
"""

import operator


def gpu_token_budget(
        budget_gb, dim, num_layers, ar_interval, sp_size, n_branches,
        dtype_bytes=2, *, batch_size):
    """Self/ar KV sequence tokens to keep resident on GPU per cache so the total
    resident KV on THIS rank fits ``budget_gb``. Returns 0 (full offload) when budget_gb <= 0.

    A cache tensor is ``[B, S, N_local, D]``. One sequence token therefore costs
    ``batch_size * (dim // sp_size) * dtype_bytes`` bytes per layer, K/V side, and
    branch after the Ulysses head-scatter. There are ``2 (k+v) * n_subs * n_branches``
    such tensors per token, where
    ``n_subs = num_layers`` self-attn caches ``+ num_layers // ar_interval`` ar-attn caches, and
    ``n_branches`` is the number of growing caches (cond/uncond x high/low). ``batch_size``
    is the local cache batch dimension, not the global batch size or packed view count.
    """
    try:
        batch_size = operator.index(batch_size)
    except TypeError as exc:
        raise TypeError(f'batch_size must be an integer, got {batch_size!r}') from exc
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    if budget_gb <= 0:
        return 0
    n_subs = num_layers + num_layers // ar_interval
    bytes_per_token = (
        batch_size * (dim // sp_size) * dtype_bytes
        * 2 * n_subs * n_branches
    )
    return int(budget_gb * 1e9 / bytes_per_token) if bytes_per_token > 0 else 0


def resolve_kv_offload(
        config, force_zero, dim, num_layers, ar_interval, sp_size, n_branches,
        *, batch_size):
    """Resolve the runtime KV-offload knobs from ``config`` into the concrete values stamped
    onto each cache dict and read by the write paths: ``(offload_cpu, offload_crossattn,
    gpu_tokens, budget_gb)``.

    ``force_zero`` (set by the pipeline when a latent/chunk eviction path is active) drops the
    budget to 0 so every chunk offloads WHOLE (no token-split), keeping the eviction
    bookkeeping consistent. Keeping this next to ``gpu_token_budget`` means the whole GB->knobs
    policy lives in one place rather than inline in the pipeline. ``batch_size`` is the
    actual local cache B for this request.
    """
    offload_cpu = bool(getattr(config, 'kv_offload', False))
    offload_crossattn = bool(getattr(config, 'kv_offload_crossattn', True))
    budget_gb = 0.0 if force_zero else float(getattr(config, 'kv_gpu_budget_gb', 0.0))
    gpu_tokens = gpu_token_budget(
        budget_gb, dim, num_layers, ar_interval, sp_size,
        n_branches=n_branches, batch_size=batch_size)
    return offload_cpu, offload_crossattn, gpu_tokens, budget_gb
