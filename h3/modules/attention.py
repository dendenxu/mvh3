"""Explicit H3 attention kernels: SDPA for dense masks, compiled flex for sparse masks."""

import torch
from torch.nn.attention.flex_attention import flex_attention


def _flex(query, key, value, block_mask, kernel_options=None):
    # Dynamo may fall back after its variant limit; never run quadratic eager flex.
    if not torch.compiler.is_compiling():
        raise RuntimeError("Sparse H3 attention requires compilation; increase the variant limit or fix the graph")
    return flex_attention(query, key, value, block_mask=block_mask, kernel_options=kernel_options)


compiled_flex_attention = torch.compile(_flex, dynamic=False, fullgraph=True)


def dispatch_attention_fn(query,
                          key,
                          value,
                          attn_mask=None,
                          dropout_p=0.,
                          is_causal=False,
                          backend=None,
                          parallel_config=None):
    if parallel_config is not None:
        raise ValueError("Use the explicit WorldViews Ulysses path for sequence parallelism")
    if backend not in (None, "native", "sdpa"):
        raise ValueError(f"Unsupported dense attention backend: {backend}")
    result = torch.nn.functional.scaled_dot_product_attention(query.transpose(1, 2),
                                                              key.transpose(1, 2),
                                                              value.transpose(1, 2),
                                                              attn_mask=attn_mask,
                                                              dropout_p=dropout_p,
                                                              is_causal=is_causal)
    return result.transpose(1, 2)
