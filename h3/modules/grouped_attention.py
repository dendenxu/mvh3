"""Exact visibility groups for native FA4 varlen backward."""

import math
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import AuxRequest, flex_attention


@dataclass(frozen=True)
class VisibilityGroups:
    query: torch.Tensor
    key: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    max_query: int
    max_key: int
    sizes: torch.Tensor | None = None
    deterministic: bool = False

    def to(self, device, dynamic=False, deterministic=False):
        values = [x.to(device) for x in (self.query, self.key, self.cu_query, self.cu_key)]
        sizes = None
        if dynamic:
            # Only backward indexing varies here. Sparse forward Q/K/V and
            # mask compilation retain their original specialization policy.
            for value in values[1:]:
                torch._dynamo.mark_dynamic(value, 0)
            sizes = torch.tensor([self.max_query, self.max_key], dtype=torch.int64)
        return type(self)(*values, self.max_query, self.max_key, sizes, deterministic)


def visibility_groups(layout):
    """Partition queries by identical visible KV sets without an S-by-S mask."""
    layout = layout.to("cpu")
    size = layout.kind.numel()
    indices = torch.arange(size)
    active = torch.ones(size, dtype=torch.bool) if layout.active is None else layout.active
    selected = indices[active]
    fields = torch.stack((layout.kind[selected], layout.chunk[selected], layout.scope[selected]), dim=-1)
    _, group_ids = torch.unique(fields, dim=0, return_inverse=True)
    groups = {}
    for group in range(int(group_ids.max()) + 1 if group_ids.numel() else 0):
        queries = selected[group_ids == group]
        keys = indices[layout.mask_mod(0, 0, queries[0], indices)]
        if not keys.numel():
            raise ValueError("Grouped attention requires a visible key for each active query")
        identity = keys.numpy().tobytes()
        if identity in groups:
            groups[identity][0].append(queries)
        else:
            groups[identity] = ([queries], keys)
    pairs = [(torch.cat(queries), keys) for queries, keys in groups.values()]
    pairs.extend((index.reshape(1), index.reshape(1)) for index in indices[~active])
    q_sizes, k_sizes = ([pair[i].numel() for pair in pairs] for i in (0, 1))
    cu_q = torch.tensor([0, *q_sizes], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    cu_k = torch.tensor([0, *k_sizes], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return VisibilityGroups(
        torch.cat([q for q, _ in pairs]),
        torch.cat([k for _, k in pairs]),
        cu_q,
        cu_k,
        max(q_sizes),
        max(k_sizes),
    )


@torch.library.custom_op("mvh3::grouped_attention_backward", mutates_args=())
def grouped_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    q_order: torch.Tensor,
    k_order: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flash_attn.cute.interface import _flash_attn_bwd

    batch, heads, size, dim = query.shape
    offsets = torch.arange(batch, device=query.device)[:, None] * size
    q_indices, k_indices = (order[None].add(offsets).flatten() for order in (q_order, k_order))

    def gather(x, indices):
        return x.transpose(1, 2).reshape(batch * size, heads, dim).index_select(0, indices)

    q, k, v = (
        gather(x, indices) for x, indices in ((query, q_indices), (key, k_indices), (value, k_indices))
    )
    out, dout = (gather(x, q_indices) for x in (output, grad_output))
    grouped_lse = lse.transpose(1, 2).reshape(batch * size, heads).index_select(0, q_indices).T.contiguous()

    def batch_cu(cu, length):
        starts = cu[:-1][None] + torch.arange(batch, device=cu.device, dtype=torch.int32)[:, None] * length
        return torch.cat((starts.flatten(), cu.new_full((1,), batch * length)))

    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        grouped_lse,
        cu_seqlens_q=batch_cu(cu_q, q_order.numel()),
        cu_seqlens_k=batch_cu(cu_k, k_order.numel()),
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        deterministic=deterministic,
    )
    full_q = torch.empty_like(dq).index_copy_(0, q_indices, dq)
    restored = [full_q]

    # KV can occur in several groups. Sum in FP32 before the final BF16 cast.
    for gradient in (dk, dv):
        full = torch.zeros(batch * size, heads, dim, device=query.device, dtype=torch.float32)
        restored.append(full.index_add_(0, k_indices, gradient.float()))
    return tuple(
        torch.empty_like(original).copy_(gradient.reshape(batch, size, heads, dim).transpose(1, 2))
        for original, gradient in zip((query, key, value), restored)
    )


@grouped_backward.register_fake
def grouped_backward_fake(
    query,
    key,
    value,
    output,
    grad_output,
    lse,
    q_order,
    k_order,
    cu_q,
    cu_k,
    max_q,
    max_k,
    deterministic=False,
):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


class GroupedFlexAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, query, key, value, block_mask, q_order, k_order, cu_q, cu_k, max_q, max_k, deterministic
    ):
        output, auxiliary = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={"BACKEND": "FLASH"},
            return_aux=AuxRequest(lse=True),
        )

        # The pinned Torch FLASH template writes native ln-LSE; public flex
        # still applies its log2-to-ln conversion. Undo it for native backward.
        native_lse = auxiliary.lse / math.log(2.0)
        ctx.save_for_backward(query, key, value, output, native_lse, q_order, k_order, cu_q, cu_k)
        ctx.max_q, ctx.max_k = max_q, max_k
        ctx.deterministic = deterministic
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, lse, q_order, k_order, cu_q, cu_k = ctx.saved_tensors
        gradients = grouped_backward(
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            q_order,
            k_order,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            ctx.deterministic,
        )
        return (*gradients, *(None for _ in range(8)))


@torch.library.custom_op("mvh3::grouped_attention_backward_metadata", mutates_args=())
def grouped_backward_metadata(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    q_order: torch.Tensor,
    k_order: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    sizes: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CPU launch metadata avoids both scalar graph guards and GPU synchronization.
    max_q, max_k = sizes.tolist()
    return grouped_backward(
        query, key, value, output, grad_output, lse, q_order, k_order, cu_q, cu_k, max_q, max_k, deterministic
    )


@grouped_backward_metadata.register_fake
def grouped_backward_metadata_fake(
    query, key, value, output, grad_output, lse, q_order, k_order, cu_q, cu_k, sizes, deterministic=False
):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


class MetadataGroupedFlexAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, query, key, value, block_mask, q_order, k_order, cu_q, cu_k, sizes, deterministic):
        output, auxiliary = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={"BACKEND": "FLASH"},
            return_aux=AuxRequest(lse=True),
        )
        native_lse = auxiliary.lse / math.log(2.0)
        ctx.save_for_backward(query, key, value, output, native_lse, q_order, k_order, cu_q, cu_k, sizes)
        ctx.deterministic = deterministic
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, output, lse, q_order, k_order, cu_q, cu_k, sizes = ctx.saved_tensors
        gradients = grouped_backward_metadata(
            query,
            key,
            value,
            output,
            grad_output,
            lse,
            q_order,
            k_order,
            cu_q,
            cu_k,
            sizes,
            ctx.deterministic,
        )
        return (*gradients, *(None for _ in range(7)))
