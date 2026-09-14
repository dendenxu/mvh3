import itertools

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from h3.modules.grouped_attention import visibility_groups
from h3.modules.masking import CONDITION, TokenLayout


def make_layout(joint, cross_view, history, dropped, padded):
    ids = torch.arange(67)
    kind, chunk, scope = ids % 3, (ids // 3) % 4, (ids // 12) % 2
    scope[(kind == CONDITION) & (ids % 6 == 0)] = -1
    active = ids % 17 != 0 if padded else None
    dropout = torch.eye(4, dtype=torch.bool).roll(1, 0) if dropped else None
    return TokenLayout(kind, chunk, scope, cross_view, active, dropout, history, joint)


@pytest.mark.parametrize("options", list(itertools.product((False, True), repeat=5)))
def test_visibility_groups_preserve_all_edges(options):
    layout = make_layout(*options)
    plan = visibility_groups(layout)
    reconstructed = torch.zeros(67, 67, dtype=torch.bool)
    for i in range(plan.cu_query.numel() - 1):
        queries = plan.query[plan.cu_query[i] : plan.cu_query[i + 1]]
        keys = plan.key[plan.cu_key[i] : plan.cu_key[i + 1]]
        reconstructed[queries[:, None], keys] = True
    assert torch.equal(reconstructed, layout.dense())
    assert torch.equal(plan.query.sort().values, torch.arange(67))
    assert plan.max_query == int(plan.cu_query.diff().max())
    assert plan.max_key == int(plan.cu_key.diff().max())


@pytest.mark.parametrize(
    "options",
    [
        (False, False, True, True, True),
        (False, True, False, False, True),
        (True, True, True, False, False),
        (True, False, True, True, True),
    ],
)
def test_grouped_forward_and_gradients_match_dense(options):
    torch.manual_seed(174)
    layout = make_layout(*options)
    plan = visibility_groups(layout)
    qkv = tuple(torch.randn(2, 2, 67, 8, dtype=torch.float64, requires_grad=True) for _ in range(3))
    q, k, v = qkv
    expected = scaled_dot_product_attention(q, k, v, attn_mask=layout.dense())
    actual = torch.zeros_like(q)
    for i in range(plan.cu_query.numel() - 1):
        queries = plan.query[plan.cu_query[i] : plan.cu_query[i + 1]]
        keys = plan.key[plan.cu_key[i] : plan.cu_key[i + 1]]
        block = scaled_dot_product_attention(q[:, :, queries], k[:, :, keys], v[:, :, keys])
        actual = actual.index_copy(2, queries, block)
    upstream = torch.randn_like(actual)
    expected_grad = torch.autograd.grad(expected, qkv, upstream)
    actual_grad = torch.autograd.grad(actual, qkv, upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_joint_visibility_merges_equivalent_query_classes():
    layout = make_layout(True, True, True, True, True)
    plan = visibility_groups(layout)
    assert plan.cu_query.numel() - 1 == 1 + int((~layout.active).sum())
    assert plan.key.numel() == layout.kind.numel()


def test_inactive_rows_keep_independent_self_edges():
    layout = TokenLayout(
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        active=torch.zeros(3, dtype=torch.bool),
    )
    plan = visibility_groups(layout)
    assert torch.equal(plan.query, torch.arange(3))
    assert torch.equal(plan.key, plan.query)
    assert torch.equal(plan.cu_query, torch.arange(4, dtype=torch.int32))
