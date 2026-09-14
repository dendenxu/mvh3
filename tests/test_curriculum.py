from pathlib import Path
import json

import torch
import yaml

from dataset.curriculum import short_mono_windows
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout


def test_all_views_and_tails_retained():
    counts = [297, 77, 10, 150]
    windows = short_mono_windows(counts)
    for view, count in enumerate(counts):
        covered = []
        for window in windows:
            if window.view == view:
                assert 0 < window.frame_count <= 77
                covered.extend(range(window.start, window.stop))
        assert covered == list(range(count))


def test_both_stages_use_the_same_full_sources():
    configs = Path(__file__).resolve().parents[1] / "configs"
    from omegaconf import OmegaConf
    from utils.config import load_config
    short = load_config(configs / "stage1_short_mono.yaml")
    full = load_config(configs / "stage2_long_multiview.yaml")
    assert OmegaConf.to_container(short.dataset) == OmegaConf.to_container(full.dataset)
    assert len(short.dataset.datasets) == 19
    assert short.h3.stage == 1 and full.h3.stage == 2
    assert short.model == full.model
    assert short.ar_lr == full.ar_lr == 1e-5
    assert short.h3.short_frames == 77


def test_masks_obey_teacher_forcing_and_view_isolation():
    kind = torch.tensor([CONDITION, CONDITION, CONDITION, CLEAN, CLEAN, NOISY, NOISY, CLEAN])
    chunk = torch.tensor([-1, 0, 1, 0, 1, 0, 1, 0])
    scope = torch.tensor([-1, 0, 0, 0, 0, 0, 0, 1])
    mask = TokenLayout(kind, chunk, scope, cross_view=False).dense()
    assert mask[0, 0] and mask[0].sum() == 1
    assert mask[5, 0] and mask[5, 1] and mask[5, 5]
    assert not mask[5, 2] and not mask[5, 3] and not mask[5, 4] and not mask[5, 6]
    assert mask[6, 3] and not mask[6, 4] and not mask[6, 5]
    assert not mask[3, 7] and not mask[7, 3]
    assert TokenLayout(kind, chunk, scope, cross_view=True).dense()[3, 7]
    assert mask.any(dim=-1).all()


def test_split_block_mask_matches_reference_with_padding_and_subsets():
    from torch.nn.attention.flex_attention import create_block_mask

    ids = torch.arange(389)
    kind = (ids // 47) % 3
    chunk = (ids // 29) % 4
    scope = (ids // 97) % 2
    scope[(kind == CONDITION) & (ids < 31)] = -1
    dropout = torch.eye(4, dtype=torch.bool).roll(-1, dims=0)
    for joint, cross_view, history in ((False, False, True), (False, True, True),
                                       (False, False, False), (True, True, True), (True, False, True)):
        for active in (None, ids % 31 != 0):
            layout = TokenLayout(kind, chunk, scope, cross_view, active, dropout, history, joint)
            for indices in (None, ids[::2].flip(0)):
                for block_size in (128, (640, 128)):
                    actual = layout.block_mask(indices, block_size=block_size)
                    size = len(ids) if indices is None else len(indices)
                    expected = create_block_mask(actual.mask_mod, None, None, size, size,
                                                 device="cpu", BLOCK_SIZE=block_size, _compile=False)
                    for prefix in ("kv", "q", "full_kv", "full_q"):
                        for suffix in ("num_blocks", "indices"):
                            field = f"{prefix}_{suffix}"
                            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
                    torch.testing.assert_close(layout.dense(indices),
                                               actual.mask_mod(0, 0, torch.arange(size)[:, None],
                                                               torch.arange(size)[None, :]))


def test_rectangular_cache_mask_matches_native_metadata_and_padding():
    from torch.nn.attention.flex_attention import create_block_mask
    from h3.modules.masking import build_block_mask

    for query_length, key_length in ((37, 533), (257, 1207), (389, 389)):
        ids = torch.arange(key_length)
        old_count = key_length - query_length
        kind = torch.where(ids < old_count, CLEAN, NOISY)
        chunk, scope = ids // 73, (ids // 137) % 2
        for single, joint, cross_view in ((False, False, True), (True, False, True),
                                          (True, False, False), (False, True, False)):
            layout = TokenLayout(kind, chunk, scope, cross_view, ids % 31 != 0,
                                 torch.eye(17, dtype=torch.bool), joint=joint, single_sequence=single)

            def mask_mod(b, h, q, k):
                return ((q < query_length) & (k < key_length)
                        & layout.mask_mod(b, h, (q + old_count).clamp_max(key_length - 1),
                                           k.clamp_max(key_length - 1)))

            actual = build_block_mask(mask_mod, query_length, key_length, "cpu")
            expected = create_block_mask(mask_mod, None, None, query_length, key_length, device="cpu")
            assert actual.BLOCK_SIZE == expected.BLOCK_SIZE
            for prefix in ("kv", "q", "full_kv", "full_q"):
                for suffix in ("num_blocks", "indices"):
                    field = f"{prefix}_{suffix}"
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
            q, k = torch.arange(query_length + 17)[:, None], torch.arange(key_length + 17)[None, :]
            visible = actual.mask_mod(0, 0, q, k)
            torch.testing.assert_close(visible, expected.mask_mod(0, 0, q, k), rtol=0, atol=0)
            assert not visible[query_length:].any() and not visible[:, key_length:].any()


def test_nested_compile_keeps_mask_reduction_separate_from_grid_sort():
    from h3.modules.masking import build_block_mask

    labels = torch.arange(533) // 128
    graphs = []

    def predicate(b, h, q, k):
        return ((q < 37) & (k < 533)
                & (labels[k.clamp_max(532)] <= labels[(q + 496).clamp_max(532)]))

    def outer():
        mask = build_block_mask(predicate, 37, 533, "cpu")
        return mask.kv_num_blocks, mask.kv_indices, mask.q_num_blocks, mask.q_indices

    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward

    expected = outer()
    actual = torch.compile(outer, backend=capture)()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for graph in graphs:
        nodes = list(graph.graph.nodes)
        has_reduction = any(node.target == "sum" and node.kwargs.get("dim") == (3, 5) for node in nodes)
        has_sort = any("argsort" in str(node.target) for node in nodes)
        assert not (has_reduction and has_sort), "Outer compilation fused the quadratic reduction with grid sorting"
