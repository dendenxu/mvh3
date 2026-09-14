from copy import deepcopy

import torch
import pytest
from test_worldviews import recipe, dense_inputs
from fixtures_h3 import tiny_model, feature_document

from utils.config import validate_config
from h3.modules.kv_cache import HistoryCache
from model.diffusion import DiffusionObjective
from h3.modules.masking import CLEAN, NOISY, TokenLayout
from utils.checkpoint import load_checkpoint, save_checkpoint
from model.chunks import caption_chunk, view_chunk_ids, chunk_intervals, source_chunk_ids, prepare_chunk_plan


def random_recipe():
    cfg = recipe()
    cfg.h3.chunk_group_range = [1, 4]
    return cfg


@pytest.mark.parametrize("frames", [1, 5, 10, 17, 22, 37, 77, 137, 297])
def test_grouping_keeps_whole_source_chunks_and_all_decoder_support(frames):
    cfg, doc = random_recipe(), feature_document(frames=frames)
    view = doc["views"][0]
    source = source_chunk_ids(view, cfg.chunk_size)
    for seed in range(16):
        torch.manual_seed(seed)
        planned = prepare_chunk_plan(doc, cfg, "cpu")
        plan = planned["views"][0]["generation_chunks"]
        assert plan.shape == view["valid"].shape and plan[0] == 0
        assert ((plan[1:] - plan[:-1]) >= 0).all()
        mapping = []
        for chunk in range(int(source.max()) + 1):
            group = plan[source == chunk].unique()
            assert group.numel() == 1
            mapping.append(int(group[0]))
        counts = torch.bincount(torch.tensor(mapping))
        assert counts.sum() == int(source.max()) + 1 and ((counts >= 1) & (counts <= 4)).all()
        assert plan[~view["valid"]].eq(plan[view["valid"]][-1]).all()
        assert prepare_chunk_plan(planned, cfg, "cpu") is planned


def test_one_chunk_groups_exactly_reproduce_fixed_objective_without_rng_drift():
    cfg, doc = recipe(), feature_document(frames=77)
    torch.manual_seed(52)
    fixed, target, weights, _, _ = DiffusionObjective(cfg).pack(doc, "cpu")
    state = torch.get_rng_state()
    cfg.h3.chunk_group_range = [1, 1]
    torch.manual_seed(52)
    grouped, new_target, new_weights, _, _ = DiffusionObjective(cfg).pack(doc, "cpu")
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(dense_inputs(fixed), dense_inputs(grouped), rtol=0, atol=0)
    torch.testing.assert_close((target, weights), (new_target, new_weights), rtol=0, atol=0)


def test_disabled_partition_retains_original_recipe_rng_and_document():
    cfg, doc = recipe(), feature_document(frames=77)
    state = torch.get_rng_state()
    assert prepare_chunk_plan(doc, cfg, "cpu") is doc
    assert torch.equal(state, torch.get_rng_state())
    assert view_chunk_ids(doc["views"][0], cfg.chunk_size).max() == 3


def test_joint_views_share_partition_and_independent_videos_draw_separately():
    cfg, doc = random_recipe(), feature_document(views=4, frames=297)
    torch.manual_seed(31)
    joint = prepare_chunk_plan(doc, cfg, "cpu")
    plans = [view["generation_chunks"] for view in joint["views"]]
    assert all(torch.equal(plan, plans[0]) for plan in plans)
    assert all("generation_chunks" not in view for view in doc["views"])
    doc["isolated"] = True
    torch.manual_seed(31)
    independent = prepare_chunk_plan(doc, cfg, "cpu")
    plans = [view["generation_chunks"] for view in independent["views"]]
    assert any(not torch.equal(plan, plans[0]) for plan in plans[1:])
    assert any(len(torch.unique(torch.bincount(plan))) > 1 for plan in plans)


def test_source_captions_keep_distinct_features_tags_and_time_origins():
    cfg, doc = random_recipe(), feature_document(frames=77)
    view = doc["views"][0]
    view["generation_chunks"] = torch.tensor([0, 1, 1, 1])[source_chunk_ids(view, cfg.chunk_size)]
    view["texts"] = [(i, torch.full((1, i + 2, 32), float(i))) for i in range(4)]
    view["text_tag_specs"] = {i: torch.ones(i + 2, dtype=torch.long) for i in range(4)}
    assert [caption_chunk(view, i, cfg.chunk_size) for i in range(4)] == [0, 1, 1, 1]
    inputs, *_ = DiffusionObjective(cfg).pack(doc, "cpu", evaluation_sigma=0.5)
    n = sum(i + 2 for i in range(4))
    assert torch.equal(inputs["encoder_hidden_states"], torch.cat([x for _, x in view["texts"]], 1))
    assert inputs["attention_mask"].chunk[:n].tolist() == [0, 0] + [1] * (n - 2)

    # Newly merged future captions must not shift the established media origin.
    assert inputs["position_ids"][n, 0] == 2


def test_merged_block_is_bidirectional_but_cannot_read_the_next_block():
    cfg, doc = random_recipe(), feature_document(frames=77)
    view = doc["views"][0]
    source = source_chunk_ids(view, cfg.chunk_size)
    view["generation_chunks"] = torch.tensor([0, 0, 1, 1])[source]
    inputs, *_ = DiffusionObjective(cfg).pack(doc, "cpu", evaluation_sigma=0.5)
    layout = inputs["attention_mask"]
    noisy = torch.nonzero(layout.kind == NOISY).flatten()
    mask = layout.dense()[noisy[:, None], noisy]
    assert mask[source == 0][:, source == 1].all()
    assert mask[source == 1][:, source == 0].all()
    assert not mask[source < 2][:, source >= 2].any()
    assert not mask[source >= 2][:, source < 2].any()


def test_future_caption_changes_do_not_change_earlier_random_chunks():
    torch.manual_seed(173)
    cfg, doc, model = random_recipe(), feature_document(frames=77), tiny_model()
    model.configure_attention(cfg)
    view = doc["views"][0]
    view["generation_chunks"] = torch.tensor([0, 0, 1, 1])[source_chunk_ids(view, cfg.chunk_size)]
    view["texts"] = [(i, torch.randn(1, 3, 32)) for i in range(4)]
    changed = deepcopy(doc)
    changed["views"][0]["texts"][-1] = (3, torch.randn(1, 7, 32) * 10)
    results, positions = [], []
    for sample in (doc, changed):
        torch.manual_seed(65)
        inputs, *_ = DiffusionObjective(cfg).pack(sample, "cpu", evaluation_sigma=0.5)
        layout = inputs["attention_mask"]
        selected = (layout.kind == NOISY) & (layout.chunk == 0)
        positions.append(inputs["position_ids"][selected])
        with torch.no_grad():
            prediction = model(**dense_inputs(inputs)).sample
        results.append(prediction[:, selected[inputs["video_indices"]]])
    torch.testing.assert_close(positions[0], positions[1], rtol=0, atol=0)
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("legacy_fields", [False, True])
def test_pending_resampling_forcing_partition_is_retained_after_checkpoint_restore(tmp_path, legacy_fields):
    cfg, doc = random_recipe(), feature_document(frames=77)
    planned = prepare_chunk_plan(doc, cfg, "cpu")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    pending_key = "pending_rf" if legacy_fields else "pending_resampling_forcing"
    depth_key = "depth" if legacy_fields else "resampling_forcing_depth"
    runtime = {pending_key: (planned, [planned["views"][0]["latent"]]), depth_key: 2}
    checkpoint = save_checkpoint(model, optimizer, cfg, 1, 1, runtime, tmp_path)
    restored = load_checkpoint(model, optimizer, cfg, checkpoint)
    assert restored["runtime"]["resampling_forcing_depth"] == 2
    assert "pending_rf" not in restored["runtime"] and "depth" not in restored["runtime"]
    pending = restored["runtime"]["pending_resampling_forcing"][0]
    state = torch.get_rng_state()
    assert prepare_chunk_plan(pending, cfg, "cpu") is pending
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(pending["views"][0]["generation_chunks"], planned["views"][0]["generation_chunks"])


def test_variable_chunk_cache_window_uses_source_time_instead_of_chunk_count():
    view = feature_document(frames=77)["views"][0]
    view["generation_chunks"] = torch.tensor([0, 0, 1, 2])[source_chunk_ids(view, 5)]
    intervals = chunk_intervals(view, 5)
    assert intervals.tolist() == [[0, 37], [37, 57], [57, 77]]
    cache = HistoryCache(intervals=[intervals], sink_frames=1, window_frames=20)
    layout = TokenLayout(
        torch.full((3,), CLEAN),
        torch.arange(3),
        torch.zeros(3, dtype=torch.long),
        True,
        torch.ones(3, dtype=torch.bool),
    )
    key = torch.arange(3).reshape(1, 3, 1, 1).float()
    cache.segments = [(key, key.clone(), layout)]
    cache.trim_history(2)
    expected = (intervals[:, 0] < 1) | (intervals[:, 1] > 77 - 20)
    assert torch.equal(cache.segments[0][2].chunk, torch.arange(3)[expected])


def test_joint_views_with_different_lengths_share_source_chunk_boundaries():
    cfg, doc = random_recipe(), feature_document(views=2, frames=137)
    doc["views"][0] = feature_document(frames=77)["views"][0]
    planned = prepare_chunk_plan(doc, cfg, "cpu")
    for chunk in range(4):
        assert caption_chunk(planned["views"][0], chunk, 5) == caption_chunk(planned["views"][1], chunk, 5)
    assert prepare_chunk_plan(planned, cfg, "cpu") is planned


def test_saved_plan_cannot_split_an_original_caption_chunk():
    cfg, doc = random_recipe(), feature_document(frames=77)
    view = doc["views"][0]
    view["generation_chunks"] = torch.repeat_interleave(torch.arange(3), torch.tensor([3, 19, 5]))
    with pytest.raises(ValueError, match="cannot split"):
        prepare_chunk_plan(doc, cfg, "cpu")


def test_saved_plan_cannot_merge_more_than_the_configured_chunk_count():
    cfg, doc = random_recipe(), feature_document(frames=137)
    view = doc["views"][0]
    view["generation_chunks"] = torch.zeros_like(view["valid"], dtype=torch.long)
    with pytest.raises(ValueError, match="configured chunk range"):
        prepare_chunk_plan(doc, cfg, "cpu")


@pytest.mark.parametrize("bounds", [[0, 20], [20, 3], [3], [3.0, 20], True])
def test_invalid_chunk_ranges_fail_closed(bounds):
    cfg = recipe()
    cfg.h3.chunk_group_range = bounds
    with pytest.raises(ValueError, match="chunk_group_range"):
        validate_config(cfg)


def test_native_latent_range_requires_single_sequence():
    cfg = recipe()
    cfg.h3.chunk_size_range = [3, 20]
    with pytest.raises(ValueError, match="require single_sequence"):
        validate_config(cfg)
