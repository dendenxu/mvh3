from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from fixtures_h3 import feature_document, tiny_model
from test_worldviews import recipe, dense_inputs
from h3.modules.grouped_attention import visibility_groups
from h3.modules.kv_cache import HistoryCache
from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout
from model.chunks import prepare_chunk_plan, prepare_clean_prefix
from model.diffusion import WorldViewsObjective
from utils.captions import caption_specs
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import validate_config


def df_recipe():
    cfg = recipe()
    cfg.h3.single_sequence = True
    cfg.h3.chunk_size_range = [3, 20]
    cfg.h3.clean_prefix_probability = 1.
    cfg.h3.caption_overlap_threshold = .5
    cfg.context_noise = cfg.context_noise_std = cfg.inference_context_noise = 0.
    cfg.history_dropout_ratio = 0.
    cfg.h3.condition_noise = 0.
    cfg.resampling_forcing_clean_chunks = 0
    return cfg


def planned_document(views=1):
    doc = feature_document(views=views, frames=77)
    for view in doc["views"]:
        view["generation_chunks"] = torch.repeat_interleave(torch.arange(4), torch.tensor([5, 5, 5, 12]))
        view["clean_prefix_chunks"] = 1
        view["texts_by_bd"] = True
        view["texts"] = [(i, torch.randn(1, 3, 32)) for i in range(4)]
    return doc


@pytest.mark.parametrize("frames", [1, 5, 10, 17, 22, 37, 77, 137, 297])
def test_random_sizes_are_final_before_the_clean_cut(frames):
    cfg, original = df_recipe(), feature_document(frames=frames)
    for seed in range(12):
        torch.manual_seed(seed)
        planned = prepare_chunk_plan(original, cfg, "cpu", synchronize=False)
        chunks = planned["views"][0]["generation_chunks"].clone()
        document = prepare_clean_prefix(planned, cfg, "cpu")
        view = document["views"][0]
        assert torch.equal(view["generation_chunks"], chunks)
        sizes = torch.bincount(chunks)
        assert (sizes <= 20).all() and ((sizes >= 3).all() or len(chunks) < 3)
        assert 0 <= view["clean_prefix_chunks"] < len(sizes)
        assert chunks[~view["valid"]].eq(chunks[view["valid"]][-1]).all()
        state = torch.get_rng_state()
        assert WorldViewsObjective(cfg).prepare_document(document, "cpu") is document
        assert torch.equal(state, torch.get_rng_state())


def test_single_video_copy_clean_prefix_and_per_chunk_noise():
    cfg, doc = df_recipe(), planned_document(2)
    doc["isolated"] = True
    objective = WorldViewsObjective(cfg)
    objective.sample_sigmas = lambda count, device: (torch.arange(1, count + 1) / (count + 1), torch.ones(count), False)
    inputs, targets, weights, records, _ = objective.pack(doc, "cpu")
    layout = inputs["attention_mask"]
    assert inputs["hidden_states"].shape[1] == sum(v["latent"].shape[2] for v in doc["views"])
    assert len(records) == 2 and layout.single_sequence
    for i, (view, record) in enumerate(zip(doc["views"], records)):
        clean = view["generation_chunks"] < view["clean_prefix_chunks"]
        assert torch.equal(record["noisy"][:, :, clean], view["latent"][:, :, clean])
        assert record["sigmas"][clean].eq(0).all()
        selected_weights = weights[record["start"]:record["stop"]]
        assert selected_weights[clean].eq(0).all() and selected_weights[~clean].gt(0).all()
        assert record["sigmas"][~clean].unique().numel() == 3
        assert selected_weights[~view["valid"]].gt(0).all()
        token_ids = inputs["video_indices"][record["start"]:record["stop"]]
        assert layout.kind[token_ids][clean].eq(CLEAN).all()
        assert layout.kind[token_ids][~clean].eq(NOISY).all()
    assert not torch.equal(records[0]["sigmas"], records[1]["sigmas"])
    prediction = torch.randn_like(targets, requires_grad=True)
    loss = ((prediction - targets).square().mean(-1)[0] * weights).sum() / weights.sum()
    loss.backward()
    assert prediction.grad[:, weights == 0].eq(0).all()
    assert prediction.grad[:, weights > 0].abs().sum() > 0


def test_joint_variable_lengths_keep_chunk_bounds_and_decoder_support():
    cfg, doc = df_recipe(), feature_document(views=3, frames=137)
    doc["views"][0] = feature_document(frames=37)["views"][0]
    doc["views"][1] = feature_document(frames=77)["views"][0]
    for seed in range(20):
        torch.manual_seed(seed)
        planned = prepare_chunk_plan(doc, cfg, "cpu")
        longest = planned["views"][-1]["generation_chunks"]
        for view in planned["views"]:
            plan, valid = view["generation_chunks"], view["valid"]
            sizes = torch.bincount(plan)
            assert ((sizes >= 3) & (sizes <= 20)).all()
            assert torch.equal(plan, longest[:len(plan)])
            assert plan[~valid].eq(plan[valid][-1]).all()
        assert prepare_chunk_plan(planned, cfg, "cpu") is planned


@pytest.mark.parametrize("sizes", [[21, 6], [13, 13, 1]])
def test_saved_native_partition_validates_all_chunk_sizes(sizes):
    cfg, doc = df_recipe(), feature_document(views=2, frames=77)
    for view in doc["views"]:
        view["valid"] = torch.ones(27, dtype=torch.bool)
        view["generation_chunks"] = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    with pytest.raises(ValueError, match="configured chunk range"):
        prepare_chunk_plan(doc, cfg, "cpu")


def test_shared_causality_all_modalities_and_grouped_backward_edges():
    kind = torch.tensor([CONDITION, CONDITION, CLEAN, CONDITION, NOISY, NOISY, CONDITION, NOISY])
    chunk = torch.tensor([-1, 0, 0, 1, 1, 1, 2, 2])
    scope = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])
    layout = TokenLayout(kind, chunk, scope, cross_view=False, single_sequence=True)
    mask = layout.dense()
    assert mask[3, 4] and mask[4, 3] and mask[5, 4]
    assert mask[6, 4] and not mask[4, 6] and not mask[6, 7]
    assert not mask[0, 1] and mask.diag().all()
    plan = visibility_groups(layout)
    actual = torch.zeros_like(mask)
    for i in range(len(plan.cu_query) - 1):
        q = plan.query[plan.cu_query[i]:plan.cu_query[i + 1]]
        k = plan.key[plan.cu_key[i]:plan.cu_key[i + 1]]
        actual[q[:, None], k] = True
    assert torch.equal(actual, mask)
    dropped = replace(layout, history_dropout=torch.ones(3, 3, dtype=torch.bool)).dense()
    assert not dropped[6, 4] and dropped[4, 3] and dropped[4, 0]


def test_future_video_and_caption_length_cannot_change_earlier_blocks():
    cfg, doc, model = df_recipe(), planned_document(), tiny_model()
    model.configure_attention(cfg)
    changed = deepcopy(doc)
    changed["views"][0]["texts"][-1] = (3, torch.randn(1, 11, 32) * 50)
    changed["views"][0]["latent"][:, :, 15:] *= 50
    results, masks = [], []
    for sample in (doc, changed):
        torch.manual_seed(98)
        inputs, *_ = WorldViewsObjective(cfg).pack(sample, "cpu", evaluation_sigma=.5)
        layout = inputs["attention_mask"]
        text = inputs["text_indices"]
        masks.append(layout.dense(text))
        with torch.no_grad():
            output = model(**dense_inputs(inputs)).sample
        results.append(output[:, layout.chunk[inputs["video_indices"]] < 3])
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-6)
    assert not masks[0][:9, 9:].any() and not masks[1][:9, 9:].any()


def test_caption_majority_is_strict_and_uses_original_caption_duration():
    cfg = df_recipe()
    # A synthetic frame clock makes exact majority/tie boundaries explicit.
    view = dict(frames=torch.tensor([0., 17., 27., 37., 48., 57.]),
                generation_chunks=torch.arange(6), source_frames=77, source_start=0,
                caption_source_frames=77, caption_scene="A room.",
                caption_motions=["Walk in.", "Turn left.", "Sit down.", "Wave."])
    assert caption_specs(view, cfg) == [
        (0, "A room.\nWalk in."), (1, "A room."), (2, "A room."),
        (3, "A room.\nSit down."), (4, "A room."), (5, "A room.\nWave.")]
    # Slicing a clip never shrinks the original window's majority denominator.
    view.update(frames=torch.tensor([0.]), generation_chunks=torch.tensor([0]),
                source_start=37, source_frames=10)
    assert caption_specs(view, cfg) == [(0, "A room.")]
    view["source_frames"] = 11
    assert caption_specs(view, cfg) == [(0, "A room.\nSit down.")]


def test_short_piece_uses_its_sliced_motions_without_parent_narrative_or_labels():
    cfg, view = df_recipe(), planned_document()["views"][0]
    view.update(caption_scene="A street.", caption_motions=["Turn.", "Stop.", "Wait.", "Leave."],
                caption_source_frames=77, prompt="Wrong unsliced parent story.")
    specs = caption_specs(view, cfg)
    assert len(specs) == 4
    assert all("parent" not in text and "[CHUNK]" not in text for _, text in specs)
    view["caption_motions"] = []
    assert all(text == "A street." for _, text in caption_specs(view, cfg))
    view.update(caption_motions=None, chunk_prompts=None, prompt="A stationary object.")
    assert all(text == "A stationary object." for _, text in caption_specs(view, cfg))


def test_rf_promotes_predicted_block_and_resume_keeps_both_plan_and_cut(tmp_path):
    cfg, document = df_recipe(), planned_document()
    cfg.resampling_forcing = True
    cfg.resampling_forcing_warmup_steps = 0
    cfg.resampling_forcing_clean_chunks = 0
    objective = WorldViewsObjective(cfg)
    objective.sample_sigmas = lambda count, device: (torch.full((count,), .5), torch.ones(count), False)
    _, log = objective(lambda **x: SimpleNamespace(sample=torch.zeros_like(x["hidden_states"])), document, "cpu", 0)
    assert log["rf"]
    pending = objective.resample_document(document)
    assert pending["views"][0]["clean_prefix_chunks"] == 2
    assert torch.equal(pending["views"][0]["generation_chunks"], document["views"][0]["generation_chunks"])
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = save_checkpoint(model, optimizer, cfg, 1, 1, {"pending_rf": (pending, log["x0"])}, tmp_path)
    saved, override = load_checkpoint(model, optimizer, cfg, path)["runtime"]["pending_rf"]
    rng = torch.get_rng_state()
    assert objective.prepare_document(saved, "cpu") is saved
    assert torch.equal(rng, torch.get_rng_state())
    _, _, weights, records, _ = objective.pack(saved, "cpu", override=override)
    promoted = saved["views"][0]["generation_chunks"] == 1
    torch.testing.assert_close(records[0]["noisy"][:, :, promoted], override[0][:, :, promoted], rtol=0, atol=0)
    assert not torch.equal(override[0][:, :, promoted], saved["views"][0]["latent"][:, :, promoted])
    assert weights[promoted].eq(0).all()


def test_default_df_rf_does_not_replace_its_first_prediction_with_ground_truth():
    from pathlib import Path
    from utils.config import load_config

    default = load_config(Path(__file__).resolve().parents[1] / "configs/diffusion_forcing.yaml")
    assert default.resampling_forcing_clean_chunks == 0
    cfg, document = df_recipe(), planned_document(2)
    cfg.resampling_forcing = True
    cfg.resampling_forcing_warmup_steps = 0
    cfg.resampling_forcing_clean_chunks = default.resampling_forcing_clean_chunks
    for view in document["views"]:
        view["clean_prefix_chunks"] = 0
    objective = WorldViewsObjective(cfg)
    objective.sample_sigmas = lambda count, device: (torch.full((count,), .5), torch.ones(count), False)
    _, log = objective(lambda **x: SimpleNamespace(sample=torch.zeros_like(x["hidden_states"])), document, "cpu", 0)
    assert log["rf"]
    pending = objective.resample_document(document)
    _, _, weights, records, _ = objective.pack(pending, "cpu", override=log["x0"])
    for i, (view, record) in enumerate(zip(pending["views"], records)):
        prefix = view["generation_chunks"] == 0
        assert view["clean_prefix_chunks"] == 1
        torch.testing.assert_close(record["noisy"][:, :, prefix], log["x0"][i][:, :, prefix], rtol=0, atol=0)
        assert not torch.equal(record["noisy"][:, :, prefix], view["latent"][:, :, prefix])
        assert weights[record["start"]:record["stop"]][prefix].eq(0).all()


def test_inference_never_samples_or_reads_ground_truth_clean_prefix():
    cfg, doc = df_recipe(), planned_document()
    view = doc["views"][0]
    view.pop("clean_prefix_chunks")
    objective = WorldViewsObjective(cfg)
    rng = torch.get_rng_state()
    planned = objective.prepare_document(doc, "cpu", training=False)
    assert "clean_prefix_chunks" not in planned["views"][0]
    assert torch.equal(rng, torch.get_rng_state())
    current = [torch.randn_like(view["latent"])]
    state = dict(chunk=2, sigma=.5, current=current, conditions=[None], cached=True)
    inputs, *_ = objective.pack(planned, "cpu", inference=state)
    layout = inputs["attention_mask"]
    assert layout.chunk.max() == 2
    assert inputs["text_attention_mask"][-1, 0]
    assert not layout.active[:6].any() and layout.active[6:9].all()
    changed = deepcopy(planned)
    changed["views"][0]["latent"].fill_(float("nan"))
    other, *_ = objective.pack(changed, "cpu", inference=state)
    torch.testing.assert_close(inputs["hidden_states"], other["hidden_states"], rtol=0, atol=0)


def test_source_stream_encodes_the_plan_before_sampling_a_clean_prefix(monkeypatch):
    from collections import deque
    from trainer.diffusion import SourceStream
    from utils import distributed as groups

    cfg, raw = df_recipe(), feature_document(frames=77)
    cfg.h3.text_conditioning = "text_only"
    cfg.cond_text_dropout_ratio = 0.
    view = raw["views"][0]
    view.update(caption_scene="A person.", caption_motions=["Walk.", "Stop.", "Sit.", "Wave."],
                caption_source_frames=77)
    requests = []
    def encode(captions):
        requests.extend(captions)
        return [torch.zeros(1, 2, 32) for _ in captions]
    captured = []
    def gather(document):
        captured.append(deepcopy(document))
        return [document]
    monkeypatch.setattr(groups, "gather_mixed_batch", gather)
    stream = SourceStream.__new__(SourceStream)
    stream.cfg, stream.validation = cfg, False
    stream.video = SimpleNamespace(device="cpu", prepare=lambda document, *args: document)
    stream.text, stream.negative = encode, None
    stream.pending, stream.mixed = deque([raw]), deque()
    document = stream.next()
    planned = captured[0]["views"][0]
    assert "generation_chunks" in planned and "clean_prefix_chunks" not in planned
    assert planned["texts_by_bd"]
    assert requests[:-1] == [caption for _, caption in caption_specs(planned, cfg)]
    trained = WorldViewsObjective(cfg).prepare_document(document, "cpu")
    assert torch.equal(trained["views"][0]["generation_chunks"], planned["generation_chunks"])


def test_cache_keeps_each_past_caption_once_with_its_clean_media(monkeypatch):
    import torch.nn.attention.flex_attention as flex
    original = flex.create_block_mask
    monkeypatch.setattr(flex, "create_block_mask", lambda *a, **kw: original(*a, **{**kw, "_compile": False}))
    cache = HistoryCache(offload=False)
    cfg, document = df_recipe(), planned_document()
    objective = WorldViewsObjective(cfg)
    with torch.no_grad():
        for chunk in range(4):
            inputs, *_ = objective.pack(document, "cpu", inference=dict(
                chunk=chunk, sigma=0., current=[document["views"][0]["latent"]],
                conditions=[None], cached=True, update_cache=True))
            layout = inputs["attention_mask"]
            key = torch.zeros(1, len(layout.kind), 1, 2)
            cache.read_and_append(key, key, layout, True)
    kinds = torch.cat([segment[2].kind for segment in cache.segments])
    assert (kinds == CONDITION).sum() == 12
    assert (kinds == CLEAN).sum() == 27
    assert all(segment[2].single_sequence for segment in cache.segments)


def test_independent_and_joint_views_cut_only_after_their_shared_partition():
    cfg, document = df_recipe(), feature_document(views=3, frames=137)
    torch.manual_seed(31)
    joint = WorldViewsObjective(cfg).prepare_document(document, "cpu")
    first = joint["views"][0]
    assert all(torch.equal(v["generation_chunks"], first["generation_chunks"]) for v in joint["views"])
    assert all(v["clean_prefix_chunks"] == first["clean_prefix_chunks"] for v in joint["views"])
    document["isolated"] = True
    torch.manual_seed(31)
    isolated = WorldViewsObjective(cfg).prepare_document(document, "cpu")
    assert any(not torch.equal(v["generation_chunks"], isolated["views"][0]["generation_chunks"])
               for v in isolated["views"][1:])


@pytest.mark.parametrize("views,isolated,offload", [(1, True, False), (2, False, False), (2, True, False), (2, False, True)])
def test_df_cache_and_recompute_rollout_agree_with_causal_text(monkeypatch, views, isolated, offload):
    import h3.modules.model as module
    import torch.nn.attention.flex_attention as flex
    from pipeline.ar_inference import generate

    original = flex.create_block_mask
    monkeypatch.setattr(flex, "create_block_mask", lambda *a, **kw: original(*a, **{**kw, "_compile": False}))
    def dense_attention(q, k, v, block_mask, **kw):
        qi, ki = torch.arange(q.shape[-2]), torch.arange(k.shape[-2])
        mask = block_mask.mask_mod(0, 0, qi[:, None], ki[None, :])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    monkeypatch.setattr(module, "compiled_flex_attention", dense_attention)
    cfg, doc, model = df_recipe(), planned_document(views), tiny_model()
    doc["isolated"] = isolated
    if not isolated:
        for view in doc["views"][1:]:
            view["texts"] = doc["views"][0]["texts"]
    model.configure_attention(cfg)
    cfg.kv_sink_size, cfg.kv_window_size = 1, 5
    cfg.sampling_steps = 3
    cfg.kv_offload = offload
    outputs = []
    for cached in (True, False):
        torch.manual_seed(114)
        outputs.append(generate(model, doc, None, cfg, "cpu", use_cache=cached))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_single_sequence_recipe_rejects_caption_modes_and_conflicting_partitions():
    cfg = df_recipe()
    validate_config(cfg)
    cfg.h3.caption_mode = "global"
    with pytest.raises(ValueError, match="Caption modes"):
        validate_config(cfg)
    del cfg.h3.caption_mode
    cfg.h3.chunk_group_range = [1, 4]
    with pytest.raises(ValueError, match="no chunk_group_range"):
        validate_config(cfg)


def test_raw_i2v_request_uses_actual_fps_and_the_training_caption_partition(tmp_path):
    import numpy as np
    from PIL import Image
    from h3.data import temporal_layout
    from pipeline.i2v_input import prepare_request

    cfg = df_recipe()
    Image.new("RGB", (32, 32)).save(tmp_path / "image.png")
    camera = np.zeros((77, 10), np.float32)
    camera[:, :2] = 1.
    np.save(tmp_path / "camera.npy", camera)
    request = dict(fps=16, prompt="A room.", scene="A room.", chunks=["Enter.", "Turn.", "Sit.", "Wave."],
                   views=[dict(image="image.png", camera="camera.npy")])
    image_calls, captions = [], []
    def encode_image(pixels, generator):
        image_calls.append(pixels.shape)
        return torch.zeros(1, 24, 1, 2, 2), temporal_layout(1), torch.ones(1, 1)
    def encode_text(pairs):
        captions.extend(caption for caption, _ in pairs)
        return [dict(features=torch.zeros(1, 3, 32), tags=torch.ones(3, dtype=torch.long)) for _ in pairs]
    video = SimpleNamespace(device="cpu", encode=encode_image)
    text = SimpleNamespace(i2v=encode_text)
    document = prepare_request(request, tmp_path, video, text, cfg)
    view = document["views"][0]
    assert view["fps"] == 16 and "clean_prefix_chunks" not in view
    assert image_calls == [torch.Size([1, 3, 32, 32])]
    assert view["latent"].eq(0).all()
    assert captions == [caption for _, caption in caption_specs(view, cfg)]
    assert [i for i, _ in view["texts"]] == list(range(int(view["generation_chunks"].max()) + 1))
    with pytest.raises(ValueError, match="24 FPS"):
        prepare_request(request, tmp_path, video, text, cfg, chunked=False)


def test_fixed_video_feature_bank_rebinds_captions_after_each_new_partition():
    cfg, doc = df_recipe(), feature_document(frames=77)
    view = doc["views"][0]
    motions = ["Walk.", "Stop.", "Turn.", "Wave."]
    captions = ["A room."] + ["\n".join(["A room.", *motions[start:stop]])
                              for start in range(4) for stop in range(start + 1, 5)]
    view.update(caption_scene="A room.", caption_motions=motions, caption_source_frames=77,
                caption_feature_bank={text: dict(features=torch.full((1, 3, 32), float(i)),
                                                tags=torch.ones(3, dtype=torch.long))
                                      for i, text in enumerate(captions)})
    objective = WorldViewsObjective(cfg)
    plans = set()
    for seed in range(12):
        torch.manual_seed(seed)
        prepared = objective.prepare_document(doc, "cpu")["views"][0]
        plans.add(tuple(prepared["generation_chunks"].tolist()))
        assert "generation_chunks" not in view
        for (chunk, caption), (text_chunk, embedding) in zip(caption_specs(prepared, cfg), prepared["texts"]):
            assert chunk == text_chunk
            assert torch.equal(embedding, view["caption_feature_bank"][caption]["features"])
    assert len(plans) > 1
