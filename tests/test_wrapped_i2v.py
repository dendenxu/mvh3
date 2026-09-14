import copy

import pytest
import torch

from fixtures_h3 import tiny_model
from test_overfit_native import conditioned_document, native_recipe
from h3.distributed.fsdp import configure_model
from h3.modules.camera import camera_projection
from model.diffusion import WorldViewsObjective
from pipeline.joint_inference import joint_inputs


def recipe():
    cfg = native_recipe()
    cfg.model.prope_unwrapped = False
    cfg.h3.scale_conditioning = "camera_geometry"
    cfg.model.fa4 = False
    return cfg


def static_document(frames=22):
    doc = conditioned_document(frames)
    view = doc["views"][0]
    pose = torch.tensor([.83, 1.4, -.13, .06, .23, -.41, .12, 1.1, -2.4, 3.7])
    for part in (view, view["condition"]):
        part["pose"][:] = pose
        matrix = camera_projection(part["pose"][None])
        part["projection"], part["inverse"] = matrix.projection[0], matrix.inverse[0]
    return doc


def test_wrapped_static_camera_matches_initialized_native_all_layers():
    torch.manual_seed(419)
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    signature = configure_model(model, cfg)
    conditions = WorldViewsObjective(cfg).condition_latents(doc, "cpu")
    current = [torch.randn_like(doc["views"][0]["latent"])]
    for sigma in (.999, .5, .001):
        inputs, _ = joint_inputs(doc, current, conditions, sigma, cfg, "cpu")
        inputs["attention_mask"] = inputs["attention_mask"].dense()
        base = {k: v for k, v in inputs.items() if not k.startswith("camera_")}
        with torch.no_grad():
            actual = model(**inputs).sample
            expected = model(**base).sample
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert signature == {n: tuple(p.shape) for n, p in model.named_parameters()}


@pytest.mark.parametrize("mode", ["decomposed", "matrix"])
def test_independent_static_cameras_each_match_native_initialization(mode):
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    cfg.model.prope_mode = cfg.model.mv_prope_mode = mode
    doc["isolated"] = True
    doc["views"].append(copy.deepcopy(doc["views"][0]))
    for part in (doc["views"][1], doc["views"][1]["condition"]):
        part["pose"][:, 0] *= 1.5
        part["pose"][:, 7] += 2
        matrix = camera_projection(part["pose"][None])
        part["projection"], part["inverse"] = matrix.projection[0], matrix.inverse[0]
    configure_model(model, cfg)
    inputs, *_ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    assert inputs["camera_reference"].shape == inputs["camera_pose"].shape
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    native = {key: value for key, value in inputs.items() if not key.startswith("camera_")}
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs).sample, model(**native).sample, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["decomposed", "matrix"])
def test_independent_video_prediction_does_not_depend_on_other_camera(mode):
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    cfg.model.prope_mode = cfg.model.mv_prope_mode = mode
    doc["isolated"] = True
    doc["views"].append(copy.deepcopy(doc["views"][0]))
    for view in doc["views"]:
        view["pose"][-1, 7] += .4
        matrix = camera_projection(view["pose"][None])
        view["projection"], view["inverse"] = matrix.projection[0], matrix.inverse[0]
    changed = copy.deepcopy(doc)
    first = changed["views"][0]
    for part in (first, first["condition"]):
        part["pose"][:, 0] *= 1.5
        part["pose"][:, 7] += 2
    first["pose"][-1, 7] += .8
    for part in (first, first["condition"]):
        matrix = camera_projection(part["pose"][None])
        part["projection"], part["inverse"] = matrix.projection[0], matrix.inverse[0]
    configure_model(model, cfg)
    predictions = []
    for value in (doc, changed):
        torch.manual_seed(27)
        inputs, _, _, records, _ = WorldViewsObjective(cfg).pack(value, "cpu", evaluation_sigma=.5)
        inputs["attention_mask"] = inputs["attention_mask"].dense()
        with torch.no_grad():
            output = model(**inputs).sample
        predictions.append([output[:, record["start"]:record["stop"]] for record in records])
    assert (predictions[0][0] - predictions[1][0]).abs().max() > 1e-4
    torch.testing.assert_close(predictions[0][1], predictions[1][1], rtol=0, atol=0)


def test_independent_video_clock_and_prediction_ignore_other_caption_length():
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    doc["isolated"] = True
    doc["views"].append(copy.deepcopy(doc["views"][0]))
    changed = copy.deepcopy(doc)
    first = changed["views"][0]
    first["text"] = torch.cat((first["text"], first["text"][:, :1] + 10), 1)
    configure_model(model, cfg)
    predictions, positions = [], []
    for value in (doc, changed):
        torch.manual_seed(27)
        inputs, _, _, records, _ = WorldViewsObjective(cfg).pack(value, "cpu", evaluation_sigma=.5)
        layout = inputs["attention_mask"]
        positions.append(inputs["position_ids"][layout.scope == 1])
        inputs["attention_mask"] = layout.dense()
        with torch.no_grad():
            output = model(**inputs).sample
        record = records[1]
        predictions.append(output[:, record["start"]:record["stop"]])
    torch.testing.assert_close(positions[0], positions[1], rtol=0, atol=0)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("view_count", [1, 2])
def test_future_caption_length_does_not_shift_earlier_video(isolated, view_count):
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    doc["isolated"] = isolated
    view = doc["views"][0]
    view["texts"] = [(0, view["text"]), (1, view["text"] + 1)]
    if view_count == 2:
        doc["views"].append(copy.deepcopy(view))
    changed = copy.deepcopy(doc)
    first, future = changed["views"][0]["texts"]
    changed["views"][0]["texts"] = [first, (1, torch.cat((future[1], future[1][:, :1] + 10), 1))]
    configure_model(model, cfg)
    predictions, positions = [], []
    for value in (doc, changed):
        torch.manual_seed(27)
        inputs, _, _, records, _ = WorldViewsObjective(cfg).pack(value, "cpu", evaluation_sigma=.5)
        record = records[0]
        layout = inputs["attention_mask"]
        indices = inputs["video_indices"][record["start"]:record["stop"]]
        earlier = layout.chunk[indices] == 0
        positions.append(inputs["position_ids"][indices[earlier]])
        inputs["attention_mask"] = layout.dense()
        with torch.no_grad():
            output = model(**inputs).sample
        predictions.append(output[:, record["start"]:record["stop"]][:, earlier])
    torch.testing.assert_close(positions[0], positions[1], rtol=0, atol=0)
    torch.testing.assert_close(predictions[0], predictions[1], rtol=1e-6, atol=1e-6)


def test_wrapped_moving_camera_changes_output_and_preserves_temporal_rope():
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    cfg.model.prope_mode = "matrix"
    configure_model(model, cfg)
    inputs, _ = joint_inputs(doc, [doc["views"][0]["latent"]],
                             WorldViewsObjective(cfg).condition_latents(doc, "cpu"), .5, cfg, "cpu")
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    with torch.no_grad():
        still = model(**inputs).sample
        p, pi = (x.clone() for x in inputs["camera_projections"])
        p[:, 2:, 0, 3] += .2
        pi = torch.linalg.inv(p)
        inputs["camera_projections"] = (p, pi)
        moved = model(**inputs).sample
    assert (still - moved).abs().max() > 1e-5


def test_neutral_camera_uses_native_graph_but_small_motion_keeps_overlay():
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    configure_model(model, cfg)
    inputs, _ = joint_inputs(doc, [doc["views"][0]["latent"]],
                             WorldViewsObjective(cfg).condition_latents(doc, "cpu"), .5, cfg, "cpu")
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    cameras = []
    hook = model.transformer_blocks[0].register_forward_pre_hook(lambda module, arguments: cameras.append(arguments[5]))
    with torch.no_grad():
        model(**inputs)
        projection, inverse = [value.clone() for value in inputs["camera_projections"]]
        projection[:, -1, 0, 3] += 1e-6
        inputs["camera_projections"] = (projection, torch.linalg.inv(projection))
        model(**inputs)
    hook.remove()
    assert cameras[0] is None
    assert cameras[1] is not None and cameras[1].wrapped


def test_decomposed_uses_later_frame_camera_in_every_original_layer():
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    cfg.model.prope_mode = cfg.model.mv_prope_mode = "decomposed"
    configure_model(model, cfg)
    view = doc["views"][0]
    view["pose"][-1, 4] += .2
    view["pose"][-1, 7] += .3
    matrices = camera_projection(view["pose"][None])
    view["projection"], view["inverse"] = matrices.projection[0], matrices.inverse[0]
    inputs, _, _, records, _ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    assert "camera_pose_f0" not in inputs
    record = records[0]
    media_ids = inputs["camera_indices"][inputs["video_indices"]][record["start"]:record["stop"]]
    torch.testing.assert_close(inputs["camera_pose"][0, media_ids][-1], view["pose"][-1], rtol=0, atol=0)
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    cameras = []
    hooks = [block.register_forward_pre_hook(lambda module, args: cameras.append(args[5]))
             for block in model.transformer_blocks]
    with torch.no_grad():
        model(**inputs)
    for hook in hooks:
        hook.remove()
    assert len(cameras) == len(model.transformer_blocks)
    for block, camera in zip(model.transformer_blocks, cameras):
        assert block.attn.processor.camera_mode == "decomposed"
        rotation = camera.decomposed.rotation[0, media_ids]
        assert not torch.equal(rotation[0], rotation[-1])
        phases = camera.decomposed.h_sin[0, media_ids]
        assert not torch.equal(phases[0], phases[-1])


def test_video_only_padding_cannot_relay_into_video_or_text(monkeypatch):
    import model.packing as packing
    monkeypatch.setattr(packing, "get_sp_size", lambda: 8)
    cfg, doc, model = recipe(), static_document(), tiny_model().eval()
    configure_model(model, cfg)
    inputs, _ = joint_inputs(doc, [doc["views"][0]["latent"]],
                             WorldViewsObjective(cfg).condition_latents(doc, "cpu"), .5, cfg, "cpu")
    assert inputs["audio_hidden_states"].shape[1] > 0
    layout = inputs["attention_mask"]
    assert not layout.active[inputs["audio_indices"]].any()
    inputs["attention_mask"] = layout.dense()
    with torch.no_grad():
        a = model(**inputs).sample
        inputs["audio_hidden_states"] = torch.randn_like(inputs["audio_hidden_states"]) * 1000
        b = model(**inputs).sample
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_joint_inference_never_reads_future_ground_truth():
    cfg, doc = recipe(), static_document()
    current = [torch.randn_like(doc["views"][0]["latent"])]
    conditions = WorldViewsObjective(cfg).condition_latents(doc, "cpu")
    changed = copy.deepcopy(doc)
    changed["views"][0]["latent"].fill_(float("nan"))
    a, _ = joint_inputs(doc, current, conditions, .5, cfg, "cpu")
    b, _ = joint_inputs(changed, current, conditions, .5, cfg, "cpu")
    for key in ("hidden_states", "encoder_hidden_states", "position_ids", "timestep"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_identical_captions_keep_distinct_image_and_chunk_conditioning():
    cfg, doc = recipe(), static_document()
    doc["isolated"] = False
    doc["views"].append(copy.deepcopy(doc["views"][0]))
    first, second = doc["views"]
    second["text"] = second["text"] + 1
    inputs, *_ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    assert len(inputs["text_indices"]) == first["text"].shape[1] * 2
    torch.testing.assert_close(inputs["encoder_hidden_states"], torch.cat([first["text"], second["text"]], 1))
    second["text"] = first["text"].clone()
    inputs, *_ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    assert len(inputs["text_indices"]) == first["text"].shape[1]
    second["texts"] = [(1, second["text"])]
    inputs, *_ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    assert len(inputs["text_indices"]) == first["text"].shape[1] * 2


@pytest.mark.parametrize("frames", [22, 77, 115])
def test_joint_i2v_layout_and_visual_text_tags_match_pinned_native(frames):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    from diffusers import MiniMaxH3Transformer3DModel
    cfg, doc, model = recipe(), static_document(frames), tiny_model().eval()
    view = doc["views"][0]
    view["fps"] = 24
    view["text_tags"] = torch.tensor([1, 0, 0])
    configure_model(model, cfg)
    native = tiny_model(MiniMaxH3Transformer3DModel).eval()
    native.load_state_dict(model.state_dict(), strict=True)
    inputs, _ = joint_inputs(doc, [view["latent"]], WorldViewsObjective(cfg).condition_latents(doc, "cpu"), .5, cfg, "cpu")
    expected = MiniMaxH3PrepareLayoutStep.build_packed_sequence(view["text_tags"], view["latent"].shape[2], 2, 2,
                                                              0, (1, 2, 2), 2, 2, 0, ("first",))
    for key, value in zip(("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"), expected):
        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)
    inputs["attention_mask"] = inputs["attention_mask"].dense()
    base = {k: v for k, v in inputs.items() if not k.startswith("camera_") and k not in ("scale_log", "attention_mask")}
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs).sample, native(**base).sample, rtol=0, atol=0)


@pytest.mark.parametrize("frames", [17, 57, 77, 115])
def test_decoder_support_tail_is_conditioned_and_supervised(frames):
    from model.diffusion import view_chunk_ids
    cfg, doc = recipe(), static_document(frames)
    view = doc["views"][0]
    inputs, target, weights, records, _ = WorldViewsObjective(cfg).pack(doc, "cpu", evaluation_sigma=.5)
    record = records[0]
    layout = inputs["attention_mask"]
    indices = inputs["video_indices"][record["start"]:record["stop"]]
    tail = ~view["valid"]
    assert tail.any()
    assert layout.active[indices].all()
    assert weights[record["start"]:record["stop"]][tail].gt(0).all()
    chunks = view_chunk_ids(view, cfg.chunk_size)
    assert chunks[tail].eq(chunks[view["valid"]][-1]).all()
    assert layout.dense()[indices[-1], inputs["text_indices"]].all()
    clean = record["noisy"] + .5 * target[:, record["start"]:record["stop"]].reshape(
        1, len(tail), 24, 2, 2).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(clean, view["latent"], atol=1e-6, rtol=1e-5)


def test_chunk_rollout_writes_every_decoder_support_latent():
    from types import SimpleNamespace
    from pipeline.ar_inference import generate
    cfg, doc = recipe(), static_document(77)
    cfg.use_kv_cache = False

    class ConstantVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = [torch.nn.Identity()]

        def forward(self, **inputs):
            return SimpleNamespace(sample=torch.ones_like(inputs["hidden_states"]))

    outputs = generate(ConstantVelocity(), doc, None, cfg, "cpu", steps=2)
    assert outputs[0].shape == doc["views"][0]["latent"].shape
    assert outputs[0][:, :, ~doc["views"][0]["valid"]].square().mean() > .1
