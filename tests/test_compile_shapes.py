from copy import deepcopy

import torch
import pytest
from fixtures_h3 import tiny_model
from test_worldviews import dense_inputs
from test_diffusion_forcing import df_recipe, planned_document

from h3.compile_shapes import pad_camera
from utils.config import validate_config
from h3.distributed.fsdp import compile_blocks
from model.diffusion import DiffusionObjective
from h3.modules.camera import CameraBundle, camera_projection, precompute_camera


@pytest.mark.parametrize("isolated", [False, True])
def test_training_padding_preserves_data_noise_mask_predictions_and_gradients(isolated):
    torch.manual_seed(531)
    cfg, doc = df_recipe(), planned_document(2)
    doc["isolated"] = isolated
    cfg.history_dropout_ratio = 0.2
    cfg.context_noise, cfg.context_noise_std = 0.2, 0.1
    model = tiny_model()
    model.configure_attention(cfg)
    padded_cfg = deepcopy(cfg)
    padded_cfg.h3.training_shape_buckets = dict(tokens=128, timesteps=16, cameras=64, chunks=16)
    padded_model = deepcopy(model)
    padded_model.configure_attention(padded_cfg)
    packed = []
    for config in (cfg, padded_cfg):
        torch.manual_seed(945)
        packed.append(DiffusionObjective(config).pack(doc, "cpu"))
    original, padded = packed[0][0], packed[1][0]
    for key in ("hidden_states", "encoder_hidden_states", "camera_pose", "video_indices", "text_indices"):
        torch.testing.assert_close(original[key], padded[key], rtol=0, atol=0)
    for index in (1, 2):
        torch.testing.assert_close(packed[0][index], packed[1][index], rtol=0, atol=0)
    count = len(original["token_tags"])
    original_mask, padded_mask = original["attention_mask"].dense(), padded["attention_mask"].dense()
    assert torch.equal(original_mask, padded_mask[:count, :count])
    assert not padded_mask[:count, count:].any()
    assert torch.equal(padded_mask[count:, count:], torch.eye(128 - count, dtype=torch.bool))
    for inputs in (original, padded):
        torch.testing.assert_close(
            inputs["timestep"][inputs["timestep_indices"]][:count],
            original["timestep"][original["timestep_indices"]],
            rtol=0,
            atol=0,
        )
    outputs, gradients = [], []
    for network, inputs, data in zip((model, padded_model), (original, padded), packed):
        output = network(**dense_inputs(inputs)).sample
        loss = ((output - data[1]).square().mean(-1)[0] * data[2]).sum() / data[2].sum()
        loss.backward()
        outputs.append(output)
        gradients.append(
            {name: value.grad for name, value in network.named_parameters() if value.requires_grad}
        )
    torch.testing.assert_close(*outputs, rtol=1e-5, atol=1e-6)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=2e-4, atol=2e-6)


def test_camera_padding_keeps_existing_rows_and_parameter_free_structure():
    pose = torch.zeros(1, 7, 10)
    pose[..., :2] = 1
    pose[0, :, 7] = torch.arange(7) * 0.1
    camera = CameraBundle(precompute_camera(pose), camera_projection(pose), True)
    padded = pad_camera(camera, 32)
    assert padded.wrapped is True
    for original, actual in (
        (camera.matrix.projection, padded.matrix.projection),
        (camera.decomposed.rotation3, padded.decomposed.rotation3),
        (camera.decomposed.h_cos, padded.decomposed.h_cos),
    ):
        assert actual.shape[1] == 32
        torch.testing.assert_close(actual[:, :7], original, rtol=0, atol=0)
        assert not actual.requires_grad


def test_checkpointed_blocks_reuse_graphs_when_caption_lengths_change(monkeypatch):
    from torch.utils.checkpoint import DefaultDeviceType

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    compiled_graphs = []
    original_compile = torch.compile

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(
        torch, "compile", lambda function, **kwargs: original_compile(function, backend=backend, **kwargs)
    )
    cfg, doc = df_recipe(), planned_document()
    cfg.attn_block_compile = cfg.gradient_checkpointing = True
    cfg.h3.checkpoint_outside_compile = True
    cfg.h3.training_shape_buckets = dict(tokens=128, timesteps=16, cameras=64, chunks=16)
    torch.manual_seed(824)
    model = tiny_model()
    model.configure_attention(cfg)
    reference = deepcopy(model)
    compile_blocks(model, cfg)
    objective = DiffusionObjective(cfg)
    counts = []
    for length in (3, 7, 11, 5):
        doc["views"][0]["texts"] = [(i, torch.randn(1, length, 32)) for i in range(4)]
        inputs, *_ = objective.pack(doc, "cpu")
        inputs = dense_inputs(inputs)
        for network in (model, reference):
            network.zero_grad(set_to_none=True)
        actual, expected = model(**inputs).sample, reference(**inputs).sample
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (_, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
        counts.append(len(compiled_graphs))
    assert counts[0] > 0
    assert counts == [counts[0]] * len(counts)


@pytest.mark.parametrize("buckets", [{"tokens": 0}, {"tokens": True}, {"cameras": -1}, {"unknown": 8}, [8]])
def test_invalid_shape_buckets_fail_before_training(buckets):
    cfg = df_recipe()
    cfg.h3.training_shape_buckets = buckets
    with pytest.raises(ValueError, match="[Bb]ucket"):
        validate_config(cfg)
