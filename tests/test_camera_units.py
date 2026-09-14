import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from test_native_flow import conditioned_document
from test_worldviews import recipe

from h3.data import temporal_layout
from model.diffusion import DiffusionObjective
from pipeline.i2v_input import prepare_request
from utils.camera import prepare_camera_geometry
from utils.config import load_config, stage_dataset_config


def scaled_copy(document, factor):
    scaled = copy.deepcopy(document)
    for view in scaled["views"]:
        view["scale"] = factor
        for geometry in (view, view["condition"]):
            geometry["pose"][..., 7:10] /= factor
            for key in ("projection", "inverse"):
                geometry[key][..., :3, 3] /= factor
    return scaled


@pytest.mark.parametrize("factor", [0.1, 10.0, 100.0])
def test_old_geometry_cache_matches_unscaled_live_inputs(factor):
    cfg, document = recipe(), conditioned_document(77)
    cached = scaled_copy(document, factor)
    original = copy.deepcopy(cached)
    restored = prepare_camera_geometry(cached, cfg)
    for geometry, expected in (
        (restored["views"][0], document["views"][0]),
        (restored["views"][0]["condition"], document["views"][0]["condition"]),
    ):
        for key in ("pose", "projection", "inverse"):
            torch.testing.assert_close(geometry[key], expected[key])
    assert restored["views"][0]["scale"] == 1.0
    assert restored["views"][0]["latent"] is cached["views"][0]["latent"]
    assert prepare_camera_geometry(restored, cfg) is restored
    for view, expected in zip(cached["views"], original["views"]):
        assert view["scale"] == expected["scale"]
        for geometry, before in ((view, expected), (view["condition"], expected["condition"])):
            for key in ("pose", "projection", "inverse"):
                torch.testing.assert_close(geometry[key], before[key], rtol=0, atol=0)
    torch.manual_seed(29)
    live, *_ = DiffusionObjective(cfg).pack(document, "cpu", evaluation_sigma=0.5)
    torch.manual_seed(29)
    replay, *_ = DiffusionObjective(cfg).pack(cached, "cpu", evaluation_sigma=0.5)
    for key in (
        "camera_pose",
        "camera_reference",
        "camera_projections",
        "camera_projection_reference",
        "hidden_states",
        "timestep",
        "position_ids",
    ):
        torch.testing.assert_close(replay[key], live[key])
    assert live["scale_log"] is None and replay["scale_log"] is None


def test_image_request_restores_recorded_psf_without_changing_conditioning(tmp_path):
    cfg = recipe()
    Image.new("RGB", (32, 32), color=(50, 70, 90)).save(tmp_path / "input.png")
    pose = np.zeros((22, 10), dtype=np.float32)
    pose[:, :2] = 1
    pose[:, 4] = 0.2
    pose[:, 7:10] = np.arange(22)[:, None] * np.array([0.1, 0.2, 0.3])
    np.save(tmp_path / "live.npy", pose)
    pose[:, 7:10] /= 10
    np.save(tmp_path / "cached.npy", pose)

    class ImageEncoder:
        device = "cpu"

        def encode(self, pixels, generator):
            assert pixels.shape == (1, 3, 32, 32)
            layout = temporal_layout(1)
            latent = torch.randn(1, 24, len(layout.valid), 2, 2, generator=generator)
            return latent, layout, None

    class TextEncoder:

        def i2v(self, requests):
            assert len(requests) == 1 and len(requests[0][1]) == 1
            return [dict(features=torch.ones(1, 3, 32), tags=torch.zeros(3, dtype=torch.long))]

    def request(camera, scale):
        return prepare_request(
            dict(prompt="A moving camera", views=[dict(image="input.png", camera=camera, scale=scale)]),
            tmp_path,
            ImageEncoder(),
            TextEncoder(),
            cfg,
        )["views"][0]

    live, restored = request("live.npy", 1), request("cached.npy", 10)
    for geometry, expected in ((restored, live), (restored["condition"], live["condition"])):
        for key in ("pose", "projection", "inverse", "latent"):
            torch.testing.assert_close(geometry[key], expected[key])
    assert restored["scale"] == 1 and restored["source_pose_stable_factor"] == 10
    torch.testing.assert_close(restored["text"], live["text"], rtol=0, atol=0)


def test_both_stages_and_validation_disable_psf_but_matrix_preserves_it(monkeypatch):
    monkeypatch.setenv("MVH3_DATA_ROOT", "/data")
    monkeypatch.setenv("MVH3_DATA_ROOT3", "/data3")
    configs = Path(__file__).resolve().parents[1] / "configs"
    cfg = recipe()
    assert not cfg.model.scale_cond
    for stage in (1, 2):
        for validation in (False, True):
            assert list(stage_dataset_config(cfg, stage, validation).pose_stable_factors) == [1.0]
    for name in ("camera_matrix.yaml", "camera_alternating.yaml"):
        matrix = load_config(configs / name)
        assert matrix.model.scale_cond
        for stage in (1, 2):
            for validation in (False, True):
                assert list(stage_dataset_config(matrix, stage, validation).pose_stable_factors) == [
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                    1000.0,
                ]
        document = scaled_copy(conditioned_document(), 10.0)
        assert prepare_camera_geometry(document, matrix) is document
