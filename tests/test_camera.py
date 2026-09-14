import torch
import pytest

from h3.modules.camera import apply_camera, wigner_rotation, rotvec_to_matrix, precompute_camera


def poses(batch=2, count=3):
    pose = torch.zeros(batch, count, 10)
    pose[..., :2] = 1
    return pose


def test_identity_and_nonvideo_parity():
    torch.manual_seed(3)
    x = torch.randn(2, 4, 2, 128)
    ids = torch.tensor([-1, 0, 1, 2])
    identity = apply_camera(x, precompute_camera(poses()), ids)
    torch.testing.assert_close(identity, x, atol=2e-6, rtol=2e-6)
    pose = poses()
    pose[:, 1, 4:7] = torch.tensor([0.1, 0.3, -0.2])
    pose[:, 2, 7:10] = torch.tensor([0.1, 0.2, 0.3])
    y = apply_camera(x, precompute_camera(pose), ids)
    assert torch.equal(y[:, 0], x[:, 0])
    assert not torch.allclose(y[:, 2:], x[:, 2:])
    for start, stop in ((0, 16), (48, 64), (96, 128), (30, 32), (46, 48), (78, 80), (94, 96)):
        assert torch.equal(y[..., start:stop], x[..., start:stop])
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=5e-6, rtol=2e-6)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_rotation_relative_convention(order):
    left = rotvec_to_matrix(torch.tensor([0.13, -0.04, 0.23]))
    right = rotvec_to_matrix(torch.tensor([-0.11, 0.09, 0.18]))
    dl, dr = wigner_rotation(left, order), wigner_rotation(right, order)
    torch.testing.assert_close(dl.T @ dr, wigner_rotation(left.T @ right, order), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_camera_gradients_and_repeated_use(dtype):
    x = torch.randn(2, 4, 2, 128, dtype=dtype, requires_grad=True)
    pose = poses()
    pose[1, :, 7] = 0.1
    encoding = precompute_camera(pose)
    ids = torch.tensor([-1, 0, 1, 2])
    first = apply_camera(x, encoding, ids)
    second = apply_camera(x, encoding, ids)
    assert torch.equal(first, second)
    first.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_focal_fails():
    pose = poses()
    pose[0, 0, 0] = 0
    with pytest.raises(ValueError, match="focal"):
        precompute_camera(pose)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_wide_translation_keeps_small_motion_and_breaks_old_period(axis):
    pose = poses(batch=1, count=5)
    pose[0, :, 7 + axis] = torch.tensor([0.0, 0.01, 2 * torch.pi, 100.0, 500.0])
    features = torch.ones(1, 5, 1, 128)
    encoded = apply_camera(features, precompute_camera(pose, pose[:, :1]), torch.arange(5))
    torch.testing.assert_close(encoded.norm(dim=-1), features.norm(dim=-1), atol=5e-6, rtol=2e-6)

    # Centimeter motion remains visible, and a full turn of the old integer
    # frequency bank no longer aliases to the reference camera.
    for index in range(1, 5):
        assert (encoded[:, index] - encoded[:, 0]).abs().max() > 0.1
    for start, stop in ((0, 16), (48, 64), (96, 128)):
        assert torch.equal(encoded[..., start:stop], features[..., start:stop])


def test_recipe_digest_rejects_old_camera_frequency_bank(monkeypatch):
    from omegaconf import OmegaConf

    from h3.modules import camera
    from utils.config import recipe_digest

    cfg = OmegaConf.create({"h3": {"stage": 1}})
    current = recipe_digest(cfg)
    cfg.h3.stage = 2
    assert recipe_digest(cfg) == current
    monkeypatch.setattr(camera, "TRANSLATION_FREQUENCIES", (1.0, 2.0, 4.0, 8.0, 16.0))
    assert recipe_digest(cfg) != current
