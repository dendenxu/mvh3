import pytest
import torch

from mvh3.camera import apply_camera, precompute_camera, rotvec_to_matrix, wigner_rotation


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
