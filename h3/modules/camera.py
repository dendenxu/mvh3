"""Parameter-free camera transforms on H3's existing spatial Q/K channels.

The 56-channel decomposed PRoPE allocation and frozen Wigner bases follow
WorldGen's pure-temporal encoding. H3 uses split-half RoPE, so its H/W pairs
must be gathered and interleaved before applying that allocation.
"""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path

import torch

FROZEN_BASES = {
    int(order): torch.tensor(values, dtype=torch.float32)
    for order, values in json.loads(Path(__file__).with_name("wigner_bases.json").read_text()).items()
}
TRANSLATION_FREQUENCIES = (1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class CameraEncoding:
    rotation: torch.Tensor
    rotation2: torch.Tensor
    rotation3: torch.Tensor
    h_cos: torch.Tensor
    h_sin: torch.Tensor
    w_cos: torch.Tensor
    w_sin: torch.Tensor

    def to(self, device, non_blocking=False):
        return type(self)(*(getattr(self, field.name).to(device, non_blocking=non_blocking) for field in fields(self)))


@dataclass(frozen=True)
class MatrixCameraEncoding:
    projection: torch.Tensor
    inverse: torch.Tensor

    def to(self, device, non_blocking=False):
        return type(self)(self.projection.to(device, non_blocking=non_blocking),
                          self.inverse.to(device, non_blocking=non_blocking))


@dataclass(frozen=True)
class CameraBundle:
    decomposed: CameraEncoding
    matrix: MatrixCameraEncoding

    def to(self, device, non_blocking=False):
        return type(self)(self.decomposed.to(device, non_blocking), self.matrix.to(device, non_blocking))


def camera_projection(pose):
    """Normalized K @ w2c; production passes the dataset's independent matrices."""
    rotation = rotvec_to_matrix(pose[..., 4:7].float()).mT
    extrinsic = torch.eye(4, device=pose.device).expand(*pose.shape[:-1], 4, 4).clone()
    extrinsic[..., :3, :3] = rotation
    extrinsic[..., :3, 3] = -(rotation @ pose[..., 7:10, None].float()).squeeze(-1)
    intrinsic = torch.eye(4, device=pose.device).expand_as(extrinsic).clone()
    intrinsic[..., 0, 0], intrinsic[..., 1, 1] = pose[..., 0], pose[..., 1]
    intrinsic[..., 0, 2], intrinsic[..., 1, 2] = pose[..., 2], pose[..., 3]
    projection = intrinsic @ extrinsic
    return MatrixCameraEncoding(projection, torch.linalg.inv(projection))


def matrix_rotary(rotary, indices):
    """Drop only the slow ten H/W pairs for video; native T and tail remain."""
    cos, sin = rotary
    keep = torch.ones(cos.shape[-1], device=cos.device, dtype=torch.bool)
    for start in (22, 38, 70, 86):
        keep[start:start + 10] = False
    keep = keep[None] | (indices < 0)[:, None]
    return torch.where(keep, cos, 1.0), torch.where(keep, sin, 0.0)


def apply_matrix(features, matrices, indices, head_offset=0, total_heads=None):
    """Column-vector projection on 20 existing slow H and 20 slow W channels."""
    first, second = features[..., :48].float(), features[..., 48:96].float()
    paired = torch.stack((first, second), dim=-1).unflatten(-2, (3, 16)).flatten(-2)
    t, h, w = paired.unbind(-2)
    matrix = matrices[:, indices.clamp_min(0)]
    per_head = matrix.ndim == 5
    if per_head:
        heads = features.shape[2]
        subframes = matrix.shape[2]
        assigned = (torch.arange(heads, device=features.device) + head_offset) * subframes // (total_heads or heads)
        matrix = matrix.index_select(2, assigned)

    def project(x):
        points = x[..., 12:].unflatten(-1, (5, 4))
        equation = "bshij,bshpj->bshpi" if per_head else "bsij,bshpj->bshpi"
        projected = torch.einsum(equation, matrix, points).flatten(-2)
        return torch.cat((x[..., :12], projected), -1)

    paired = torch.stack((t, project(h), project(w)), -2).unflatten(-1, (16, 2)).flatten(-3, -2)
    result = torch.cat((paired[..., 0], paired[..., 1], features[..., 96:].float()), -1).to(features.dtype)
    return torch.where((indices >= 0)[None, :, None, None], result, features)


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula with a finite zero-angle limit."""
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*rotvec.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(rotvec, dim=-1)[..., None, None]
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return eye + torch.sinc(theta / math.pi) * skew + 0.5 * torch.sinc(theta / (2 * math.pi)).square() * (skew @ skew)


def wigner_rotation(rotation: torch.Tensor, order: int) -> torch.Tensor:
    """Use the committed basis; never regenerate a basis during model loading."""
    if order == 1:
        return rotation
    basis = FROZEN_BASES[order].to(device=rotation.device, dtype=rotation.dtype)
    if order == 2:
        return torch.einsum("Iab,...ac,...bd,Jcd->...IJ", basis, rotation, rotation, basis)
    if order == 3:
        return torch.einsum("Iabc,...ad,...be,...cf,Jdef->...IJ", basis, rotation, rotation, rotation, basis)
    raise ValueError(f"Unsupported Wigner order: {order}")


def precompute_camera(pose_10d: torch.Tensor) -> CameraEncoding:
    """Compute once per forward from [B, cameras, 10] canonical c2w poses.

    Rows are [fx, fy, cx, cy, rotvec(R_c2w), camera_center_world]. Intrinsics
    are normalized to the image dimensions; geometry is already world-locked
    and scaled by the data adapter. Pose rows may represent different times.
    """
    if pose_10d.ndim != 3 or pose_10d.shape[-1] != 10 or pose_10d.shape[1] == 0:
        raise ValueError("camera_pose must have shape [batch, num_camera_poses, 10] with at least one pose")
    if not torch.isfinite(pose_10d).all() or not (pose_10d[..., :2] > 0).all():
        raise ValueError("Camera poses must be finite and normalized focal lengths must be positive")
    with torch.autocast(device_type=pose_10d.device.type, enabled=False):
        pose = pose_10d.float()
        rotation = rotvec_to_matrix(pose[..., 4:7])
        frequencies = pose.new_tensor(TRANSLATION_FREQUENCIES)
        translation = pose[..., 7:10, None] * frequencies
        intrinsics = torch.cat((pose[..., :2].log(), pose[..., 2:4]), dim=-1) * 4.0
        # H: D1(3) + D2(5) + tx(10) + tz{1,4,16}(6) + fx,cx(4).
        # W: D1(3) + D3(7) + ty(10) + tz{2,8}(4) + fy,cy(4).
        h_angles = torch.cat((translation[..., 0, :], translation[..., 2, (0, 2, 4)], intrinsics[..., (0, 2)]), dim=-1)
        w_angles = torch.cat((translation[..., 1, :], translation[..., 2, (1, 3)], intrinsics[..., (1, 3)]), dim=-1)
        return CameraEncoding(
            rotation,
            wigner_rotation(rotation, 2),
            wigner_rotation(rotation, 3),
            h_angles.cos(),
            h_angles.sin(),
            w_angles.cos(),
            w_angles.sin(),
        )


def apply_camera(features: torch.Tensor, camera: CameraEncoding, camera_indices: torch.Tensor) -> torch.Tensor:
    """Overlay camera encoding after native RoPE on [B, S, heads, 128] Q or K.

    camera_indices is [S], indexing the pose table; -1 leaves a token exact.
    Text/audio tokens must use -1. The temporal pairs (0:16, 48:64), four
    remaining spatial channels per axis, and the 32-channel tail stay exact.
    """
    if features.shape[-1] != 128:
        raise ValueError("The camera allocation requires H3's 128-channel attention heads")
    if camera_indices.shape != (features.shape[1], ):
        raise ValueError("camera_indices must match the local packed sequence length")
    ids = camera_indices.clamp_min(0)

    def project(block, rotation):
        return torch.einsum("bshj,bsij->bshi", block, rotation[:, ids])

    def rotate_pairs(block, cos, sin):
        pairs = block.unflatten(-1, (-1, 2))
        x, y = pairs.unbind(-1)
        cos, sin = cos[:, ids, None, :], sin[:, ids, None, :]
        return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1).flatten(-2)

    with torch.autocast(device_type=features.device.type, enabled=False):
        first, second = features[..., :48].float(), features[..., 48:96].float()
        paired = torch.stack((first, second), dim=-1).unflatten(-2, (3, 16))
        t, h, w = paired.flatten(-2).unbind(-2)
        h = torch.cat((
            project(h[..., :3], camera.rotation),
            project(h[..., 3:8], camera.rotation2),
            rotate_pairs(h[..., 8:28], camera.h_cos, camera.h_sin),
            h[..., 28:],
        ),
                      dim=-1)
        w = torch.cat((
            project(w[..., :3], camera.rotation),
            project(w[..., 3:10], camera.rotation3),
            rotate_pairs(w[..., 10:28], camera.w_cos, camera.w_sin),
            w[..., 28:],
        ),
                      dim=-1)
        paired = torch.stack((t, h, w), dim=-2).unflatten(-1, (16, 2)).flatten(-3, -2)
        rotated = torch.cat((paired[..., 0], paired[..., 1]), dim=-1).to(features.dtype)
        result = torch.cat((rotated, features[..., 96:]), dim=-1)
    return torch.where((camera_indices >= 0)[None, :, None, None], result, features)
