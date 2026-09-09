import numpy as np
from scipy.spatial.transform import Rotation


def parse_poses(pose_flat):
    """Parse flat pose array into c2w matrices (single-view). Vectorized.

    intrinsics is per-frame (n, 4) [fx, fy, cx, cy] so time-varying zoom/FoV
    is preserved (callers index it by frame, in lockstep with c2ws).
    """
    n = len(pose_flat) // 10
    pose = np.array(pose_flat, dtype=np.float64).reshape(n, 10)
    intrinsics = pose[:, :4]  # (n, 4) per-frame [fx, fy, cx, cy]
    Rs = Rotation.from_rotvec(pose[:, 4:7]).as_matrix()  # (n, 3, 3)
    c2ws = np.zeros((n, 4, 4), dtype=np.float64)
    c2ws[:, :3, :3] = Rs
    c2ws[:, :3, 3] = pose[:, 7:10]
    c2ws[:, 3, 3] = 1.0
    return intrinsics, c2ws
