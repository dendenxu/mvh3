# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, Sampler, get_worker_info

import os
import time
import math
import torch
import random
import torch.nn.functional as F
from functools import partial

import numpy as np
import pyarrow.parquet as pq


from utils.console import *
from utils.easyvolcap import read_camera_minimal
from utils.data import as_numpy_func
from utils.data import export_camera
from dataset.fps_remap import resolve_fps_remap
from utils.math_utils import affine_padding
from utils.math_utils import affine_inverse
from utils.math_utils import ixt_inverse
from utils.math_utils import ixt_padding
from utils.parallel import parallel_execution
from utils.distributed import get_rank
from utils.distributed import get_world_size
from utils.distributed import is_main_process
from utils.distributed import is_node_main
from utils.video import CFRVideoReader
from utils.video import write_video
from utils.misc import set_seed
from utils.mvgame import draw_view_acc_abs
from utils.mvgame import build_main_walk
from utils.mvgame import compute_view_offsets
from utils.mvgame import apply_view_chaos


def random_move(
    length: int, n_views: int, n_frames: int,
    frame_velo_min: float = 1.0, frame_velo_max: float = 1.0,
    view_velo_min: float = -3.0, view_velo_max: float = 3.0,
    frame_acc_min: float = -1.0, frame_acc_max: float = 1.0,
    view_acc_min: float = -1.0, view_acc_max: float = 1.0,  # -1 - 1, movement tuning
    frame_velo_buffer_min: float = -3.0, frame_velo_buffer_max: float = 3.0,  # smaller buffer size
    view_velo_buffer_min: float = -3.0, view_velo_buffer_max: float = 3.0,  # smaller buffer size
    acc_update_iter: int = 20,  # 20 - 50, controls how static the video would be
    drag_coefficient: float = 0.8,  # 0.5 - 0.8, controls how static the video would be
    return_numpy: bool = True,
    seed: int = -1,
):
    """
    Randomly construct the indices for each frame of the video to get a new video
    return:
        length of tuple: (view_idx, frame_idx)

    1. Randomly select a starting view, if fps can be negative, also randomly select a starting frame.
    2. Randomly select the next frame index, view index, constrained by the view acceleration and view velocity.
    """
    # Fast path: C++ extension (Torch JIT extension) that runs this loop with the GIL released.
    # For maximal speed, request return_numpy=True to avoid Python tuple construction.
    try:
        from utils._dataset_ext import randomly_construct_video_np_cpp

        arr = randomly_construct_video_np_cpp(
            length=length,
            n_views=n_views,
            n_frames=n_frames,
            frame_velo_min=frame_velo_min,
            frame_velo_max=frame_velo_max,
            view_velo_min=view_velo_min,
            view_velo_max=view_velo_max,
            frame_acc_min=frame_acc_min,
            frame_acc_max=frame_acc_max,
            view_acc_min=view_acc_min,
            view_acc_max=view_acc_max,
            frame_velo_buffer_min=frame_velo_buffer_min,
            frame_velo_buffer_max=frame_velo_buffer_max,
            view_velo_buffer_min=view_velo_buffer_min,
            view_velo_buffer_max=view_velo_buffer_max,
            acc_update_iter=acc_update_iter,
            drag_coefficient=drag_coefficient,
            seed=seed,
        )
        if return_numpy:
            return arr
        return list(map(tuple, arr.tolist()))
    except Exception:
        # If compilation/import fails, fall back to pure Python/Numpy implementation below.
        pass

    view, frame = 0, 0
    view_velo, frame_velo = 0.0, 0.0
    if frame_velo_min < 0 and frame_velo_max < 0:
        # Backward only
        frame = n_frames - 1
    elif frame_velo_min < 0:
        frame = np.random.randint(0, n_frames)
    else:
        frame = 0
    view = np.random.randint(0, n_views)
    indices = []
    for i in range(length):
        indices.append((view, frame))

        # Randomly select acceleration for frame and view every acc_update_iter
        if i % acc_update_iter == 0:
            mean = (view_acc_min + view_acc_max) / 2
            std = (view_acc_max - view_acc_min) / 6
            view_acc = np.random.normal(mean, std)
            mean = (frame_acc_min + frame_acc_max) / 2
            std = (frame_acc_max - frame_acc_min) / 6
            frame_acc = np.random.normal(mean, std)

        # Update velocity based on acceleration
        view_velo += view_acc
        frame_velo += frame_acc

        # Consider drag coefficient
        view_velo *= drag_coefficient
        frame_velo *= drag_coefficient

        # Use this as the buffer values for the next frame
        view_velo = np.clip(view_velo, view_velo_buffer_min, view_velo_buffer_max)
        frame_velo = np.clip(frame_velo, frame_velo_buffer_min, frame_velo_buffer_max)

        # Update view and frame index, set a clipped max or min value for the actual velocity used
        view += np.round(np.clip(view_velo, view_velo_min, view_velo_max)).astype(np.int64)
        frame += np.round(np.clip(frame_velo, frame_velo_min, frame_velo_max)).astype(np.int64)

        # Wrap around if out of bounds
        view %= n_views
        frame %= n_frames  # frame should not be wrapped around, otherwise the video will be broken

    return np.asarray(indices)


def load_view(inds: Tuple[int, List[int]], vr: CFRVideoReader, cam: Dict[str, Any], ratio: float = 1.0, align_corners: bool = False):
    view_idx, frame_inds = inds
    try:
        view_frames = vr.get_batch(frame_inds, unique_and_sorted=True, return_unstacked=True, ratio=ratio)  # F, H, W, 3
        view_cameras = [cam[f'{f:06d}'] for f in frame_inds]  # F, 3, 4, camera parameters
        view_cameras = [{k: np.copy(v) for k, v in cam.items()} for cam in view_cameras]  # manual deep copy

        if ratio != 1.0:
            for cam in view_cameras:
                cam['K'][:2] *= ratio
        return view_frames, view_cameras
    except Exception as e:
        wi = get_worker_info()
        vr_path = getattr(vr, "video_path", None)
        log(red(
            f"[load_constructed_video] view decode failed: rank={get_rank()} "
            f"worker={wi.id if wi is not None else 0} view_idx={view_idx} "
            f"n_inds={len(frame_inds)} first={frame_inds[0] if len(frame_inds) else None} "
            f"last={frame_inds[-1] if len(frame_inds) else None} video_path={vr_path} err={e}"
        ))
        stacktrace()
        raise


def load_constructed_video(indices: np.ndarray, vrs: List[CFRVideoReader], cams: List[Dict[str, Any]], num_workers: int = 8, ratio: float = 1.0):
    """
    Load a constructed video from the given indices, video readers, and camera parameters.
    """
    indices_per_view = [[]for _ in range(len(vrs))]
    frames = []
    cameras = []
    frames_per_view = []
    cameras_per_view = []
    local_inds = []  # the index inside the loaded view_frames, sorted by their global indices
    for idx, (view_idx, frame_idx) in enumerate(indices):
        local_inds.append(len(indices_per_view[view_idx]))
        indices_per_view[view_idx].append(frame_idx)

    # Load videos and cameras
    inds = [(i, v) for i, v in enumerate(indices_per_view)]
    rets = parallel_execution(inds, vrs, cams, action=load_view, num_workers=num_workers, ratio=ratio)
    frames_per_view, cameras_per_view = zip(*rets)

    for idx, ((view_idx, frame_idx), local_idx) in enumerate(zip(indices, local_inds)):
        frames.append(frames_per_view[view_idx][local_idx])
        cameras.append(cameras_per_view[view_idx][local_idx])

    return frames, cameras


def worker_init_fn(worker_id, seed=-1, dataset=None):
    if seed == -1:
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
    else:
        worker_seed = seed
    set_seed(worker_seed)

    # Eagerly init all sub-datasets at worker spawn so the first getitem
    # doesn't trigger lazy disk reads (which cause straggler iterations).
    if dataset is not None:
        datasets = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
        for ds in datasets:
            if hasattr(ds, 'init_loader'):
                ds.init_loader()


def normalize_ixt(K, h, w):
    # Batch compute projections
    if isinstance(K, np.ndarray):
        K_n = K.copy()
    else:
        K_n = K.clone()
    K_n[:, 0, 0] /= w
    K_n[:, 1, 1] /= h
    K_n[:, 0, 2] = K_n[:, 0, 2] / w - 0.5
    K_n[:, 1, 2] = K_n[:, 1, 2] / h - 0.5
    return K_n


def unnormalize_ixt(K_n, h, w):
    # Batch compute projections
    if isinstance(K_n, np.ndarray):
        K = K_n.copy()
    else:
        K = K_n.clone()
    K[:, 0, 0] *= w
    K[:, 1, 1] *= h
    K[:, 0, 2] = (K[:, 0, 2] + 0.5) * w
    K[:, 1, 2] = (K[:, 1, 2] + 0.5) * h
    return K


def apply_affine_2d(corners: torch.Tensor, M: torch.Tensor):
    """
    corners: B, N, 2
    M: B, 3, 3
    """
    corners = corners @ M[:, :2, :2].mT + M[:, None, :2, 2]
    return corners


def get_aa_bounds(corners: torch.Tensor):
    """
    corners: B, N, 2
    """
    x0 = corners[..., 0].min(dim=-1).values  # B
    y0 = corners[..., 1].min(dim=-1).values  # B
    x1 = corners[..., 0].max(dim=-1).values  # B
    y1 = corners[..., 1].max(dim=-1).values  # B
    return x0, y0, x1, y1


def smooth_aug_path(mi, ma, power=1.0, *, length, acc=0.5, base=100, device=None, dtype=None):
    """Temporally-smooth random trajectory in [mi, ma] for one aug parameter.

    Reuses `random_move` purely as a 1-D smooth-random-walk generator: feed it
    n_views=base (=100) and read only the view component (path[:, 0]) — a
    drag/acceleration-smoothed integer sequence in [0, base) repurposed as the
    per-frame aug-parameter trajectory. `acc` is the view-acceleration bound;
    bigger acc => the path sweeps its range faster.
    """
    path = random_move(length, n_views=base, n_frames=length,
                       view_acc_min=-acc, view_acc_max=acc, seed=np.random.randint(1e9))
    path = path[:, 0].astype(np.float32)
    # random_move wraps the view index modulo base, producing a sawtooth with
    # ±base discontinuities. Undo the wrap into a continuous triangle path: a step
    # of magnitude > base/2 is a wraparound, not real motion; `flips` toggles parity
    # at each wrap, and flipped segments reflect via (base - path) so the path folds
    # back continuously instead of jumping.
    diffs = np.diff(path, prepend=path[0])
    mask = np.abs(diffs) > base / 2
    flips = np.cumsum(mask) % 2
    path_continuous = np.where(flips == 1, base - path, path)
    # Map continuous [0, base] -> t in [-1, 1] centered at base/2, apply an odd power
    # curve (sign-preserving |t|**power: power>1 eases toward the center value,
    # power<1 pushes toward the extremes), then remap [-1, 1] -> [0, 1] -> [mi, ma].
    t = (path_continuous / (base / 2)) - 1.0
    t_mapped = np.sign(t) * (np.abs(t) ** power)
    norm_path = (t_mapped + 1.0) / 2.0
    return torch.as_tensor(mi + (ma - mi) * norm_path, device=device, dtype=dtype)


def video_augmentation(
    frames: Union[np.ndarray, torch.Tensor], cameras: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    Ho: int = 0, Wo: int = 0,
    # FIXME: ADD GLOBAL CONTROL FOR ENABLING IMAGE SPACE AUGMENTATION
    # s_min: float = 0.65, s_max: float = 1.25, s_power=1.0,
    # cx_min: float = 0, cx_max: float = 0, cx_power=1.5,
    # cy_min: float = 0, cy_max: float = 0, cy_power=1.5,
    # r_min: float = 0.0, r_max: float = 0.0, r_power=2.0,
    # acc: float = 0.0, # fixed in place
    # CAVEAT (2026-05-28 audit): s_max=1.0 here diverges from static.py / dynamic.py
    # / multiview.py which all pass s_max=1.25 explicitly via aug_kwargs. mvgame.py
    # does NOT pass s_min/s_max in its __getitem__ (line ~1287-1303), so this
    # signature default is what mvgame ends up using. Likely unintentional —
    # commented-out line above shows original was 1.25.
    s_min: float = 0.65, s_max: float = 1.0, s_power=1.0,
    cx_min: float = -0.1, cx_max: float = 0.1, cx_power=1.5,
    cy_min: float = -0.1, cy_max: float = 0.1, cy_power=1.5,
    r_min: float = -15.0, r_max: float = 15.0, r_power=2.0,
    acc: float = 0.5,
    max_fov_h_deg: float = None,
):
    """
    frames: N, C, H, W
    o: the output size of the image, should always be smaller or equal to the input size, will first perform center crop of the original image to match the output size
    s: scale to apply, relative to principal point of the image
    c: the offset to the principal point of the image, this is relative to the output size
    r: relative rotation, in degrees, positive value means clockwise rotation of the image content
    """
    is_tensor = isinstance(frames, torch.Tensor)
    N, C, H, W = frames.shape
    Ho, Wo = Ho or H, Wo or W

    # Ensure float32 and move to device once
    frames: torch.Tensor = torch.as_tensor(frames, dtype=torch.float32)
    device, dtype = frames.device, frames.dtype

    # Vectorized camera parameter extraction
    new_Ks = torch.as_tensor(np.stack([c['K'] for c in cameras]), dtype=dtype, device=device)
    new_Rs = torch.as_tensor(np.stack([c['R'] for c in cameras]), dtype=dtype, device=device)
    new_Ts = torch.as_tensor(np.stack([c['T'] for c in cameras]), dtype=dtype, device=device)
    new_Ks = normalize_ixt(new_Ks, H, W)

    # Each aug parameter follows its own temporally-smooth random trajectory
    # (length N, shared acceleration bound `acc`); see smooth_aug_path.
    smooth = partial(smooth_aug_path, length=N, acc=acc, device=device, dtype=dtype)
    scales = smooth(s_min, s_max, s_power)
    rolls = smooth(r_min, r_max, r_power) * (np.pi / 180.0)
    cx_rel_offs = smooth(cx_min, cx_max, cx_power)
    cy_rel_offs = smooth(cy_min, cy_max, cy_power)

    # Used for centering the principal points for cropping, rotation and scaling.
    # normalize_ixt put cx/cy in [-0.5, 0.5] (frac of W/H, minus 0.5 to center);
    # the affine matrices work in [-1, 1] NDC, so *2 converts to that range. The
    # `_11` suffix means "expressed in the [-1, 1] grid_sample coordinate frame".
    cx_rel_11, cy_rel_11 = new_Ks[:, 0, 2] * 2, new_Ks[:, 1, 2] * 2  # N

    # Optimization: Pre-allocate a single identity batch to avoid multiple clones
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)  # N, 3, 3

    # 1. Principal Point Centering Matrices
    M_pp_center = I.clone()
    M_pp_center[:, 0, 2] = -cx_rel_11
    M_pp_center[:, 1, 2] = -cy_rel_11

    M_pp_uncenter = I.clone()
    M_pp_uncenter[:, 0, 2] = cx_rel_11
    M_pp_uncenter[:, 1, 2] = cy_rel_11

    # 2. Crop Matrix
    M_crop = I.clone()
    M_crop[:, 0, 0] = Wo / W
    M_crop[:, 1, 1] = Ho / H

    # 3. Rotation Matrix (Aspect aware)
    aspect = W / H
    sin_t, cos_t = rolls.sin(), rolls.cos()
    M_rot = I.clone()
    M_rot[:, 0, 0] = cos_t
    M_rot[:, 0, 1] = -sin_t * (1.0 / aspect)  # Combined Aspect Unnorm @ Rot @ Aspect Norm
    M_rot[:, 1, 0] = sin_t * aspect
    M_rot[:, 1, 1] = cos_t

    # 4.0 Compute forward scales
    inv_s = 1 / scales

    # 4.1 Bound the scaling matrix to avoid black bars or pixels.
    # corners_base are the four corners of the output frame in the [-1, 1]
    # normalized device coords that F.affine_grid / grid_sample operate in (the
    # affine matrices M_* all live in this NDC space, not pixel space).
    corners_base = torch.as_tensor(
        [[-1, -1], [1, -1], [-1, 1], [1, 1]],
        dtype=dtype).to(device, non_blocking=True)  # 4, 2
    corners_base = corners_base.unsqueeze(0).repeat(N, 1, 1)  # N, 4, 2

    # Push the output corners through crop+rotation (no scale yet), measure how
    # far the rotated quad's axis-aligned bbox reaches vs the unrotated [-1,1]
    # box. inv_s_max is the largest zoom-OUT (inv_s = 1/scale) still allowed
    # before the rotated/cropped frame would expose black border: per-corner
    # ratio base/rotated, taking the min (tightest) corner. Done in pp-centered
    # space so scaling pivots about the principal point.
    M_crop_rot = M_pp_uncenter @ M_rot @ M_crop @ M_pp_center  # N, 3, 3
    corners_crop_rot = apply_affine_2d(corners_base, M_crop_rot)  # N, 4, 2
    corners_crop_rot_center = apply_affine_2d(corners_crop_rot, M_pp_center)  # scale in centered space
    corners_base_center = apply_affine_2d(corners_base, M_pp_center)  # scale in centered space
    x0, y0, x1, y1 = get_aa_bounds(corners_crop_rot_center)  # N
    x0_b, y0_b, x1_b, y1_b = get_aa_bounds(corners_base_center)  # N
    inv_s_max = torch.stack([x0_b / x0, y0_b / y0, x1_b / x1, y1_b / y1], dim=-1).min(dim=-1).values  # N

    # FoV cap: per-frame s floor so output h-FoV ≤ max_fov_h_deg.
    # new_Ks[:,0,0] is fx normalized to input (= fx_pix/W after normalize_ixt).
    # After aug, fx_norm_out = fx_norm_in * W/Wo * s, so output fx_pix = fx_norm_out*Wo.
    # s_req = Wo² / (2*W²*new_Ks[:,0,0]*tan(max_fov/2)), clamped to ≤ s_max (no upscale).
    if max_fov_h_deg is not None:
        t = math.tan(math.radians(max_fov_h_deg / 2))
        s_req = (Wo * Wo) / (2.0 * W * W * new_Ks[:, 0, 0] * t)
        s_req = s_req.clamp(max=s_max)
        inv_s_max = torch.minimum(inv_s_max, 1.0 / s_req)

    # 4.2 Clip the scales
    inv_s = inv_s.clip(max=inv_s_max)  # N

    # 4.3 Computing reverted scales
    scales = 1 / inv_s  # N

    # 4. Scaling matrix
    M_scale = I.clone()
    M_scale[:, 0, 0] = inv_s
    M_scale[:, 1, 1] = inv_s

    # 4.0 Compute forward offsets
    offs = torch.stack([(-cx_rel_offs * 2) * Wo / W, (-cy_rel_offs * 2) * Ho / H], dim=-1)  # N, 2
    rot_offs = (M_rot[:, :2, :2] @ offs.unsqueeze(-1)).squeeze(-1)  # N, 2, 1 -> N, 2

    # 4.1. Bound the offsets to avoid black bars or pixels
    M_crop_rot_scale = M_pp_uncenter @ M_scale @ M_rot @ M_crop @ M_pp_center  # N, 3, 3
    corners_crop_rot_scale = apply_affine_2d(corners_base, M_crop_rot_scale)  # N, 4, 2
    x0, y0, x1, y1 = get_aa_bounds(corners_crop_rot_scale)  # N
    x0_b, y0_b, x1_b, y1_b = get_aa_bounds(corners_base)  # N
    x_min, x_max = x0_b - x0, x1_b - x1  # N
    y_min, y_max = y0_b - y0, y1_b - y1  # N
    rot_offs_min = torch.stack([x_min, y_min], dim=-1)  # N, 2
    rot_offs_max = torch.stack([x_max, y_max], dim=-1)  # N, 2

    # 4.2 Clip the offsets
    rot_offs = rot_offs.clip(rot_offs_min, rot_offs_max)

    # (No reverted-offset step: the intrinsics are derived by inverting the exact
    # image transform M below, which stays consistent for any roll+offset combo.)

    # 5. Offset Matrix
    M_offs = I.clone()
    M_offs[:, 0, 2] = rot_offs[:, 0]
    M_offs[:, 1, 2] = rot_offs[:, 1]

    # Chain the transformations: M_offs @ (Uncenter @ Scale @ Center) @ (Uncenter @ Rot @ Center) @ (Uncenter @ Crop @ Center)
    M = M_offs @ M_pp_uncenter @ M_scale @ M_rot @ M_crop @ M_pp_center

    # Affine grid and sample. When the aug ranges are non-degenerate (real aug,
    # not the disable_aug path that pins everything to identity via fixed_*), pick
    # the grid_sample filter randomly per call. This decorrelates the alias
    # realization across epochs without changing camera parameters.
    is_real_aug = (s_min != s_max) or (cx_min != cx_max) or (cy_min != cy_max) or (r_min != r_max)
    mode = random.choice(['bilinear', 'bicubic']) if is_real_aug else 'bilinear'
    grid = F.affine_grid(M[:, :2, :3], (N, C, Ho, Wo), align_corners=False)
    new_frames = F.grid_sample(frames, grid, mode=mode, padding_mode='zeros', align_corners=False)

    # Rotation update. Rolling the image content clockwise by theta is equivalent
    # to rotating the camera frame counter-clockwise, so the extrinsic update uses
    # the transpose of the 2D image-space rotation (note the swapped signs of the
    # sin terms vs M_rot above). Left-multiplying R and T rotates the world->cam
    # frame about the optical (z) axis; the 3rd row stays identity (z unchanged).
    M_rot_3d = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    M_rot_3d[:, 0, 0] = cos_t
    M_rot_3d[:, 0, 1] = sin_t  # Transposed 2D rotation for camera coord change
    M_rot_3d[:, 1, 0] = -sin_t
    M_rot_3d[:, 1, 1] = cos_t

    new_Rs = M_rot_3d @ new_Rs
    new_Ts = M_rot_3d @ new_Ts

    # Update the intrinsics by INVERTING the exact image transform M, so K stays
    # consistent with the pixels grid_sample produced — by construction, for any
    # combination of crop/scale/roll/offset. (The previous per-term updates —
    # fx *= scale; cx += reverted_offset — desynced K from the image when roll and
    # principal-point offset combined, shifting the conditioning camera ~tens of
    # px from the actual frame on essentially every augmented sample.)
    #
    # affine_grid(align_corners=False) maps an output pixel p_out to the input
    # sample location  P_in @ M @ P_out^{-1} @ p_out  (P_* convert NDC<->pixel:
    # pix = S/2 * ndc + (S/2 - 0.5)). The induced input->output pixel homography
    # is therefore H = P_out @ M^{-1} @ P_in^{-1}, and the intrinsics that
    # reproduce it after the camera roll are  K_new = H @ K_orig @ M_rot_3d^{-1}.
    def ndc_to_pix(sw, sh):
        P = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        P[:, 0, 0] = sw / 2.0; P[:, 0, 2] = sw / 2.0 - 0.5
        P[:, 1, 1] = sh / 2.0; P[:, 1, 2] = sh / 2.0 - 0.5
        return P

    K_orig_pix = torch.as_tensor(np.stack([c['K'] for c in cameras]),
                                 dtype=dtype, device=device)
    H_in2out = ndc_to_pix(Wo, Ho) @ torch.linalg.inv(M) @ torch.linalg.inv(ndc_to_pix(W, H))
    K_new = H_in2out @ K_orig_pix @ torch.linalg.inv(M_rot_3d)
    K_new = K_new / K_new[:, 2:3, 2:3]
    new_Ks = torch.zeros_like(K_new)
    new_Ks[:, 0, 0] = K_new[:, 0, 0]
    new_Ks[:, 1, 1] = K_new[:, 1, 1]
    new_Ks[:, 0, 2] = K_new[:, 0, 2]
    new_Ks[:, 1, 2] = K_new[:, 1, 2]
    new_Ks[:, 2, 2] = 1.0

    if not is_tensor:
        new_frames, new_Ks, new_Rs, new_Ts = new_frames.numpy(), new_Ks.numpy(), new_Rs.numpy(), new_Ts.numpy()

    return new_frames, new_Ks, new_Rs, new_Ts


def gamma_correct(frames: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply a pre-computed gamma to frames in [0, 1] range, shape (F, C, H, W).

    Gamma is computed ONCE per sequence by the caller (see ``compute_sequence_gamma``)
    so all views/frames share the same correction — matching aug_views.py which
    samples across the scene before the per-view rendering pass. Per-view median
    would give each view its own gamma and break cross-view appearance consistency.
    gamma==1.0 is a no-op identity; skip the pow.
    """
    if gamma == 1.0:
        return frames
    return frames.clamp(min=1e-8) ** gamma


def compute_sequence_gamma(vrs, indices: np.ndarray, mv: int, n_views: int,
                           band: dict = None,
                           n_sample_frames: int = 5, n_sample_views: int = 3,
                           *, indices_by_view=None) -> float:
    """One gamma for the whole sequence, pushing its brightness toward ``band``.

    Single mechanism for BOTH the mvgame aggressive lift (``MVGAME_LIFT_BAND``, the
    default — raw/game renders run systematically dark, so lift any median<0.25 up to
    0.25) and the loose real-data clamp (``LOOSE_EXPOSURE_BAND``). They differ ONLY in
    band params:
      - dark  : median < dark_lo AND p95 < dark_p95_gate -> gamma<1 so median -> dark_target
      - bright: median > bright_hi                       -> gamma>1 so median -> bright_target
      - else 1.0
    ``band['use_luma']`` picks the brightness stat: Rec.601 luma (real data) or flat RGB
    (mvgame legacy). With MVGAME_LIFT_BAND the result is byte-identical to the old
    lift-only function: its gate/bright edges sit above 1.0 so the p95 gate never blocks
    and the bright side never fires, leaving exactly "lift any median<0.25 to 0.25".

    Sampling: ``n_sample_views`` evenly-spaced views x ``n_sample_frames`` frames at
    ratio=0.1, pooled as flattened 1D so mixed-resolution rigs (waymo front/side) pool
    cleanly. ``indices`` is an (F,2) view/frame map (mvgame random_move) or 1D frame
    indices (multiview sync). Callers with independently indexed readers can pass
    ``indices_by_view`` so each sampled view decodes its own absolute frame indices.
    ``mv`` accepted for signature stability but unused.
    Returns one shared gamma (cross-view appearance consistency).
    """
    band = band if band is not None else MVGAME_LIFT_BAND
    if indices_by_view is not None and len(indices_by_view) != len(vrs):
        raise ValueError(
            f'indices_by_view has {len(indices_by_view)} entries for {len(vrs)} readers')
    n_sv = max(1, min(n_sample_views, len(vrs)))
    n_sf = max(1, min(n_sample_frames, len(indices)))
    sv_inds = np.linspace(0, len(vrs) - 1, n_sv).astype(int)
    sf_pos = np.linspace(0, len(indices) - 1, n_sf).astype(int)
    indices = np.asarray(indices)
    sf_inds = (indices[sf_pos, 1] if indices.ndim == 2 else indices[sf_pos]).astype(int)

    pooled = []
    for sv in sv_inds:
        view_sf_inds = sf_inds
        if indices_by_view is not None:
            view_indices = np.asarray(indices_by_view[int(sv)])
            view_n_sf = max(1, min(n_sample_frames, len(view_indices)))
            view_sf_pos = np.linspace(0, len(view_indices) - 1, view_n_sf).astype(int)
            view_sf_inds = (view_indices[view_sf_pos, 1]
                            if view_indices.ndim == 2 else view_indices[view_sf_pos]).astype(int)
        # Quick decode at small resolution (ratio=0.1 like aug_views.py).
        sampled = np.asarray(vrs[int(sv)].get_batch(view_sf_inds, ratio=0.1)).astype(np.float32) / 255.0
        pooled.append((sampled @ LUMA_BT601).ravel() if band.get('use_luma', True)
                      else sampled.ravel())
    stat = np.concatenate(pooled)
    median = float(np.median(stat))
    p95 = float(np.percentile(stat, 95))
    if median < band['dark_lo'] and p95 < band['dark_p95_gate']:
        return float(np.log(band['dark_target']) / np.log(max(median, 1e-3)))
    if median > band['bright_hi']:
        return float(np.log(band['bright_target']) / np.log(min(median, 1.0 - 1e-3)))
    return 1.0


# Rec.601 luma weights for brightness statistics. Kept identical to
# scripts/data/cleanup/measure_brightness_distribution.py so the runtime clamp
# triggers line up with the per-dataset distribution we measured offline.
LUMA_BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# MVGAME_LIFT_BAND — the original mvgame correction (default of compute_sequence_gamma,
# used by mvgame_raw + aug mvgame via their gamma_correction flag). Raw/game renders run
# systematically darker than the aug_views-baked views, so lift ANY sequence with median
# < 0.25 up to 0.25. use_luma=False keeps the legacy flat-RGB median; the >1.0 gate/bright
# edges disable the p95 gate and the bright side, so this reproduces the old one-sided
# lift-only behaviour byte-for-byte.
MVGAME_LIFT_BAND = dict(dark_lo=0.25, dark_p95_gate=1.01, dark_target=0.25,
                        bright_hi=1.01, bright_target=0.66, use_luma=False)

# Loose two-sided exposure-clamp band, shared across ALL real datasets (one
# global correction keeps brightness consistent across co-training). Tuned from
# measure_brightness_distribution.py over configs/448.yaml so only the extreme
# tails fire — in practice waymo_e2e's dark side/rear cams, plus the rare night
# clip elsewhere; every normal scene is a no-op.
#   dark_lo / dark_p95_gate / dark_target : luma median < dark_lo AND p95 < gate
#                                           -> brighten (gamma<1) so median -> dark_target
#   bright_hi / bright_target             : luma median > bright_hi
#                                           -> darken (gamma>1) so median -> bright_target
#
# The DARK side needs the p95 gate, not median alone: a low median with a HIGH
# p95 is a dark-dominant-but-correctly-exposed scene (night sky/water with bright
# city lights) — global gamma lifting that floods the blacks into gray noise.
# Gating on p95 means "only lift when even the brightest 5% is dim", i.e. the
# whole histogram is compressed dark = genuinely underexposed (the waymo night
# case: median 0.03, p95 0.05-0.13). Verified on debug/exposure_clamp viz.
#
# The BRIGHT side keys on MEDIAN (genuine full-frame washout), deliberately NOT
# on p95: a high p95 with a normal median is a bright sky over a normally-exposed
# scene (also the waymo case), and a global curve pulling that down would crush
# the legit road/foreground midtones. Full-frame washouts are rare in this data,
# so the bright side is mostly a dormant safety net.
LOOSE_EXPOSURE_BAND = dict(dark_lo=0.10, dark_p95_gate=0.30, dark_target=0.16,
                           bright_hi=0.80, bright_target=0.66, use_luma=True)


# Luma weights + YIQ basis for the fused color jitter below. _YIQ2RGB is the
# EXACT inverse of _RGB2YIQ so a zero-angle hue rotation is the identity.
_LUMA_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.5959, -0.2746, -0.3213],
                         [0.2115, -0.5227, 0.3112]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)
def image_augmentation(frames: torch.Tensor) -> torch.Tensor:
    """Image-space augmentation that leaves camera parameters untouched. Adds per-clip
    random gaussian blur, color jitter, and per-frame additive gaussian noise. Frames
    in [0, 1] range, shape (F, C, H, W).

    The color jitter (brightness/contrast/saturation/hue) is FUSED into a couple of
    tensor passes instead of four torchvision transforms — each of those does an
    rgb<->hsv or rgb->grayscale roundtrip (~20 full-tensor passes + allocations),
    which made this ~53s/sample on raw mvgame (vs 1.9s for the geometric warp on the
    SAME pixels). brightness/contrast/saturation are bit-exact to torchvision (same
    luma + blend math); hue is a YIQ chroma rotation (≈ HSV for the tiny ±0.02
    jitter, mean diff ~5%). RNG draw order is preserved for reproducibility."""
    import math
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    sigma = float(np.random.uniform(0.0, 0.6))
    if sigma > 0.05:
        ks = 2 * int(np.ceil(3 * sigma)) + 1
        frames = TF.gaussian_blur(frames, kernel_size=ks, sigma=sigma)

    brightness = float(np.random.uniform(0.9, 1.1))
    contrast = float(np.random.uniform(0.9, 1.1))
    saturation = float(np.random.uniform(0.9, 1.1))
    hue = float(np.random.uniform(-0.02, 0.02))

    w = _LUMA_W.to(frames.dtype)
    frames = frames * brightness
    gray = (frames * w).sum(1, keepdim=True)
    frames = frames * contrast + gray.mean(dim=(1, 2, 3), keepdim=True) * (1 - contrast)
    gray = (frames * w).sum(1, keepdim=True)
    frames = frames * saturation + gray * (1 - saturation)
    ang = hue * 2 * math.pi
    cs, sn = math.cos(ang), math.sin(ang)
    rot = torch.tensor([[1, 0, 0], [0, cs, -sn], [0, sn, cs]], dtype=torch.float32)
    M = (_YIQ2RGB @ rot @ _RGB2YIQ).to(frames.dtype)
    frames = torch.einsum('mc,nchw->nmhw', M, frames)

    if np.random.random() < 0.5:
        # Additive photometric noise generated at 1/4 resolution and bilinearly
        # upsampled. A full-res randn_like dominated this pass (~7s/sample); a
        # smooth low-res field is an equivalent tiny (σ=0.01) perturbation.
        nf, cf, hf, wf = frames.shape
        noise = torch.randn(nf, cf, max(1, hf // 4), max(1, wf // 4), dtype=frames.dtype) * 0.01
        frames = frames + F.interpolate(noise, size=(hf, wf), mode='bilinear', align_corners=False)

    return frames.clamp(0, 1)


def load_posed_video(
    indices: np.ndarray, vrs: List[CFRVideoReader], cams: List[Dict[str, Any]],
    height: int, width: int,
    num_workers: int = 0, ratio: float = 1.0,
    R0: Optional[np.ndarray] = None,
    T0: Optional[np.ndarray] = None,
    fixed_s: float = None,
    fixed_cx: float = None,
    fixed_cy: float = None,
    fixed_r: float = None,
    force_crop_h: int = None,
    force_crop_w: int = None,
    image_aug: bool = False,
    gamma_value: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    # Load the video frames and cameras, then run the shared finish (gamma,
    # augment, world-lock, PRoPE projs). Split out so PresampledDataset can feed
    # its own pre-decoded frames + baked cameras through the EXACT same path.
    frames, cameras = load_constructed_video(indices, vrs, cams, num_workers, ratio=ratio)
    return finish_posed_video(
        frames, cameras, height, width, ratio=ratio, R0=R0, T0=T0,
        fixed_s=fixed_s, fixed_cx=fixed_cx, fixed_cy=fixed_cy, fixed_r=fixed_r,
        force_crop_h=force_crop_h, force_crop_w=force_crop_w,
        image_aug=image_aug, gamma_value=gamma_value, indices=indices, **kwargs)


def finish_posed_video(
    frames, cameras, height: int, width: int, ratio: float = 1.0,
    R0: Optional[np.ndarray] = None, T0: Optional[np.ndarray] = None,
    fixed_s: float = None, fixed_cx: float = None, fixed_cy: float = None, fixed_r: float = None,
    force_crop_h: int = None, force_crop_w: int = None,
    image_aug: bool = False, gamma_value: float = 1.0, indices=None, **kwargs,
) -> Dict[str, Any]:
    """Post-decode pipeline shared by load_posed_video and PresampledDataset:
    tensorize -> gamma -> video_augmentation -> image_aug -> world-lock to (R0,T0)
    -> PRoPE projs/projs_inv + world-locked Rs/Ts. `frames` is the raw list from
    load_constructed_video (or pre-decoded numpy frames); `cameras` a per-frame list
    of {K,R,T} (w2c). Factored out verbatim so the spec-driven loader stays
    caption<->pixel aligned."""
    # Convert to torch tensor with [0, 1] range and permute to F, C, H, W
    frames = torch.stack([torch.as_tensor(f, dtype=torch.float32) for f in frames])
    frames = frames.permute(0, 3, 1, 2)
    frames = frames / 255.0  # F, 3, H, W

    if gamma_value != 1.0:
        frames = gamma_correct(frames, gamma_value)

    # Manual deep copy
    cameras = [{k: torch.as_tensor(v, dtype=torch.float32).clone() for k, v in cam.items()} for cam in cameras]

    # Apply augmentation (includes scaling, cx cy, and roll)
    roll_scale_move_kwargs = {**kwargs}
    if fixed_s is not None:
        roll_scale_move_kwargs['s_min'] = fixed_s
        roll_scale_move_kwargs['s_max'] = fixed_s
    if fixed_cx is not None:
        roll_scale_move_kwargs['cx_min'] = fixed_cx
        roll_scale_move_kwargs['cx_max'] = fixed_cx
    if fixed_cy is not None:
        roll_scale_move_kwargs['cy_min'] = fixed_cy
        roll_scale_move_kwargs['cy_max'] = fixed_cy
    if fixed_r is not None:
        roll_scale_move_kwargs['r_min'] = fixed_r
        roll_scale_move_kwargs['r_max'] = fixed_r

    if force_crop_h is not None and force_crop_w is not None:
        # Cropping from top left corner doesn't change camera parameters
        frames = frames[:, :, :force_crop_h, :force_crop_w]

    frames, Ks, Rs, Ts = video_augmentation(frames, cameras, Ho=int(height * ratio), Wo=int(width * ratio), **roll_scale_move_kwargs)

    if image_aug:
        frames = image_augmentation(frames)

    # # Baseline, without image space augmentations
    # Ks = np.stack([c['K'] for c in cameras]).astype(np.float32)
    # Rs = np.stack([c['R'] for c in cameras]).astype(np.float32)
    # Ts = np.stack([c['T'] for c in cameras]).astype(np.float32)
    # H, W = frames.shape[-2:]
    # y = (H - height) // 2
    # x = (W - width) // 2
    # frames = frames[:, :, y:y + height, x:x + width]
    # Ks[..., 0, 2] -= x
    # Ks[..., 1, 2] -= y

    # # Downsample if needed
    # if ratio != 1.0:
    #     H, W = frames.shape[-2:]
    #     Ho, Wo = int(H * ratio + 0.5), int(W * ratio + 0.5)
    #     frames = F.interpolate(torch.as_tensor(frames), size=(Ho, Wo), mode='area')
    #     Ks[:, 0] *= Wo / W
    #     Ks[:, 1] *= Ho / H
    #     height, width = Ho, Wo

    if R0 is None or T0 is None:
        R0, T0 = Rs[0], Ts[0]

    # World-lock to camera 0: re-express the world frame so cam 0 sits at the
    # origin with a fixed canonical orientation (R_target), while every camera's
    # relative pose is preserved. R,T are world->cam (x_cam = R x_world + T).
    # Substituting view 0 confirms the intent: Rs_new[0] = R0 R0^T R_target =
    # R_target and Ts_new[0] = T0 - R0 R0^T T0 = 0. PRoPE is relative, so this
    # only changes numerics, not the geometry the model sees (see caller note).
    # Target: Camera 0 at origin, looking at +Y (Z-forward), with Z-up (-Y-down)
    R_target = torch.as_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    Rs_new = Rs @ R0.mT @ R_target
    Ts_new = Ts - Rs @ R0.mT @ T0

    # Build the 4x4 PRoPE projection: intrinsics (normalized to output H,W so the
    # matrix is resolution-agnostic) @ world->cam extrinsics. projs_inv is its
    # decomposed inverse (cam->world @ inverse-intrinsics) for the reverse map.
    Ks_n = normalize_ixt(Ks, int(height * ratio), int(width * ratio))
    RTs = torch.cat([Rs_new, Ts_new], dim=-1)  # F, 3, 4
    RTs = affine_padding(RTs)
    projs = ixt_padding(Ks_n) @ RTs

    # Batch compute inverse projections
    RTs_inv = affine_padding(affine_inverse(RTs))
    Ks_inv = ixt_padding(ixt_inverse(Ks_n))  # arguments
    projs_inv = RTs_inv @ Ks_inv

    return {
        'frames': frames,
        'projs': projs,
        'projs_inv': projs_inv,
        'indices': indices,
        'Ks': Ks,
        # Return world-locked w2c Rs/Ts in the same frame as projs. The trainer derives
        # canonical c2w pose_10d (R_c2w, camera center C) from these values, keeping the
        # matrix and decomposed streams aligned in one v0/f0-anchored per-clip frame.
        'Rs': Rs_new,
        'Ts': Ts_new,
    }


def aggregate_cams(cams: List[Dict[str, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
    agg_cams = {}
    for k in cams[0]['000000'].keys():
        param = []
        for v in range(len(cams)):
            view = []
            for f in cams[v].keys():
                view.append(cams[v][f][k])
            view = np.stack(view)
            param.append(view)
        param = np.stack(param)
        agg_cams[k] = param
    return agg_cams


def select_pose_stable_factor(centers, factors, target=1.58):
    """Pick a translation-rescale factor that keeps PRoPE's bf16 attention safe.

    PRoPE is RELATIVE: the attention score between two cameras depends on their
    relative translation, and `apply_proj` casts the projected q/k back to bf16 where
    that relative score grows ~2.4*T^2 (worst case = 27/sqrt(128): 27 of 36 prope dims
    pick up T via the homogeneous column). Stable softmax = no hard overflow, but large
    T saturates attention, inflates grads ~T^2 (clipped), and bf16 rounding erodes the
    O(1) rotation signal riding on the T-scaled offset. So the quantity that matters
    is the MAX RELATIVE camera translation in the clip = the MAX PAIRWISE camera-center
    distance (the window's diameter).

    We use max pairwise (not the mean) so a one-way trajectory's far end is captured.
    Pairwise distance is translation-invariant (depends only on relative positions), so
    it gives the true movement whether the input poses are world-locked (the current
    default — load_*view returns Rs_new/Ts_new, v0/f0 at origin) or raw absolute. Keeping
    pairwise rather than a plain norm makes the metric robust to either convention.

    centers: [N, 3] camera centers (= -R^T @ T).
    target: geometric center of the post-division band = [target/sqrt10, target*sqrt10]
        (always 10x wide). Default 1.58 -> post-div in [0.5, 5.0].
    Returns (psf, max_t) where max_t is the pre-division max pairwise distance.
    """
    factors = sorted(factors)
    n = centers.shape[0] if centers.ndim >= 1 else 0
    if n < 2:
        return (float(factors[0]) if factors else 1.0), 0.0
    # Max pairwise distance (translation-invariant). N = F*mv is small (~100-300).
    # float64: raw absolute coords can be ~1000s (mvgame game-world), and float32
    # cdist loses the small relative spread to catastrophic cancellation at that
    # magnitude. double() keeps the pairwise distances exact regardless of offset.
    c = centers.double()
    max_t = float(torch.cdist(c, c).amax().item())
    if len(factors) == 1:
        return float(factors[0]), max_t
    # Center the post-division band on `target`: factor closest to max_t/target -> post-div
    # in [target/sqrt10, target*sqrt10] (always 10x wide). psf stays in `factors`, so
    # scale_tokens=log(psf) vocabulary is unchanged.
    log_mt = math.log(max(max_t, 1e-6) / max(target, 1e-6))
    psf = min(factors, key=lambda f: abs(math.log(f) - log_mt))
    return float(psf), max_t


def normalize_cam_translation(cams, target: float = 1.0):
    """Normalize camera translations so max absolute camera center coordinate = target.

    Camera center C = -R^T @ T (world-space position). We find max(|C|) across
    all views and frames, then scale T uniformly so that max center coordinate = target.
    This preserves multi-view consistency (all views share the same scale factor).

    Supports two formats:
    - Aggregated (MultiViewDataset): cams is a dict with R=[V,F,3,3], T=[V,F,3,1]
    - List-of-dicts (StaticDataset): cams is a list of {K:[3,3], R:[3,3], T:[3,1]} dicts (numpy)

    If target <= 0 or max coordinate is 0, no normalization is applied.
    """
    if target <= 0:
        return cams

    if isinstance(cams, dict):
        # Aggregated format: R=[V,F,3,3], T=[V,F,3,1]
        R, T = cams['R'], cams['T']
        # C = -R^T @ T, using einsum for batched transpose-matmul
        C = -np.einsum('...ji,...jk->...ik', R, T)  # [..., 3, 1]
        max_val = np.abs(C).max()
        if max_val > 0:
            cams['T'] = T * (target / max_val)
    else:
        # List-of-dicts format: stack into arrays for vectorized computation
        Rs = np.stack([cam['R'] for cam in cams])  # [N, 3, 3]
        Ts = np.stack([cam['T'] for cam in cams])  # [N, 3, 1]
        C = -np.einsum('nji,njk->nik', Rs, Ts)     # [N, 3, 1]
        max_val = np.abs(C).max()
        if max_val > 0:
            scale = target / max_val
            Ts *= scale  # scale in-place on the stacked copy
            for i, cam in enumerate(cams):
                cam['T'] = Ts[i].copy()  # independent copy, no shared backing array

    return cams


def deaggregate_cams(agg_cams: Dict[str, np.ndarray]) -> List[Dict[str, Dict[str, np.ndarray]]]:
    V, F = next(iter(agg_cams.values())).shape[:2]
    return [
        {f"{f:06d}": {k: agg_cams[k][v, f] for k in agg_cams} for f in range(F)}
        for v in range(V)
    ]


# Eight views in total:
#         1
#   5          7
#
# 2               3
#
#   6          4
#         0 (main view)
# ---------------------------------
# |               |       |       |
# |               |   1   |   2   |
# |               |       |       |
# |       0       |-------|-------|
# |               |       | 4 | 5 |
# |               |   3   |-------|
# |               |       | 6 | 7 |
# ---------------------------------
#
# Five views in total
# 3   2
# 4   1
#   0 (main view)
# |-------|-------|
# |       | 1 | 2 |
# |   0   |-------|
# |       | 3 | 4 |
# -----------------
#
# Three views in total
# 2   1
#   0 (main view)
# |-------|----
# |       | 1 |
# |   0   |----
# |       | 2 |
# -------------
#
# Two views in total
#   1
#   0 (main view)
# |-------|
# | 0 | 1 |
# |-------|
PACK_FACTORY_MANUAL = {
    1: {
        "mv": 1,
        "pack_size": np.asarray([1.0, 1.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0], dtype=np.float32),  # logical order
    },
    2: {
        "mv": 2,
        "pack_size": np.asarray([1.0, 2.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 1], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 1.0], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0], dtype=np.float32),  # logical order
    },
    3: {
        "mv": 3,
        "pack_size": np.asarray([1.0, 1.5], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 1, 2], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 0.5, 0.5], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0, 1.0], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0, 0.5], dtype=np.float32),  # logical order
    },
    4: {
        "mv": 4,
        "pack_size": np.asarray([1.0, 4.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 1, 2, 3], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # logical order
    },
    5: {
        "mv": 5,
        "pack_size": np.asarray([1.0, 2.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 1, 2, 3, 4], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0, 1.5, 1.0, 1.5], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32),  # logical order
    },
    6: {
        "mv": 6,
        "pack_size": np.asarray([1.0, 6.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # logical order
    },
    8: {
        "mv": 8,
        "pack_size": np.asarray([1.0, 2.0], dtype=np.float32),  # multiply width by 2 (ratio for height and width)
        "pack_inds": np.asarray([0, 4, 7, 1, 5, 2, 6, 3], dtype=np.int32),  # geometry order
        'rs': np.asarray([1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25], dtype=np.float32),  # logical order
        "xs": np.asarray([0.0, 1.0, 1.5, 1.0, 1.5, 1.75, 1.5, 1.75], dtype=np.float32),  # logical order
        "ys": np.asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.75, 0.75], dtype=np.float32),  # logical order
    },

}


def make_strip_pack(mv: int) -> dict:
    """1xmv full-resolution horizontal strip layout.

    Default fallback for any mv not in the manual table. View-iso rescue
    paths put a different video in each slot — no shared spatial structure
    to exploit, so they just lay out side by side at full res. Packed
    canvas = height x (mv * width).
    """
    return {
        "mv": mv,
        "pack_size": np.asarray([1.0, float(mv)], dtype=np.float32),
        "pack_inds": np.asarray(list(range(mv)), dtype=np.int32),
        'rs': np.asarray([1.0] * mv, dtype=np.float32),
        "xs": np.asarray([float(i) for i in range(mv)], dtype=np.float32),
        "ys": np.asarray([0.0] * mv, dtype=np.float32),
    }


class PackFactory:
    """Hand-tuned mixed-resolution layouts for mv ∈ {1, 2, 3, 4, 5, 6, 8};
    everything else auto-falls-back to a 1xmv full-resolution strip.

    The manual layouts pack a primary view + supporting half/quarter views
    into a constrained canvas (e.g. mv=5 → 1x2 with 1 full + 4 half;
    mv=8 → 1x2 with 1 full + 3 half + 4 quarter). Useful for cross-view
    static / mvgame samples where one view is "primary".

    Strip layouts (1xN at full res) are what every view-iso rescue path
    wants — different videos in different slots, no primary view, no shared
    spatial structure. We auto-generate them on demand rather than hand-
    listing 10 / 12 / 14 / 16 / 20 / 30 / ...

    `mv in pack_factory` returns True for any positive int — the strip
    fallback can serve any mv ≥ 1. `keys()` reports only the manual
    entries (used in error messages and the static fallback's pack_size
    compatibility check at static.py).
    """

    def __init__(self, manual: dict):
        self.manual = dict(manual)
        self.auto: dict = {}

    def __contains__(self, mv) -> bool:
        return isinstance(mv, int) and mv >= 1

    def __getitem__(self, mv: int) -> dict:
        if mv in self.manual:
            return self.manual[mv]
        if not isinstance(mv, int) or mv < 1:
            raise KeyError(f'pack_factory: mv must be a positive int, got {mv!r}')
        if mv not in self.auto:
            self.auto[mv] = make_strip_pack(mv)
        return self.auto[mv]

    def keys(self):
        return self.manual.keys()


pack_factory = PackFactory(PACK_FACTORY_MANUAL)


class MultiViewDataset(Dataset):
    """Dataset that returns a video and a text prompt per sample.
    """

    def __init__(self,
                 data_path: str = '/mnt/bn/foundation-ads3/zhenxu.zx/datasets/mvgame/qwen3_filtered.parquet',
                 gen_size: int = 124,
                 height: int = 448,  # target height to center crop to
                 width: int = 832,  # target width to center crop to

                 # Data loading setting
                 drop_last: int = 3,  # for some weird reasons the last frame in the raw data is blank -> bad taa propagation
                 view_acc_abs_min: float = 0.0,  # randomly select the acceleration value, easier motion
                 view_acc_abs_max: float = 0.0,  # randomly select the acceleration value, easier motion
                 per_worker_threads: int = 4,  # sequential, not spawning new workers

                 mv_size: int = 8,  # total number of views to generate, default to 8 for mvgame
                 mv_chaos: float = 0.1,  # the acceleration for extra random offsets in low-res views
                 off_perturb_std: float = 0.05,  # gaussian sample the indices, global fixed offset for every frame
                 # TODO: Design attn mask to add variation in resolution

                 #  assume_file_names: bool = True,
                 #  assume_n_views: int = 25,  # only used for faster data loading
                 #  assume_n_interps: int = 4,  # only used for faster data loading

                 # Sequence sampling for faster loading
                 seq_sample: List[int] = (0, None, 1),

                 # Overfitting setting
                 overfit: bool = False,
                 overfit_seq_sample: List[int] = (0, 1, 1),  # overfit the first sample (sequence of mvgame)
                 overfit_vid_sample: List[int] = (0, None, 1),  # just overfit the first 8 views for debugging purposes
                 # When overfit=True, setting this to True keeps augmentation on (random_move,
                 # multi-view sampling, random start_idx) and only restricts metadata to
                 # overfit_seq_sample. For mvgame, one metadata row = one scene folder with many
                 # views x many frames, so this gives a meaningful "one-seq" overfit rather than
                 # the vid1-style single-clip overfit that `overfit=True` alone produces.
                 overfit_keep_aug: bool = False,

                 config=dotdict({'sp_size': 1, 'model': {'vae_stride': [4, 8, 8]}}),
                 pose_norm_target: float = 1.0,
                 pose_stable_factors=1.0,  # float or list of floats; picks closest to max_t in log space

                 disable_augmentation_ratio: float = 1.0,  # for 0.25 of all samples, disable augmentation completely
                 image_aug: bool = False,  # image-space aug (gblur, color jitter, noise); gated by the same disable_aug switch as video aug
                 gamma_correction: bool = False,  # adaptive gamma on loaded raw frames (matches aug_views behaviour); applied regardless of disable_aug
                 max_fov_h_deg: float = None,  # cap output h-FoV by per-frame s_min floor (None=off; see video_augmentation)
                 sp_sharding: bool = False,

                 # FPS resampling
                 # FIXME: READ FROM VIDEO DYNAMICALLY DURING TRAINING
                 # WE ALREADY HAVE THE VIDEO READER OBJECT CONSTRUCTED
                 dataset_fps: int = None,  # source video fps (None = same as model_fps, no resampling)
                 model_fps: int = 24,      # model's target fps

                 # Per-sample shape pool (mirrors StaticDataset.shape_pool). None
                 # = disabled (preserves all existing behavior). When set, every
                 # __getitem__ idx-deterministically picks one (mv, gen) and
                 # forces a uniform-rs strip pack so all views stay full-res.
                 shape_pool=None,
                 shape_pool_weights=None,

                 # Aggregator uses `effective_samples ** sampling_weight_power`
                 # as sampling weight. Must be explicit (not **kwargs) — without
                 # an attribute, aggregator's getattr falls back to 0.8 and yaml
                 # overrides are silently ignored. This default MUST stay in sync
                 # with that 0.8 fallback (the attr is always set, so the fallback
                 # never actually fires — a 0.6 here silently overrode the 0.8).
                 sampling_weight_power: float = 0.8,

                 *args,
                 **kwargs
                 ):
        """
        data_path: Path to a jsonl file containing the metadata for the videos.
        Each JSON object looks like this:
            {"video_path": "path/to/video(/xxxxxx-xxxxxx.mp4)", "caption": "a cat"}
        """
        self.config = config  # global config object
        self.data_path = data_path
        self.dataset_name = splitext(basename(data_path))[0]
        self.gen_size = gen_size
        self.vae_stride_t = config.model.vae_stride[0]
        self.height = height
        self.width = width
        self.pose_norm_target = pose_norm_target
        psf = pose_stable_factors
        if isinstance(psf, str):
            psf = [float(x) for x in psf.split(',')]
        self.pose_stable_factors = sorted(psf) if isinstance(psf, (list, tuple)) else [float(psf)]
        self.sp_sharding = sp_sharding
        self.sampling_weight_power = sampling_weight_power
        self.model_fps = model_fps
        self.dataset_fps = dataset_fps
        self.fps_ratio = dataset_fps / model_fps if dataset_fps is not None else 1.0
        if is_main_process():
            log(f'Creating dataset from {blue(data_path)}, gen_size={gen_size}, mv_size={mv_size}, '
                f'dataset_fps={dataset_fps}, model_fps={model_fps}, height={height}, width={width}')
            if self.fps_ratio < 1:
                log(yellow(f'Dataset FPS: {dataset_fps}, model FPS: {model_fps}, forcing FPS ratio to be 1'))
                self.fps_ratio = 1

        self.view_acc_abs_min = view_acc_abs_min
        self.view_acc_abs_max = view_acc_abs_max
        self.per_worker_threads = per_worker_threads

        # Multi-view related
        self.mv_size = mv_size
        self.mv_chaos = mv_chaos
        self.off_perturb_std = off_perturb_std
        self.disable_augmentation_ratio = disable_augmentation_ratio
        self.image_aug = image_aug
        self.gamma_correction = gamma_correction
        self.max_fov_h_deg = max_fov_h_deg

        mv = self.mv_size
        assert mv in pack_factory, f"We only support {list(pack_factory.keys())} packing for now, but got {mv}"
        pack = pack_factory[mv]
        # After extracting shape from the given pack ratios, make sure the new size is divisible by 16
        assert (self.height * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        assert (self.width * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        assert (self.height * pack['xs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        assert (self.width * pack['ys'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        self.pack = pack

        # shape_pool: see StaticDataset.shape_pool. Per __getitem__ pick one
        # (mv, gen) with idx-seeded weighted choice; force strip pack so each
        # entry has uniform per-view resolution. None = disabled.
        self.shape_pool = [tuple(s) for s in (shape_pool or [])]
        if self.shape_pool:
            assert all(len(s) == 2 for s in self.shape_pool), \
                f"shape_pool entries must be (mv, gen) pairs, got {self.shape_pool}"
            assert (self.height % 16 == 0) and (self.width % 16 == 0), \
                f"shape_pool requires height/width divisible by 16 (strip pack)"
        self.shape_pool_weights = list(shape_pool_weights) if shape_pool_weights else None
        if self.shape_pool_weights is not None:
            assert len(self.shape_pool_weights) == len(self.shape_pool), \
                f"shape_pool_weights length {len(self.shape_pool_weights)} != shape_pool length {len(self.shape_pool)}"
        self.shape_pool_active = False

        # # Faster videos / cameras directory enumeration
        # self.use_faster_aa_videos_cameras_pattern = assume_file_names
        # self.n_views_pattern = assume_n_views
        # self.n_interps_pattern = assume_n_interps

        # Overfitting settings
        self.overfit = overfit
        self.overfit_seq_sample = overfit_seq_sample
        self.overfit_vid_sample = overfit_vid_sample
        self.overfit_keep_aug = overfit_keep_aug

        # Every object is a path to the mvgame dataset folder, containing camera parameters
        # This is the global metadata before sharding across workers
        # Sharding should only happen after getting the worker info
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.metadata = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.metadata = [json.loads(line) for line in f if line.strip()]
        elif data_path.endswith('.parquet'):
            pf = pq.ParquetFile(data_path)
            # Read metadata WITHOUT the pose column to avoid pyarrow int32
            # list-index overflow on large multi-cam parquets (38k rows ×
            # 125k floats/row > 2^31). Pose is loaded lazily per-row in
            # __getitem__ via the parquet-backed self._pf handle.
            non_pose_cols = [c for c in pf.schema_arrow.names if c != 'pose']
            self.metadata = []
            for batch in pf.iter_batches(batch_size=500, columns=non_pose_cols):
                self.metadata.extend(batch.to_pylist())

            # Read prompt_embeds_shape from parquet schema metadata if available
            schema_meta = pf.schema_arrow.metadata or {}
            if b'prompt_embeds_shape' in schema_meta:
                self.prompt_embeds_shape = json.loads(schema_meta[b'prompt_embeds_shape'])
            elif 'prompt_embeds' in pf.schema_arrow.names:
                # Fallback for parquets that have a prompt_embeds column but no
                # shape metadata (e.g. part2 cut_bb_clean). T5 XXL caches are
                # always (512, 4096). Matches static.py / multiview.py behavior.
                self.prompt_embeds_shape = [512, 4096]
        else:
            raise NotImplementedError(f'Unrecognized metadata type for file: {data_path}')

        n_total = len(self.metadata)
        # Tag each meta with its original parquet row index (parity with
        # static.py / multiview.py) so the vis meta panel can show which row a
        # sample came from. Tag the full list before slicing so the index is the
        # absolute parquet row; slicing/sharding keep the same dict refs.
        for i, m in enumerate(self.metadata):
            m['pose_idx'] = i
        if overfit:
            self.metadata = self.metadata[slice(*overfit_seq_sample)]
        else:
            self.metadata = self.metadata[slice(*seq_sample)]  # only load the selected samples

        # Distributed dispatch sanity check: each (rank, worker) needs ≥1 row
        # to avoid ZeroDivision in shard_meta. Warn loudly if the parquet/slice
        # leaves the dataset empty or below threshold rather than silently
        # crashing the worker later.
        threshold = max(1, get_world_size() * int(kwargs.get('num_workers', 1) or 1))
        cls_name = type(self).__name__
        self.is_empty = (len(self.metadata) == 0)
        if is_node_main():
            tag = green('OK') if len(self.metadata) >= threshold else red('TOO FEW')
            log(f'MultiViewDataset init: {green(len(self.metadata))} scenes '
                f'from {blue(data_path)} (parquet rows={n_total}, '
                f'threshold={threshold}) [{tag}]')
            if len(self.metadata) == 0:
                log(red(
                    f'[{cls_name} EMPTY] {data_path}: 0 rows after slice — '
                    f'this dataset will be DROPPED from co-training (weight=0). '
                    f'Check the source file and seq_sample/overfit_seq_sample.'
                ))
            elif len(self.metadata) < threshold:
                log(red(
                    f'[{cls_name} BELOW THRESHOLD] {data_path}: '
                    f'{len(self.metadata)} rows < num_workers*world_size = {threshold}. '
                    f'Some workers will reuse rows; not fatal but may distort sampling.'
                ))

        # Resolve relative video_path / prompt_embeds against the parquet's own
        # directory. Convention: paths in parquets are relpaths so data can be remounted; we
        # make them absolute here so downstream code doesn't care.
        data_root = dirname(data_path)
        for m in self.metadata:
            for key in ('video_path', 'prompt_embeds'):
                v = m.get(key)
                if isinstance(v, str) and v and not isabs(v):
                    m[key] = join(data_root, v)

        # Optional frame-range cut support (backward compatible):
        # Rows may contain frame_start/frame_end (integer) to use only a sub-range of the
        # underlying video/camera files. Both default to "full video" when absent/null.
        for m in self.metadata:
            m.setdefault('frame_start', 0)
            if m.get('frame_start') is None:
                m['frame_start'] = 0
            m.setdefault('frame_end', None)  # None = use real video length at load time

    def init_loader(self):
        # Should only be called inside __getitem__
        if hasattr(self, 'video_readers') and hasattr(self, 'camera_params'):
            return

        sharded_metadata = self.shard_meta()

        # CANNOT pickle a PyVideoReader object
        self.video_readers = {}  # these are already sharded
        self.camera_params = {}  # these are already sharded

        rank = get_rank()
        pid = os.getpid()
        wid = get_worker_info().id if get_worker_info() is not None else 0
        show_log = (wid == 0 and rank == 0)

        if show_log:
            log(f'[Rank {cyan(rank)} worker {cyan(wid)} pid {cyan(pid)}] '
                f'init_loader: loading {len(sharded_metadata)} scenes...')
        t0 = time.time()

        for idx, meta in enumerate(sharded_metadata):
            video_path = meta['video_path']
            fpaths = sorted(os.listdir(join(video_path, 'videos')))
            fpaths = [join(video_path, 'videos', f) for f in fpaths]

            def create_vr(path):
                return CFRVideoReader(path, thread_type='NONE')

            vrs = parallel_execution(fpaths, action=create_vr)
            self.video_readers[idx] = vrs

            fpaths = sorted(os.listdir(join(video_path, 'cameras')))
            fpaths = [join(video_path, 'cameras', f) for f in fpaths]

            cams = parallel_execution(fpaths, action=read_camera_minimal)
            cams = aggregate_cams(cams)
            cams = normalize_cam_translation(cams, self.pose_norm_target)
            self.camera_params[idx] = cams

        if show_log:
            log(f'[Rank {cyan(rank)} worker {cyan(wid)} pid {cyan(pid)}] '
                f'init_loader: done in {time.time()-t0:.1f}s')

    def pick_stable_factor(self, max_t: float) -> float:
        """Pick the factor closest to max_t in log space from pose_stable_factors."""
        factors = self.pose_stable_factors
        if len(factors) == 1:
            return factors[0]
        log_mt = math.log(max(max_t, 1e-6))
        return min(factors, key=lambda f: abs(math.log(f) - log_mt))

    def shard_meta(self):
        # Return sharding results for THIS WORKER
        if hasattr(self, 'sharded_metadata'):
            return self.sharded_metadata

        # Empty metadata (shouldn't normally happen for mvgame — guard anyway
        # so a misconfigured parquet doesn't ZeroDivision the worker init).
        if not self.metadata:
            self.sharded_metadata = []
            return self.sharded_metadata

        # Return sharding results for THIS RANK
        wi = get_worker_info()
        rank = get_rank()
        world = get_world_size()

        # Worker
        if wi is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = wi.id, wi.num_workers

        # Global worker id
        g_workers = world * num_workers
        g_id = rank * num_workers + worker_id

        # No sp related sharding required since we're performing gathering
        if self.config.sp_size != 1 and self.sp_sharding:
            g_workers = world // self.config.sp_size
            g_id = rank // self.config.sp_size  # use the same shard for each of the sp group

        # Stride based sharding, controls memory usage
        self.sharded_metadata = self.metadata[g_id % len(self.metadata)::g_workers]  # every worker should have at least one sample
        self.metadata = self.sharded_metadata  # cleanup prompt memory

        return self.sharded_metadata  # should not access this directly

    def __len__(self):
        # Just report a large enough number here
        # For 1 metadata, would have 500 samples
        # Although the real number is very large
        #
        # Rough sample count per scene = (~100 cameras) * (~500 frames/scene) /
        # (gen_size * vae_stride_t frames per sample); times n_seqs scenes. The
        # final * 1e9 deliberately over-reports: this dataset behaves like an
        # infinite stream (getitem reseeds per idx), so __len__ only needs to be
        # large enough that the sampler never wraps within a run, not exact.
        mult = int(1e9)  # a large value for easier randomness management
        return len(self.metadata) * 100 * 500 // (self.gen_size * self.vae_stride_t) * mult  # this will report the total sample if calling from inside the main process of each gpu

    @property
    def n_seqs(self):
        return len(self.metadata)

    # Typical game-capture scene length in frames (~20s at 25fps).
    # Used to estimate effective training samples per scene for aggregator weighting.
    AVG_SCENE_FRAMES = 500

    @property
    def effective_samples(self):
        """sum(scene_frames / fps) * num_cameras * model_fps / tfs / 8.

        Per-row scene length uses (frame_end - frame_start) when available (a cut
        segment), otherwise falls back to AVG_SCENE_FRAMES. This makes aggregator
        sampling weights proportional to the actual amount of training content
        in each row, not the number of rows.

        Returns 0 when empty so the aggregator drops this dataset (weight=0).
        """
        if not self.metadata:
            return 0
        tfs = self.gen_size * self.vae_stride_t - 3
        default_fps = self.dataset_fps if self.dataset_fps else self.model_fps
        # Sampling-weight multiplier ONLY (feeds effective_samples below). Keep at
        # the tuned 25 — do NOT bump to match how many views shape_pool can draw.
        # The view-draw capability (drawing up to 100/110 views per scene) is
        # independent of this weight; conflating them silently triples mvgame's
        # sampling share and oversamples the slow raw-mvgame loader.
        num_cams = 25  # mvgame has ~100 cameras but training samples ~25 views per scene
        total = 0.0  # sum(scene_frames / row_fps) over rows
        for m in self.metadata:
            fs = int(m.get('frame_start') or 0)
            fe = m.get('frame_end')
            if fe is None or fe <= 0:
                scene_frames = self.AVG_SCENE_FRAMES
            else:
                scene_frames = max(0, int(fe) - fs)
            row_fps = m.get('fps') or default_fps
            total += scene_frames / row_fps
        total = total * num_cams * self.model_fps
        return max(1, int(total / tfs / 8))

    def getitem_impl(self, idx: int, **kwargs):
        """
        The randomness in play here:
        1. The order of all the sequences is shuffled randomly
        2. When constructing the main video, the view indices are randomly selected with acceleration
        3. The starting frame index is randomly selected
        4. The offset of supporting views has a small global perturbation
        5. There's another set of chaos offsets (every frame is different) added onto the already perturbed offsets
        6. The zooming factor, principal points of the camera and the roll of the camera are all selected using similar acceleration-based criteria
            - camera roll
            - zooming factor (also takes into account the minimum zooming required for the rolling)
            - cx offset
            - cy offset

        When a row's effective length is shorter than the requested
        total_frame_size, we keep advancing idx and (if shape_pool is active)
        re-pick a new (mv, gen) from the pool until we find a fit. No retry
        cap, no raise — gather_mixed_batch tolerates per-rank shape differences
        so divergent shape attempts across ranks are safe.
        """

        # Just use the underlying sharded dataset
        self.init_loader()  # make sure everything is initialized
        if not self.sharded_metadata:
            raise RuntimeError(
                f'{type(self).__name__} has no usable samples — should never '
                f'be picked by DatasetAggregator (weight=0). '
                f'Check data_path={self.data_path}'
            )

        # Resampling loop: advance idx (and re-pick shape from shape_pool, if
        # active) until we find a row whose effective length fits the picked
        # (mv, gen). Loops forever by design — log every retry so a starved
        # configuration is loud rather than silent.
        shape_attempt = 0
        retry_count = 0
        while True:
            # idx decomposes into (epoch, row): idx // n is how many full passes
            # over this worker's n sharded rows we've done — used as the RNG seed,
            # so each pass reseeds and produces fresh augmentation / view sampling
            # for the same row, yet stays deterministic per idx. idx % n (below)
            # picks the row. With sp_sharding (shard_meta), an sp group shares one
            # shard and thus one seed per idx, keeping their augmentation in lockstep.
            seed = idx // len(self.sharded_metadata)  # unique to every sp group
            seed = seed % (2 ** 32 - 1)  # make it a valid seed
            set_seed(seed)  # set the random seed for this particular getitem
            local_idx = idx % len(self.sharded_metadata)
            meta = self.sharded_metadata[local_idx]
            vrs = self.video_readers[local_idx]
            cams = self.camera_params[local_idx]
            cams = deaggregate_cams(cams)

            # Frame-range window: use only [frame_start, frame_end) of the underlying files.
            # Both default to full video when absent (backward compatible with old parquets).
            actual_len = min(len(vrs[0]), len(cams[0]))
            frame_start = int(meta.get('frame_start') or 0)
            frame_end = meta.get('frame_end')
            if frame_end is None or frame_end <= 0:
                frame_end = actual_len
            else:
                frame_end = min(int(frame_end), actual_len)
            n_frames_src = max(0, frame_end - frame_start)  # effective source length inside the window

            # Per-row source fps (parquet column). Overrides self.dataset_fps when
            # present — mirrors svreal static/dynamic convention. Used e.g. to tag
            # watch_dogs_legion as 32fps while other mvgame rows stay at 25fps.
            row_src_fps = meta.get('fps') or self.dataset_fps
            # Snap (model_fps, source_fps) to a clean ratio via the remap factory:
            # e.g. (16, 25) → (16, 24) → ratio 1.5 (period-2 alternating Δf instead of
            # the chaotic period-16 pattern from raw 25/16=1.5625). See dataset/fps_remap.py.
            row_eff_model_fps, row_snapped_src_fps = resolve_fps_remap(self.model_fps, row_src_fps) \
                if row_src_fps else (self.model_fps, None)
            row_fps_ratio = row_snapped_src_fps / row_eff_model_fps if row_snapped_src_fps else 1.0
            if row_fps_ratio < 1:
                row_fps_ratio = 1.0  # don't upsample (matches ctor clamp)

            n_frames = int(n_frames_src / row_fps_ratio)    # effective frames at model fps
            n_views = len(vrs)

            # total_latent_size is fixed now, defined in config/passed through dataset
            total_latent_size = self.gen_size  # In training, we use gen_size as the total latent sequence length
            # Wan's causal video VAE maps L latent frames to (L-1)*vae_stride_t + 1
            # pixel frames (first frame coded alone, then groups of vae_stride_t).
            # That equals L*vae_stride_t - (vae_stride_t - 1); with vae_stride_t=4
            # the constant is -3. So we must decode this many source frames.
            total_frame_size = total_latent_size * self.vae_stride_t - 3  # +1 to pass in the first frame for the wan vae

            if total_frame_size <= n_frames:
                idx = local_idx  # commit the row that fits
                break

            retry_count += 1
            # Re-pick a new shape from shape_pool on every miss; if shape_pool is
            # inactive maybe_pick_shape is a no-op and we just advance idx.
            old_mv, old_gen = self.mv_size, self.gen_size
            if self.shape_pool:
                shape_attempt += 1
                self.maybe_pick_shape(idx, attempt=shape_attempt)
            log(yellow(
                f"Video {meta['video_path']} too short ({n_frames}<{total_frame_size}, "
                f"mv={old_mv}, gen={old_gen}) on {self.data_path}: advancing idx, "
                f"re-pick → (mv={self.mv_size}, gen={self.gen_size}) [retry={retry_count}]"
            ))
            idx = local_idx + 1

        # overfit_keep_aug=True means "restrict metadata only; leave everything else
        # at its normal (augmented) behavior" — see ctor docstring.
        overfit_disables_aug = self.overfit and not self.overfit_keep_aug
        disable_aug = random.random() < self.disable_augmentation_ratio or overfit_disables_aug

        # Make the size of kv divisible by the chunk latent size (q)
        prompts = meta['caption']

        # Main walk: one random_move through the camera ring x time, sliced to a
        # random total_frame_size window, frame col remapped to absolute source
        # frames. Shared with make_sample_specs via utils.mvgame so the
        # presampled aug-mvgame indices match what training builds (the index
        # construction is the SINGLE source of truth; pixel aug stays below).
        view_acc_abs = draw_view_acc_abs(self.view_acc_abs_min, self.view_acc_abs_max, disable_aug)
        overfit_view_sample = list(range(*self.overfit_vid_sample)) if self.overfit else None
        indices = build_main_walk(n_frames, n_views, total_frame_size, view_acc_abs, disable_aug,
                                  row_fps_ratio, frame_start, n_frames_src, overfit_view_sample)

        # Total number of frames in the loaded video
        n_frames = len(indices)
        mv = self.mv_size
        batch = {'cpu': {}}
        # Basic batch misc info
        batch['mv'] = mv
        # These are things we want to keep on the cpu
        batch['cpu']['seed'] = int(seed)
        batch['cpu']['prompts'] = prompts
        batch['cpu']['dataset_name'] = self.dataset_name
        batch['cpu']['parquet'] = basename(self.data_path)
        batch['cpu']['indices'] = indices
        batch['cpu']['video_path'] = meta['video_path']
        # Row + source-frame span surfaced on the vis meta panel (parity with
        # static/multiview/dynamic). indices[:, 1] is the source frame column
        # (post fps-remap + frame_start); mv chaos only perturbs the view column
        # indices[:, 0], so min/max here is the true frame range actually read.
        batch['cpu']['rows'] = np.asarray([int(meta.get('pose_idx', -1))], dtype=np.int64)
        batch['cpu']['start_frames'] = np.asarray([int(indices[:, 1].min())], dtype=np.int64)
        batch['cpu']['end_frames'] = np.asarray([int(indices[:, 1].max())], dtype=np.int64)
        # Effective post-subsample fps (matches static.py / dynamic.py convention).
        # The remap factory has already chosen a clean (eff_model, snapped_src)
        # pair, so we report the eff_model as what the model "sees" — for some
        # source rates this is < the outer self.model_fps (e.g. 30→15, 60→15)
        # to keep the ratio integer.
        effective_fps = int(round(row_eff_model_fps)) if row_src_fps else self.model_fps
        batch['fps'] = effective_fps

        # We're asked to pack multiview information into a single batch (using the predefined patterns)
        pack_size = self.pack['pack_size']
        pack_inds = self.pack['pack_inds']
        ratios = self.pack['rs']
        xs = self.pack['xs']
        ys = self.pack['ys']

        # Compute the view index offsets. Spread mv views evenly across the source
        # camera ring as fractions of a full turn, jitter each by off_perturb_std,
        # then map fraction -> integer source-view index. The +0.5 before int cast
        # is round-to-nearest (np int cast truncates); %n_views wraps the ring.
        # pack_inds reorders from "geometry order" (the spatial slot layout drawn
        # in the PACK_FACTORY_MANUAL diagram above) into "logical order" (the order
        # rs/xs/ys iterate when packing), so offsets[i] aligns with ratios[i] etc.
        offsets = compute_view_offsets(mv, n_views, self.off_perturb_std, pack_inds)
        batch['cpu']['offsets'] = offsets  # logical

        height_pack = int(self.height * pack_size[0])
        width_pack = int(self.width * pack_size[1])
        frames = torch.zeros((n_frames, 3, height_pack, width_pack), dtype=torch.float32)  # match the shape of the loaded frames before feeding into the vae
        projs = []
        projs_inv = []
        Ks, Rs, Ts = [], [], []

        # This ensures all camera parameters are world locked.
        # Anchor the shared world reference at the window's FIRST SAMPLED frame
        # (view 0), matching multiview/static/dynamic which bootstrap R0 from the
        # first sampled frame. Anchoring at the full-video frame 0 (cams[0]['000000'])
        # leaves |T| = distance from the video start; with a random start_idx that can
        # be >> the window's own movement, and since apply_proj casts the projected
        # q/k back to bf16 at magnitude ~|T|, the large |T| swamps the relative camera
        # signal (precision loss / score overflow). PRoPE is relative, so re-anchoring
        # only changes the numerics, not the geometry the model sees.
        # .get() with the frame-0 fallback guards against a missing key (never crash).
        ref_key = f'{int(indices[0, 1]):06d}'
        ref_cam = cams[0].get(ref_key, cams[0]['000000'])
        R0, T0 = ref_cam['R'].astype(np.float32), ref_cam['T'].astype(np.float32)

        # Sequence-level gamma: compute ONCE here from a small sample across
        # views, then apply the same gamma value to every view's frames. This
        # matches aug_views.py and preserves cross-view brightness consistency
        # (per-view median would give each view its own gamma → broken).
        gamma_value = 1.0
        if self.gamma_correction:
            gamma_value = compute_sequence_gamma(vrs, indices, mv, n_views)

        # Pack the loaded images and camera parameters
        for off, ratio, x, y in zip(offsets, ratios, xs, ys):
            # Per-view index map: copy the main walk, add this view's mv_chaos
            # view-jitter + ring offset. apply_view_chaos draws the chaos HERE
            # (inside the load loop) so the RNG stream stays interleaved with each
            # view's video_augmentation draw below — identical to the old inline code.
            indices_v = apply_view_chaos(indices, int(off), n_views, view_acc_abs, self.mv_chaos, disable_aug)
            if disable_aug:
                kwargs.update({
                    "fixed_s": 1.0,
                    "fixed_cx": 0.0,
                    "fixed_cy": 0.0,
                    "fixed_r": 0.0,
                })
            kwargs['image_aug'] = self.image_aug and not disable_aug
            kwargs['gamma_value'] = gamma_value
            kwargs['max_fov_h_deg'] = self.max_fov_h_deg
            batch_v = load_posed_video(indices_v, vrs, cams, self.height, self.width, self.per_worker_threads, ratio, R0, T0, **kwargs)  # resized according predefined ratio

            h, w = batch_v['frames'].shape[-2:]
            x, y = int(x * self.width), int(y * self.height)  # x, y are float offsets
            frames[:, :, y:y + h, x:x + w] = batch_v['frames']

            projs.append(batch_v['projs'])
            projs_inv.append(batch_v['projs_inv'])
            Ks.append(batch_v['Ks'])
            Rs.append(batch_v['Rs'])
            Ts.append(batch_v['Ts'])

        # Regular info
        batch['frames'] = frames  # F, 3, H, 2W, packed
        # Normally the camera parameters should be considered metadata, but since we want to move them to the gpu, we keep them raw

        batch['projs'] = torch.stack(projs, dim=1).reshape(-1, 4, 4)  # F8, 4, 4
        batch['projs_inv'] = torch.stack(projs_inv, dim=1).reshape(-1, 4, 4)  # F8, 4, 4
        batch['Ks'] = torch.stack(Ks, dim=1).reshape(-1, 3, 3)  # F8, 3, 3
        batch['Rs'] = torch.stack(Rs, dim=1).reshape(-1, 3, 3)  # F8, 3, 3
        batch['Ts'] = torch.stack(Ts, dim=1).reshape(-1, 3, 1)  # F8, 3, 1

        # Adaptive pose stable factor sized by MAX PAIRWISE camera distance (window
        # diameter = largest relative translation), not mean: a one-way trajectory's far
        # end is what overflows bf16 PRoPE (score ≈ 2.4 * T², need T < ~5). See
        # select_pose_stable_factor. batch['Rs']/['Ts'] are now WORLD-LOCKED (v0/f0 at
        # origin), consistent with projs + pose_10d; pairwise is translation-invariant
        # so it stays correct + robust regardless of frame.
        centers = -torch.bmm(batch['Rs'].mT, batch['Ts']).squeeze(-1)  # [F*mv, 3] world-locked
        pose_stable_factor, pose_max_t = select_pose_stable_factor(centers, self.pose_stable_factors)
        if pose_stable_factor != 1.0:
            # Scale BOTH projection translations and w2c T. The trainer derives
            # pose_10d camera center C=-R^T T, so both PRoPE streams see the same scale.
            batch['projs'][:, :3, 3] /= pose_stable_factor
            batch['projs_inv'][:, :3, 3] /= pose_stable_factor
            batch['Ts'] /= pose_stable_factor

        # Multi-view info, should not be batched
        # TODO: Different size in different batch? Dynamically change batch size
        batch['cpu']['pack'] = self.pack  # pack info dict
        batch['cpu']['pack']['width'] = self.width  # target width
        batch['cpu']['pack']['height'] = self.height  # target height
        batch['cpu']['pose_stable_factor'] = pose_stable_factor
        batch['cpu']['pose_max_t'] = pose_max_t  # pre-division max pairwise dist (bf16 diagnostic)

        return batch

    def get_worker_iterator(self):
        """
        Internal helper: creates an infinite, shuffling iterator for the current worker.
        This simulates an IterableDataset behavior within a Map-style dataset.
        """
        # Ensure metadata is loaded for this worker
        self.init_loader()
        indices = np.arange(len(self.sharded_metadata))

        while True:
            # SHUFFLE: Randomize the order for this "local epoch"
            # This ensures we see every sample exactly once before repeating (Sampling Without Replacement)
            np.random.shuffle(indices)

            for idx in indices:
                yield idx

    def maybe_pick_shape(self, idx: int, attempt: int = 0):
        """Idx-seeded weighted pick from self.shape_pool. See
        StaticDataset.maybe_pick_shape for semantics. Mutates self.mv_size,
        self.gen_size, self.pack (strip layout), and self.shape_pool_active.

        `attempt`: salt added to the seed so getitem_impl can re-pick a
        different shape after exhausting its row-skip retries for the
        currently picked shape. attempt=0 preserves the original seed for
        backward compatibility with existing runs.
        """
        if not self.shape_pool:
            self.shape_pool_active = False
            return None
        seed_str = f'shape_pool_{idx}' if attempt == 0 else f'shape_pool_{idx}_attempt_{attempt}'
        rng = random.Random(seed_str)
        weights = self.shape_pool_weights or [1.0] * len(self.shape_pool)
        mv_p, gen_p = rng.choices(self.shape_pool, weights=weights, k=1)[0]
        self.mv_size = int(mv_p)
        self.gen_size = int(gen_p)
        self.pack = make_strip_pack(self.mv_size)
        self.shape_pool_active = True
        return self.mv_size, self.gen_size

    def __getitem__(self, idx: int):
        try:
            # # ------------------------------------------------------------------
            # # Hack: Simulate IterableDataset behavior
            # # We completely IGNORE the incoming 'idx' from the global sampler.
            # # Instead, we maintain a persistent iterator state on the worker.
            # # ------------------------------------------------------------------
            # iterator = self.get_worker_iterator()

            # # Get the next valid local index from our shuffled generator
            # # This index is guaranteed to be within [0, len(local_metadata) - 1]
            # local_idx = next(iterator)
            # meta = self.sharded_metadata[local_idx]

            # # We pass this local_idx to getitem_impl.
            # # Since local_idx < len(metadata), the logic inside getitem_impl:
            # # "idx = idx % len(metadata)" becomes a no-op (identity operation).
            # return self.getitem_impl(local_idx)

            self.maybe_pick_shape(idx)
            return self.getitem_impl(idx)

        except Exception as e:
            wi = get_worker_info()
            meta = locals().get('meta', {})
            meta_video_path = None
            meta_keys = None
            try:
                if isinstance(meta, dict):
                    meta_video_path = meta.get('video_path', None)
                    meta_keys = list(meta.keys())
            except Exception:
                meta_video_path = None

            log(red(
                f"[Dataset __getitem__] failed: rank={get_rank()} worker={wi.id if wi is not None else 0} "
                f"local_idx={locals().get('local_idx', 'N/A')} "
                f"meta_video_path={meta_video_path} meta_keys={meta_keys} err={e}"
            ))
            stacktrace()
            raise


def cycle(dl):
    """
    Yield a new iterator after finishing one epoch
    """
    while True:
        for data in dl:
            yield data


@catch_throw
def test_video_augmentation():
    # Test augmentations
    from utils.console import log
    from utils.console import blue
    from utils.console import run_parser
    from utils.video import write_video
    args = dotdict(
        data_path='/mnt/bn/foundation-ads3/zhenxu.zx/datasets/mvgame/cp77_of_re.parquet',
        gen_size=30,
        per_worker_threads=8,
        output_dir='data/dataaug/cp77_of_re'
    )
    args = run_parser(args, __doc__)

    dataset = MultiViewDataset(
        data_path=args.data_path,
        gen_size=args.gen_size,
        per_worker_threads=args.per_worker_threads,
        height=448,
        width=832,
        overfit=True,
        overfit_seq_ind=0,
        overfit_vid_sample=(0, 1, 1),
        mv_chaos=0.0,
        off_perturb_std=0.0,
    )  # faster loading

    os.makedirs(args.output_dir, exist_ok=True)
    frames_uint8_list = []
    out_path_list = []
    KRTs_list = []

    # Baseline
    h, w = dataset.height, dataset.width
    dataset.height, dataset.width = 720, 1280
    batch = dataset.getitem_impl(0, fixed_s=1.0, fixed_cx=0.0, fixed_cy=0.0, fixed_r=0.0, force_crop_h=720, force_crop_w=1280)
    frames, Ks, Rs, Ts = batch['frames'], batch['Ks'], batch['Rs'], batch['Ts']  # F, 3, H, W (packed)
    # Convert from float [0, 1] to uint8 [0, 255] and (F, C, H, W) -> (F, H, W, C)
    frames_uint8 = (frames.permute(0, 2, 3, 1) * 255).clip(0, 255).to(torch.uint8)
    out_path = join(args.output_dir, f'baseline.mp4')
    frames_uint8_list.append(frames_uint8)
    out_path_list.append(out_path)
    KRTs_list.append([Ks, Rs, Ts])
    dataset.height, dataset.width = h, w

    # Test settings
    s_cx_cy_r_list = [
        [1.0, 0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [832 / 1280, 0.0, 0.0, 0.0],
        [448 / 720, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0],
        [2.0, 0.5, 0.5, 0.0],
        [2.0, -0.5, -0.5, 0.0],
        [1.0, 0.0, 0.0, -45],
        [1.0, 0.0, 0.0, 45],
        [1.0, 0.5, 0.5, 45],
        [2.0, -0.5, -0.5, 45],
    ]

    # Virtually no change in dataloading speed for now
    for s, cx, cy, r in tqdm(s_cx_cy_r_list, desc='Loading samples'):
        # s, cx, cy, r = 2.0, 0.25, 0.25, 0.0
        batch = dataset.getitem_impl(0, fixed_s=s, fixed_cx=cx, fixed_cy=cy, fixed_r=r, force_crop_h=720, force_crop_w=1280)
        frames, Ks, Rs, Ts = batch['frames'], batch['Ks'], batch['Rs'], batch['Ts']  # F, 3, H, W (packed)
        # Convert from float [0, 1] to uint8 [0, 255] and (F, C, H, W) -> (F, H, W, C)
        frames_uint8 = (frames.permute(0, 2, 3, 1) * 255).clip(0, 255).to(torch.uint8)
        out_path = join(args.output_dir, f'sample_fixed_s{s:.2f}_cx{cx:.2f}_cy{cy:.2f}_r{r:.2f}.mp4')

        frames_uint8_list.append(frames_uint8)
        out_path_list.append(out_path)
        KRTs_list.append([Ks, Rs, Ts])

    def write(filename, frames_uint8, KRTs, **kwargs):
        Ks, Rs, Ts = KRTs  # unroll list
        write_video(filename, frames_uint8, **kwargs)
        np.savez_compressed(filename.replace('.mp4', '.npz'), Ks=Ks, Rs=Rs, Ts=Ts)

    parallel_execution(out_path_list, frames_uint8_list, KRTs_list, action=write, desc=f'Writing samples to {blue(args.output_dir)}', print_progress=True, fps=16)


@catch_throw
def test_random_augmentation():
    # Test augmentations
    from utils.console import log
    from utils.console import blue
    from utils.console import run_parser
    from utils.video import write_video
    args = dotdict(
        data_path='/mnt/bn/foundation-ads3/zhenxu.zx/datasets/mvgame/cp77_of_re.parquet',
        gen_size=30,
        per_worker_threads=8,
        num_samples=12,
        output_dir='data/dataaug/cp77_of_re'
    )
    args = run_parser(args, __doc__)

    dataset = MultiViewDataset(
        data_path=args.data_path,
        gen_size=args.gen_size,
        per_worker_threads=args.per_worker_threads,
        height=448,
        width=832,
    )  # faster loading

    os.makedirs(args.output_dir, exist_ok=True)
    frames_uint8_list = []
    out_path_list = []
    KRTs_list = []

    # Virtually no change in dataloading speed for now
    for i in tqdm(range(args.num_samples), desc='Loading samples'):
        batch = dataset[i]
        frames, Ks, Rs, Ts = batch['frames'], batch['Ks'], batch['Rs'], batch['Ts']  # F, 3, H, W (packed)
        # Convert from float [0, 1] to uint8 [0, 255] and (F, C, H, W) -> (F, H, W, C)
        frames_uint8 = (frames.permute(0, 2, 3, 1) * 255).clip(0, 255).to(torch.uint8)
        out_path = join(args.output_dir, f'sample_{i}.mp4')

        frames_uint8_list.append(frames_uint8)
        out_path_list.append(out_path)
        KRTs_list.append([Ks, Rs, Ts])

    def write(filename, frames_uint8, KRTs, **kwargs):
        Ks, Rs, Ts = KRTs  # unroll list
        write_video(filename, frames_uint8, **kwargs)
        # np.savez_compressed(filename.replace('.mp4', '.npz'), Ks=Ks, Rs=Rs, Ts=Ts)
        c2ws = affine_inverse(torch.cat([Rs, Ts], dim=-1))
        # np.savez_compressed(filename.replace('.mp4', '.npz'), Ks=Ks, c2ws=c2ws)
        export_camera(c2ws, Ks, filename=filename.replace('.mp4', '.ply'))

    parallel_execution(out_path_list, frames_uint8_list, KRTs_list, action=write, desc=f'Writing samples to {blue(args.output_dir)}', print_progress=True, fps=16)


if __name__ == '__main__':
    # test_video_augmentation()
    test_random_augmentation()
