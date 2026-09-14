"""Camera-ring and time indices used by multi-view video augmentation.

Keep random draws in their original order: view jitter is interleaved with each
view's pixel augmentation by the dataset reader.
"""

from typing import Optional

import numpy as np

# ===========================================================================
# Augmented multi-view index construction (shared training <-> sampler).
#
# Mirror of dataset/mvgame.py MultiViewDataset.getitem_impl, index parts only
# (no pixel aug). Each helper maps 1:1 to a block of getitem_impl and draws the
# SAME np.random calls in the SAME order, so a caller that runs them in order
# reproduces training's construction bit-for-bit under a shared seed.
#
# random_move lives in dataset/mvgame.py (it carries the C++/video_sampling_ext fast
# path + numpy fallback); we lazy-import it inside the helpers so this module
# stays import-cheap and dataset/mvgame.py can import these helpers at top level
# without a circular import.
# ===========================================================================


def draw_view_acc_abs(view_acc_abs_min: float, view_acc_abs_max: float, disable_aug: bool = False) -> float:
    """Clipped-normal view-acceleration magnitude (getitem_impl 1432-1440).
    Returns 0.0 when aug is disabled (the straight single-view path)."""
    if disable_aug:
        return 0.0
    mean = (view_acc_abs_max + view_acc_abs_min) / 2
    std = (view_acc_abs_max - view_acc_abs_min) / 6
    view_acc_abs = np.random.normal(mean, std)

    # clip to avoid cpp errors on the tail (matches getitem_impl)
    return float(np.clip(view_acc_abs, view_acc_abs_min, view_acc_abs_max))


def build_main_walk(
    n_frames_full: int,
    n_views: int,
    total_frame_size: int,
    view_acc_abs: float,
    disable_aug: bool,
    row_fps_ratio: float = 1.0,
    frame_start: int = 0,
    n_frames_src: Optional[int] = None,
    overfit_view_sample: Optional[list] = None,
) -> np.ndarray:
    """The shared main walk (getitem_impl 1430-1454): one random_move through
    the camera ring x time over the FULL effective row length, sliced to a
    random total_frame_size window, then frame column remapped from model-fps
    space back to ABSOLUTE source frames ([frame_start, frame_end)).

    Returns indices_main: (total_frame_size, 2) int64 [src_view, src_frame].
    RNG: main random_move seed, then start_idx (aug branch); a single view
    choice (disable_aug branch)."""
    from dataset.mvgame import random_move

    if not disable_aug:
        indices = random_move(
            length=n_frames_full,
            n_views=n_views,
            n_frames=n_frames_full,
            view_acc_min=-view_acc_abs,
            view_acc_max=view_acc_abs,
            seed=np.random.randint(1e9),
        )  # F, 2: view, frame
        start_idx = np.random.randint(0, n_frames_full - total_frame_size + 1)
        indices = indices[start_idx : start_idx + total_frame_size]
    else:
        view_sample = overfit_view_sample if overfit_view_sample is not None else list(range(n_views))
        view = np.random.choice(view_sample)
        indices = np.asarray([(view, frame) for frame in range(total_frame_size)])
    indices = np.array(indices, copy=True)  # own the slice before in-place remap

    # Remap frame indices model-fps -> source-fps space, then offset to the window.
    if row_fps_ratio != 1.0:
        cap = (n_frames_src - 1) if n_frames_src is not None else None
        remapped = np.round(indices[:, 1] * row_fps_ratio).astype(indices.dtype)
        indices[:, 1] = np.minimum(remapped, cap) if cap is not None else remapped
    if frame_start:
        indices[:, 1] = indices[:, 1] + frame_start
    return indices


def compute_view_offsets(
    mv: int, n_views: int, off_perturb_std: float, pack_inds: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-view ring offsets (getitem_impl 1507-1519): mv views spread evenly
    around the source camera ring as fractions of a turn, each jittered by
    off_perturb_std, mapped to integer source-view indices, reordered by
    pack_inds (geometry->logical; identity for the shape_pool strip pack), with
    view 0 pinned to the ring center (offset 0).
    RNG: one np.random.normal of length mv."""
    offsets = np.arange(mv) / mv
    offsets = np.random.normal(offsets, off_perturb_std)
    offsets = (offsets * n_views + 0.5).astype(np.int32) % n_views
    if pack_inds is not None:
        offsets = offsets[pack_inds]
    offsets = np.array(offsets, copy=True)
    offsets[0] = 0  # fix first view to the center view
    return offsets


def apply_view_chaos(
    indices_main: np.ndarray, off: int, n_views: int, view_acc_abs: float, mv_chaos: float, disable_aug: bool
) -> np.ndarray:
    """One output view's index map (getitem_impl 1551-1558): copy the main walk,
    add a small per-view mv_chaos walk to the VIEW column only (frame column is
    shared across views -> the views are time-synchronized, camera-offset), then
    add this view's ring offset and wrap. Window length F = len(indices_main)
    (the chaos random_move runs at the WINDOW length, matching the post-slice
    n_frames reassignment in getitem_impl).
    RNG: one chaos random_move seed (aug branch only)."""
    from dataset.mvgame import random_move

    indices_v = np.copy(indices_main)
    F = len(indices_main)
    if not disable_aug:
        indices_chaos = random_move(
            length=F,
            n_views=n_views,
            n_frames=F,
            view_acc_min=-view_acc_abs * mv_chaos,
            view_acc_max=view_acc_abs * mv_chaos,
            seed=np.random.randint(1e9),
        )  # F, 2: view, frame
        indices_v[:, 0] += indices_chaos[:, 0] - indices_chaos[0, 0]
    indices_v[:, 0] += off
    indices_v[:, 0] %= n_views
    return indices_v
