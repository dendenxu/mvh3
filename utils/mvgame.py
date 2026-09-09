"""
Shared helpers for mvgame quality-filter / cut scripts, plus the augmented
multi-view INDEX CONSTRUCTION shared between training (dataset/mvgame.py
MultiViewDataset.getitem_impl) and the presample/caption sampler
(scripts/data/seedpro/make_sample_specs.py).

Keeps the per-script files focused on their filter logic and avoids four copies
of the same game-name / game-dir / EXR-truncation utilities drifting apart.

Index construction (construct_aug_indices + the four sub-helpers it wraps) is
the SINGLE source of truth for how an aug-mvgame sample is built from a scene:
one `random_move` main walk through the camera-ring x time, then per output
view a mv_chaos view-jitter + a ring offset. Returning the per-view (mv, F, 2)
[src_view, src_frame] index map lets the sampler store a concrete construction
in the spec and the captioner/renderer replay the EXACT frames training builds.
The sub-helpers are exposed so getitem_impl can call apply_view_chaos INSIDE
its per-view load loop, preserving the original RNG draw order (chaos draw
interleaved with each view's video_augmentation draw) — see construct_aug_indices.
"""

from typing import Optional

import numpy as np


def canonical_game(game_dir_name: str) -> str:
    """Collapse `game_dir` (raw directory name) into a canonical snake_case key.

    Rules:
    - Lowercase + spaces → underscores
    - Strip the part2 'the_second_batch' suffixes so first/second-batch merge
    - Normalize 'deadisland_2' (as seen on disk) to 'dead_island_2'
    """
    n = game_dir_name.lower().replace(' ', '_')
    for suf in ('_the_second_batch', '_second_batch'):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    n = n.replace('deadisland_2', 'dead_island_2')
    return n


def game_dir_from_seq_path(seq_path: str) -> str:
    """Extract `<game_dir>` from a `.../capture/<game_dir>/<seq_id>` seq_path.

    Returns 'unknown' if the path doesn't contain `/capture/`.
    """
    if '/capture/' not in seq_path:
        return 'unknown'
    return seq_path.split('/capture/')[-1].split('/')[0]


def resolve_video_paths(paths, parquet_path):
    """Resolve relative `video_path` entries against the parquet file's dirname.

    Base / cut parquets store `video_path` as relative (e.g. 'capture/<game>/<seq>')
    for portability; filter scripts need absolute paths to open cameras/depths/videos.
    Any path already absolute is returned unchanged (backward-compat for older parquets).
    """
    from os.path import abspath, dirname, isabs, join
    parent = abspath(dirname(parquet_path) or '.')
    return [p if isabs(p) else join(parent, p) for p in paths]


def exr_is_broken(depth) -> bool:
    """Heuristic for truncated EXR writes: frame 0 has content but frame -1 is all-zero.

    Works with numpy arrays and torch tensors. Call only when the EXR read
    itself succeeded; an exception during read is a stronger broken signal.
    Expected shape: (N, ...) where N is the frame axis.
    """
    if depth.shape[0] < 2:
        return False
    first_has_content = bool((depth[0] != 0).any())
    last_all_zero = bool((depth[-1] == 0).all())
    # torch bool tensor .item() fallback for consistency
    if hasattr(first_has_content, 'item'):
        first_has_content = first_has_content.item()
    if hasattr(last_all_zero, 'item'):
        last_all_zero = last_all_zero.item()
    return bool(first_has_content and last_all_zero)


# ===========================================================================
# Augmented multi-view index construction (shared training <-> sampler).
#
# Mirror of dataset/mvgame.py MultiViewDataset.getitem_impl, index parts only
# (no pixel aug). Each helper maps 1:1 to a block of getitem_impl and draws the
# SAME np.random calls in the SAME order, so a caller that runs them in order
# reproduces training's construction bit-for-bit under a shared seed.
#
# random_move lives in dataset/mvgame.py (it carries the C++/_dataset_ext fast
# path + numpy fallback); we lazy-import it inside the helpers so this module
# stays import-cheap and dataset/mvgame.py can import these helpers at top level
# without a circular import.
# ===========================================================================


def draw_view_acc_abs(view_acc_abs_min: float, view_acc_abs_max: float,
                      disable_aug: bool = False) -> float:
    """Clipped-normal view-acceleration magnitude (getitem_impl 1432-1440).
    Returns 0.0 when aug is disabled (the straight single-view path)."""
    if disable_aug:
        return 0.0
    mean = (view_acc_abs_max + view_acc_abs_min) / 2
    std = (view_acc_abs_max - view_acc_abs_min) / 6
    view_acc_abs = np.random.normal(mean, std)
    # clip to avoid cpp errors on the tail (matches getitem_impl)
    return float(np.clip(view_acc_abs, view_acc_abs_min, view_acc_abs_max))


def build_main_walk(n_frames_full: int, n_views: int, total_frame_size: int,
                    view_acc_abs: float, disable_aug: bool,
                    row_fps_ratio: float = 1.0, frame_start: int = 0,
                    n_frames_src: Optional[int] = None,
                    overfit_view_sample: Optional[list] = None) -> np.ndarray:
    """The shared main walk (getitem_impl 1430-1454): one random_move through
    the camera ring x time over the FULL effective row length, sliced to a
    random total_frame_size window, then frame column remapped from model-fps
    space back to ABSOLUTE source frames ([frame_start, frame_end)).

    Returns indices_main: (total_frame_size, 2) int64 [src_view, src_frame].
    RNG: main random_move seed, then start_idx (aug branch); a single view
    choice (disable_aug branch)."""
    from dataset.mvgame import random_move
    if not disable_aug:
        indices = random_move(length=n_frames_full, n_views=n_views, n_frames=n_frames_full,
                              view_acc_min=-view_acc_abs, view_acc_max=view_acc_abs,
                              seed=np.random.randint(1e9))  # F, 2: view, frame
        start_idx = np.random.randint(0, n_frames_full - total_frame_size + 1)
        indices = indices[start_idx:start_idx + total_frame_size]
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


def compute_view_offsets(mv: int, n_views: int, off_perturb_std: float,
                         pack_inds: Optional[np.ndarray] = None) -> np.ndarray:
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


def apply_view_chaos(indices_main: np.ndarray, off: int, n_views: int,
                     view_acc_abs: float, mv_chaos: float,
                     disable_aug: bool) -> np.ndarray:
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
        indices_chaos = random_move(length=F, n_views=n_views, n_frames=F,
                                    view_acc_min=-view_acc_abs * mv_chaos,
                                    view_acc_max=view_acc_abs * mv_chaos,
                                    seed=np.random.randint(1e9))  # F, 2: view, frame
        indices_v[:, 0] += indices_chaos[:, 0] - indices_chaos[0, 0]
    indices_v[:, 0] += off
    indices_v[:, 0] %= n_views
    return indices_v


def construct_aug_indices(mv: int, n_views: int, n_frames_full: int, total_frame_size: int,
                          view_acc_abs_min: float, view_acc_abs_max: float,
                          mv_chaos: float, off_perturb_std: float,
                          row_fps_ratio: float = 1.0, frame_start: int = 0,
                          n_frames_src: Optional[int] = None,
                          disable_aug: bool = False,
                          pack_inds: Optional[np.ndarray] = None) -> np.ndarray:
    """Build the full per-view index map for ONE aug-mvgame sample.

    Returns indices_all: (mv, total_frame_size, 2) int [src_view, src_frame] —
    output view i, frame t reads source camera indices_all[i, t, 0] at source
    frame indices_all[i, t, 1]. This is what make_sample_specs stores in the
    spec and the captioner/renderer replays.

    Draws all mv chaos walks up front (after the main walk + offsets). This is
    the SAMPLER-facing entry point. getitem_impl does NOT call this wrapper: it
    calls build_main_walk + compute_view_offsets once, then apply_view_chaos
    INSIDE its per-view load loop, so each view's chaos draw stays interleaved
    with that view's video_augmentation draw (preserving the training RNG
    stream). Both routes produce identical indices under the same seed because
    the chaos draw order across views is the same; only the interleaving with
    the (sampler-absent) pixel-aug draws differs."""
    view_acc_abs = draw_view_acc_abs(view_acc_abs_min, view_acc_abs_max, disable_aug)
    indices_main = build_main_walk(n_frames_full, n_views, total_frame_size, view_acc_abs,
                                   disable_aug, row_fps_ratio, frame_start, n_frames_src)
    offsets = compute_view_offsets(mv, n_views, off_perturb_std, pack_inds)
    indices_all = np.stack([
        apply_view_chaos(indices_main, int(off), n_views, view_acc_abs, mv_chaos, disable_aug)
        for off in offsets
    ])  # (mv, F, 2)
    return indices_all
