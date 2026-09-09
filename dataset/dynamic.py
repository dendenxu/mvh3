# Dynamic multi-video dataset.
#
# Each __getitem__ loads `mv_size` INDEPENDENT videos as separate views, packed
# into one batch via the standard pack_factory layout (different views may use
# different sub-resolutions). Views are isolated from each other in attention
# (via metadata['view_isolated']) and each view has its OWN text caption,
# pose normalization, and i2v cond_image first frame.
#
# Reuses StaticDataset machinery for parquet loading, pose parsing, video
# reading and `load_view`. Adds:
#   - `__getitem__` picks mv_size random rows
#   - per-view independent pose normalization (each view's R0/T0 is its own)
#   - per-view prompt embeddings stacked as [mv, L, D]
#   - sets `batch['cpu']['view_isolated'] = True` so the trainer/model take the
#     view-isolated path. Other dataset types are unaffected.

from typing import List
import os
import random
import torch
import numpy as np

from torch.utils.data import get_worker_info
from torch.utils.data._utils.collate import default_collate

from utils.console import *
from utils.distributed import get_rank
from utils.distributed import is_main_process
from utils.distributed import is_node_main
from utils.video import TorchCodecVideoReader
from utils.misc import set_seed
from dataset.mvgame import pack_factory
from dataset.mvgame import make_strip_pack
from dataset.mvgame import normalize_cam_translation
from dataset.mvgame import select_pose_stable_factor
from dataset.mvgame import compute_sequence_gamma
from dataset.mvgame import LOOSE_EXPOSURE_BAND
from dataset.static import StaticDataset
from dataset.static import parse_pose_column
from dataset.fps_remap import resolve_fps_remap


class DynamicDataset(StaticDataset):
    """Dataset that loads `mv_size` independent videos as separate views per sample.

    Single-video parquets (sekai_walking, drone, dl3dv, ...) are valid sources —
    each row is treated as one independent monocular sample. The dataset picks
    `mv_size` random rows per __getitem__ and packs them into a single multi-view
    batch with view-isolated attention.

    TODO (seedpro re-screening — NOT yet wired in; user 2026-06-29): single-view
    static-prior samples that seedpro flagged non-static must be routed HERE and
    loaded as dynamic — chunk-level caption, gen ratio 0.9. See the full routing
    policy in dataset/static.py StaticDataset's docstring (case 1b).
    """

    def __init__(self, *args, fps_targets=None, dynamic_gen_size=0, dynamic_mv_size=None,
                 short_paths=None, **kwargs):
        # Must set fps_targets + dynamic_gen_size + dynamic_mv_size + short_paths
        # BEFORE super().__init__ so StaticDataset's is_sample_viable() filter
        # sees them when computing the per-row viability threshold.
        self.fps_targets = fps_targets  # explicit fps target list, e.g. [16]; None = default graded fallback
        self.dynamic_gen_size = dynamic_gen_size  # gen_size used in dynamic path (falling back from static); 0 = use self.gen_size
        self.dynamic_mv_size = dynamic_mv_size    # mv_size used in dynamic path (falling back from static); None = use self.mv_size
        # short_paths: list of (mv, gen_size) tuples, tried as a graded
        # rescue chain when the main multi-video path can't pick `mv_size`
        # videos at `dynamic_gen_size`. e.g. [[4, 20], [10, 10]] tries
        # mv=4 gen=20 first (longer per-video), then mv=10 gen=10 (shortest).
        # Each entry must satisfy gen_size >= 2*chunk_size to keep teacher
        # forcing's context_size > 0; gen_size == chunk_size degenerates TF
        # to "no context" which mismatches inference.
        self.short_paths = [tuple(p) for p in (short_paths or [])]
        # Pure DynamicDataset has only one path, so dynamic_gen_size (if set)
        # IS the gen_size. Inject into kwargs BEFORE super().__init__ so
        # StaticDataset's is_sample_viable filter sees the runtime gen_size —
        # otherwise the filter uses the smaller of (top-level gen_size,
        # dynamic_gen_size) and lets through rows too short for the actual
        # runtime gen_size, causing "exhausted all paths" spam at runtime.
        # StaticDynamicDataset has two paths and keeps self.gen_size for the
        # static path, so it is excluded by the type() check.
        #
        # effective_samples is a cross-dataset WEIGHTING unit and must use the
        # SAME gen_size for every dataset (static/mvgame/multiview never swap it).
        # Capture the config gen_size HERE, before the swap below, so a pure
        # DynamicDataset is weighted on the same tfs unit as the rest — otherwise
        # its longer dynamic clip (dynamic_gen_size, e.g. 75 vs gen_size 20)
        # inflates tfs ~3.9x and silently under-weights all dynamic data ~2.2x.
        self.eff_gen_size = kwargs.get('gen_size')
        if type(self) is DynamicDataset and dynamic_gen_size:
            kwargs['gen_size'] = dynamic_gen_size
        # Same for mv: pure DynamicDataset's single path uses dynamic_mv_size AS
        # the mv_size (e.g. mv=1 long single-view main at dynamic_gen_size, then
        # laddering down through short_paths as the anchor gets shorter). Falls
        # back to the config mv_size when dynamic_mv_size is unset. StaticDynamic-
        # Dataset is excluded (type check): its static path keeps mv_size, and its
        # dynamic fallback already applies dynamic_mv_size in handle_static_exhausted.
        if type(self) is DynamicDataset and dynamic_mv_size is not None:
            kwargs['mv_size'] = dynamic_mv_size
        super().__init__(*args, **kwargs)
        # Path-keyed video reader cache (since each getitem accesses arbitrary rows,
        # not the per-shard fixed set that StaticDataset assumes).
        self.dyn_video_readers = {}

        if is_node_main():
            log(f'DynamicDataset: {green(len(self.metadata))} videos, mv_size={self.mv_size}, '
                f'each sample picks {self.mv_size} independent videos')

    # NOTE: we intentionally do NOT override init_loader anymore.
    #
    # Earlier an override existed to skip the parent's bulk pose preload because
    # the old `load_poses` used `pf.read_row_groups([all_target_rgs])` which
    # combined many row groups into a single pyarrow ListArray whose int32
    # offsets overflowed for large datasets (span/spand ~600k rows). That fix
    # broke performance: each `load_one_video_view` call lazy-loaded one
    # pose_idx at a time, and each lazy load read an entire row group from disk
    # but only cached the single requested row — so repeated accesses to the
    # same row group re-read it every time. On dynamic datasets (which pick
    # random rows per getitem) this resulted in O(mv × n_row_groups) disk reads
    # per training step instead of the O(1) amortized cache hit rate the old
    # bulk-preload path had.
    #
    # `load_poses` has since been rewritten to iterate row groups one at a time
    # (`pf.read_row_group(rg_idx)`), so each ListArray is bounded to a single
    # row group's worth of elements and the int32 overflow cannot happen. The
    # parent's bulk preload is therefore safe again, and faster for dynamic
    # because it caches the shard's entire pose set at worker init.

    @property
    def effective_samples(self):
        """sum(num_frames / src_fps) * model_fps / tfs / 8. Each video = 1 view.
        Returns 0 when empty after prefilter so the aggregator drops it."""
        if not self.metadata:
            return 0
        # Use the config (pre-swap) gen_size as the common weighting unit, NOT
        # self.gen_size: for a pure DynamicDataset self.gen_size was swapped to
        # dynamic_gen_size in __init__, which would penalize it ~2.2x vs the
        # static/mv datasets that keep gen_size. eff_gen_size is set in __init__.
        tfs = (self.eff_gen_size or self.gen_size) * self.vae_stride_t - 3
        nf = np.array([m.get('num_frames', 0) or 0 for m in self.metadata], dtype=np.float64)
        fps = np.array([m.get('fps', self.model_fps) or self.model_fps for m in self.metadata], dtype=np.float64)
        total = np.sum(nf / fps) * self.model_fps
        return max(1, int(total / tfs / 8))

    def get_video_reader_for_path(self, abs_path: str) -> TorchCodecVideoReader:
        """Path-keyed lazy video reader cache."""
        vr = self.dyn_video_readers.get(abs_path)
        if vr is None:
            if isfile(abs_path):
                video_file = abs_path
            elif isfile(abs_path + '.mp4'):
                video_file = abs_path + '.mp4'
            elif isdir(abs_path):
                video_files = sorted([f for f in os.listdir(abs_path) if f.endswith(('.mp4', '.avi', '.mov'))])
                if not video_files:
                    raise FileNotFoundError(f'No video file found in {abs_path}')
                video_file = join(abs_path, video_files[0])
            else:
                raise FileNotFoundError(f'Video path not found: {abs_path}')
            # Construct via the config-selected reader class inherited from
            # StaticDataset.__init__ (`video_reader` cfg → TorchCodecVideoReader
            # or CFRVideoReader), NOT the TorchCodecVideoReader imported above —
            # that import is only the nominal return type hint.
            vr = self.video_reader_cls(video_file)
            self.dyn_video_readers[abs_path] = vr
        return vr

    def resolve_video_path(self, video_path: str) -> str:
        if isabs(video_path):
            return video_path
        return join(self.data_root, video_path)

    def load_one_video_view(self, meta_row, target_h: int, target_w: int,
                              total_frame_size: int, fps_ratio: float, aug_kwargs: dict):
        """Load a single video as one view.

        Returns (view_data, prompt_embed, caption, resolved_video_path, seg_start).
        view_data is the dict from `load_view`. Each view normalizes pose to its
        OWN first frame by passing R0=None to load_view (no cross-view sharing).
        The last two return values (resolved path + source seg_start) are what
        the sidecar needs to reproduce this view later.
        """
        # Resolve path & get reader
        video_path = self.resolve_video_path(meta_row['video_path'])
        vr = self.get_video_reader_for_path(video_path)

        # Lazy-load camera params for this row's pose_idx
        pose_idx = meta_row['pose_idx']
        if pose_idx not in self.camera_params:
            self.load_poses([pose_idx])
        # camera_params stores raw flat array; parse + normalize on access
        pose_flat = self.camera_params[pose_idx]
        cameras = parse_pose_column(pose_flat)
        if self.pose_norm_target > 0:
            normalize_cam_translation(cameras, self.pose_norm_target)

        # Apply optional [frame_start, frame_end) window from parquet metadata
        # (mvgame convention, see StaticDataset.__init__). Sampling is in
        # window-space [0, n_frames_src); frame_start is added back when
        # building seg_frame_inds so vr.get_batch / video_to_cam_indices
        # receive original source frame indices.
        #
        # Pose-video length mismatch — mirrors StaticDataset.getitem_impl
        # logic at static.py:787. Windowed-pose datasets (EPIC-Fields:
        # n_cam == num_frames per segment, frame_start in absolute mp4 frame
        # space) need n_vid as the source range, NOT min(n_vid, n_cam) —
        # else min collapses to the per-segment slice and the absolute
        # frame_start overshoots, giving n_frames_src=0 for every row.
        n_vid = len(vr)
        n_cam = len(cameras)
        if abs(n_cam - n_vid) <= 1:
            full_src = min(n_vid, n_cam)  # aligned datasets, preserves old behavior
        else:
            full_src = n_vid              # windowed-pose or subsampled-pose: use video length
        frame_start = int(meta_row.get('frame_start') or 0)
        frame_end = meta_row.get('frame_end')
        frame_end_eff = full_src if (frame_end is None or frame_end <= 0) else min(int(frame_end), full_src)
        n_frames_src = max(0, frame_end_eff - frame_start)
        src_frame_size = int(round(total_frame_size * fps_ratio))
        if n_frames_src < src_frame_size:
            return None  # too short

        # Pick a random temporal window of src_frame_size source frames, then
        # fps-subsample it down to total_frame_size output frames.
        max_start = n_frames_src - src_frame_size
        seg_start = random.randint(0, max_start) if max_start > 0 else 0
        # Output frame i maps to source offset round(i * fps_ratio) within the
        # window. fps_ratio == 1.0 → identity (no subsample); fps_ratio > 1.0 →
        # stride > 1 (e.g. 60fps source → 15fps model, ratio 4, every 4th frame).
        # The np.minimum(..., src_frame_size-1) is a defensive clamp that keeps
        # every offset inside the chosen window [0, src_frame_size-1]. (With
        # fps_ratio >= 1.0 — guaranteed by max(1.0, ...) at the tier source —
        # round((total_frame_size-1)*fps_ratio) already stays <= src_frame_size-1,
        # so the clamp is a safety bound rather than a fix for an observed
        # off-by-one.)
        model_idx = np.arange(total_frame_size)
        src_offsets = np.minimum(
            np.round(model_idx * fps_ratio).astype(np.int64),
            src_frame_size - 1,
        )
        # Add window base (seg_start) and parquet window offset (frame_start) to
        # land in absolute source-mp4 frame space for vr.get_batch.
        seg_frame_inds = (seg_start + frame_start) + src_offsets

        # Loose exposure clamp, per video: dynamic views are INDEPENDENT clips
        # (no cross-view appearance to keep consistent), so each computes its own
        # gamma. No-op outside the extreme dark/overexposed tails.
        gamma_value = 1.0
        if self.exposure_clamp:
            win_inds = np.linspace(frame_start, frame_start + n_frames_src - 1,
                                   num=min(8, n_frames_src)).astype(np.int64)
            gamma_value = compute_sequence_gamma([vr], win_inds, 1, 1, band=LOOSE_EXPOSURE_BAND)

        view_data = self.load_view(
            vr, cameras, seg_frame_inds,
            target_h, target_w,
            R0=None, T0=None,
            frame_start=frame_start,
            n_frames_src=n_frames_src,
            gamma_value=gamma_value,
            **aug_kwargs,
        )

        # Load prompt embedding for THIS row
        # H3 re-encodes the raw caption with Qwen3; Wan/T5 caches are incompatible.
        embed = torch.empty((0, 5120), dtype=torch.bfloat16)


        # Return seg_start in original source frame space (window-offset added)
        # so callers (batch['cpu']['start_frames']) get a frame index that points
        # at the actual mp4, not a window-local offset.
        # Also return the parquet row (pose_idx) and the last source frame index
        # (seg_frame_inds is monotonic, so [-1] is the max) so callers can surface
        # (row, start, end) on the vis meta panel.
        return (view_data, embed, meta_row.get('caption', ''), video_path,
                int(seg_start + frame_start), int(meta_row['pose_idx']),
                int(seg_frame_inds[-1]))

    def try_long_single_view(self, anchor_meta: dict, aug_kwargs: dict):
        """Single-video, mv=1, gen_size=long_gen_size sample.

        Tries graded fps tiers (primary remap → native) on the given anchor.
        Returns a fully built batch on success, or None if neither tier loads.

        The result is NOT marked view_isolated: with mv=1 there is exactly one
        view, so isolation semantics collapse to the standard single-view path
        (one prompt, one cond_image, no per-view text expansion).
        """
        long_lat = self.long_gen_size
        long_tfs = long_lat * self.vae_stride_t - 3

        # mv=1 pack: [1,1] layout, full-resolution single view.
        pack = pack_factory[1]
        pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}
        target_h = int(self.height * pack['rs'][0])
        target_w = int(self.width * pack['rs'][0])

        src_fps = float(anchor_meta.get('fps', self.model_fps))
        # Factory-snapped primary tier; native fps fallback only allowed in
        # graded mode (self.fps_targets is None — used by StaticDynamicDataset's
        # dynamic fallback). Strict mode (fps_targets=[16] for pure
        # DynamicDataset) skips native to preserve the model_fps contract;
        # rows that don't fit primary return None and the caller falls through
        # to short_paths.
        eff_model_fps, eff_src_fps = resolve_fps_remap(self.model_fps, src_fps)
        primary_ratio = max(1.0, float(eff_src_fps) / float(eff_model_fps)) if eff_src_fps else 1.0
        tiers = [(primary_ratio, eff_model_fps)]
        if primary_ratio > 1.0 and self.fps_targets is None:
            tiers.append((1.0, int(round(src_fps))))

        result = None
        fps_ratio = None
        level_eff_model = self.model_fps
        for ratio, level_em in tiers:
            try:
                result = self.load_one_video_view(
                    anchor_meta, target_h, target_w,
                    long_tfs, ratio, aug_kwargs,
                )
            except Exception:
                result = None
                continue
            if result is not None:
                fps_ratio = ratio
                level_eff_model = level_em
                break
        if result is None:
            return None

        view_data, embed, caption, video_path, seg_start, row_idx, seg_end = result
        effective_fps = int(round(level_eff_model)) if fps_ratio > 1.0 else int(round(src_fps))

        batch = {'cpu': {}}
        batch['mv'] = 1
        batch['cpu']['prompts'] = caption
        # Single view: scalar `video_path` is canonical; build_meta_dicts and the
        # error logs fall back to it, so no redundant (and default_collate-
        # transposed) `video_paths` list here. The view_as_batch path keeps
        # `video_paths` because there it is genuinely per-view.
        batch['cpu']['video_path'] = video_path
        batch['cpu']['dataset_name'] = self.dataset_name
        batch['cpu']['parquet'] = basename(self.data_path)
        batch['cpu']['start_frames'] = np.asarray([seg_start], dtype=np.int64)
        batch['cpu']['end_frames'] = np.asarray([seg_end], dtype=np.int64)
        batch['cpu']['rows'] = np.asarray([row_idx], dtype=np.int64)
        batch['cpu']['fps_ratios'] = np.asarray([fps_ratio], dtype=np.float32)
        batch['cpu']['aug_kwargs'] = dict(aug_kwargs)
        batch['fps'] = effective_fps
        batch['prompt_embeds'] = embed  # [L, D] — static-style single-view shape

        # NOT setting view_isolated: mv=1 collapses to the standard path.
        batch['frames'] = view_data['frames']  # [F, 3, H, W]
        # mv=1 stack-and-reshape is a no-op shape-wise; keep the form for
        # symmetry with multi-view batches.
        batch['projs'] = torch.stack([view_data['projs']], dim=1).reshape(-1, 4, 4)
        batch['projs_inv'] = torch.stack([view_data['projs_inv']], dim=1).reshape(-1, 4, 4)
        batch['Ks'] = torch.stack([view_data['Ks']], dim=1).reshape(-1, 3, 3)
        batch['Rs'] = torch.stack([view_data['Rs']], dim=1).reshape(-1, 3, 3)
        batch['Ts'] = torch.stack([view_data['Ts']], dim=1).reshape(-1, 3, 1)

        # Pose stable factor — single-view: size by MAX |T| (farthest point on the
        # camera trajectory from the anchor), not mean drift; the far point is what
        # overflows bf16 PRoPE (score ≈ 2.4 * T², need T < ~5).
        R_v, T_v = view_data['Rs'], view_data['Ts']
        centers_v = -torch.bmm(R_v.mT, T_v).squeeze(-1)
        pose_stable_factor, pose_max_t = select_pose_stable_factor(centers_v, self.pose_stable_factors)
        if pose_stable_factor != 1.0:
            batch['projs'][:, :3, 3] /= pose_stable_factor
            batch['projs_inv'][:, :3, 3] /= pose_stable_factor
            batch['Ts'] /= pose_stable_factor

        batch['cpu']['pack'] = pack
        batch['cpu']['pack']['width'] = self.width
        batch['cpu']['pack']['height'] = self.height
        batch['cpu']['pose_stable_factor'] = pose_stable_factor
        batch['cpu']['pose_max_t'] = pose_max_t  # pre-division max pairwise dist (bf16 diagnostic)
        return batch

    def try_view_iso(self, target_mv: int, total_latent_size: int, aug_kwargs: dict,
                      anchor_meta: dict = None, force_strip: bool = False):
        """View-isolated multi-video sampling helper.

        Picks `target_mv` rows and loads each as one independent view at
        `total_latent_size` latents. If `anchor_meta` is given it occupies
        view 0 (failure → return None so the caller can drop to a shorter
        path); the remaining `target_mv - 1` slots are random-sampled from
        the same dataset. Returns a fully built batch on success, or None
        if it can't fill all views within the attempt budget.

        `force_strip`: bypass pack_factory's manual mixed-resolution layouts
        and use a uniform strip pack (full-res per view). Set by callers that
        intentionally pick a `target_mv` outside the main config (shape_pool,
        short_paths) so the picked shape isn't reshaped by an unrelated
        mixed-res layout.
        """
        n_meta = len(self.sharded_metadata)
        if n_meta == 0 or target_mv not in pack_factory:
            return None

        total_frame_size = total_latent_size * self.vae_stride_t - 3

        # Pack layout for target_mv. mv=2/5/8 share pack_size=[1,2]; mv=4
        # uses [1,4]; mv=10 uses [1,10]. gather_mixed_batch tolerates per-rank
        # shape differences, so co-training with mixed mv values is OK — each
        # rank just produces a different-shape sample, broadcast individually.
        pack = make_strip_pack(target_mv) if (self.shape_pool_active or force_strip) else pack_factory[target_mv]
        pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}
        pack_size = pack['pack_size']
        ratios = pack['rs']
        xs = pack['xs']
        ys = pack['ys']

        height_pack = int(self.height * pack_size[0])
        width_pack = int(self.width * pack_size[1])
        n_frames_out = total_frame_size

        # view_as_batch: when every view has the same full-resolution size
        # (rs all 1.0 — the case for every strip layout: mv=1/2/4/6/10/12/...),
        # we can return per-view tensors stacked on a leading dim instead of
        # packing them spatially. Downstream this turns mv into a batch
        # dimension, which keeps the attention block_mask at single-view
        # size (F*h*w)² instead of (F*h*w*mv)² — critical for high mv at
        # short gen, where the dense N×N mask intermediate would OOM.
        view_as_batch = bool(np.allclose(ratios, 1.0))
        if view_as_batch:
            # Collect per-view frames directly (no packed canvas).
            frames_list: List[torch.Tensor] = []
        else:
            frames = torch.zeros((n_frames_out, 3, height_pack, width_pack), dtype=torch.float32)
        projs_list = []
        projs_inv_list = []
        Ks_list, Rs_list, Ts_list = [], [], []
        embeds_list = []
        captions_list = []
        effective_fps_list = []
        video_paths_list: List[str] = []
        start_frames_list: List[int] = []
        end_frames_list: List[int] = []
        rows_list: List[int] = []
        fps_ratios_list: List[float] = []

        # Shuffled pool ensures distinct rows while available; once exhausted
        # (n_meta < target_mv or many rejects), fall back to with-replacement.
        # When anchor_meta is provided, it owns slot 0 unconditionally; the
        # pool fills slots 1..target_mv-1.
        #
        # Replay mode (`_replay_force_row_order=True`, set by load_batch_via_seed):
        # sharded_metadata has been pinned to the exact mv rows in view order.
        # Pool = indices 1..mv-1 (anchor consumes idx 0 via is_anchor branch),
        # reversed so pop() yields 1, 2, …, mv-1 — byte-identical fill order.
        # No random fallback so a missing row hard-fails instead of silently
        # substituting.
        force_order = bool(getattr(self, '_replay_force_row_order', False))
        if force_order:
            row_pool = list(range(1, n_meta))[::-1]
        else:
            row_pool = list(range(n_meta))
            random.shuffle(row_pool)
        picked = 0
        attempts = 0
        # Budget = 20 row tries per requested view. Each attempt either fills a
        # slot or rejects a row (too short / load error), so this tolerates a
        # ~95% reject rate before giving up and returning None to the caller
        # (which then drops to a shorter path). Large enough to ride out a
        # cluster of short/broken rows; bounded so a fully-broken shard fails
        # fast instead of spinning.
        max_attempts = target_mv * 20
        view_idx = 0
        while picked < target_mv and attempts < max_attempts:
            attempts += 1
            is_anchor = anchor_meta is not None and picked == 0
            if is_anchor:
                meta_row = anchor_meta
            elif row_pool:
                row_idx = row_pool.pop()
                meta_row = self.sharded_metadata[row_idx]
            else:
                if force_order:
                    raise RuntimeError(
                        'try_view_iso replay (force_order=True) exhausted its '
                        'pre-ordered row pool — a pinned row failed to load. '
                        'Replay assumes the original training succeeded with '
                        'these exact rows; a load failure here means the data '
                        'or config has drifted since the original step.'
                    )
                row_idx = random.randrange(n_meta)
                meta_row = self.sharded_metadata[row_idx]

            src_fps = float(meta_row.get('fps', self.model_fps))
            # Factory-snapped primary tier; native fps fallback only allowed in
            # graded mode (fps_targets is None — used by StaticDynamicDataset's
            # dynamic fallback). Strict mode (fps_targets=[16] for pure
            # DynamicDataset) skips native to preserve the model_fps contract.
            eff_model_fps, eff_src_fps = resolve_fps_remap(self.model_fps, src_fps)
            primary_ratio = max(1.0, float(eff_src_fps) / float(eff_model_fps)) if eff_src_fps else 1.0
            tiers = [(primary_ratio, eff_model_fps)]
            if primary_ratio > 1.0 and self.fps_targets is None:
                tiers.append((1.0, int(round(src_fps))))

            target_h = int(self.height * ratios[view_idx])
            target_w = int(self.width * ratios[view_idx])

            result = None
            fps_ratio = None
            level_eff_model = self.model_fps
            for ratio, level_em in tiers:
                try:
                    result = self.load_one_video_view(
                        meta_row, target_h, target_w,
                        total_frame_size, ratio, aug_kwargs,
                    )
                except Exception as e:
                    if attempts <= 3:
                        log(yellow(f"DynamicDataset: load_one_video_view failed for "
                                   f"{meta_row.get('video_path', '?')}: {type(e).__name__}: {e}"))
                    # CAVEAT (2026-05-28 audit): exception logged only for first 3
                    # attempts per anchor; subsequent failures silent. If 100% of
                    # pool fails (broken parquet, missing embeds, codec error), the
                    # caller in getitem_impl just logs "exhausted all paths" with
                    # no per-row failure trace. Consider summary log when
                    # picked < target_mv at end-of-loop.
                    result = None
                    continue
                if result is not None:
                    fps_ratio = ratio
                    level_eff_model = level_em
                    break

            if result is None:
                # Anchor failure is terminal: caller picks a shorter path
                # rather than papering over by retrying random rows.
                if is_anchor:
                    return None
                continue

            view_data, embed, caption, view_video_path, view_seg_start, view_row, view_seg_end = result

            if view_as_batch:
                # Each view kept as its own [F, 3, H, W] tensor; stacked at the
                # end so collate_fn can flatten leading [B, mv] → batch dim.
                frames_list.append(view_data['frames'])
            else:
                h_v, w_v = view_data['frames'].shape[-2:]
                px = int(xs[view_idx] * self.width)
                py = int(ys[view_idx] * self.height)
                frames[:, :, py:py + h_v, px:px + w_v] = view_data['frames']

            projs_list.append(view_data['projs'])
            projs_inv_list.append(view_data['projs_inv'])
            Ks_list.append(view_data['Ks'])
            Rs_list.append(view_data['Rs'])
            Ts_list.append(view_data['Ts'])
            embeds_list.append(embed)
            captions_list.append(caption)
            effective_fps_list.append(
                int(round(level_eff_model)) if fps_ratio > 1.0 else int(round(src_fps))
            )
            video_paths_list.append(view_video_path)
            start_frames_list.append(view_seg_start)
            end_frames_list.append(view_seg_end)
            rows_list.append(view_row)
            fps_ratios_list.append(float(fps_ratio))

            picked += 1
            view_idx += 1

        if picked < target_mv:
            return None

        batch = {'cpu': {}}
        batch['cpu']['video_path'] = video_paths_list[0]
        batch['cpu']['video_paths'] = list(video_paths_list)
        # Per-view parquet (same file for all views). Length-mv so view_as_batch
        # collate folds it sample-major to B*mv (mirrors video_paths) → shows on
        # every view panel. With video_paths, grep the basename to find the row.
        batch['cpu']['parquets'] = [basename(self.data_path)] * len(video_paths_list)
        batch['cpu']['start_frames'] = np.asarray(start_frames_list, dtype=np.int64)
        # Per-view source-frame end + parquet row (length-mv → view_as_batch
        # collate folds them sample-major to B*mv, one per view panel; in the
        # strip path each view is a DIFFERENT row, unlike static/multiview).
        batch['cpu']['end_frames'] = np.asarray(end_frames_list, dtype=np.int64)
        batch['cpu']['rows'] = np.asarray(rows_list, dtype=np.int64)
        batch['cpu']['fps_ratios'] = np.asarray(fps_ratios_list, dtype=np.float32)
        batch['cpu']['aug_kwargs'] = dict(aug_kwargs)
        batch['fps'] = int(round(sum(effective_fps_list) / len(effective_fps_list)))

        if view_as_batch:
            # Layout: leading dim = mv; collate_fn flattens [B, mv, ...] → [B*mv, ...]
            # so prepare_batch sees a B*mv batch of plain single-view samples.
            batch['mv'] = 1
            batch['cpu']['view_as_batch'] = True
            batch['cpu']['orig_mv'] = target_mv  # for logging only
            batch['cpu']['prompts'] = list(captions_list)            # one caption per view
            batch['frames'] = torch.stack(frames_list, dim=0)        # [mv, F, 3, H, W]
            batch['projs'] = torch.stack(projs_list, dim=0)          # [mv, F, 4, 4]
            batch['projs_inv'] = torch.stack(projs_inv_list, dim=0)  # [mv, F, 4, 4]
            batch['Ks'] = torch.stack(Ks_list, dim=0)                # [mv, F, 3, 3]
            batch['Rs'] = torch.stack(Rs_list, dim=0)                # [mv, F, 3, 3]
            batch['Ts'] = torch.stack(Ts_list, dim=0)                # [mv, F, 3, 1]
            batch['prompt_embeds'] = torch.stack(embeds_list, dim=0)  # [mv, L, D]
            # mv=1 single-view pack — every batch element packs as a single full view.
            mv1_pack = pack_factory[1]
            mv1_pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in mv1_pack.items()}
            mv1_pack['width'] = self.width
            mv1_pack['height'] = self.height
            mv1_pack['ws'] = np.asarray([self.width], dtype=np.int32)
            mv1_pack['hs'] = np.asarray([self.height], dtype=np.int32)
            mv1_pack['xs'] = (mv1_pack['xs'] * self.width).astype(np.int32)
            mv1_pack['ys'] = (mv1_pack['ys'] * self.height).astype(np.int32)
            batch['cpu']['pack'] = mv1_pack
        else:
            batch['mv'] = target_mv
            batch['cpu']['view_isolated'] = True
            batch['cpu']['prompts'] = captions_list[0] if captions_list else ''
            batch['frames'] = frames
            batch['projs'] = torch.stack(projs_list, dim=1).reshape(-1, 4, 4)
            batch['projs_inv'] = torch.stack(projs_inv_list, dim=1).reshape(-1, 4, 4)
            batch['Ks'] = torch.stack(Ks_list, dim=1).reshape(-1, 3, 3)
            batch['Rs'] = torch.stack(Rs_list, dim=1).reshape(-1, 3, 3)
            batch['Ts'] = torch.stack(Ts_list, dim=1).reshape(-1, 3, 1)
            batch['prompt_embeds'] = torch.stack(embeds_list, dim=0)
            batch['cpu']['pack'] = pack
            batch['cpu']['pack']['width'] = self.width
            batch['cpu']['pack']['height'] = self.height

        # View-isolated: one global factor must keep EVERY view safe, so size by the
        # global MAX |T| across all views (each view is anchored to its own first
        # frame), not the per-view mean averaged across views.
        all_centers = torch.cat([
            -torch.bmm(R_v.mT, T_v).squeeze(-1) for R_v, T_v in zip(Rs_list, Ts_list)
        ], dim=0)
        pose_stable_factor, pose_max_t = select_pose_stable_factor(all_centers, self.pose_stable_factors)
        if pose_stable_factor != 1.0:
            batch['projs'][..., :3, 3] /= pose_stable_factor
            batch['projs_inv'][..., :3, 3] /= pose_stable_factor
            batch['Ts'] /= pose_stable_factor

        batch['cpu']['pose_stable_factor'] = pose_stable_factor
        batch['cpu']['pose_max_t'] = pose_max_t  # pre-division max pairwise dist (bf16 diagnostic)

        return batch

    def getitem_impl(self, idx: int, **kwargs):
        self.init_loader()
        n_meta = len(self.sharded_metadata)
        if n_meta == 0:
            raise RuntimeError(
                f'{type(self).__name__} has no usable samples after length '
                f'prefilter — should never be picked by DatasetAggregator '
                f'(weight=0). Check data_path={self.data_path}'
            )

        while True:
            seed = idx // n_meta
            seed = seed % (2 ** 32 - 1)
            set_seed(seed)
            self.last_seed = int(seed)
            local_idx = idx % n_meta

            anchor_meta = self.sharded_metadata[local_idx]
            n_anchor = anchor_meta.get('num_frames', 0)
            if isinstance(n_anchor, list):
                n_anchor = min(n_anchor) if n_anchor else 0

            disable_aug = random.random() < self.disable_augmentation_ratio or self.overfit
            if disable_aug:
                aug_kwargs = dict(s_min=1.0, s_max=1.0, cx_min=0.0, cx_max=0.0,
                                  cy_min=0.0, cy_max=0.0, r_min=0.0, r_max=0.0)
            else:
                aug_kwargs = dict(s_min=0.65, s_max=1.25,
                                  cx_min=-0.1, cx_max=0.1,
                                  cy_min=-0.1, cy_max=0.1,
                                  r_min=-15.0, r_max=15.0)
            aug_kwargs.update(kwargs)

            long_tfs = self.long_gen_size * self.vae_stride_t - 3 if self.long_gen_size > 0 else 0
            main_tfs = self.gen_size * self.vae_stride_t - 3

            if self.long_gen_size > 0 and n_anchor >= long_tfs:
                long_batch = self.try_long_single_view(anchor_meta, aug_kwargs)
                if long_batch is not None:
                    return long_batch

            if n_anchor >= main_tfs:
                main_batch = self.try_view_iso(self.mv_size, self.gen_size, aug_kwargs,
                                                anchor_meta=anchor_meta)
                if main_batch is not None:
                    return main_batch

            for short_mv, short_gen in self.short_paths:
                short_tfs = short_gen * self.vae_stride_t - 3
                if n_anchor < short_tfs:
                    continue
                short_batch = self.try_view_iso(short_mv, short_gen, aug_kwargs,
                                                 anchor_meta=anchor_meta, force_strip=True)
                if short_batch is not None:
                    return short_batch

            log(yellow(
                f"DynamicDataset: anchor {anchor_meta.get('video_path', '?')} "
                f"exhausted all paths (nf={n_anchor}), advancing idx"
            ))
            # CAVEAT (2026-05-28 audit): `idx = local_idx + 1` re-enters the
            # while-loop where `seed = idx // n_meta`. For local_idx + 1 < n_meta
            # this collapses retry seed to 0 regardless of original idx, biasing
            # the distribution of retry samples. Not an NCCL hazard but
            # reproducibility-affecting under short-row pressure. Same pattern in
            # static.py:770 (getitem_impl recursive call) and multiview.py:618.
            idx = local_idx + 1

    def __getitem__(self, idx: int):
        # Pure dynamic-type datasets are NOT affected by shape_pool — the
        # view-iso path uses self.mv_size / self.gen_size from config (the
        # latter overridden by dynamic_gen_size in __init__), with short_paths
        # and long_gen rescues unchanged. We override StaticDataset.__getitem__
        # to skip maybe_pick_shape so self.gen_size stays at the config value.
        # StaticDynamicDataset (which DOES want shape_pool for its static
        # path) re-overrides to call StaticDataset.__getitem__ directly.
        try:
            self.shape_pool_active = False
            batch = self.getitem_impl(idx)
            batch['cpu']['seed'] = int(self.last_seed)
            return batch
        except Exception as e:
            wi = get_worker_info()
            log(red(
                f"[DynamicDataset __getitem__] failed: rank={get_rank()} "
                f"worker={wi.id if wi is not None else 0} idx={idx} err={e}"
            ))
            stacktrace()
            raise

    @staticmethod
    def collate_fn(samples):
        """Same view_as_batch flatten as the aggregator's collate (gated on
        per-sample flag). Provided here so single-`type: dynamic` configs
        (no aggregator wrap) still get the flatten."""
        from dataset.aggregator import view_as_batch_collate
        return view_as_batch_collate(samples)


class StaticDynamicDataset(DynamicDataset):
    """Static dataset with dynamic fallback for short videos.

    fps_targets is intentionally forced to None here: static_dynamic should use
    the graded fps fallback ([model_fps, 24, 30, 60, native]) on BOTH paths so
    short videos can still be loaded at a higher fps when 16fps would demand
    more frames than the clip has. A top-level `fps_targets: [16]` in the config
    still applies to pure `dynamic` entries (which should be strict 16fps), but
    static_dynamic opts out.

    Long enough videos: multi-view from single trajectory (StaticDataset path)
    at self.gen_size with self.mv_size views from one trajectory.
    Too short for requested views: pick mv independent videos (DynamicDataset
    path) at self.dynamic_gen_size (0 = fall back to self.gen_size) and
    self.dynamic_mv_size views (falls back to self.mv_size if unset).

    dynamic_mv_size is typically larger than mv_size to increase batch size on
    the dynamic path — since view-isolated attention is cheaper per view than
    the static path's full cross-view attention, packing more independent
    videos per step balances the compute budget between the two paths.
    """

    def __init__(self, *args, **kwargs):
        # Force graded fps fallback — ignore any top-level fps_targets.
        kwargs['fps_targets'] = None
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        # static_dynamic uses cross-view static path → shape_pool applies for
        # the static path. Re-route through StaticDataset.__getitem__ which
        # calls maybe_pick_shape (DynamicDataset.__getitem__ override skips it).
        return StaticDataset.__getitem__(self, idx)

    def handle_static_exhausted(self, idx, **kwargs):
        # Temporarily swap gen_size/mv_size so DynamicDataset.getitem_impl
        # uses the dynamic-path values instead of the static-path ones.
        # When shape_pool was active for the static path, self.mv_size has
        # been mutated to the picked-pool mv. Use self.init_mv_size (the
        # config-time value) as the default fallback mv so dynamic-path
        # behavior matches the no-shape_pool case.
        orig_gs = self.gen_size
        orig_mv = self.mv_size
        try:
            if self.dynamic_gen_size:
                self.gen_size = self.dynamic_gen_size
            if self.dynamic_mv_size is not None:
                self.mv_size = self.dynamic_mv_size
            else:
                self.mv_size = self.init_mv_size
            return DynamicDataset.getitem_impl(self, idx, **kwargs)
        finally:
            self.gen_size = orig_gs
            self.mv_size = orig_mv

    def getitem_impl(self, idx, **kwargs):
        return StaticDataset.getitem_impl(self, idx, **kwargs)
