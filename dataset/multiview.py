# Real multi-view dataset (EgoExo4D and similar).
# Loads multiple synchronized real-world camera streams per scene from a parquet
# metadata file. Conceptually multi-view (like MVGame) but storage-shaped like
# StaticDataset (parquet + TorchCodec + lazy pose loading).
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, get_worker_info

import os
import json
import math
import time
import torch
import random
import numpy as np
import pyarrow.parquet as pq

from utils.console import *
from utils.math_utils import affine_padding
from utils.math_utils import affine_inverse
from utils.math_utils import ixt_inverse
from utils.math_utils import ixt_padding
from utils.distributed import get_rank
from utils.distributed import get_world_size
from utils.distributed import is_main_process
from utils.distributed import is_node_main
import utils.video as video_utils
from utils.video import TorchCodecVideoReader
from utils.misc import set_seed
from utils.parallel import parallel_execution
from dataset.mvgame import video_augmentation
from dataset.mvgame import image_augmentation
from dataset.mvgame import gamma_correct
from dataset.mvgame import compute_sequence_gamma
from dataset.mvgame import LOOSE_EXPOSURE_BAND
from dataset.mvgame import normalize_ixt
from dataset.mvgame import normalize_cam_translation
from dataset.mvgame import select_pose_stable_factor
from dataset.mvgame import pack_factory
from dataset.mvgame import make_strip_pack
from dataset.static import parse_pose_column
from dataset.fps_remap import resolve_fps_remap


def parse_pose_column_egoexo4d(pose_flat: np.ndarray, num_frames_per_cam: List[int]):
    """Parse the EgoExo4D-style flat pose array.

    pose_flat is shape (n_cams * max_frames * 10,), zero-padded for cameras
    with fewer frames. Reshape to (n_cams, max_frames, 10), slice each
    camera's valid range, and reuse parse_pose_column from static.py to
    convert each camera's [N*10] flat slice into per-frame {K,R,T} dicts.

    max_frames is inferred from pose_flat.size rather than max(num_frames),
    because the pose column occasionally has slightly fewer entries than
    what num_frames reports (off by a few frames due to sync/rounding).

    Returns: list[n_cams] of list[N_cam_i] of {K, R, T} dicts.
    """
    n_cams = len(num_frames_per_cam)
    # Infer max_frames from actual pose size instead of trusting num_frames
    max_nf = pose_flat.size // (n_cams * 10)
    pose_5d = pose_flat[:n_cams * max_nf * 10].reshape(n_cams, max_nf, 10)

    cams_per_view = []
    for cam_idx in range(n_cams):
        # Clip to whichever is smaller: reported num_frames or actual pose rows
        nf = min(num_frames_per_cam[cam_idx], max_nf)
        cam_flat = pose_5d[cam_idx, :nf, :].reshape(nf * 10)
        cams_per_view.append(parse_pose_column(cam_flat))
    return cams_per_view


class MultiViewRealDataset(Dataset):
    """Real multi-view dataset for synchronized multi-camera scenes.

    Supports EgoExo4D (5 cams), Waymo Perception (5 cams), Waymo E2E (8 cams),
    and similar datasets where each scene directory contains 0.mp4 … N-1.mp4.

    Each scene = mv_size synchronized real cameras of (possibly) different
    resolutions and per-camera frame counts. A training sample picks one
    temporal window and packs all cameras as mv-views. Camera-to-pack-view
    assignment is randomized per sample so the model sees each physical camera
    in different pack positions over time.

    Outputs the same batch schema as MVGame / StaticDataset for drop-in
    compatibility with gather_mixed_batch co-training.
    """

    def __init__(self,
                 data_path: str,
                 data_root: str = None,  # base dir for relative video_path; defaults to parquet's parent dir
                 gen_size: int = 60,
                 height: int = 448,
                 width: int = 832,

                 mv_size: int = 5,  # number of views to produce per sample
                 num_cameras: int = None,  # physical cameras in data (defaults to mv_size; set higher to randomly subsample)
                 dataset_fps: int = 30,
                 model_fps: int = 24,
                 per_worker_threads: int = 4,

                 # Sequence sampling
                 seq_sample: List[int] = (0, None, 1),

                 # Overfitting
                 overfit: bool = False,
                 overfit_seq_sample: List[int] = (0, 1, 1),

                 config=dotdict({'sp_size': 1, 'model': {'vae_stride': [4, 8, 8]}}),
                 pose_norm_target: float = 1.0,
                 pose_stable_factors=1.0,

                 # Augmentation: disabled by default for real data
                 disable_augmentation_ratio: float = 1.0,
                 image_aug: bool = False,         # mvgame-style image aug (gblur/color jitter/noise); gated by disable_aug
                 gamma_correction: bool = False,  # mvgame-style aggressive lift-to-0.25 (mvgame_raw); applied regardless of disable_aug
                 exposure_clamp: bool = False,    # loose two-sided exposure clamp for real data (only extreme tails fire); see compute_sequence_gamma + LOOSE_EXPOSURE_BAND
                 max_fov_h_deg: float = None,     # cap output h-FoV by per-frame s_min floor (None=off)
                 sp_sharding: bool = False,

                 sampling_weight_power: float = 0.8,  # default MUST match aggregator's getattr fallback (0.8); the attr is always set here so the fallback never fires

                 # Per-camera video filename template relative to video_path.
                 # Default matches EgoExo4D/Waymo layout ('<scene>/0.mp4' ... '<scene>/N-1.mp4').
                 # Override for mvgame-raw: 'video/{c:06d}.mp4'.
                 video_path_template: str = '{c}.mp4',

                 # If True, resize frames so they just cover the target (max-ratio).
                 # Default True preserves egoexo4d/waymo behavior. Set False to
                 # match mvgame.py semantics (no pre-resize; video_augmentation
                 # sees native-resolution frames and has full scale headroom).
                 pre_resize: bool = True,

                 # Per-sample shape pool (mirrors StaticDataset.shape_pool).
                 # Each entry's mv is auto-capped to NUM_CAMERAS. None = disabled.
                 shape_pool=None,
                 shape_pool_weights=None,

                 # Video reader class name (looked up in utils.video).
                 # 'TorchCodecVideoReader' (default): handles VFR/B-frames but
                 # full file scan in __init__ — slow for multi-GB videos on NFS.
                 # 'CFRVideoReader': av.open + moov keyframe_pts, sub-second init
                 # at any file size; CFR only. Use for epic_fields, nymeria.
                 video_reader: str = 'TorchCodecVideoReader',

                 *args, **kwargs):
        self.sampling_weight_power = sampling_weight_power
        self.video_path_template = video_path_template
        self.video_reader_cls = getattr(video_utils, video_reader)
        self.pre_resize = pre_resize
        # Threaded per-view decode switch (default ON). Distinct per-cam readers are
        # decoded across <=8 threads (parallel_execution) — cuts serial 8-view 720p
        # decode from ~17s to ~3s, and ~5x more at 20 views. The earlier "regression"
        # that motivated turning this off was node-confounded AND driven by the 53s
        # CPU image_augmentation hogging cores; with that fused down to ~4s the decode
        # threads have headroom. Set false to restore the serial baseline. NOTE:
        # aug-mvgame (MultiViewDataset) has its own long-standing parallel decode,
        # untouched by this flag.
        self.parallel_view_decode = bool(kwargs.get('parallel_view_decode', True))
        self.config = config
        self.data_path = data_path
        self.data_root = data_root or dirname(data_path)
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
        self.per_worker_threads = per_worker_threads
        self.NUM_CAMERAS = num_cameras if num_cameras is not None else mv_size
        # If physical cameras are fewer than the requested mv_size, cap mv_size
        # to NUM_CAMERAS — the view loop in getitem_impl iterates `zip(view_to_cam,
        # ratios, ...)` which has length min(NUM_CAMERAS, mv_size). Without this
        # cap, batch['mv']=mv_size but batch['Rs'] is sized for NUM_CAMERAS views,
        # so the trainer's padding (mv_size-sized) produces a mis-shaped tensor
        # (e.g. mv=8 + NUM_CAMERAS=5 + F=157 → 785+24=809 entries, not 1280).
        # This worked at mv=4 only because min(5,4)=4 matched mv_size; surfaces
        # at mv=8 when NUM_CAMERAS<mv_size.
        # Pack layout side: pack_factory[5] and pack_factory[8] both use
        # pack_size=[1.0, 2.0] (same packed image shape) so co-training stays
        # consistent at the gather-batch level.
        self.mv_size = min(mv_size, self.NUM_CAMERAS)
        self.disable_augmentation_ratio = disable_augmentation_ratio
        self.image_aug = image_aug
        self.gamma_correction = gamma_correction
        self.exposure_clamp = exposure_clamp
        self.max_fov_h_deg = max_fov_h_deg
        self.dataset_fps = dataset_fps
        self.model_fps = model_fps
        # Snap (model_fps, dataset_fps) via the fps_remap factory so the per-step
        # frame stride round(i*ratio) lands on a clean p/q (q≤2) pattern instead
        # of chaotic period-q patterns. e.g. (16, 30) → (15, 30) → ratio 2.0.
        # The model receives `effective_model_fps` in `batch['fps']` (may differ
        # from the outer self.model_fps for some sources, e.g. 15 for source 30/60).
        if dataset_fps is not None:
            eff_m, eff_s = resolve_fps_remap(model_fps, dataset_fps)
            self.effective_model_fps = eff_m
            self.fps_ratio = float(eff_s) / float(eff_m)
        else:
            self.effective_model_fps = model_fps
            self.fps_ratio = 1.0
        if self.fps_ratio < 1:
            self.fps_ratio = 1  # don't upsample, mirrors mvgame.py:723-725

        mv = self.mv_size
        assert mv in pack_factory, f"Supported mv sizes: {list(pack_factory.keys())}, got {mv}"
        pack = pack_factory[mv]
        assert (self.height * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        assert (self.width * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        self.pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}

        # shape_pool: see StaticDataset.shape_pool. Per __getitem__ pick one
        # (mv, gen); when the picked mv exceeds this dataset's NUM_CAMERAS,
        # maybe_pick_shape re-picks from the subset with mv ≤ NUM_CAMERAS so
        # the dropped entry's share is redistributed across other valid shapes
        # (NOT silently collapsed to (NUM_CAMERAS, gen), which created an
        # off-pool shape and biased sampling).
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

        self.overfit = overfit
        self.overfit_seq_sample = overfit_seq_sample

        if is_node_main():
            log(f'Creating MultiViewRealDataset from {blue(data_path)}, gen_size={gen_size}, '
                f'mv_size={mv_size}, dataset_fps={dataset_fps}, model_fps={model_fps}, '
                f'height={height}, width={width}')

        # Load parquet — only read SMALL columns at init (skip the huge `pose` column).
        self.pf = pq.ParquetFile(data_path)
        schema = self.pf.schema_arrow
        small_cols = [c for c in schema.names if c != 'pose']
        small_table = self.pf.read(columns=small_cols)

        # Schema metadata: accept both plural (correct) and singular (typo in
        # egoexo4d parquet), with [512, 4096] fallback for any prompt_embeds column.
        schema_meta = schema.metadata or {}
        if b'prompt_embeds_shape' in schema_meta:
            self.prompt_embeds_shape = json.loads(schema_meta[b'prompt_embeds_shape'])
        elif b'prompt_embed_shape' in schema_meta:
            self.prompt_embeds_shape = json.loads(schema_meta[b'prompt_embed_shape'])
        elif 'prompt_embeds' in schema.names:
            self.prompt_embeds_shape = [512, 4096]

        full_metadata = small_table.to_pylist()

        # Tag each meta with its original parquet row index for pose lookup
        for i, m in enumerate(full_metadata):
            m['pose_idx'] = i

        n_total = len(full_metadata)

        if overfit:
            self.metadata = full_metadata[slice(*overfit_seq_sample)]
        else:
            self.metadata = full_metadata[slice(*seq_sample)]

        # Pose cache keyed by pose_idx.
        self.camera_params = {}

        # Visibility into the per-dataset row count BEFORE sharding. Real-data
        # datasets (egoexo4d, waymo) don't currently apply a length prefilter
        # — but if the parquet ends up empty (or below distributed dispatch
        # threshold) we still want a loud warning so the user knows which
        # entry needs attention rather than a cryptic ZeroDivisionError later.
        threshold = max(1, get_world_size() * int(kwargs.get('num_workers', 1) or 1))
        cls_name = type(self).__name__
        self.is_empty = (len(self.metadata) == 0)
        if is_node_main():
            tag = green('OK') if len(self.metadata) >= threshold else red('TOO FEW')
            log(f'MultiViewRealDataset init: {green(len(self.metadata))} scenes '
                f'from {blue(data_path)} (parquet rows={n_total}, '
                f'threshold={threshold}) [{tag}]')
            if len(self.metadata) == 0:
                log(red(
                    f'[{cls_name} EMPTY] {data_path}: 0 rows after slice — '
                    f'this dataset will be DROPPED from co-training (weight=0). '
                    f'Check the parquet contents and seq_sample/overfit_seq_sample.'
                ))
            elif len(self.metadata) < threshold:
                log(red(
                    f'[{cls_name} BELOW THRESHOLD] {data_path}: '
                    f'{len(self.metadata)} rows < num_workers*world_size = {threshold}. '
                    f'Some workers will reuse rows; not fatal but may distort sampling.'
                ))


    def load_poses(self, pose_indices):
        """Read and parse poses for specific parquet row indices.

        Reads BOTH `pose` and `num_frames` columns since EgoExo4D pose is
        layout-dependent on per-camera num_frames. Cross-camera scale
        normalization is applied via a single flattened list to preserve
        relative scene geometry — per-camera independent normalization would
        give each camera its own scale and break multi-view consistency.
        """
        needed = [i for i in pose_indices if i not in self.camera_params]
        if not needed:
            return

        t0 = time.time()
        needed_set = set(needed)

        # pyarrow ParquetFile is NOT fork-safe — re-open in workers.
        wi = get_worker_info()
        pf = pq.ParquetFile(self.data_path) if wi is not None else self.pf
        n_rg = pf.metadata.num_row_groups

        rg_offsets = []
        offset = 0
        for i in range(n_rg):
            n = pf.metadata.row_group(i).num_rows
            rg_offsets.append((offset, offset + n))
            offset += n

        target_rgs = []
        for rg_idx, (rg_start, rg_end) in enumerate(rg_offsets):
            if any(rg_start <= idx < rg_end for idx in needed_set):
                target_rgs.append(rg_idx)

        # Iterate row groups in batches to avoid pyarrow int32 offset overflow
        # on the list<pose> column. Mirrors dataset/static.py:load_poses — when
        # a single row group holds more than ~17k scenes (n_cams*frames*10
        # per row), read_row_group overflows the list offsets buffer.
        POSE_BATCH_SIZE = 2000

        # Read the target row groups CONCURRENTLY (reusing utils.parallel).
        # A strided shard scatters the needed rows across ~all row groups, and each
        # (multi-cam) pose chunk must be fully decompressed to extract its 1-3 needed
        # rows, so the cost is dominated by decompress + network I/O — both of which
        # pyarrow runs with the GIL released, so threads give near-linear speedup
        # (e.g. mvgame_raw's 25-cam ~10GB pose column: ~240s cold -> tens of s).
        # pyarrow ParquetFile is NOT thread-safe, so each task opens its own handle.
        def read_one_rg(rg_idx):
            pf_local = pq.ParquetFile(self.data_path)
            rg_start, rg_end = rg_offsets[rg_idx]
            rg_needed_local = sorted(
                orig_idx - rg_start
                for orig_idx in needed
                if rg_start <= orig_idx < rg_end
            )
            out = {}
            if not rg_needed_local:
                return out
            cursor = 0
            batch_start = 0
            for batch in pf_local.iter_batches(
                batch_size=POSE_BATCH_SIZE,
                row_groups=[rg_idx],
                columns=['pose', 'num_frames'],
            ):
                n = batch.num_rows
                batch_end = batch_start + n
                if cursor < len(rg_needed_local) \
                        and rg_needed_local[cursor] < batch_end:
                    pose_col = batch.column('pose')
                    nf_col = batch.column('num_frames')
                    while cursor < len(rg_needed_local) \
                            and rg_needed_local[cursor] < batch_end:
                        local_idx = rg_needed_local[cursor]
                        orig_idx = local_idx + rg_start
                        pose_flat = np.asarray(
                            pose_col[local_idx - batch_start].as_py(),
                            dtype=np.float32,
                        )
                        num_frames = nf_col[local_idx - batch_start].as_py()
                        if isinstance(num_frames, (int, float)):
                            num_frames = [int(num_frames)] * self.NUM_CAMERAS
                        out[orig_idx] = (pose_flat, num_frames)
                        cursor += 1
                    del pose_col, nf_col
                del batch
                batch_start = batch_end
                if cursor >= len(rg_needed_local):
                    break
            return out

        if target_rgs:
            # num_workers capped (8) to bound concurrent decompress memory and FS
            # contention when many dataloader workers init at once; tune as needed.
            per_rg = parallel_execution(
                list(target_rgs), action=read_one_rg,
                num_workers=min(8, len(target_rgs)),
                sequential=len(target_rgs) <= 1,
            )
            for rg_out in per_rg:
                self.camera_params.update(rg_out)

        dt = time.time() - t0
        rank = get_rank()
        wid = get_worker_info()
        wid = wid.id if wid else 0
        if rank == 0 and wid == 0:
            log(f'Loaded {green(len(needed))}/{offset} {self.dataset_name} poses '
                f'({n_rg} row groups, {len(target_rgs)} read) in {dt:.2f}s')


    def init_loader(self):
        if hasattr(self, 'video_paths'):
            return

        sharded_metadata = self.shard_meta()
        self.video_readers = {}
        self.video_paths = {}

        for idx, meta in enumerate(sharded_metadata):
            base_dir = meta['video_path']
            if not isabs(base_dir):
                base_dir = join(self.data_root, base_dir)
            # Per-camera mp4 path; see video_path_template docstring for default.
            self.video_paths[idx] = [join(base_dir, self.video_path_template.format(c=c)) for c in range(self.NUM_CAMERAS)]

        self.load_poses([m['pose_idx'] for m in sharded_metadata])

        rank = get_rank()
        wid = get_worker_info().id if get_worker_info() is not None else 0
        if wid == 0 and rank == 0:
            log(f'MultiViewRealDataset init_loader: {green(len(sharded_metadata))} scenes')

    def get_cameras(self, idx: int):
        """Get parsed cameras for scene at sharded index. Returns list[NUM_CAMERAS].

        camera_params stores raw (pose_flat, num_frames) tuples. Parsing into
        per-camera per-frame {K, R, T} dicts + cross-camera normalization is
        done here on every access — CPU cost is negligible vs video decode.
        """
        meta = self.sharded_metadata[idx]
        pose_idx = meta['pose_idx']
        if pose_idx not in self.camera_params:
            self.load_poses([pose_idx])
        pose_flat, num_frames = self.camera_params[pose_idx]
        cams_per_view = parse_pose_column_egoexo4d(pose_flat, num_frames)
        if self.pose_norm_target > 0:
            flat_cams = [c for view in cams_per_view for c in view]
            normalize_cam_translation(flat_cams, self.pose_norm_target)
        return cams_per_view

    def pick_stable_factor(self, max_t: float) -> float:
        """Pick the configured stable factor closest to the scene scale `max_t`
        in LOG space (multiplicative distance), so a scene with mean camera
        distance ~3 m maps to the factor that brings translations near O(1).
        Single-factor configs short-circuit. max(max_t, 1e-6) guards log(0)."""
        factors = self.pose_stable_factors
        if len(factors) == 1:
            return factors[0]
        log_mt = math.log(max(max_t, 1e-6))
        return min(factors, key=lambda f: abs(math.log(f) - log_mt))

    def get_video_readers(self, idx):
        """Lazily create the NUM_CAMERAS video readers on first access."""
        if idx not in self.video_readers:
            paths = self.video_paths[idx]
            for p in paths:
                if not isfile(p):
                    raise FileNotFoundError(f'EgoExo4D video missing: {p}')
            self.video_readers[idx] = [self.video_reader_cls(p) for p in paths]
        return self.video_readers[idx]

    def shard_meta(self):
        if hasattr(self, 'sharded_metadata'):
            return self.sharded_metadata

        # Empty after slice/filter — keep worker alive (no ZeroDivisionError);
        # aggregator will not pick this dataset (weight=0).
        if not self.metadata:
            self.sharded_metadata = []
            return self.sharded_metadata

        wi = get_worker_info()
        rank = get_rank()
        world = get_world_size()

        if wi is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = wi.id, wi.num_workers

        # Default: every (rank, dataloader-worker) is an independent shard, so
        # the global shard count = world * num_workers and g_id is this
        # worker's flat global index.
        g_workers = world * num_workers
        g_id = rank * num_workers + worker_id

        # Sequence-parallel sharding: all ranks inside one SP group must read
        # the SAME rows (the sequence is split across the group, so they must
        # agree on which sample to load). Collapse to world//sp_size shards
        # keyed by SP-group index instead of per-rank.
        if self.config.sp_size != 1 and self.sp_sharding:
            g_workers = world // self.config.sp_size
            g_id = rank // self.config.sp_size

        # Strided round-robin assignment: this shard owns rows g_id, g_id+
        # g_workers, ... The `g_id % len` start guards the case where there are
        # more shards than rows (start index stays in range; rows get reused).
        self.sharded_metadata = self.metadata[g_id % len(self.metadata)::g_workers]
        self.metadata = self.sharded_metadata
        return self.sharded_metadata

    def __len__(self):
        # Report a deliberately huge length (rows × 100*500/(gen_size*vae_stride_t)
        # × 1e9): the RandomSampler draws idx in [0, len), and getitem_impl maps
        # idx → (local_idx = idx % rows, seed = idx // rows) so each idx is a
        # distinct deterministic (row, augmentation-seed) draw. The exact value
        # is not a real sample count — it just needs to be large enough to never
        # exhaust. The 100*500 and 1e9 factors are arbitrary inflators;
        # gen_size*vae_stride_t in the denominator makes longer windows report
        # proportionally fewer samples.
        # CAVEAT: gen_size is mutated by maybe_pick_shape on every __getitem__
        # when shape_pool is active. RandomSampler(replacement=True) caches len
        # at construction so this is safe today, but any sampler that re-queries
        # len mid-epoch would see a fluctuating size.
        mult = int(1e9)
        return len(self.metadata) * 100 * 500 // (self.gen_size * self.vae_stride_t) * mult

    @property
    def n_seqs(self):
        return len(self.metadata)

    @property
    def effective_samples(self):
        """sum(min_nf / row_fps) * src_cams * model_fps / tfs / 8.
        Per-row fps from parquet when available, else dataset_fps fallback.
        Returns 0 when empty so the aggregator drops the dataset."""
        if not self.metadata:
            return 0
        # tfs = pixel frames per training sample (the 4F-3 VAE decode length).
        # Final `/8` is the fixed per-sample normalizer shared by every
        # dataset's effective_samples so DatasetAggregator weights stay
        # comparable across sources (see docs/DATASET.md sampling formula).
        tfs = self.gen_size * self.vae_stride_t - 3
        default_fps = self.dataset_fps if self.dataset_fps else self.model_fps
        total = 0.0
        for m in self.metadata:
            nf_val = m.get('num_frames', 0)
            nf = min(nf_val) if isinstance(nf_val, list) else (nf_val or 0)
            fps_val = m.get('fps', default_fps)
            row_fps = float(min(fps_val) if isinstance(fps_val, list) else fps_val) or default_fps
            total += nf / row_fps
        total = total * self.NUM_CAMERAS * self.model_fps
        return max(1, int(total / tfs / 8))

    def load_view(self, vr: TorchCodecVideoReader, cameras: list,
                   frame_indices: np.ndarray,
                   target_h: int, target_w: int,
                   R0=None, T0=None,
                   gamma_value: float = 1.0,  # mvgame-style adaptive gamma (sequence-level)
                   image_aug: bool = False,   # mvgame-style image-space aug (per-view)
                   frame_start: int = 0,
                   n_frames_src: int = None,
                   frames=None,
                   **aug_kwargs):
        """Load frames + cameras for the given indices, resize/crop to target.

        Identical to StaticDataset.load_view: pre-resize via max-ratio so the
        resized frame covers the target in BOTH dims, then video_augmentation
        does a small aspect-ratio crop. Per-camera vr.h/vr.w means this works
        for both Aria (1408x1408) and GoPro (1280x720) without special-casing.
        Aria into a landscape pack view loses ~46% vertical content (cropped);
        this is a known cost — random shuffling means each Aria sample lands
        in a different pack position over time so coverage averages out.

        `frame_indices` are ABSOLUTE mp4 frame indices (caller must add
        `frame_start` before calling). `frame_start` + `n_frames_src` enable
        windowed-pose detect (sliced pose parquets like Nymeria): when
        `frame_start > 0` AND `abs(len(cameras) - n_frames_src) <= 5`, pose
        is indexed window-relative (`fi - frame_start`); otherwise pose is
        treated as full and indexed absolutely (`fi` directly). See
        memory:project_windowed_pose_convention.
        """
        resize_ratio = max(target_h / vr.h, target_w / vr.w) if self.pre_resize else 1.0
        if frames is None:  # caller may pass pre-decoded frames (parallel decode)
            frames = vr.get_batch(frame_indices.tolist(), return_channel_first=True,
                                  return_tensor=True, ratio=resize_ratio)
        frames = frames.float() / 255.0  # F, C, H, W in [0, 1]

        # Sequence-level gamma (constant across views) applied BEFORE any geometric aug.
        if gamma_value != 1.0:
            frames = gamma_correct(frames, gamma_value)

        # Windowed-pose detect: sliced pose ↔ window-relative cam indices.
        # CAVEAT (2026-05-28 audit): tolerance `<= 5` is empirical (allows few-
        # frame sync drift between mp4 and pose). If any future windowed parquet
        # has `|len(cams) - n_frames_src| > 5`, the detect silently falls
        # through to absolute indexing and reads wrong poses with no error. For
        # Nymeria / EPIC-Fields difference is always 0; if a new windowed
        # source has larger drift, tighten the check or add an assert.
        if n_frames_src is not None and frame_start > 0 and abs(len(cameras) - n_frames_src) <= 5:
            cam_indices = (frame_indices - frame_start).astype(np.int64)
        else:
            cam_indices = frame_indices.astype(np.int64)
        cams = [{k: torch.as_tensor(v, dtype=torch.float32).clone() for k, v in cameras[fi].items()}
                for fi in cam_indices]
        if resize_ratio != 1.0:
            for cam in cams:
                cam['K'][:2] *= resize_ratio

        # Capture pre-augmentation R0/T0 BEFORE video_augmentation applies
        # random roll/scale (otherwise per-view augmentation diff leaks into
        # relative poses).
        #
        # CAVEAT (2026-05-28 audit): R0/T0 anchor convention diverges across
        # dataloaders:
        #   - multiview.py / static.py / dynamic.py: R0 = first SAMPLED frame
        #     of view 0 (per-batch, depends on temporal segment + random
        #     shuffle of `view_to_cam`)
        #   - mvgame.py:1370: R0 = cams[0]['000000'] = first frame of FULL
        #     recording (segment-invariant)
        # PRoPE/training is insensitive to the basis choice as long as it's
        # consistent within a batch — but if downstream visualization or eval
        # assumes view 0 = a specific physical camera (e.g. ego head), the
        # multiview.py `random.shuffle(view_to_cam)` upstream randomizes which
        # camera defines the world frame.
        if R0 is None or T0 is None:
            R0 = cams[0]['R'].clone()
            T0 = cams[0]['T'].clone()

        frames, Ks, Rs, Ts = video_augmentation(
            frames, cams, Ho=target_h, Wo=target_w, **aug_kwargs
        )

        if image_aug:
            frames = image_augmentation(frames)

        # Rebase all poses into a canonical world frame anchored at view 0's
        # first sampled frame (see R0/T0 capture above). Rs/Ts here are w2c
        # (parse_pose_column returns R_w2c = R_c2w^T, T_w2c = -R_w2c @ t_c2w),
        # so right-multiplying by R0.mT moves the world basis onto camera 0's
        # frame-0 axes, and R_target then re-expresses it as: camera 0 at the
        # origin looking at +Y (Z-forward), Z-up (-Y-down). R_target is a +90°
        # rotation about X. Ts_new subtracts the same world-origin shift so the
        # translation stays consistent with the rotated basis. Identical
        # convention to static.py / mvgame.py load_view so all dataloaders emit
        # poses in the same frame for co-training.
        R_target = torch.as_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        Rs_new = Rs @ R0.mT @ R_target
        Ts_new = Ts - Rs @ R0.mT @ T0

        Ks_n = normalize_ixt(Ks, target_h, target_w)
        RTs = torch.cat([Rs_new, Ts_new], dim=-1)
        RTs = affine_padding(RTs)
        projs = ixt_padding(Ks_n) @ RTs

        RTs_inv = affine_padding(affine_inverse(RTs))
        Ks_inv = ixt_padding(ixt_inverse(Ks_n))
        projs_inv = RTs_inv @ Ks_inv

        return {
            'frames': frames,
            'projs': projs,
            'projs_inv': projs_inv,
            'Ks': Ks,
            # World-locked w2c Rs/Ts in the same frame as projs. The trainer derives
            # canonical c2w pose_10d (R_c2w, camera center C) from these values.
            'Rs': Rs_new,
            'Ts': Ts_new,
            'R0': R0,
            'T0': T0,
        }

    def getitem_impl(self, idx: int, **kwargs):
        self.init_loader()
        if not self.sharded_metadata:
            raise RuntimeError(
                f'{type(self).__name__} has no usable samples — should never '
                f'be picked by DatasetAggregator (weight=0). '
                f'Check data_path={self.data_path}'
            )
        shape_attempt = 0
        while True:
            # Decompose the flat sampler idx: local_idx selects the scene row,
            # seed = idx // n_rows acts as the "epoch" number so the same scene
            # gets a fresh-but-reproducible augmentation draw each time the
            # sampler revisits it. set_seed makes every random.* below (window
            # start, view shuffle, aug params) deterministic per idx — required
            # so all SP ranks that share an idx produce identical batches.
            seed = idx // len(self.sharded_metadata)
            seed = seed % (2 ** 32 - 1)  # clamp into the valid numpy/torch seed range
            set_seed(seed)
            local_idx = idx % len(self.sharded_metadata)

            meta = self.sharded_metadata[local_idx]

            # Fast length check from parquet metadata — no video I/O.
            nf_meta = meta.get('num_frames', 0)
            if isinstance(nf_meta, list):
                n_frames_src = min(nf_meta) if nf_meta else 0
            else:
                n_frames_src = int(nf_meta or 0)

            # Per-row source fps (multi-cam rows may carry a list → take the
            # slowest cam). resolve_fps_remap snaps it to a clean stride pattern;
            # max(1.0, ...) forbids upsampling (ratio<1 would duplicate frames),
            # so for sources slower than the model rate we just consume natively.
            fps_val = meta.get('fps', self.dataset_fps or self.model_fps)
            row_src_fps = float(min(fps_val) if isinstance(fps_val, list) else fps_val)
            row_eff_model, row_eff_src = resolve_fps_remap(self.model_fps, row_src_fps)
            row_fps_ratio = max(1.0, float(row_eff_src) / float(row_eff_model)) \
                if row_eff_src else 1.0
            n_frames = int(n_frames_src / row_fps_ratio)  # usable length in model-fps frames

            total_latent_size = self.gen_size
            # Pixel frames a gen_size-latent clip decodes to. VAE temporal conv
            # is causal: F latents → (F-1)*vae_stride_t + 1 frames; with
            # vae_stride_t=4 that is 4F-3, hence the -3.
            total_frame_size = total_latent_size * self.vae_stride_t - 3
            mv = self.mv_size

            if total_frame_size <= n_frames:
                # Scene is long enough: collapse idx to local_idx so the rest of
                # getitem_impl keys video readers / sharded_metadata by the row
                # index (the high "epoch" part of idx has done its job seeding).
                idx = local_idx
                break

            # Scene too short to fill one window. Advance to the next row
            # (local_idx + 1) and loop; the next iteration recomputes seed from
            # the new idx so the retry isn't a duplicate draw. With shape_pool,
            # also re-pick a (possibly smaller) shape first so a long-window pick
            # on a short clip can downsize instead of skipping forever.
            old_mv, old_gen = self.mv_size, self.gen_size
            if self.shape_pool:
                shape_attempt += 1
                self.maybe_pick_shape(idx, attempt=shape_attempt)
            log(yellow(
                f"MultiViewRealDataset: scene {meta['video_path']} too short "
                f"({n_frames}<{total_frame_size}, mv={old_mv}, gen={old_gen}): "
                f"re-pick → (mv={self.mv_size}, gen={self.gen_size})"
            ))
            idx = local_idx + 1

        # Now load actual video readers + cameras for the selected row.
        vrs = self.get_video_readers(idx)
        cams_per_view = self.get_cameras(idx)

        # Reconcile metadata num_frames with actual file lengths.
        # CAVEAT (2026-05-28 audit): `min` across cams collapses to the SHORTEST
        # per-cam value. For homogeneous parquets (Nymeria/mvgame_raw: all cams
        # sliced to same num_frames per row) this equals num_frames. For a
        # heterogeneous cohort (some cams full pose + others sliced), this would
        # silently truncate to the sliced cam's length and the windowed-detect
        # in load_view (line ~522) may misclassify per-cam. Currently no such
        # cohort exists in production parquets.
        n_frames_src = min(min(len(vrs[c]), len(cams_per_view[c]))
                           for c in range(self.NUM_CAMERAS))
        n_frames = int(n_frames_src / row_fps_ratio)

        # Augmentation flags (same as StaticDataset)
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
        if self.max_fov_h_deg is not None:
            aug_kwargs['max_fov_h_deg'] = self.max_fov_h_deg

        # Sample one temporal window in MODEL fps space, then remap to source
        # indices. Mirrors mvgame.py:915-916, 958-959.
        if n_frames > total_frame_size:
            start_model = random.randint(0, n_frames - total_frame_size)
        else:
            start_model = 0
        model_idx = np.arange(start_model, start_model + total_frame_size)
        if row_fps_ratio != 1.0:
            # round(model_i * ratio) is the source-frame stride. Clamp to
            # n_frames_src-1: the last window step can round past the final
            # decodable frame (n_frames is the floor of n_frames_src/ratio, so
            # model_idx.max()*ratio may exceed n_frames_src-1 by <1 frame).
            src_idx = np.minimum(np.round(model_idx * row_fps_ratio).astype(np.int64),
                                 n_frames_src - 1)
        else:
            src_idx = model_idx.astype(np.int64)

        # Windowed-pose convention (Nymeria etc.): parquet `pose` is sliced
        # per-segment; mp4 is full; `src_idx` is window-relative. Add
        # frame_start so vr.get_batch reads the correct absolute frames.
        # load_view subtracts it back for cameras lookup when n_frames_src
        # matches len(cameras) (auto-detect). frame_start=0 rows (egoexo4d,
        # waymo) pass through unchanged.
        frame_start = int(meta.get('frame_start') or 0)
        src_idx_abs = src_idx + frame_start

        # Random camera-to-view shuffle so the model sees each physical camera
        # in different pack positions over time. The Aria square (1408x1408)
        # gets vertically cropped when placed in a landscape pack view —
        # randomization means coverage averages out across samples.
        view_to_cam = list(range(self.NUM_CAMERAS))
        random.shuffle(view_to_cam)

        # Build batch. gather_mixed_batch tolerates per-dataset schema
        # differences (see utils/distributed.py / DatasetAggregator docs).
        batch = {'cpu': {}}
        batch['mv'] = mv
        batch['cpu']['seed'] = int(seed)
        batch['cpu']['prompts'] = meta['caption']
        batch['cpu']['video_path'] = meta['video_path']
        batch['cpu']['dataset_name'] = self.dataset_name
        batch['cpu']['parquet'] = basename(self.data_path)
        # Reproduction info: shared time window + per-view camera selection.
        # All views share the same source frame indices; they differ only in
        # which physical camera they read from (view_to_cam is a shuffle of
        # range(NUM_CAMERAS)). Together with video_path + seed these replay
        # the exact sample.
        batch['cpu']['start_frame'] = int(start_model)
        # Row + source-frame span for the vis meta panel (parity with
        # static/mvgame/dynamic). src_idx_abs is the absolute source frame indices
        # actually read (start_model remapped to source fps + frame_start); all
        # views share this one window, so a single (row, start, end) applies.
        batch['cpu']['rows'] = np.asarray([int(meta.get('pose_idx', -1))], dtype=np.int64)
        batch['cpu']['start_frames'] = np.asarray([int(src_idx_abs.min())], dtype=np.int64)
        batch['cpu']['end_frames'] = np.asarray([int(src_idx_abs.max())], dtype=np.int64)
        batch['cpu']['view_to_cam'] = np.asarray(view_to_cam, dtype=np.int64)  # (NUM_CAMERAS,)
        batch['cpu']['fps_ratio'] = float(row_fps_ratio)
        batch['cpu']['aug_kwargs'] = dict(aug_kwargs)
        if row_fps_ratio == 1.0:
            batch['fps'] = int(row_src_fps)
        else:
            batch['fps'] = row_eff_model

        # Prompt embeddings — must be cached, otherwise rank divergence
        # H3 re-encodes the raw caption with Qwen3; Wan/T5 caches are incompatible.
        embed = torch.empty((0, 5120), dtype=torch.bfloat16)
        batch['prompt_embeds'] = embed


        # Pack layout fields (see pack_factory / make_strip_pack):
        #   pack_size [sh, sw]: scales base (height, width) to the full canvas.
        #   rs[v]:  per-view resolution scale relative to base h/w.
        #   xs[v], ys[v]: top-left of view v's tile as a FRACTION of base
        #                 (width, height); multiplied by self.width/height below.
        pack_size = self.pack['pack_size']
        ratios = self.pack['rs']
        xs = self.pack['xs']
        ys = self.pack['ys']

        height_pack = int(self.height * pack_size[0])
        width_pack = int(self.width * pack_size[1])
        n_frames_out = total_frame_size
        frames = torch.zeros((n_frames_out, 3, height_pack, width_pack), dtype=torch.float32)
        projs_list = []
        projs_inv_list = []
        Ks_list, Rs_list, Ts_list = [], [], []

        # World reference: first frame of whichever camera is assigned to view 0
        R0, T0 = None, None

        # Sequence-level gamma: compute ONCE across views before the view loop so all
        # views share the same correction (per-view median would break cross-view
        # brightness consistency). Sample uses ratio=0.1 small-thumb decode (fast).
        gamma_value = 1.0
        if self.gamma_correction:  # mvgame_raw: aggressive lift (default MVGAME_LIFT_BAND)
            gamma_value = compute_sequence_gamma(vrs, src_idx_abs, mv, self.NUM_CAMERAS)
        elif self.exposure_clamp:  # real data: loose two-sided clamp
            gamma_value = compute_sequence_gamma(vrs, src_idx_abs, mv, self.NUM_CAMERAS,
                                                 band=LOOSE_EXPOSURE_BAND)
        # image_aug is per-view but gated by the same disable_aug switch as video_aug.
        view_image_aug = self.image_aug and not disable_aug

        # Decode every REALIZED view's frames in PARALLEL — each view reads its OWN
        # camera file (vrs[cam_idx]), so the decodes are independent. The rebase
        # loop below stays SERIAL (it threads R0/T0 view0->viewN) and consumes the
        # pre-decoded frames via load_view(frames=...), same path as StaticDataset.
        # view_specs is built in the SAME zip order as the loop so
        # decoded_per_view[view_idx] aligns. A SINGLE index list is passed to
        # parallel_execution — multi-list positional args mis-broadcast when
        # view_to_cam is longer than ratios. Parallel only when every realized view
        # maps to a distinct reader (no shared-cursor race); num_workers capped to
        # bound concurrent decode memory; else fall back to per-view decode.
        view_specs = []  # (cam_idx, resize_ratio) per realized view, in loop order
        for _ci, _r, _xo, _yo in zip(view_to_cam, ratios, xs, ys):
            _rr = (max(int(self.height * _r) / vrs[_ci].h, int(self.width * _r) / vrs[_ci].w)
                   if self.pre_resize else 1.0)
            view_specs.append((_ci, _rr))
        decoded_per_view = None
        _cams = [c for c, _ in view_specs]
        # Default OFF: leave decoded_per_view=None so the view loop below decodes
        # serially via load_view(frames=None) — the pre-2026-06-17 baseline for THIS
        # dataset. Only enabled when parallel_view_decode=True AND every realized view
        # maps to a distinct reader (no shared-cursor race). NOTE: mvgame.py's aug
        # load_constructed_video has its OWN long-standing parallel decode — that is
        # baseline and is intentionally NOT gated here.
        if self.parallel_view_decode and len(set(_cams)) == len(_cams):
            _inds = src_idx_abs.tolist()
            def _decode_view(i):
                c, rr = view_specs[i]
                return vrs[c].get_batch(_inds, return_channel_first=True,
                                        return_tensor=True, ratio=rr)
            decoded_per_view = parallel_execution(
                list(range(len(view_specs))), action=_decode_view,
                num_workers=min(8, len(view_specs)))

        for view_idx, (cam_idx, ratio, x_off, y_off) in enumerate(
                zip(view_to_cam, ratios, xs, ys)):
            target_h = int(self.height * ratio)
            target_w = int(self.width * ratio)

            view_data = self.load_view(
                vrs[cam_idx], cams_per_view[cam_idx], src_idx_abs,
                target_h, target_w,
                R0=R0, T0=T0,
                gamma_value=gamma_value,
                image_aug=view_image_aug,
                frame_start=frame_start,
                n_frames_src=n_frames_src,
                frames=(decoded_per_view[view_idx] if decoded_per_view is not None else None),
                **aug_kwargs
            )

            if R0 is None:
                R0 = view_data['R0']
                T0 = view_data['T0']

            h, w = view_data['frames'].shape[-2:]
            px = int(x_off * self.width)
            py = int(y_off * self.height)
            frames[:, :, py:py + h, px:px + w] = view_data['frames']

            projs_list.append(view_data['projs'])
            projs_inv_list.append(view_data['projs_inv'])
            Ks_list.append(view_data['Ks'])
            Rs_list.append(view_data['Rs'])
            Ts_list.append(view_data['Ts'])

        batch['frames'] = frames
        # Each *_list[v] is per-view [F, ...]. stack(dim=1) → [F, V, ...], then
        # reshape(-1, ...) flattens to FRAME-MAJOR order [f0v0, f0v1, ..., f1v0,
        # ...] (all views of frame 0, then frame 1). Downstream PRoPE/packing
        # relies on this view-within-frame interleave.
        batch['projs'] = torch.stack(projs_list, dim=1).reshape(-1, 4, 4)
        batch['projs_inv'] = torch.stack(projs_inv_list, dim=1).reshape(-1, 4, 4)
        batch['Ks'] = torch.stack(Ks_list, dim=1).reshape(-1, 3, 3)
        batch['Rs'] = torch.stack(Rs_list, dim=1).reshape(-1, 3, 3)
        batch['Ts'] = torch.stack(Ts_list, dim=1).reshape(-1, 3, 1)

        # Adaptive pose stable factor sized by MAX PAIRWISE camera distance (window
        # diameter), not mean. Rs/Ts are w2c so center C = -R^T @ T; the far end
        # overflows bf16 PRoPE. batch['Rs']/['Ts'] are now WORLD-LOCKED (v0/f0),
        # consistent with projs + pose_10d; pairwise is translation-invariant.
        centers = -torch.bmm(batch['Rs'].mT, batch['Ts']).squeeze(-1)  # world-locked
        pose_stable_factor, pose_max_t = select_pose_stable_factor(centers, self.pose_stable_factors)
        if pose_stable_factor != 1.0:
            # Scale BOTH projection translations and w2c T. The trainer derives
            # pose_10d camera center C=-R^T T, so both PRoPE streams see the same scale.
            batch['projs'][:, :3, 3] /= pose_stable_factor
            batch['projs_inv'][:, :3, 3] /= pose_stable_factor
            batch['Ts'] /= pose_stable_factor

        batch['cpu']['pack'] = self.pack
        batch['cpu']['pack']['width'] = self.width
        batch['cpu']['pack']['height'] = self.height
        batch['cpu']['pose_stable_factor'] = pose_stable_factor
        batch['cpu']['pose_max_t'] = pose_max_t  # pre-division max pairwise dist (bf16 diagnostic)

        return batch

    def maybe_pick_shape(self, idx: int, attempt: int = 0):
        """Idx-seeded weighted pick from self.shape_pool. Mutates self.mv_size,
        self.gen_size, self.pack (strip layout), and self.shape_pool_active.

        When the picked mv exceeds this dataset's NUM_CAMERAS, re-pick from
        the subset with mv ≤ NUM_CAMERAS using a redirect-seeded RNG. The
        redirect is deterministic per idx but distinct from the primary pick
        seed so the redistribution doesn't correlate with the primary pick.

        `attempt`: salt so getitem_impl can re-pick a different shape after
        a too-short miss. attempt=0 preserves the original seed.
        """
        if not self.shape_pool:
            self.shape_pool_active = False
            return None
        seed_str = f'shape_pool_{idx}' if attempt == 0 else f'shape_pool_{idx}_attempt_{attempt}'
        rng = random.Random(seed_str)
        weights = self.shape_pool_weights or [1.0] * len(self.shape_pool)
        mv_p, gen_p = rng.choices(self.shape_pool, weights=weights, k=1)[0]
        if int(mv_p) > self.NUM_CAMERAS:
            valid = [(s, w) for s, w in zip(self.shape_pool, weights)
                     if s[0] <= self.NUM_CAMERAS]
            if not valid:
                # No pool entry fits this dataset's camera budget; fall back to
                # the legacy cap so we still produce a sample rather than raise.
                self.mv_size = self.NUM_CAMERAS
                self.gen_size = int(gen_p)
            else:
                rng2 = random.Random(f'shape_pool_redirect_{idx}')
                mv_p, gen_p = rng2.choices([s for s, _ in valid],
                                           weights=[w for _, w in valid], k=1)[0]
                self.mv_size = int(mv_p)
                self.gen_size = int(gen_p)
        else:
            self.mv_size = int(mv_p)
            self.gen_size = int(gen_p)
        self.pack = make_strip_pack(self.mv_size)
        self.shape_pool_active = True
        return self.mv_size, self.gen_size

    def __getitem__(self, idx: int):
        try:
            self.maybe_pick_shape(idx)
            return self.getitem_impl(idx)
        except Exception as e:
            wi = get_worker_info()
            log(red(
                f"[MultiViewRealDataset __getitem__] failed: rank={get_rank()} "
                f"worker={wi.id if wi is not None else 0} idx={idx} err={e}"
            ))
            stacktrace()
            raise
