# Static scene dataset for DL3DV, RealEstate10K, and similar trajectory-based datasets
# Creates multi-view by sampling non-overlapping trajectory segments from a single video
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
import torch.nn.functional as F

from utils.console import *
from utils.data import as_numpy_func
from utils.math_utils import affine_padding
from utils.math_utils import affine_inverse
from utils.math_utils import ixt_inverse
from utils.math_utils import ixt_padding
from utils.distributed import get_rank
from utils.distributed import get_world_size
from utils.distributed import is_node_main
import utils.video as video_utils
from utils.video import TorchCodecVideoReader
from utils.misc import set_seed
from utils.parallel import parallel_execution
from dataset.mvgame import video_augmentation
from dataset.mvgame import gamma_correct
from dataset.mvgame import compute_sequence_gamma
from dataset.mvgame import LOOSE_EXPOSURE_BAND
from dataset.mvgame import normalize_ixt
from dataset.mvgame import normalize_cam_translation
from dataset.mvgame import select_pose_stable_factor
from dataset.mvgame import pack_factory
from dataset.mvgame import make_strip_pack
from dataset.fps_remap import resolve_fps_remap

try:
    import roma
    rotvec_to_rotmat = as_numpy_func(roma.rotvec_to_rotmat)
except ImportError:
    from scipy.spatial.transform import Rotation
    def rotvec_to_rotmat(rotvec):
        return Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)


def parse_pose_column(pose_flat: np.ndarray):
    """
    Parse the flat pose array from parquet into per-frame camera dicts.
    pose_flat: [N*10] where each frame is [fx, fy, cx, cy, ax, ay, az, x, y, z]
    Returns: list of dicts [{K: 3x3, R: 3x3, T: 3x1}, ...] for N frames

    CAVEAT (2026-05-28 audit): No NaN/Inf check on rotvec or translation.
    `Rotation.from_rotvec` silently returns NaN matrices for NaN inputs and
    wraps mod-2π otherwise — a single corrupted parquet row poisons Rs/Ts/projs
    and the training loss with no warning. Consider `assert np.all(np.isfinite(pose))`.
    """
    pose = pose_flat.reshape(-1, 10)
    N = pose.shape[0]
    fx, fy, cx, cy = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    rotvecs = pose[:, 4:7]  # N, 3 (c2w rotation as angle-axis)
    trans = pose[:, 7:10]   # N, 3 (c2w translation)

    # Build intrinsics
    Ks = np.zeros((N, 3, 3), dtype=np.float32)
    Ks[:, 0, 0] = fx
    Ks[:, 1, 1] = fy
    Ks[:, 0, 2] = cx
    Ks[:, 1, 2] = cy
    Ks[:, 2, 2] = 1.0

    # c2w rotation matrices
    c2w_R = rotvec_to_rotmat(rotvecs)  # N, 3, 3

    # Convert c2w to w2c: R_w2c = R_c2w^T, T_w2c = -R_c2w^T @ t_c2w
    Rs = np.transpose(c2w_R, (0, 2, 1))  # N, 3, 3 (w2c)
    Ts = -Rs @ trans[..., None]  # N, 3, 1 (w2c)

    cameras = []
    for i in range(N):
        cameras.append({
            'K': Ks[i],
            'R': Rs[i],
            'T': Ts[i],
        })
    return cameras


class StaticDataset(Dataset):
    """Dataset for static scenes (DL3DV, RealEstate10K, etc.).

    Creates multi-view by sampling non-overlapping trajectory segments as different views.
    Outputs the same batch format as MultiViewDataset for drop-in compatibility.

    TODO (seedpro static/dynamic re-screening — NOT yet wired in; user 2026-06-29):
    seedpro re-checks every static-prior sample and, when it finds real world motion,
    emits a quality entry {"type":"other","severity":"high","desc":"non-static scene:..."}
    (see scripts/data/seedpro/caption_sample.py static-check + static_violation_stats.py).
    Manual review (HDFS debug/caption_mismatch_m1) concluded seedpro's verdicts are CORRECT
    -> treat them as ground truth and route each captioned static-prior sample as:
      1. static=true BUT seedpro flagged non-static (it disagrees):
         a. MULTIVIEW  (mv>1)  -> DISCARD — the synchronized-views assumption is broken,
            unrecoverable as multiview.
         b. SINGLE-VIEW(mv==1) -> treat as DYNAMIC: load via the dynamic path
            (dataset/dynamic.py DynamicDataset), chunk-level caption, gen ratio 0.9.
      2. static=true AND seedpro did NOT disagree -> use AS static, regardless of which
         source dataset the row came from (incl. rows from a `_dynamic_` parquet that were
         genuinely static for that clip).
    NB: a separate upstream bug (dynamic rows mislabeled static=true from a stray parquet
    `static` column) is fixed for NEW specs in make_sample_specs.build_spec; the data-side
    correction for ALREADY-captioned rows is pending — see the KNOWN-BUG block in
    scripts/data/seedpro/merge_captions.py.
    """

    def __init__(self,
                 data_path: str,
                 data_root: str = None,  # base dir for relative video_path; defaults to parquet's parent dir
                 gen_size: int = 60,
                 height: int = 448,
                 width: int = 832,

                 mv_size: int = 8,
                 per_worker_threads: int = 4,

                 # Sequence sampling
                 seq_sample: List[int] = (0, None, 1),

                 # Overfitting
                 overfit: bool = False,
                 overfit_seq_sample: List[int] = (0, 1, 1),

                 config=dotdict({'sp_size': 1, 'model': {'vae_stride': [4, 8, 8]}}),
                 pose_norm_target: float = 1.0,
                 pose_stable_factors=1.0,  # float or list of floats

                 # Augmentation: disabled by default for static real data,
                 # but can be enabled via these params or **kwargs
                 disable_augmentation_ratio: float = 1.0,  # 1.0 = always disable aug
                 # Loose two-sided exposure clamp for real data: only the extreme
                 # dark/overexposed tails are nudged toward a comfortable band,
                 # everything else is a no-op (see compute_sequence_gamma + LOOSE_EXPOSURE_BAND).
                 # Applied regardless of disable_aug (it's a data-quality fix, not
                 # an augmentation). Inherited by Dynamic/StaticDynamic subclasses.
                 exposure_clamp: bool = False,
                 sp_sharding: bool = False,

                 # Per-video fps remap. SpatialVID has mixed source fps (24/25/30/50/60)
                 # and direct consumption at 60fps looks too slow. Each video's source
                 # fps is read from the parquet `fps` column; frames are subsampled by
                 # `max(1, src_fps / model_fps)` to match the model rate.
                 model_fps: int = 24,
                 sampling_weight_power: float = 0.8,  # default MUST match aggregator's getattr fallback (0.8); the attr is always set here so the fallback never fires

                 # Single-view long-temporal alternative. When set, every successful
                 # main-path pick has probability `long_gen_split` of being swapped
                 # to a mv=1 × long_gen_size monocular sample on the same video.
                 # Without this, long videos would always land on mv=mv_size×gen_size
                 # and the long-context training distribution would never be sampled.
                 # Also serves as the rescue tier when the chain exhausts (no mv≥2
                 # tier fits, but the video is long enough for mv=1).
                 long_gen_size: int = 0,         # 0 = disabled
                 long_gen_split: float = 0.5,    # P(swap to long path) when long_fits

                 # Per-sample shape pool (orthogonal to the per-anchor fallback chain).
                 # When set, each __getitem__ idx-deterministically picks one (mv, gen)
                 # from the pool and runs the existing path logic at that shape. All
                 # pool entries use a uniform-rs strip pack (every view at full
                 # height x width — no mixed-resolution layouts) so the per-view
                 # resolution stays constant across pool entries. None = disabled
                 # (preserves all existing behavior — assertions, fallback chain,
                 # short_paths, long_gen, etc. unchanged).
                 shape_pool=None,
                 shape_pool_weights=None,

                 # Video reader class name (looked up in utils.video).
                 # 'TorchCodecVideoReader' (default): handles VFR/B-frames but
                 # full file scan in __init__ — slow for multi-GB videos on NFS.
                 # 'CFRVideoReader': av.open + moov keyframe_pts, sub-second init
                 # at any file size; CFR only. Use for epic_fields-style sources.
                 video_reader: str = 'TorchCodecVideoReader',

                 *args, **kwargs):
        self.sampling_weight_power = sampling_weight_power
        self.video_reader_cls = getattr(video_utils, video_reader)
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
        self.mv_size = mv_size
        # Keep config-time mv_size separately — maybe_pick_shape mutates self.mv_size
        # at runtime so handle_static_exhausted can't use self.mv_size as the
        # config default for dynamic_mv_size=None case.
        self.init_mv_size = mv_size
        self.disable_augmentation_ratio = disable_augmentation_ratio
        self.exposure_clamp = exposure_clamp
        self.model_fps = model_fps
        self.long_gen_size = int(long_gen_size or 0)
        self.long_gen_split = float(long_gen_split)

        # shape_pool: list of (mv, gen) tuples. When non-empty, each __getitem__
        # samples one entry (idx-seeded, optional weights) and overrides
        # self.mv_size / self.gen_size for that call; pack lookups also switch
        # to make_strip_pack(mv) so all views stay at full height x width.
        self.shape_pool = [tuple(s) for s in (shape_pool or [])]
        if self.shape_pool:
            assert all(len(s) == 2 for s in self.shape_pool), \
                f"shape_pool entries must be (mv, gen) pairs, got {self.shape_pool}"
            for mv_p, _gen_p in self.shape_pool:
                assert mv_p >= 1 and (self.height % 16 == 0) and (self.width % 16 == 0), \
                    f"shape_pool entry mv={mv_p} requires height/width divisible by 16"
        self.shape_pool_weights = list(shape_pool_weights) if shape_pool_weights else None
        if self.shape_pool_weights is not None:
            assert len(self.shape_pool_weights) == len(self.shape_pool), \
                f"shape_pool_weights length {len(self.shape_pool_weights)} != shape_pool length {len(self.shape_pool)}"
        # Set when a __getitem__ call has been routed through shape_pool — toggles
        # strip pack lookup at the runtime pack site.
        self.shape_pool_active = False

        mv = self.mv_size
        assert mv in pack_factory, f"Supported mv sizes: {list(pack_factory.keys())}, got {mv}"
        pack = pack_factory[mv]
        assert (self.height * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        assert (self.width * pack['rs'] % 16 == 0).all(), "Packed sizes must be divisible by 16"
        self.pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}

        self.overfit = overfit
        self.overfit_seq_sample = overfit_seq_sample

        if is_node_main():
            log(f'Creating StaticDataset from {blue(data_path)}, gen_size={gen_size}, '
                f'mv_size={mv_size}, height={height}, width={width}, model_fps={model_fps}')

        # Load parquet — only read SMALL columns at init (skip the huge `pose`
        # column). pose is loaded lazily on first getitem. This keeps init fast
        # even for parquets with 333+ MB of pose data (DL3DV: 9343 rows × 35k floats).
        self.pf = pq.ParquetFile(data_path)
        schema = self.pf.schema_arrow
        small_cols = [c for c in schema.names if c != 'pose']
        small_table = self.pf.read(columns=small_cols)

        schema_meta = schema.metadata or {}
        if b'prompt_embeds_shape' in schema_meta:
            self.prompt_embeds_shape = json.loads(schema_meta[b'prompt_embeds_shape'])
        elif 'prompt_embeds' in schema.names:
            # Fallback: some older parquets have a prompt_embeds column but no
            # shape metadata. T5 XXL caches are always (512, 4096).
            self.prompt_embeds_shape = [512, 4096]

        full_metadata = small_table.to_pylist()  # fast: no pose

        # Tag each meta with its original parquet row index for pose lookup.
        # Also normalize optional frame_start/frame_end window columns (mirror
        # of dataset/mvgame.py:942-946): both default to "full video" when
        # absent/null — backward compatible with parquets that don't have
        # these columns. The window is applied in __getitem__ /
        # DynamicDataset.load_one_video_view to restrict sampling to
        # [frame_start, frame_end) in the original video's frame space.
        for i, m in enumerate(full_metadata):
            m['pose_idx'] = i
            fs = m.get('frame_start')
            m['frame_start'] = 0 if fs is None else int(fs)
            fe = m.get('frame_end')
            m['frame_end'] = None if (fe is None or fe <= 0) else int(fe)

        # Filter unsamplable entries: videos too short to produce any valid
        # sample under this dataset's fps target constraints. Avoids wasted
        # pick attempts (DynamicDataset) and infinite recursion (StaticDataset
        # skipping to next idx). Filter criterion is per-sample: an entry is
        # unsamplable if it can't produce even 1 view at the tightest allowed
        # fps target — the mv-view requirement (static path) is checked at
        # sample time, with StaticDynamicDataset falling back to dynamic for
        # shorter videos.
        n_total = len(full_metadata)
        viable = [m for m in full_metadata if self.is_sample_viable(m)]
        n_viable = len(viable)
        n_filtered = n_total - n_viable

        # Length-based prefilter happens BEFORE sharding. Report which dataset
        # was filtered, by how much, and warn loudly when the result is empty
        # or too small for distributed dispatch. Threshold = num_workers ×
        # world_size: each (rank, worker) needs ≥1 row to avoid ZeroDivision in
        # shard_meta, and stride-based sharding works best when every shard has
        # at least one sample.
        threshold = max(1, get_world_size() * int(kwargs.get('num_workers', 1) or 1))
        self.is_empty = (n_viable == 0)
        if is_node_main():
            cls_name = type(self).__name__
            tag = green('OK') if n_viable >= threshold else red('TOO FEW')
            log(f'[{cls_name} filter] {blue(data_path)}: '
                f'{n_total} total → {green(n_viable)} viable '
                f'(filtered {n_filtered} too-short) [{tag}, threshold={threshold}]')
            if n_viable == 0:
                log(red(
                    f'[{cls_name} EMPTY] {data_path}: '
                    f'ALL {n_total} rows filtered as too short — this dataset '
                    f'will be DROPPED from co-training (weight=0). '
                    f'Likely cause: gen_size={self.gen_size} (×vae_stride_t {self.vae_stride_t} '
                    f'-3) requires more frames than any video provides at the '
                    f'allowed fps targets ({getattr(self, "fps_targets", None)}). '
                    f'Either reduce gen_size for this dataset or remove it from the config.'
                ))
            elif n_viable < threshold:
                log(red(
                    f'[{cls_name} BELOW THRESHOLD] {data_path}: '
                    f'only {n_viable} viable rows < num_workers*world_size = {threshold}. '
                    f'Some workers will have NO data; training will still run but '
                    f'sharding will reuse rows across workers (g_id % len). Consider '
                    f'reducing gen_size for this dataset or dropping it.'
                ))

        if overfit:
            self.metadata = viable[slice(*overfit_seq_sample)]
        else:
            self.metadata = viable[slice(*seq_sample)]

        # Pose cache keyed by pose_idx (original parquet row index).
        self.camera_params = {}

        if is_node_main():
            log(f'StaticDataset init: {green(len(self.metadata))} scenes from {blue(data_path)} '
                f'(after slice: viable={n_viable})')

    def load_poses(self, pose_indices):
        """Read and parse poses for specific parquet row indices.

        Uses row-group filtering to avoid reading the entire pose column when
        the parquet has multiple row groups. For single row group (typical for
        <100K rows), falls back to reading the full column but only parses
        requested rows.
        """
        needed = [i for i in pose_indices if i not in self.camera_params]
        if not needed:
            return

        t0 = time.time()
        needed_set = set(needed)

        # Re-open parquet in worker processes: pyarrow ParquetFile is NOT
        # fork-safe — reusing the parent's pf in a forked DataLoader worker
        # hangs on Arrow's internal I/O thread pool.
        wi = get_worker_info()
        pf = pq.ParquetFile(self.data_path) if wi is not None else self.pf
        n_rg = pf.metadata.num_row_groups

        # Build row-group offset table
        rg_offsets = []  # (start_row, end_row) per group
        offset = 0
        for i in range(n_rg):
            n = pf.metadata.row_group(i).num_rows
            rg_offsets.append((offset, offset + n))
            offset += n

        # Find row groups containing target indices
        target_rgs = []
        for rg_idx, (rg_start, rg_end) in enumerate(rg_offsets):
            if any(rg_start <= idx < rg_end for idx in needed_set):
                target_rgs.append(rg_idx)

        # Read row groups ONE AT A TIME, and within each row group stream in
        # small batches, releasing each batch before the next.
        #
        # Why not `pf.read_row_groups(target_rgs, columns=['pose'])` or even a
        # single `pf.read_row_group(rg_idx, columns=['pose'])`?
        # pyarrow materializes the selection into a single Table whose `pose`
        # column is one ListArray with a single int32 offsets buffer covering
        # the whole column. For large parquets (span ~450k, spand ~640k,
        # general_ht ~280k, spatialvid ~620k with up to 901 frames/row) the
        # cumulative offsets can exceed 2^31 elements — even within a single
        # row group — and crash with `OSError: List index overflow`. Example:
        # spatialvid_nohq_filtered_dynamic_posefiltered.parquet has one row
        # group totalling ~2.97e9 pose doubles, well past int32.
        # `iter_batches` yields RecordBatches each with their OWN independent
        # int32 offsets buffer, so as long as a single batch fits in int32
        # (batch_size=50000 * max 901 frames * 10 doubles ≈ 4.5e8 << 2^31) we
        # are safe regardless of total row group size.
        # `row_groups=[rg_idx]` scopes the iterator to one row group — same
        # IO as `read_row_group`, just chunked differently in memory.
        POSE_BATCH_SIZE = 10_000

        # Read the target row groups CONCURRENTLY (reusing utils.parallel):
        # decompress + network I/O dominate and pyarrow releases the GIL, so threads
        # give near-linear speedup. Single-row-group parquets collapse to one task
        # (sequential) and keep the original int32-overflow-safe streaming. pyarrow
        # ParquetFile is NOT thread-safe, so each task opens its own handle.
        def read_one_rg(rg_idx):
            pf_local = pq.ParquetFile(self.data_path)
            rg_start, rg_end = rg_offsets[rg_idx]
            # Indices within this row group, in local coordinates.
            rg_needed_local = sorted(
                orig_idx - rg_start
                for orig_idx in needed
                if rg_start <= orig_idx < rg_end
            )
            out = {}
            if not rg_needed_local:
                return out
            cursor = 0  # pointer into rg_needed_local
            batch_start = 0  # local row offset of current batch
            for batch in pf_local.iter_batches(
                batch_size=POSE_BATCH_SIZE,
                row_groups=[rg_idx],
                columns=['pose'],
            ):
                n = batch.num_rows
                batch_end = batch_start + n
                if cursor < len(rg_needed_local) \
                        and rg_needed_local[cursor] < batch_end:
                    pose_col = batch.column('pose')
                    while cursor < len(rg_needed_local) \
                            and rg_needed_local[cursor] < batch_end:
                        local_idx = rg_needed_local[cursor]
                        orig_idx = local_idx + rg_start
                        # Store RAW flat array only (40 bytes/frame vs 720 parsed).
                        # Parsing + normalization deferred to get_cameras() at access time.
                        out[orig_idx] = np.asarray(
                            pose_col[local_idx - batch_start].as_py(),
                            dtype=np.float32,
                        )
                        cursor += 1
                    del pose_col
                del batch
                batch_start = batch_end
                if cursor >= len(rg_needed_local):
                    break  # consumed everything we need from this row group
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
                f'({n_rg} row groups, {len(target_rgs)} read) in {dt:.2f} s')

    def init_loader(self):
        if hasattr(self, 'video_paths'):
            return

        sharded_metadata = self.shard_meta()
        self.video_readers = {}
        self.video_paths = {}

        for idx, meta in enumerate(sharded_metadata):
            video_path = meta['video_path']
            if not isabs(video_path):
                video_path = join(self.data_root, video_path)
            self.video_paths[idx] = video_path

        # Load all poses for this worker's shard. Called eagerly from
        # worker_init_fn at spawn time so the first getitem has no disk reads.
        self.load_poses([m['pose_idx'] for m in sharded_metadata])

        rank = get_rank()
        wid = get_worker_info().id if get_worker_info() is not None else 0
        if wid == 0 and rank == 0:
            log(f'{type(self).__name__} init_loader: {green(len(sharded_metadata))} scenes')

    def get_cameras(self, idx: int):
        """Get parsed cameras for scene at sharded index.

        camera_params stores raw flat arrays (compact, 40B/frame). Parsing
        into per-frame {K, R, T} dicts + translation normalization is done
        here on every access — CPU cost is negligible vs video decode.
        """
        meta = self.sharded_metadata[idx]
        pose_idx = meta['pose_idx']
        if pose_idx not in self.camera_params:
            self.load_poses([pose_idx])
        pose_flat = self.camera_params[pose_idx]
        cameras = parse_pose_column(pose_flat)
        if self.pose_norm_target > 0:
            normalize_cam_translation(cameras, self.pose_norm_target)
        return cameras

    def pick_stable_factor(self, max_t: float) -> float:
        # CAVEAT (2026-05-28 audit): callers pass `mean_dist` (mean pairwise
        # camera-center distance), but the bf16 PRoPE constraint is on max |T|
        # post-scaling (score ≈ 2.4 * T², need T < ~5). For a one-way trajectory
        # max |T| can be many σ over mean → chosen factor may not cap |T| under 5.
        # Consider passing max(|T|) instead of mean(), or adding a runtime
        # assertion that batch-max |T| < threshold after division.
        factors = self.pose_stable_factors
        if len(factors) == 1:
            return factors[0]
        log_mt = math.log(max(max_t, 1e-6))
        return min(factors, key=lambda f: abs(math.log(f) - log_mt))

    def get_video_reader(self, idx):
        """Lazily create video reader on first access."""
        if idx not in self.video_readers:
            video_path = self.video_paths[idx]
            if isfile(video_path):
                video_file = video_path
            elif isfile(video_path + '.mp4'):
                video_file = video_path + '.mp4'
            elif isdir(video_path):
                video_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.mp4', '.avi', '.mov'))])
                if video_files:
                    video_file = join(video_path, video_files[0])
                else:
                    raise FileNotFoundError(f'No video file found in {video_path}')
            else:
                raise FileNotFoundError(f'Video path not found: {video_path}')
            self.video_readers[idx] = self.video_reader_cls(video_file)
        return self.video_readers[idx]

    def shard_meta(self):
        if hasattr(self, 'sharded_metadata'):
            return self.sharded_metadata

        # Empty after the length prefilter — keep the worker alive (no
        # ZeroDivisionError) so co-training can drop this dataset and continue.
        # The aggregator filters out zero-weight datasets so __getitem__ should
        # not be reached; if it is, getitem_impl raises a clear error below.
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

        g_workers = world * num_workers
        g_id = rank * num_workers + worker_id

        if self.config.sp_size != 1 and self.sp_sharding:
            g_workers = world // self.config.sp_size
            g_id = rank // self.config.sp_size

        self.sharded_metadata = self.metadata[g_id % len(self.metadata)::g_workers]
        self.metadata = self.sharded_metadata

        return self.sharded_metadata

    def __len__(self):
        # CAVEAT (2026-05-28 audit): reads self.gen_size which is mutated by
        # maybe_pick_shape (line 1100) on every __getitem__ when shape_pool is
        # active. RandomSampler(replacement=True) caches len at construction so
        # this is currently safe — but any sampler that re-queries len mid-epoch
        # would see a fluctuating size. Fragile.
        mult = int(1e9)
        return len(self.metadata) * 100 * 500 // (self.gen_size * self.vae_stride_t) * mult

    @property
    def n_seqs(self):
        return len(self.metadata)

    @property
    def effective_samples(self):
        """sum(num_frames / src_fps) * model_fps / tfs / 8.
        Static: src_views=1 (frames shared across mv views).
        Returns 0 when the dataset is empty after prefilter — aggregator uses
        this to drop the dataset (weight=0, never picked)."""
        if not self.metadata:
            return 0
        # tfs = "target frame size": gen_size latent frames converted to source
        # pixel frames. The causal VAE maps vae_stride_t (=4) pixel frames to 1
        # latent frame, except the first latent frame which covers a single
        # pixel frame — so L latent frames span vae_stride_t*L - (vae_stride_t-1)
        # = 4*L - 3 pixel frames. /8 below is a fixed normalization shared with
        # the other datasets' effective_samples so the aggregator's eff^power
        # weights are comparable across sources.
        tfs = self.gen_size * self.vae_stride_t - 3
        nf = np.array([m.get('num_frames', 0) or 0 for m in self.metadata], dtype=np.float64)
        fps = np.array([m.get('fps', self.model_fps) or self.model_fps for m in self.metadata], dtype=np.float64)
        total = np.sum(nf / fps) * self.model_fps
        return max(1, int(total / tfs / 8))

    def sample_view_segments(self, n_frames_total: int, total_frame_size: int, mv_size: int):
        """
        Sample mv_size non-overlapping segments of total_frame_size frames from n_frames_total frames.
        For static scenes, different time segments = different viewpoints.

        The algorithm uniformly samples from all valid non-overlapping placements:
        1. Compute total slack = n_frames_total - mv_size * total_frame_size
        2. Randomly partition the slack into (mv_size + 1) gaps (before, between, after segments)
        3. Place segments sequentially with these gaps
        4. Randomly shuffle assignment to view indices (any segment can be the main view)

        Returns: list of (start_idx, end_idx) tuples, length mv_size
        """
        needed = mv_size * total_frame_size
        assert n_frames_total >= needed, \
            f"Scene too short: {n_frames_total} frames < {needed} needed ({mv_size} views x {total_frame_size} frames)"

        slack = n_frames_total - needed
        # Random partition of slack into (mv_size + 1) non-negative integer gaps.
        # This is the "stars and bars" method from combinatorics (Feller, 1968,
        # "An Introduction to Probability Theory and Its Applications", Ch. II.5).
        # Equivalent to sampling from a Dirichlet(1,...,1) distribution on the simplex
        # (uniform over all valid placements), discretized to integers.
        # Ref: https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
        if slack > 0:
            breaks = sorted(random.randint(0, slack) for _ in range(mv_size))
            gaps = [breaks[0]] + [breaks[i] - breaks[i - 1] for i in range(1, mv_size)] + [slack - breaks[-1]]
        else:
            gaps = [0] * (mv_size + 1)

        # Place segments with gaps
        segments = []
        pos = 0
        for i in range(mv_size):
            pos += gaps[i]
            segments.append((pos, pos + total_frame_size))
            pos += total_frame_size

        # Shuffle so any segment can be assigned to any view (including the main view)
        random.shuffle(segments)
        return segments

    @staticmethod
    def video_to_cam_indices(frame_indices: np.ndarray, n_vid: int, n_cam: int) -> np.ndarray:
        """Map video frame indices → pose array indices.

        Three regimes based on `n_cam` vs `n_vid`:

        1. **n_cam == n_vid**: 1:1 mapping, return `frame_indices` unchanged.
           Covers mvgame, spatialvid, general_ht, span etc.

        2. **|n_cam - n_vid| == 1**: off-by-one pose array (sekai_walk_s/d,
           drone_s/d have pose with 1 extra frame at the end). Clamp to
           `n_cam - 1`; the 1:1 mapping is correct for all frames except the
           very last boundary, which just gets clamped harmlessly.

        3. **|n_cam - n_vid| > 1**: pose is subsampled relative to video —
           the ytb_reencode.py fps bug left re10k/dynpose with video at the
           raw YouTube source fps (often 60) while pose is from the original
           RealEstate10K-style 30fps metadata. Assume uniform temporal
           sampling: video frame `vi` at time `vi / n_vid_time` corresponds
           to pose index `round(vi * (n_cam - 1) / (n_vid - 1))`. This is
           exact when pose timestamps span the same cropped range as the
           video, uniformly sampled.

        Args:
            frame_indices: int64 array of video frame indices, values in [0, n_vid).
            n_vid: total number of video frames available (`len(vr)`).
            n_cam: total number of pose entries available (`len(cameras)`).

        Returns:
            int64 array same shape as `frame_indices`, values in [0, n_cam).

        CAVEAT (2026-05-28 audit): no `n_cam == 0` guard. Regime 2 returns -1
        for empty cameras (silently picks pose[-1] = last entry via Python
        negative indexing); regime 3 returns 0 for `n_cam == 1`, silently
        broadcasting frame-0's pose across all output positions with no log.
        Add `assert n_cam > 0` on entry if you suspect degenerate parquet rows.
        """
        if n_cam == n_vid:
            return frame_indices
        if abs(n_cam - n_vid) <= 1:
            return np.minimum(frame_indices, n_cam - 1)
        # Subsampled pose: linear interpolation from video space into pose space.
        scale = (n_cam - 1) / max(1, n_vid - 1)
        return np.minimum(
            np.round(frame_indices.astype(np.float64) * scale).astype(np.int64),
            n_cam - 1,
        )

    def load_view(self, vr: TorchCodecVideoReader, cameras: list,
                   frame_indices: np.ndarray,
                   target_h: int, target_w: int,
                   R0=None, T0=None, frame_start=0,
                   n_frames_src: int = None,
                   gamma_value: float = 1.0,
                   frames=None,
                   **aug_kwargs):
        """
        Load frames and cameras for the given frame indices, resize/crop to target size.

        frame_indices: 1D np.int64 array of absolute frame indices into vr.
        frame_start: window offset. Subtracted from frame_indices for camera
            lookup so pose[0] corresponds to video frame frame_start.
        n_frames_src: window length (frame_end - frame_start). When provided
            and matches len(cameras) within tolerance, triggers windowed-pose
            mode: pose[i] is treated as corresponding to window-relative
            frame i (1:1 within window). See memory:
            project_windowed_pose_convention. Required for cut-segment
            parquets (EPIC-Fields, Nymeria) where pose is sliced per row.
        Camera indices are derived via video_to_cam_indices() to handle
        datasets where the pose array length differs from the video frame
        count (e.g. re10k/dynpose subsampled pose).

        Returns dict with frames, projs, projs_inv, Ks, Rs, Ts
        """

        # Pre-resize frames to approximately the target size before calling
        # video_augmentation. This matches MVGame's load_view(ratio=pack_ratio)
        # behavior. Without this, video_augmentation's M_crop = target/source
        # (e.g. 112/720 = 0.16) center-crops a tiny region instead of resizing
        # the whole image. With pre-resize, M_crop ≈ 1.0 and video_augmentation
        # only does a small aspect-ratio adjustment.
        # Use max ratio so the resized frame covers the target in BOTH dimensions.
        # video_augmentation will center-crop the excess (small, <=aspect mismatch).
        # e.g. 720x1280 → target 112x208: ratio_h=0.156, ratio_w=0.163 → use 0.163
        #      resized = 117x208, crop 117→112 vertically (5 pixels, not a zoom)
        # This also handles portrait sources: 1280x720 → target 112x208:
        #      ratio_h=0.088, ratio_w=0.289 → use 0.289, resized = 370x208, crop 370→112
        resize_ratio = max(target_h / vr.h, target_w / vr.w)
        if frames is None:  # caller may pass pre-decoded frames (batched decode)
            frames = vr.get_batch(frame_indices.tolist(), return_channel_first=True,
                                  return_tensor=True, ratio=resize_ratio)
        frames = frames.float() / 255.0  # F, C, H, W in [0, 1]

        # Sequence-level exposure clamp (constant gamma, computed once across the
        # views by the caller) applied BEFORE any geometric aug. gamma==1.0 no-op.
        if gamma_value != 1.0:
            frames = gamma_correct(frames, gamma_value)

        # Windowed-pose detect: when caller passes n_frames_src and pose has
        # roughly window-length entries, treat pose as window-relative
        # (regime 1 1:1 inside video_to_cam_indices). Otherwise fall back to
        # whole-video pose indexing (regime 1/2/3 by length comparison).
        #
        # MUST fire for frame_start == 0 too. telecut produces [0, frame_end)
        # windows of a LONGER, un-re-encoded video file (e.g. sekai_walk:
        # 1024-frame pose sliced from an 1800-frame mp4, frame_start=0). The
        # old `frame_start > 0` gate sent those to the else branch, where
        # cam_n_vid=len(vr)=1800 makes video_to_cam_indices treat the pose as
        # *subsampled* (regime 3) and rescale indices by ~0.57 → pose silently
        # desyncs from (and lags) the video. The `abs(...) <= 5` length match is
        # the real windowed-pose signal; frame_start is irrelevant to it.
        # See docs/POSE_WINDOW_REMAP_BUG.md.
        if n_frames_src is not None and abs(len(cameras) - n_frames_src) <= 5:
            cam_n_vid = n_frames_src
        else:
            cam_n_vid = len(vr) - frame_start
        cam_indices = self.video_to_cam_indices(
            frame_indices - frame_start, cam_n_vid, len(cameras))

        # Collect camera dicts for these frames and adjust K for the pre-resize
        # (same as MVGame's load_view lines 132-134)
        cams = [{k: torch.as_tensor(v, dtype=torch.float32).clone() for k, v in cameras[ci].items()}
                for ci in cam_indices]
        if resize_ratio != 1.0:
            for cam in cams:
                cam['K'][:2] *= resize_ratio

        # Capture pre-augmentation R0/T0 BEFORE video_augmentation applies random
        # roll/scale. MVGame does this via cams[0]['000000'] (raw camera). Using
        # post-aug R0 would leak per-view augmentation differences into relative
        # poses (e.g. view 0 roll=+5° vs view 1 roll=-3° → 8° error in relative R).
        if R0 is None or T0 is None:
            R0 = cams[0]['R'].clone()
            T0 = cams[0]['T'].clone()

        frames, Ks, Rs, Ts = video_augmentation(
            frames, cams, Ho=target_h, Wo=target_w, **aug_kwargs
        )

        # Relative pose computation uses pre-aug R0/T0 (captured above). Rebase
        # every camera into a canonical world frame anchored at view-0/frame-0:
        # that reference camera lands at the origin looking at +Y (Z-forward)
        # with Z-up, encoded by R_target. Rs/Ts here are w2c (see
        # parse_pose_column), so `Rs @ R0.mT` removes the reference rotation and
        # the right-multiply by R_target re-expresses it in the target axis
        # convention; `Ts - Rs @ R0.mT @ T0` shifts the world origin onto the
        # reference camera center. Same formula/axes as mvgame.py / multiview.py
        # so all datasets feed the model identically-canonicalized poses.
        R_target = torch.as_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        Rs_new = Rs @ R0.mT @ R_target
        Ts_new = Ts - Rs @ R0.mT @ T0

        # projs = full 4x4 world->pixel projection per frame: pad normalized
        # K to 4x4 (ixt_padding) and the rebased [R|T] to 4x4 (affine_padding),
        # then compose. projs_inv is the pixel->world inverse, built from the
        # separately-inverted extrinsics and intrinsics (NOT inverse(projs)) so
        # each factor stays well-conditioned. Both feed PRoPE camera embedding.
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

    def is_sample_viable(self, meta):
        """Return True if this entry can produce at least 1 view at some fps target.

        Criterion: there exists an allowed fps target t ≤ src_fps such that
        num_frames >= ceil(tfs * src_fps / t). Subclasses override via
        self.fps_targets attribute (None = graded fallback, native fps allowed).

        Multi-path semantics: a row is viable if ANY enabled runtime path can
        use it. We collect candidate gen_sizes from all paths the dataset
        runtime might exercise (static path `gen_size`, dynamic fallback
        `dynamic_gen_size`, long single-view `long_gen_size`, view-iso rescue
        `short_paths`) and use the SMALLEST — that is the most lenient
        threshold, so a clip that fits the easiest path passes.
        """
        n = meta.get('num_frames', 0)
        if isinstance(n, list):
            n = min(n) if n else 0
        if not n:
            return False

        # Collect gen_size candidates across all enabled runtime paths.
        # Smaller gen_size → fewer frames needed → easier to satisfy.
        candidates = [self.gen_size]
        dgs = getattr(self, 'dynamic_gen_size', None)
        if dgs:
            candidates.append(dgs)
        long_gs = getattr(self, 'long_gen_size', 0)
        if long_gs:
            candidates.append(long_gs)
        short_paths = getattr(self, 'short_paths', None)
        if short_paths:
            for _short_mv, short_gen in short_paths:
                if short_gen:
                    candidates.append(short_gen)
        # shape_pool gens — a row is viable if it fits the smallest pool gen.
        for _mv_p, gen_p in getattr(self, 'shape_pool', []) or []:
            if gen_p:
                candidates.append(gen_p)
        gs = min(candidates)
        tfs = gs * self.vae_stride_t - 3

        src_fps = float(meta.get('fps', self.model_fps)) or self.model_fps
        # Must mirror runtime's effective ratio (load_one_video_view uses
        # resolve_fps_remap which snaps NTSC sources UP — e.g. 59.94 → ratio
        # 60/15 = 4.0, not 59.94/16 = 3.74). Using the raw ratio here lets
        # rows pass the filter that runtime then rejects, causing per-anchor
        # "exhausted all paths" spam.
        eff_model_fps, eff_src_fps = resolve_fps_remap(self.model_fps, src_fps)
        snapped_ratio = max(1.0, float(eff_src_fps) / float(eff_model_fps)) if eff_src_fps else 1.0
        fps_targets = getattr(self, 'fps_targets', None)
        if fps_targets is not None:
            usable = [t for t in fps_targets if t <= src_fps]
            if not usable:
                return False
            min_ratio = snapped_ratio
        else:
            # Default graded fallback includes native src_fps (ratio 1.0)
            min_ratio = 1.0
        return n >= int(round(tfs * min_ratio))

    def handle_static_exhausted(self, idx: int, **kwargs):
        """Called when no fps/mv level can fill the requested views.

        Default: skip to next video. Subclasses (e.g. StaticDynamicDataset)
        override this to fall back to dynamic sampling.
        """
        return self.getitem_impl(idx + 1, **kwargs)

    def getitem_impl(self, idx: int, **kwargs):
        self.init_loader()
        if not self.sharded_metadata:
            raise RuntimeError(
                f'{type(self).__name__} has no usable samples after length '
                f'prefilter — should never be picked by DatasetAggregator '
                f'(weight=0). Check data_path={self.data_path}'
            )
        # __len__ is inflated huge so RandomSampler(replacement=True) draws idx
        # from a vast space. Decompose it: quotient = pseudo-epoch counter,
        # remainder = which scene. Seeding RNG by the quotient means each time
        # the same scene (same idx % len) is revisited it gets a fresh seed, so
        # sample_view_segments' stars-and-bars placement and the augmentation
        # draw differ every epoch instead of replaying identical frames/crops.
        # mod (2**32 - 1) keeps the seed inside numpy/torch's uint32 range.
        seed = idx // len(self.sharded_metadata)
        seed = seed % (2 ** 32 - 1)
        set_seed(seed)
        idx = idx % len(self.sharded_metadata)

        meta = self.sharded_metadata[idx]
        vr = self.get_video_reader(idx)
        cameras = self.get_cameras(idx)

        # Static scene: use native frames directly as independent viewpoints.
        #
        # Pose-video length mismatch handling:
        # - Aligned datasets (mvgame, spatialvid, general_ht, span):
        #     n_cam == n_vid → `min` is trivially n_vid
        # - Off-by-one datasets (sekai_walk, drone): n_cam = n_vid + 1 →
        #     `min` = n_vid, pose's extra frame is unused (correct)
        # - Subsampled-pose datasets (re10k, dynpose from ytb_reencode fps
        #   bug): n_cam << n_vid → we intentionally use n_vid so segments
        #   cover the full video range. `load_view` maps video indices to
        #   pose indices via `video_to_cam_indices` (linear interpolation).
        n_vid = len(vr)
        n_cam = len(cameras)
        if abs(n_cam - n_vid) <= 1:
            full_src = min(n_vid, n_cam)  # preserves prior behavior exactly
        else:
            full_src = n_vid              # use full video; pose lookup is interpolated

        # Optional [frame_start, frame_end) window — defaults to full video when
        # the parquet has no such columns (mvgame convention, see __init__).
        # Sampling happens in window-space [0, n_frames_src); the frame_start
        # offset is added back when building seg_frame_inds below so that
        # vr.get_batch and video_to_cam_indices receive original source indices.
        frame_start = int(meta.get('frame_start') or 0)
        frame_end = meta.get('frame_end')
        frame_end_eff = full_src if (frame_end is None or frame_end <= 0) else min(int(frame_end), full_src)
        n_frames_src = max(0, frame_end_eff - frame_start)

        static_latent_size = self.gen_size
        static_frame_size = static_latent_size * self.vae_stride_t - 3
        # `total_*` track the *chosen* path; updated below if we swap to long-gen.
        total_latent_size = static_latent_size
        total_frame_size = static_frame_size

        # Per-video fps remap with mv fallback.
        # Primary target comes from `dataset/fps_remap.py` factory which snaps
        # (model_fps, src_fps) to a (eff_model, eff_src) pair giving a clean
        # ratio = eff_src / eff_model with small denominator (q≤2 → period≤2
        # Δf alternation, vs the chaotic period-16 pattern raw 25/16 produces).
        # If the chosen primary tier doesn't fit (video too short), fall through:
        #   1. primary (factory) at full mv         — clean Δf, most frames needed
        #   2. native src_fps at full mv            — ratio=1, fewest frames
        #   3. native src_fps + mv=5                — reduce views
        #   4. native src_fps + mv=2                — reduce views
        #
        # Invariant: never upsample (fps_ratio >= 1.0). For src_fps < model_fps
        # the factory returns ratio<1 and the clamp below makes ratio=1.
        src_fps = float(meta.get('fps', self.model_fps))
        eff_model_fps, eff_src_fps = resolve_fps_remap(self.model_fps, src_fps)
        primary_ratio = max(1.0, float(eff_src_fps) / float(eff_model_fps)) if eff_src_fps else 1.0

        chosen = None  # (fps_ratio, mv, src_frame_size, eff_model_fps_at_this_level)

        if self.shape_pool_active:
            # shape_pool active: picked (mv, gen) is the per-iter contract. Only
            # try primary + native fps for the picked shape — do NOT walk down
            # the pool to shorter entries (that biased delivery toward shorter
            # shapes when long videos were sparse). If neither tier fits, fall
            # through to handle_static_exhausted with the picked shape preserved.
            pool_levels = [(primary_ratio, eff_model_fps)]
            if primary_ratio > 1.0:
                pool_levels.append((1.0, int(round(src_fps))))
            for fps_ratio, level_eff_model in pool_levels:
                src_frame_size = int(round(static_frame_size * fps_ratio))
                if n_frames_src >= self.mv_size * src_frame_size:
                    chosen = (fps_ratio, self.mv_size, src_frame_size, level_eff_model)
                    break
        else:
            # Original chain (unchanged): primary→native at full mv, then mv
            # reduction (5 / 2) with same pack_size as self.mv_size.
            # (fps_ratio, eff_model_fps_for_this_level, mv) tuples
            fallback_levels = [(primary_ratio, eff_model_fps, self.mv_size)]
            if primary_ratio > 1.0:
                fallback_levels.append((1.0, int(round(src_fps)), self.mv_size))
            my_pack_size = tuple(pack_factory[self.mv_size]['pack_size'])
            for reduced_mv in (5, 2):
                if reduced_mv in pack_factory and tuple(pack_factory[reduced_mv]['pack_size']) == my_pack_size:
                    fallback_levels.append((1.0, int(round(src_fps)), reduced_mv))

            for fps_ratio, level_eff_model, mv in fallback_levels:
                if mv > self.mv_size or mv not in pack_factory:
                    continue
                # Skip remap levels with reduced mv — if remap didn't help at full
                # mv, reducing mv at the same fps_ratio won't help either since the
                # native-fps path always has >= as many source frames.
                if fps_ratio > 1.0 and mv < self.mv_size:
                    continue
                src_frame_size = int(round(static_frame_size * fps_ratio))
                if n_frames_src >= mv * src_frame_size:
                    chosen = (fps_ratio, mv, src_frame_size, level_eff_model)
                    break

        # Single-view long-temporal tier. Two activation paths:
        #   (a) chain succeeded → with prob `long_gen_split`, swap any successful
        #       chosen to mv=1 + long_gen_size. Without this, long videos always
        #       land on the main mv×gen tier and the long-context distribution
        #       never gets trained. The split applies regardless of which mv tier
        #       the chain settled on (was previously gated on chosen[1]==2 only).
        #   (b) chain exhausted but long path still fits → rescue tier.
        # Native fps only (fps_ratio=1.0); a 1-view sample at long_gen_size needs
        # `long_tfs` source frames.
        long_tfs = self.long_gen_size * self.vae_stride_t - 3 if self.long_gen_size > 0 else 0
        long_fits = self.long_gen_size > 0 and n_frames_src >= long_tfs
        use_long = False
        if chosen is None:
            if long_fits and not self.shape_pool_active:
                # Rescue tier (b) for the non-shape_pool path. When shape_pool
                # is active, the picked shape is the contract — bail to
                # handle_static_exhausted rather than substituting (1, long_gen).
                use_long = True
            else:
                return self.handle_static_exhausted(idx, **kwargs)
        elif long_fits and not self.shape_pool_active and random.random() < self.long_gen_split:
            # Post-success swap (a) only fires when shape_pool is inactive.
            # Under shape_pool, (1, long_gen_size) is expected to be a regular
            # pool entry — adding a 50% swap on top double-counts it and
            # starves the other shapes' delivery.
            use_long = True

        if use_long:
            chosen = (1.0, 1, long_tfs, int(round(src_fps)))  # native fps for long path
            total_latent_size = self.long_gen_size
            total_frame_size = long_tfs

        fps_ratio, mv, src_frame_size, level_eff_model = chosen
        # Effective fps seen by the model: the eff_model_fps of whatever tier
        # the chain settled on. For the primary (factory) tier, this is the
        # snapped eff_model (e.g. 15 for source 30/60). For native fallback
        # tiers, this is src_fps (no subsample).
        effective_fps = int(round(level_eff_model)) if fps_ratio > 1.0 else int(round(src_fps))

        # Decide augmentation — same ranges as MVGame's video_augmentation defaults
        disable_aug = random.random() < self.disable_augmentation_ratio or self.overfit
        if disable_aug:
            aug_kwargs = dict(s_min=1.0, s_max=1.0, cx_min=0.0, cx_max=0.0,
                              cy_min=0.0, cy_max=0.0, r_min=0.0, r_max=0.0)
        else:
            # Match video_augmentation() signature defaults (multiview.py:229-235)
            aug_kwargs = dict(s_min=0.65, s_max=1.25,
                              cx_min=-0.1, cx_max=0.1,
                              cy_min=-0.1, cy_max=0.1,
                              r_min=-15.0, r_max=15.0)
        aug_kwargs.update(kwargs)

        # Sample non-overlapping view segments in SOURCE space.
        segments = self.sample_view_segments(n_frames_src, src_frame_size, mv)

        # Build batch. gather_mixed_batch (utils/distributed.py) uses per-rank
        # broadcasts so co-training datasets may carry different keys/shapes;
        # we only need schema consistency within a single dataset, not across.
        batch = {'cpu': {}}
        batch['mv'] = mv
        batch['cpu']['seed'] = int(seed)
        batch['cpu']['prompts'] = meta['caption']
        batch['cpu']['video_path'] = meta['video_path']
        batch['cpu']['dataset_name'] = self.dataset_name
        batch['cpu']['parquet'] = basename(self.data_path)
        # Reproduction info: the per-view (seg_start, seg_end) source-frame
        # segments chosen by sample_view_segments (stars-and-bars over slack),
        # plus the fps remap ratio and augmentation params. Together with
        # video_path + seed, these let a debug tool replay the exact frames.
        batch['cpu']['segments'] = np.asarray(segments, dtype=np.int64)  # (mv, 2)
        # Row + overall source-frame span for the vis meta panel (parity with
        # mvgame/multiview/dynamic). Static builds mv from time-disjoint segments
        # of ONE video, so the panel shows the row plus the min-start / max-end
        # across all segments (the full frame range this sample spans); `segments`
        # above keeps the per-view detail. seg bounds are window-space → + frame_start.
        seg_bounds = np.asarray(segments, dtype=np.int64)
        batch['cpu']['rows'] = np.asarray([int(meta.get('pose_idx', -1))], dtype=np.int64)
        batch['cpu']['start_frames'] = np.asarray([int(seg_bounds[:, 0].min()) + frame_start], dtype=np.int64)
        batch['cpu']['end_frames'] = np.asarray([int(seg_bounds[:, 1].max()) + frame_start], dtype=np.int64)
        batch['cpu']['fps_ratio'] = float(fps_ratio)
        batch['cpu']['aug_kwargs'] = dict(aug_kwargs)
        # fps_cond: effective fps after any subsampling. When using model_fps
        # remap (fps_ratio > 1), effective = model_fps ≈ 24. When using native
        # fps (fallback), effective = src_fps. The network needs this to know
        # the actual temporal spacing of the input frames.
        batch['fps'] = effective_fps

        # Prompt embeddings — all samples must have cached prompt_embeds.
        # Missing cache would cause rank divergence in prepare_batch
        # (one rank hits T5 FSDP, another hits gather_mixed_batch) → NCCL deadlock.
        # H3 re-encodes the raw caption with Qwen3; Wan/T5 caches are incompatible.
        embed = torch.empty((0, 5120), dtype=torch.bfloat16)
        batch['prompt_embeds'] = embed


        # Use the pack layout for the actual mv (may differ from self.mv_size
        # when fallback reduced the view count). mv=2/5/8 all have
        # pack_size=[1,2] so the output frame shape (H×2W) is consistent.
        # When shape_pool is active, force a uniform-rs strip layout so every
        # view stays at full height x width regardless of which (mv, gen) was
        # picked from the pool.
        pack = make_strip_pack(mv) if self.shape_pool_active else pack_factory[mv]
        pack = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in pack.items()}
        pack_size = pack['pack_size']
        ratios = pack['rs']
        xs = pack['xs']
        ys = pack['ys']

        height_pack = int(self.height * pack_size[0])
        width_pack = int(self.width * pack_size[1])
        n_frames_out = total_frame_size
        frames = torch.zeros((n_frames_out, 3, height_pack, width_pack), dtype=torch.float32)
        projs_list = []
        projs_inv_list = []
        Ks_list, Rs_list, Ts_list = [], [], []

        # Use first frame of first view as world reference
        R0, T0 = None, None

        # Sequence-level exposure clamp: static builds all mv views from segments
        # of the SAME video, so compute ONE gamma across the clip and share it
        # (per-segment gamma would break cross-view brightness consistency).
        # No-op for all but the extreme dark/overexposed tails.
        gamma_value = 1.0
        if self.exposure_clamp:
            win_inds = np.linspace(frame_start, frame_start + n_frames_src - 1,
                                   num=min(8, n_frames_src)).astype(np.int64)
            gamma_value = compute_sequence_gamma([vr], win_inds, mv, 1, band=LOOSE_EXPOSURE_BAND)

        # Per-view source frame indices (window-space -> source). Picks
        # total_frame_size frames spaced by fps_ratio inside each [seg_start,
        # seg_end). Mirrors multiview.py:447-452.
        model_idx = np.arange(total_frame_size)
        per_view_inds = [
            (seg_start + frame_start) + np.minimum(
                np.round(model_idx * fps_ratio).astype(np.int64), seg_end - seg_start - 1)
            for (seg_start, seg_end) in segments
        ]

        # Decode ALL views in ONE get_batch instead of one per view. shape_pool
        # static uses a uniform-rs strip pack, so every view shares the same
        # ratio -> a single-ratio batched decode is byte-identical to the
        # per-view decodes, turning mv scattered seeks into one pass. Falls back
        # to per-view decode (frames=None) when ratios differ (non-pool packs).
        batched = None
        if len(ratios) > 1 and bool(np.all(np.asarray(ratios) == ratios[0])):
            rr = max(int(self.height * ratios[0]) / vr.h, int(self.width * ratios[0]) / vr.w)
            batched = vr.get_batch(np.concatenate(per_view_inds).tolist(),
                                   return_channel_first=True, return_tensor=True, ratio=rr)

        for view_idx, ((seg_start, seg_end), ratio, x_off, y_off) in enumerate(
                zip(segments, ratios, xs, ys)):
            target_h = int(self.height * ratio)
            target_w = int(self.width * ratio)
            seg_frame_inds = per_view_inds[view_idx]
            view_frames = (batched[view_idx * total_frame_size:(view_idx + 1) * total_frame_size]
                           if batched is not None else None)

            view_data = self.load_view(
                vr, cameras, seg_frame_inds,
                target_h, target_w,
                R0=R0, T0=T0,
                frame_start=frame_start,
                n_frames_src=n_frames_src,
                gamma_value=gamma_value,
                frames=view_frames,
                **aug_kwargs
            )

            # Lock world reference to first view
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

        # Stack camera params: interleave views per frame (F, mv, ...) -> (F*mv, ...).
        # Each *_list entry is one view's (F, ...) tensor; stack(dim=1) makes the
        # view axis adjacent to frame so the flatten yields view-interleaved
        # (frame-major) order [f0v0, f0v1, ..., f0v(mv-1), f1v0, ...]. Downstream
        # PRoPE (utils/viewpack.py) recovers per-view matrices via stride-mv
        # slicing `p[:, i::mv]`, which assumes exactly this interleaving.
        batch['frames'] = frames
        batch['projs'] = torch.stack(projs_list, dim=1).reshape(-1, 4, 4)
        batch['projs_inv'] = torch.stack(projs_inv_list, dim=1).reshape(-1, 4, 4)
        batch['Ks'] = torch.stack(Ks_list, dim=1).reshape(-1, 3, 3)
        batch['Rs'] = torch.stack(Rs_list, dim=1).reshape(-1, 3, 3)
        batch['Ts'] = torch.stack(Ts_list, dim=1).reshape(-1, 3, 1)

        # Adaptive pose stable factor sized by MAX PAIRWISE camera distance (window
        # diameter), not mean — the far end overflows bf16 PRoPE. batch['Rs']/['Ts'] are
        # now WORLD-LOCKED (v0/f0), consistent with projs + pose_10d; pairwise is
        # translation-invariant so it stays correct.
        centers = -torch.bmm(batch['Rs'].mT, batch['Ts']).squeeze(-1)  # world-locked
        pose_stable_factor, pose_max_t = select_pose_stable_factor(centers, self.pose_stable_factors)
        if pose_stable_factor != 1.0:
            # Scale BOTH projection translations and w2c T. The trainer derives
            # pose_10d camera center C=-R^T T, so both PRoPE streams see the same scale.
            batch['projs'][:, :3, 3] /= pose_stable_factor
            batch['projs_inv'][:, :3, 3] /= pose_stable_factor
            batch['Ts'] /= pose_stable_factor

        batch['cpu']['pack'] = pack
        batch['cpu']['pack']['width'] = self.width
        batch['cpu']['pack']['height'] = self.height
        batch['cpu']['pose_stable_factor'] = pose_stable_factor
        batch['cpu']['pose_max_t'] = pose_max_t  # pre-division max pairwise dist (bf16 diagnostic)

        return batch

    def maybe_pick_shape(self, idx: int):
        """Idx-seeded weighted pick from self.shape_pool.

        Returns (mv, gen) and side-effect-mutates self.mv_size / self.gen_size
        + self.shape_pool_active = True; or returns None and sets the flag
        False if the pool is empty.

        The mutation is per-call (no save/restore): subsequent __getitem__ calls
        will overwrite. The original config-time mv_size / gen_size are used
        only inside __init__ (is_sample_viable, pack assertion, effective_samples
        snapshot) and are never read from self again at runtime — so the
        mutation is safe.
        """
        if not self.shape_pool:
            self.shape_pool_active = False
            return None
        rng = random.Random(f'shape_pool_{idx}')
        weights = self.shape_pool_weights or [1.0] * len(self.shape_pool)
        mv_p, gen_p = rng.choices(self.shape_pool, weights=weights, k=1)[0]
        self.mv_size = int(mv_p)
        self.gen_size = int(gen_p)
        self.shape_pool_active = True
        return self.mv_size, self.gen_size

    def __getitem__(self, idx: int):
        try:
            self.maybe_pick_shape(idx)
            return self.getitem_impl(idx)
        except Exception as e:
            wi = get_worker_info()
            log(red(
                f"[StaticDataset __getitem__] failed: rank={get_rank()} "
                f"worker={wi.id if wi is not None else 0} idx={idx} err={e}"
            ))
            stacktrace()
            raise


@catch_throw
def test_static_dataset():
    from utils.console import log
    from utils.console import blue
    from utils.console import run_parser
    from utils.video import write_video
    from utils.data import export_camera
    from utils.math_utils import affine_inverse
    from utils.parallel import parallel_execution

    args = dotdict(
        data_path='/mnt/bn/foundation-ads3/zhenxu.zx/datasets/svreal/dl3dv10k_yunzhi_filter/dl3dv10k_yunzhi_filter_cached_clean_relative_shape_vipe.parquet',
        gen_size=10,
        mv_size=2,
        height=448,
        width=832,
        num_samples=3,
        per_worker_threads=4,
        output_dir='data/static_test',
    )
    args = run_parser(args, __doc__)

    dataset = StaticDataset(
        data_path=args.data_path,
        gen_size=args.gen_size,
        mv_size=args.mv_size,
        height=args.height,
        width=args.width,
        per_worker_threads=args.per_worker_threads,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    frames_uint8_list = []
    out_path_list = []
    KRTs_list = []

    for i in tqdm(range(args.num_samples), desc='Loading samples'):
        batch = dataset[i]
        frames = batch['frames']  # F, 3, H, W (packed)
        Ks, Rs, Ts = batch['Ks'], batch['Rs'], batch['Ts']
        mv = batch['mv']

        log(f'Sample {i}: frames={frames.shape}, Ks={Ks.shape}, Rs={Rs.shape}, Ts={Ts.shape}, mv={mv}')

        # Sanity checks
        assert frames.shape[0] > 0, "No frames"
        assert Ks.shape[0] == frames.shape[0] * mv, f"Ks shape mismatch: {Ks.shape[0]} vs {frames.shape[0] * mv}"

        # Check intrinsics are reasonable (not zero, not nan)
        assert not torch.isnan(Ks).any(), "NaN in Ks"
        assert not torch.isnan(Rs).any(), "NaN in Rs"
        assert not torch.isnan(Ts).any(), "NaN in Ts"
        assert (Ks[:, 0, 0] > 0).all(), f"Bad fx: {Ks[:, 0, 0]}"
        assert (Ks[:, 1, 1] > 0).all(), f"Bad fy: {Ks[:, 1, 1]}"

        frames_uint8 = (frames.permute(0, 2, 3, 1) * 255).clip(0, 255).to(torch.uint8)
        out_path = join(args.output_dir, f'sample_{i}.mp4')
        frames_uint8_list.append(frames_uint8)
        out_path_list.append(out_path)
        KRTs_list.append([Ks, Rs, Ts])

        # Per-view camera export for visualization
        for v in range(mv):
            Ks_v = Ks[v::mv]  # F, 3, 3
            Rs_v = Rs[v::mv]
            Ts_v = Ts[v::mv]
            c2ws_v = affine_inverse(torch.cat([Rs_v, Ts_v], dim=-1))
            ply_path = join(args.output_dir, f'sample_{i}_view{v}.ply')
            export_camera(c2ws_v, Ks_v, filename=ply_path)
            log(f'  View {v}: K[0]={Ks_v[0].diag()[:2].tolist()}, exported to {blue(ply_path)}')

    def write(filename, frames_uint8, KRTs, **kwargs):
        Ks, Rs, Ts = KRTs
        write_video(filename, frames_uint8, **kwargs)

    parallel_execution(out_path_list, frames_uint8_list, KRTs_list,
                       action=write, desc=f'Writing samples to {blue(args.output_dir)}',
                       print_progress=True, fps=16)


if __name__ == '__main__':
    test_static_dataset()
