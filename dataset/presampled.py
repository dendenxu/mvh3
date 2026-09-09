"""PresampledDataset — consume the self-contained, shape-ordered presampled parquet
(scripts/data/seedpro/build_presampled_parquet.py output) DETERMINISTICALLY.

One parquet row = one complete training SAMPLE (per-view lists + baked pose). The
dataset shards whole SAMPLES (not specs) strided per (rank, worker), so there is NO
train-time random sampling and NO run-out. Pose is baked as a (mv, tfs, 10) per-view
c2w 10-vec (identical layout for every mode), so the pose path is mode-agnostic; only
the frame DECODE differs (window = cover-resize one file; aug = indices walk across the
scene's camera files). Both feed the SAME finish_posed_video (gamma → video_augmentation
→ world-lock → PRoPE projs) so the result matches the existing dataloaders / the captioner.

Per-source augmentation is honored via a source-preset map from the training config
(configs/448.yaml): disable_augmentation_ratio / image_aug / gamma_correction / pre_resize
/ max_fov_h_deg / video_reader — exactly the mvgame s_max=1.0-vs-1.25 etc. traps.
"""
import json
import random
from os.path import basename, dirname, join, normpath

import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from utils.console import log
from utils.console import blue
from utils.console import green
from utils.distributed import get_rank
from utils.distributed import get_world_size
from utils.distributed import is_main_process
from utils.distributed import is_node_main
from utils.misc import set_seed
from dataset.mvgame import finish_posed_video
from dataset.mvgame import pack_factory
from dataset.mvgame import make_strip_pack
from dataset.mvgame import select_pose_stable_factor
from dataset.mvgame import compute_sequence_gamma
from dataset.mvgame import LOOSE_EXPOSURE_BAND
from utils.parallel import parallel_execution
from utils.pose import parse_poses

try:
    from torch.utils.data import get_worker_info
except Exception:  # pragma: no cover
    get_worker_info = lambda: None

POSE_DOF = 10


def resolve_runtime_shape(mv_full, gen_full, view_isolated=False, shape_remap=None,
                          max_spatial_views=0, max_gen_size=0):
    """Resolve memory remaps, then apply ablation caps without shrinking iso batches."""
    mv_full, gen_full = int(mv_full), int(gen_full)
    remap = (shape_remap or {}).get(f'{mv_full}x{gen_full}')
    if isinstance(remap, str) and 'x' in remap:
        mv_s, gen_s = remap.split('x', 1)
        mv, gen = int(mv_s), int(gen_s)
    elif remap is not None:
        mv, gen = mv_full, int(remap)
    else:
        mv, gen = mv_full, gen_full

    if int(max_spatial_views or 0) > 0 and not bool(view_isolated):
        mv = min(mv, int(max_spatial_views))
    if int(max_gen_size or 0) > 0:
        gen = min(gen, int(max_gen_size))

    return max(1, min(mv, mv_full)), max(1, min(gen, gen_full))


def has_malformed_video_paths(mode, mv, video_paths, data_roots):
    """Detect baked paths that cannot identify the sample's requested views."""
    mv = int(mv)
    if mv <= 0:
        return True
    paths = list(video_paths or [])
    roots = list(data_roots or [])
    if len(paths) != mv or len(roots) != mv:
        return True
    if mode == 'aug':
        return any(not isinstance(path, str) or not path.strip() for path in paths)

    for index, raw_path in enumerate(paths[:mv]):
        if not isinstance(raw_path, str) or not raw_path.strip():
            return True
        path = raw_path.strip()
        root = str(roots[index] or '').strip()
        # A missing relative path was historically joined with data_root, turning
        # the source directory itself into a seemingly valid baked video path.
        if root and normpath(path) == normpath(root):
            return True
        if path.endswith(('/', '\\')):
            return True
    return False


def has_complete_embed_paths(paths, n_views, nested=False):
    """Check that every text view has all cached embed paths it will load."""
    n_views = int(n_views)
    values = list(paths or [])
    if n_views <= 0 or len(values) < n_views:
        return False

    def valid(path):
        return isinstance(path, str) and bool(path.strip()) and not path.rstrip().endswith(('/', '\\'))

    if nested:
        return all(
            values[v] and all(valid(path) for path in values[v])
            for v in range(n_views)
        )
    return all(valid(values[v]) for v in range(n_views))


def valid_baked_path_mask(table):
    """Return a row mask while preserving each valid row's physical parquet id."""
    columns = ['mode', 'mv', 'video_path', 'data_root']
    rows = table.select(columns).to_pylist()
    return np.fromiter(
        (not has_malformed_video_paths(
            row['mode'], row['mv'], row['video_path'], row['data_root'])
         for row in rows),
        dtype=np.bool_, count=len(rows),
    )


def c2w_to_w2c_cam(intr_row, c2w, resize_ratio=1.0):
    """One per-frame camera dict {K, R(w2c), T(w2c)} from a parsed c2w + intrinsics
    (fx, fy, cx, cy). K rows 0,1 scaled by resize_ratio (the cover pre-resize), matching
    multiview/static load_view (`cam['K'][:2] *= resize_ratio`)."""
    fx, fy, cx, cy = [float(x) for x in intr_row]
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
    if resize_ratio != 1.0:
        K[:2] *= resize_ratio
    Rc2w = torch.as_tensor(c2w[:3, :3], dtype=torch.float32)
    center = torch.as_tensor(c2w[:3, 3], dtype=torch.float32)
    Rw2c = Rc2w.mT
    Tw2c = (-Rw2c @ center).reshape(3, 1)
    return {'K': K, 'R': Rw2c, 'T': Tw2c}


def baked_cameras(pose_view_flat, tfs, resize_ratio=1.0):
    """Parse one view's baked flat (tfs*10) pose -> list[tfs] of {K, R, T} (w2c)."""
    intr, c2w = parse_poses(np.asarray(pose_view_flat, dtype=np.float64).reshape(-1))  # (tfs,4),(tfs,4,4)
    return [c2w_to_w2c_cam(intr[t], c2w[t], resize_ratio) for t in range(tfs)]


class PresampledDataset(Dataset):
    sampling_weight_power = 0.8  # only used if wrapped in DatasetAggregator; mix already baked

    def __init__(self, spec: str, config=None, height: int = 448, width: int = 832,
                 model_fps: int = 16, vae_stride_t: int = 4, source_config: str = 'configs/worldviews.yaml',
                 num_workers: int = 1, **kwargs):
        self.spec = spec
        self.config = config
        self.height = int(height)
        self.width = int(width)
        self.model_fps = int(model_fps)
        self.vae_stride_t = int(vae_stride_t)
        self.dataset_name = basename(spec)
        self.num_workers = int(num_workers or 1)
        self.exposure_clamp = bool(kwargs.get('exposure_clamp', False))
        # aug-mvgame decode: thread the per-source-camera opens+decodes inside decode_aug
        # (mirrors the live load_constructed_video num_workers=8). The serial per-sample
        # ~100 cold reader-opens + serial get_batch were the aug bottleneck (~30-55s/sample
        # vs the live path's ~5s); see scripts/run/bench_dataset_shapes.py --profile.
        self.aug_decode_workers = int(kwargs.get('aug_decode_workers', 8))
        self._cam_files_cache = {}  # scene_dir -> sorted(listdir(videos)); avoid per-view re-listdir
        psf = kwargs.get('pose_stable_factors', 1.0)
        self.pose_stable_factors = sorted(psf) if isinstance(psf, (list, tuple)) else [float(psf)]

        # Read everything except the heavy `pose` at init (small); pose is loaded
        # lazily per shard in init_loader (like dataset/static.py keeps pose out of init).
        pf = pq.ParquetFile(spec)
        names = pf.schema_arrow.names
        small = [c for c in names if c != 'pose']
        self.table_small = pf.read(columns=small)
        self.raw_n_samples = self.table_small.num_rows
        self.valid_row_mask = valid_baked_path_mask(self.table_small)
        self.invalid_global_rows = np.flatnonzero(~self.valid_row_mask).astype(np.int64)
        self.n_samples = int(self.valid_row_mask.sum())
        if not self.n_samples:
            raise ValueError(f'PresampledDataset has no valid baked video paths: {spec}')
        meta = pf.schema_arrow.metadata or {}
        self.shape_ranges = (json.loads(meta[b'presampled_shape_ranges'])
                             if b'presampled_shape_ranges' in meta else {})
        # Per-source aug knobs: prefer the SELF-CONTAINED baked KV metadata; fall back
        # to reading the training config (legacy / unbaked parquets).
        if b'presampled_source_knobs' in meta:
            self.source_map = json.loads(meta[b'presampled_source_knobs'])
        else:
            self.source_map = self.build_source_map(source_config)

        # Text embeds (Phase 2): present only AFTER cache_presampled_embeds has run. When
        # absent the dataset emits cpu['prompts'] text and the trainer live-encodes (current
        # path, unchanged). chunk_text_prob: for DYNAMIC samples, prob of using per-chunk
        # text (scene + [CHUNK] chunk_c) instead of global scene+motion; STATIC always global.
        self.has_embeds = 'prompt_embeds' in small
        self.embed_base = dirname(spec)
        # OFF by default (base config); pre.yaml sets chunk_text_prob: 0.9 to enable.
        self.chunk_text_prob = float(kwargs.get('chunk_text_prob', 0.0))
        # Per-shape memory remap {"<mv>x<gen>": <target>} for KNOWN-OOM shapes. Per-GPU memory ∝ the
        # visual token count = views(V) × gen(F_lat) × tokens/frame, so V and gen are SYMMETRIC levers.
        # <target> is either an int (cap GEN only, e.g. 4x30->25) or a "<V'>x<G'>" string (cap BOTH
        # views and gen, e.g. 14x10 -> "8x10" drops 14 views to 8). Default {} = off. Applied in getitem.
        sr = kwargs.get('shape_remap', {}) or {}
        self.shape_remap = {str(k): (str(v) if isinstance(v, str) else int(v))
                            for k, v in (sr.items() if isinstance(sr, dict) else {})}
        self.max_spatial_views = int(kwargs.get('max_spatial_views', 0) or 0)
        self.max_gen_size = int(kwargs.get('max_gen_size', 0) or 0)
        if self.max_spatial_views < 0 or self.max_gen_size < 0:
            raise ValueError('max_spatial_views and max_gen_size must be non-negative')
        if self.shape_remap and is_node_main():
            log(f'PresampledDataset shape_remap (memory cap): {self.shape_remap}')
        if (self.max_spatial_views or self.max_gen_size) and is_node_main():
            log('PresampledDataset ablation caps: '
                f'max_spatial_views={self.max_spatial_views or "off"}, '
                f'max_gen_size={self.max_gen_size or "off"}')
        self.prompt_embeds_shape = (json.loads(meta[b'prompt_embeds_shape'])
                                    if b'prompt_embeds_shape' in meta else [512, 4096])
        if is_node_main():
            log(f'PresampledDataset: {green(self.n_samples)} samples from {blue(spec)} '
                f'({len(self.shape_ranges)} shapes; embeds={self.has_embeds}, '
                f'chunk_text_prob={self.chunk_text_prob})')
            if len(self.invalid_global_rows):
                log(f'PresampledDataset excluded {len(self.invalid_global_rows)} malformed '
                    f'baked-path sample(s): global rows {self.invalid_global_rows.tolist()}')
        self.inited = False

    # ─── source-preset map (per-source aug knobs from the training config) ───
    def build_source_map(self, source_config):
        from omegaconf import OmegaConf
        from utils.config import load_config
        cfg = load_config(source_config)
        resolved = OmegaConf.to_container(cfg.dataset, resolve=True)
        shared = {k: v for k, v in resolved.items() if k != 'datasets'}
        entries = [{**shared, **entry} for entry in resolved['datasets']]
        smap = {}
        for e in entries:
            smap[basename(e['data_path'])] = dict(
                type=e.get('type'),
                disable_aug=float(e.get('disable_augmentation_ratio', 1.0)),
                image_aug=bool(e.get('image_aug', False)),
                gamma=bool(e.get('gamma_correction', False)),
                pre_resize=bool(e.get('pre_resize', True)),
                max_fov_h_deg=e.get('max_fov_h_deg'),
                reader=e.get('video_reader', 'torchcodec'),
            )
        return smap

    def source_preset(self, data_parquet):
        return self.source_map.get(basename(data_parquet), dict(
            disable_aug=1.0, image_aug=False, gamma=False, pre_resize=True,
            max_fov_h_deg=None, reader='torchcodec'))

    # ─── sharding / preload (mirrors dataset/static.py shard_meta) ───────────
    def shard_meta(self):
        wi = get_worker_info()
        rank, world = get_rank(), get_world_size()
        wid, nw = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        g_workers = world * nw
        g_id = rank * nw + wid
        sp = getattr(self.config, 'sp_size', 1) if self.config is not None else 1
        if sp and sp != 1:
            g_workers = world // sp
            g_id = rank // sp
        if self.n_samples == 0:
            self.shard_idx = np.array([], dtype=np.int64)
            return self.shard_idx
        # SHAPE-AWARE shard: the build shape-orders the rows, so stride WITHIN each
        # contiguous shape range and concatenate in shape order. Each worker's stream
        # is then grouped by shape (consecutive same-shape samples → far fewer
        # torch.compile recompiles), every sample is assigned to exactly one worker
        # (deterministic, no run-out), and shapes stay balanced across workers. Flat
        # stride fallback when ranges are absent (e.g. a legacy unordered parquet).
        if self.shape_ranges:
            spans = sorted((int(s), int(e)) for s, e in self.shape_ranges.values())
            parts = []
            for start, end in spans:
                raw_ids = np.arange(start, end, dtype=np.int64)
                valid_ids = raw_ids[self.valid_row_mask[start:end]]
                parts.append(valid_ids[g_id::g_workers])
            self.shard_idx = (np.concatenate(parts) if parts else np.array([], dtype=np.int64))
        else:
            valid_ids = np.flatnonzero(self.valid_row_mask).astype(np.int64)
            self.shard_idx = valid_ids[g_id::g_workers]
        return self.shard_idx

    def init_loader(self):
        if self.inited:
            return
        self.shard_meta()
        # Lazily read the baked pose column ONLY for this worker's shard rows.
        self.pose_cache = {}
        if len(self.shard_idx):
            pf = pq.ParquetFile(self.spec)
            want = set(int(i) for i in self.shard_idx)
            md = pf.metadata
            off = 0
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                if any(off <= i < off + n for i in want):
                    col = pf.read_row_group(rg, columns=['pose']).column('pose')
                    for i in range(n):
                        gi = off + i
                        if gi in want:
                            self.pose_cache[gi] = np.asarray(col[i].as_py(), dtype=np.float32)
                off += n
        self.video_readers = {}
        self.inited = True
        if get_rank() == 0 and (get_worker_info() is None or get_worker_info().id == 0):
            log(f'PresampledDataset init_loader: {green(len(self.shard_idx))} shard samples')

    # ─── interface ───────────────────────────────────────────────────────────
    @property
    def n_seqs(self):
        return self.n_samples

    @property
    def effective_samples(self):
        return max(1, self.n_samples)

    def is_sample_viable(self, *a, **k):
        return True

    def __len__(self):
        return max(1, self.n_samples) * 100 * 500  # inflated; sampler wraps via shard

    def get_reader(self, path, reader_name):
        if path not in self.video_readers:
            if str(reader_name).lower() in ('cfr', 'cfrvideoreader'):
                from utils.video import CFRVideoReader as VR
            else:
                from utils.video import TorchCodecVideoReader as VR
            self.video_readers[path] = VR(path)
        return self.video_readers[path]

    def read_row(self, gi):
        """Pull one sample row (small cols + baked pose) as a plain dict."""
        r = self.table_small.slice(gi, 1).to_pylist()[0]
        r['pose'] = self.pose_cache[gi]
        return r

    def load_embed(self, rel):
        """Load one cached T5 embed npz (bf16-as-int16) → [L, D] bfloat16 (same convention
        as dataset/static.py / cache_prompt_embeds)."""
        return torch.empty((0, 5120), dtype=torch.bfloat16)

    def load_chunk_embeds(self, rels):
        return torch.stack([self.load_embed(p) for p in rels])    # [n_chunks, L, D]

    # ─── decode (mode-specific) ──────────────────────────────────────────────
    def decode_window(self, video_path, fs, fe, fps_ratio, tfs, target_h, target_w, pre_resize, reader_name):
        vr = self.get_reader(video_path, reader_name)
        rel = np.round(np.arange(tfs) * fps_ratio).astype(np.int64)
        idx = (fs + np.minimum(rel, max(fe - fs - 1, 0)))
        idx = np.minimum(idx, len(vr) - 1)
        rr = max(target_h / vr.h, target_w / vr.w) if pre_resize else 1.0
        frames = vr.get_batch(idx.tolist(), return_channel_first=False, return_tensor=True, ratio=rr)
        return frames, rr  # (F, h, w, 3)

    def list_cam_files(self, scene_dir):
        """sorted(listdir(scene/videos)), cached per scene_dir (files are immutable).
        decode_aug is called once per output view with the same scene_dir → this avoids
        re-listing the (network-FS) directory mv times per sample."""
        import os
        cf = self._cam_files_cache.get(scene_dir)
        if cf is None:
            cf = sorted(os.listdir(join(scene_dir, 'videos')))
            self._cam_files_cache[scene_dir] = cf
        return cf

    def decode_aug(self, scene_dir, idx_arr_view, reader_name):
        """Gather raw frames for ONE output view along its (src_view, src_frame) walk,
        grouped/batched per source camera (mirrors caption_sample.read_indices_per_view).

        The per-source-camera work (reader open + get_batch) is THREADED via
        parallel_execution — mirrors the live load_constructed_video(num_workers=8).
        Both were serial before, which made aug-mvgame ~7-11x slower than the live path
        (the ~100 cold reader-opens alone were 14-35s/sample); the opens dominate on cold
        scenes. Threading is safe here: within one view every source-cam maps to a DISTINCT
        file path, so get_reader's per-path cache never races on the same key (and views are
        decoded serially by getitem_impl, so no cross-view same-path concurrency either)."""
        cam_files = self.list_cam_files(scene_dir)
        tfs = idx_arr_view.shape[0]
        out = [None] * tfs
        by_cam = {}
        for t in range(tfs):
            sv, sf = int(idx_arr_view[t, 0]), int(idx_arr_view[t, 1])
            by_cam.setdefault(sv, []).append((t, sf))

        svs = list(by_cam.keys())
        items_l = [by_cam[sv] for sv in svs]
        paths = [join(scene_dir, 'videos', cam_files[sv]) for sv in svs]

        def _decode_cam(items, path):
            vr = self.get_reader(path, reader_name)
            n = len(vr)
            uniq = sorted({min(sf, n - 1) for _, sf in items})
            frames = vr.get_batch(uniq, return_channel_first=False, return_tensor=True)
            fmap = dict(zip(uniq, frames))
            return [(t, fmap[min(sf, n - 1)]) for t, sf in items]

        if len(svs) <= 1:
            rets = [_decode_cam(items_l[0], paths[0])] if svs else []
        else:
            rets = parallel_execution(items_l, paths, action=_decode_cam,
                                      num_workers=min(self.aug_decode_workers, len(svs)))
        for chunk in rets:
            for t, fr in chunk:
                out[t] = fr
        return torch.stack(out), 1.0  # (F, H, W, 3)

    # ─── getitem ─────────────────────────────────────────────────────────────
    def getitem_impl(self, idx):
        self.init_loader()
        if not len(self.shard_idx):
            raise RuntimeError(f'PresampledDataset empty shard: {self.spec}')
        local = idx % len(self.shard_idx)
        gi = int(self.shard_idx[local])
        set_seed(idx // len(self.shard_idx))  # epoch seed → reproducible pixel-aug draw
        r = self.read_row(gi)

        mode = r['mode']; mv_full = int(r['mv']); gen_full = int(r['gen'])
        view_iso = bool(r['view_isolated'])
        tfs_full = gen_full * self.vae_stride_t - 3                  # frames as baked in the parquet
        # Shape remap (memory workaround for KNOWN-OOM shapes). Per-GPU memory ∝ VISUAL tokens =
        # views(V) × gen(F_lat) × tokens/frame — V and gen are SYMMETRIC multipliers under SP, so
        # EITHER is an equally-valid lever. Config key "<mv>x<gen>" (the DRAWN shape); value is:
        #   int          -> cap GEN only (frames):        "4x30": 25      (drawn 4x30 runs as 4x25)
        #   "<V'>x<G'>"  -> cap BOTH views and gen:        "14x10": "8x10" (drop 14 views to 8)
        # Only ever REDUCES (V'<=V, G'<=G). Dropping views keeps the FIRST V' (view 0 = cond/world-lock
        # anchor is always kept). The sample then runs as (V', G') and reuses that shape's compiled
        # graph. Pose/frames/aug-idx/cameras/per-chunk-text/prompts all slice to the first (V', G').
        mv, gen = resolve_runtime_shape(
            mv_full, gen_full, view_isolated=view_iso, shape_remap=self.shape_remap,
            max_spatial_views=self.max_spatial_views, max_gen_size=self.max_gen_size)
        tfs = gen * self.vae_stride_t - 3                            # == tfs_full when gen not remapped
        pose = np.asarray(r['pose'], dtype=np.float32).reshape(mv_full, tfs_full, POSE_DOF)[:mv, :tfs]
        src = self.source_preset(r['data_parquet'][0])
        is_aug = (mode == 'aug')
        disable_aug = (not is_aug) and src['disable_aug'] >= 1.0  # mvgame/raw=0.0 → augment
        idx_arr = np.asarray(json.loads(r['indices']))[:, :tfs] if (is_aug and r['indices']) else None  # (mv, tfs, 2), capped

        # pack layout: ALWAYS the full-res strip (make_strip_pack). Every presampled sample is
        # drawn from shape_pool (SHAPE_POOL_448), and shape_pool FORCES a uniform strip pack in
        # the live aggregator: static.py:210-213 / dynamic.py:405 switch the pack lookup to
        # make_strip_pack(mv) when shape_pool_active, and make_sample_specs.build_spec bakes
        # pack_inds=arange(mv) for the same reason. The compact mixed-resolution pack_factory[mv]
        # table (mv=3/5/8 put supporting views at 1/2-1/4 res in a smaller canvas) ONLY applies
        # when shape_pool is OFF. Using it here silently (a) downsampled supporting views to
        # half/quarter res and (b) for aug samples, mismatched the arange pack_inds the indices
        # were constructed against -> scrambled view->slot mapping. Symptom: wandb showed non-iso
        # mv3/5/8 at ~half the live run's per-shape memory (mv3 0.53x, mv5 0.44x, mv8 0.30x).
        # view_iso folds mv->batch afterwards; non-iso lays the strip into one canvas — both want
        # the SAME full-res strip geometry (make_strip_pack == pack_factory for mv 1/2/4 already).
        pack = make_strip_pack(mv)
        rs, xs, ys = pack['rs'], pack['xs'], pack['ys']

        # world-lock anchor = view 0, frame 0 (w2c R0/T0 from baked pose)
        cam00 = baked_cameras(pose[0], tfs)[0]
        R0, T0 = cam00['R'], cam00['T']

        height_pack = int(self.height * pack['pack_size'][0])
        width_pack = int(self.width * pack['pack_size'][1])
        frames_canvas = None
        view_frames, projs, projs_inv, Ks, Rs, Ts = [], [], [], [], [], []
        gamma_values = [1.0] * mv
        if not is_aug and (src['gamma'] or self.exposure_clamp):
            band = None if src['gamma'] else LOOSE_EXPOSURE_BAND

            def view_gamma(i):
                vr = self.get_reader(r['video_path'][i], src['reader'])
                rel = np.round(np.arange(tfs) * float(r['fps_ratio'][i])).astype(np.int64)
                inds = int(r['frame_start'][i]) + np.minimum(
                    rel, max(int(r['frame_end'][i]) - int(r['frame_start'][i]) - 1, 0))
                return vr, np.minimum(inds, len(vr) - 1)

            gamma_inputs = [view_gamma(i) for i in range(mv)]
            if view_iso:
                gamma_values = [compute_sequence_gamma([vr], inds, 1, 1, band=band)
                                for vr, inds in gamma_inputs]
            else:
                vrs = [vr for vr, _ in gamma_inputs]
                shared_gamma = compute_sequence_gamma(
                    vrs, gamma_inputs[0][1], mv, len(vrs), band=band,
                    indices_by_view=[inds for _, inds in gamma_inputs])
                gamma_values = [shared_gamma] * mv
        for i in range(mv):
            target_h, target_w = int(self.height * rs[i]), int(self.width * rs[i])
            if is_aug:
                f_raw, rr = self.decode_aug(r['video_path'][i], idx_arr[i], src['reader'])
            else:
                f_raw, rr = self.decode_window(
                    r['video_path'][i], int(r['frame_start'][i]), int(r['frame_end'][i]),
                    float(r['fps_ratio'][i]), tfs, target_h, target_w, src['pre_resize'], src['reader'])
            cams = baked_cameras(pose[i], tfs, resize_ratio=rr)
            aug_kw = dict(image_aug=(src['image_aug'] and not disable_aug),
                          gamma_value=gamma_values[i])
            if src.get('max_fov_h_deg'):
                aug_kw['max_fov_h_deg'] = src['max_fov_h_deg']
            if disable_aug:
                aug_kw.update(fixed_s=1.0, fixed_cx=0.0, fixed_cy=0.0, fixed_r=0.0)
            elif not is_aug:
                aug_kw.update(s_min=0.65, s_max=1.25)
            bv = finish_posed_video(list(f_raw), cams, self.height, self.width,
                                    ratio=rs[i], R0=R0, T0=T0, **aug_kw)
            view_frames.append(bv['frames'])
            projs.append(bv['projs']); projs_inv.append(bv['projs_inv'])
            Ks.append(bv['Ks']); Rs.append(bv['Rs']); Ts.append(bv['Ts'])

        batch = {'cpu': {}}
        batch['mv'] = 1 if view_iso else mv
        if view_iso:
            batch['frames'] = torch.stack(view_frames, dim=0)            # (mv, F, 3, H, W)
            batch['cpu']['view_as_batch'] = True
            batch['cpu']['orig_mv'] = mv
        else:
            f0 = view_frames[0]
            frames_canvas = torch.zeros((f0.shape[0], 3, height_pack, width_pack), dtype=torch.float32)
            for i, vf in enumerate(view_frames):
                h, w = vf.shape[-2:]
                x, y = int(xs[i] * self.width), int(ys[i] * self.height)
                frames_canvas[:, :, y:y + h, x:x + w] = vf
            batch['frames'] = frames_canvas

        # Stack per-view cameras flat ([F*mv,…], frame-major f0_v0,f0_v1,…,f1_v0,…) for the psf
        # computation (bmm needs [N,3,3]). Emit format then differs by mode (below).
        projs_t = torch.stack(projs, dim=1).reshape(-1, 4, 4)
        projs_inv_t = torch.stack(projs_inv, dim=1).reshape(-1, 4, 4)
        Ks_t = torch.stack(Ks, dim=1).reshape(-1, 3, 3)
        Rs_t = torch.stack(Rs, dim=1).reshape(-1, 3, 3)
        Ts_t = torch.stack(Ts, dim=1).reshape(-1, 3, 1)

        centers = -torch.bmm(Rs_t.mT, Ts_t).squeeze(-1)
        psf, pmax = select_pose_stable_factor(centers, getattr(self, 'pose_stable_factors', [1.0]))
        if psf != 1.0:
            projs_t[:, :3, 3] /= psf
            projs_inv_t[:, :3, 3] /= psf
            Ts_t /= psf

        if view_iso:
            # view_as_batch folds mv→batch: emit a LEADING mv axis [mv, F, …] (matching frames
            # [mv,F,…] and DynamicDataset:594-598) so view_as_batch_collate folds the cameras the
            # SAME way as frames → [B*mv, F, …]. The flat [F*mv,…] non-iso form has NO mv axis to
            # fold → prepare_batch's 4D projs padding gets a 3D tensor ("got 4 and 3").
            F_v = projs[0].shape[0]
            mvF = lambda t, a, b: t.reshape(F_v, mv, a, b).movedim(1, 0).contiguous()  # [F*mv,a,b]→[mv,F,a,b]
            batch['projs'], batch['projs_inv'] = mvF(projs_t, 4, 4), mvF(projs_inv_t, 4, 4)
            batch['Ks'], batch['Rs'], batch['Ts'] = mvF(Ks_t, 3, 3), mvF(Rs_t, 3, 3), mvF(Ts_t, 3, 1)
        else:
            batch['projs'], batch['projs_inv'] = projs_t, projs_inv_t          # [F*mv, 4, 4]
            batch['Ks'], batch['Rs'], batch['Ts'] = Ks_t, Rs_t, Ts_t           # [F*mv, 3, {3,1}]

        batch['fps'] = self.model_fps
        cpu = batch['cpu']
        cpu['seed'] = int(idx // len(self.shard_idx))
        cpu['dataset_name'] = basename(r['data_parquet'][0])
        cpu['mode'] = mode
        cpu['static'] = list(r['static'])[:mv]           # per-view, sliced to V' (folded by collate → must == mv)
        cpu['prompts'] = list(r['caption'])[:mv] if view_iso else r['caption'][0]
        cpu['video_path'] = r['video_path'][0]
        cpu['rows'] = np.asarray(r['row'], dtype=np.int64)[:mv]
        # view_as_batch folds mv→batch, so each batch row is ONE full-resolution view: emit a
        # mv=1 single-tile pack (NOT the mv-strip used to lay out the canvas). Otherwise
        # prepare_pack/unpack_encode_pack would slice mv strip tiles (x=0,W,2W,…) from a single
        # WxH frame → out-of-frame degenerate tiles → VAE conv "(…x2)". Mirrors dynamic.py:601/609.
        cpu['pack'] = {**(pack_factory[1] if view_iso else pack), 'width': self.width, 'height': self.height}
        cpu['pose_stable_factor'] = psf
        cpu['pose_max_t'] = pmax
        # NOTE: mono_iso uses view_as_batch (mv folded to batch → isolation automatic), so we
        # do NOT set view_isolated / per_view_text — the presampled path only ever uses GLOBAL
        # or PER-CHUNK text, never per-view (view_seq_lens unused; USER 2026-06-30).

        # Text embeds (Phase 2): if baked, emit prompt_embeds with the static/dynamic switch
        # (STATIC → global scene+motion; DYNAMIC → prob chunk_text_prob → per-chunk text). A
        # SHARED (non-iso) sample's mv views share ONE prompt ([0]); mono_iso emits per-view
        # (view_as_batch folds mv→batch). random.* is seeded by set_seed(idx//len) above, so the
        # switch is deterministic per idx. No embeds → cpu['prompts'] stays for live-encode.
        # Truncate per-chunk lists to the (possibly remapped) frame budget so nc stays == the video
        # chunk count (gen//chunk_size) — else the model's block-diagonal S_full % nc breaks, and the
        # vis 'C{k}' lines would out-count the fed chunk embeds.
        cap = lambda cl: cl[:max(1, len(cl) * gen // gen_full)]
        use_chunk = False
        if self.has_embeds:
            is_static = all(bool(s) for s in r['static'])
            glob = r.get('prompt_embeds'); chk = r.get('chunk_prompt_embeds')
            text_views = mv if view_iso else 1
            chunk_paths = [cap(chk[v]) for v in range(min(text_views, len(chk or [])))]
            has_global = has_complete_embed_paths(glob, text_views)
            has_chunk = has_complete_embed_paths(chunk_paths, text_views, nested=True)
            if not has_global and not has_chunk:
                raise ValueError(
                    f'Presampled row {gi} has no complete cached text embeds '
                    f'for {text_views} loaded view(s)')
            # Presence-driven text pick (USER 2026-07-06): if one slot is blank, use the OTHER and
            # ignore the dice; if both present, roll chunk_text_prob (dynamic only — static is always
            # global). Short-video split pieces (split_presampled_short.py) blank the global on purpose
            # so their SLICED [CHUNK] chunk text is used; native samples keep both -> unchanged 0.9 roll.
            use_chunk = has_chunk and ((not has_global) or (
                (not is_static) and self.chunk_text_prob > 0 and random.random() < self.chunk_text_prob))
            if use_chunk:
                batch['prompt_embeds'] = (
                    torch.stack([self.load_chunk_embeds(chunk_paths[v]) for v in range(mv)])
                    if view_iso else self.load_chunk_embeds(chunk_paths[0]))  # [mv,nc,L,D] | [nc,L,D]
                cpu['per_chunk_text'] = True
                # Preserve the selected caption policy while replacing T5 caches.
                raw_chunks = [[f"{r['scene'][v]} [CHUNK] {caption}" for caption in cap(list(r['chunks'][v] or []))]
                              for v in range(text_views)]
                cpu['chunk_prompts'] = raw_chunks if view_iso else raw_chunks[0]
            else:
                batch['prompt_embeds'] = (
                    torch.stack([self.load_embed(r['prompt_embeds'][v]) for v in range(mv)])
                    if view_iso else self.load_embed(r['prompt_embeds'][0]))                # [mv,L,D] | [L,D]

        # Caption-strip DISPLAY text (VIS ONLY; independent of the fed `prompts`/embeds — do NOT
        # feed this to T5). Mirrors scripts/data/camera/vis_parquet_poses.py:spec_caption_text so
        # the AR-LB caption panel can render per-chunk 'C{k}:' lines and highlight the active chunk:
        #   PER-CHUNK (use_chunk) -> 'scene: ..' + one 'C{k}: ..' line per (truncated) chunk, nc ==
        #     the fed chunk-embed count (so the panel's active-line index tracks the fed text);
        #   GLOBAL                -> 'scene: ..' + 'motion: ..'.
        # Fold rule matches `prompts`: per-view list under view_iso (len==mv → folded sample-major),
        # a single string otherwise.
        def _disp(v):
            sc = str((r['scene'][v] if v < len(r['scene']) else '') or '')
            if use_chunk:
                ch = cap(list(r['chunks'][v] or []))
                return '\n'.join([f'scene: {sc}'] + [f'C{k}: {c}' for k, c in enumerate(ch)])
            mo = str((r['motion'][v] if v < len(r['motion']) else '') or '')
            if not sc and not mo:  # scene/motion not split for this source -> raw fed caption
                return str((r['caption'][v] if v < len(r['caption']) else '') or '')
            return f'scene: {sc}\nmotion: {mo}'
        cpu['caption_display'] = [_disp(v) for v in range(mv)] if view_iso else _disp(0)
        return batch

    def __getitem__(self, idx):
        return self.getitem_impl(idx)

    @staticmethod
    def collate_fn(samples):
        # mono_iso samples set cpu['view_as_batch'] → fold the mv axis into batch
        # (identical rule to DatasetAggregator); window/aug samples pass through.
        from dataset.aggregator import view_as_batch_collate
        return view_as_batch_collate(samples)
