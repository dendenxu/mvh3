"""
Implements multi-view packing and unpacking

Disabling torch compile for now for ease of debugging
"""

from typing import List, Dict, Callable, Optional
from collections.abc import MutableMapping

import torch
import torch.nn as nn
import numpy as np


class OpaqueDict(MutableMapping):
    def __init__(self, data):
        self.data = data

    # Required by MutableMapping
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    # Optional: Override repr for cleaner output
    def __repr__(self) -> str:
        return f"OpaqueDict({self.data})"

    def get(self, key, default=None):
        """
        Return the value for key if key is in the dictionary, else default.
        """
        return self.data.get(key, default)


def damp_context(context_latent: torch.Tensor, context_noise: float = 0.0, context_scale: float = 1.0, num_train_timesteps: int = 1000, pack: Optional[Dict] = None, context_noise_std: float = 0.0):
    """Damp clean context latents with noise + return the matching timestep map.

    Jitter is applied here — and only here — so callers must NOT pre-jitter
    `context_noise` upstream (would cause double-jitter, std² + std² variance).

    Two regimes:
      mv > 1 && std > 0  → per-view spatial: each view's tile gets its own
                           Normal(context_noise, std) draw, and `context_t`
                           is returned as a 4D spatial map at LATENT grid
                           so the model receives the actual per-view noise
                           level via time embedding (no scalar approximation).
      else               → batch-level scalar: one Normal(context_noise, std)
                           draw (or no jitter if std==0) applied uniformly.
                           `context_t` is returned as 2D [B, F_ctx] — same as
                           the legacy interface so inference / non-MV paths
                           stay identical.

    Per-view jitter is SP-broadcast so all SP ranks agree (collective shapes).
    Batch-level jitter likewise.
    """
    batch_size, context_size, channel, height, width = context_latent.shape
    device, dtype = context_latent.device, context_latent.dtype

    mv = int(pack['mv']) if pack is not None else 1
    use_per_view = mv > 1 and context_noise_std > 0

    if use_per_view:
        from utils.distributed import broadcast_scoped
        # Per-view jitter ~ Normal(0, 1), SP-synced so all SP ranks agree.
        # The resulting noise level is a FRACTION in [0, 1] (hence clamp): it
        # doubles as the linear clean↔noise mixing coefficient below AND, scaled
        # by num_train_timesteps, as the flow-matching timestep fed to the model.
        jitter = broadcast_scoped(torch.randn(mv, device=device), scope='sp')
        per_view = (context_noise + jitter * context_noise_std).clamp_(0.0, 1.0)  # [mv]

        # Build [H, W] per-view noise-level map at LATENT grid (pack xs/ys/ws/hs
        # are pixel-space, latent /8). Used both for the noise mixing coefficient
        # AND for the timestep map (latent-grid → model downsamples to patch grid).
        xs, ys, ws, hs = pack['xs'], pack['ys'], pack['ws'], pack['hs']
        level_map = torch.full((height, width), context_noise, device=device, dtype=torch.float32)
        for i in range(mv):
            x8, y8 = xs[i] // 8, ys[i] // 8
            w8, h8 = ws[i] // 8, hs[i] // 8
            level_map[y8:y8 + h8, x8:x8 + w8] = per_view[i]

        if context_noise > 0 or context_noise_std > 0:
            noise = torch.randn_like(context_latent)  # [B, F_ctx, C, H, W]
            coef = level_map.to(dtype).view(1, 1, 1, height, width)
            context_latent = context_latent * (1 - coef) + noise * coef

        # 4D spatial timestep at latent grid; frame-broadcast over F_ctx.
        # Model downsamples to patch grid (latent /2) and shuffles to match
        # x in view-isolated mode (see ARLBWanModel.forward).
        context_t = (level_map * num_train_timesteps).view(1, 1, height, width)
        context_t = context_t.expand(batch_size, context_size, height, width).contiguous()
    else:
        # Scalar regime (mv=1 or std=0). Single SP-broadcast jitter so a single-view
        # sample still sees noise-level spread when std>0.
        eff_noise = context_noise
        if context_noise_std > 0:
            from utils.distributed import broadcast_scoped
            n = broadcast_scoped(torch.randn(1, device=device), scope='sp')
            eff_noise = max(0.0, min(1.0, context_noise + n.item() * context_noise_std))

        if eff_noise > 0:
            noise = torch.randn_like(context_latent)
            context_latent = context_latent * (1 - eff_noise) + noise * eff_noise

        context_t = torch.full((batch_size, context_size), eff_noise * num_train_timesteps,
                               device=device, dtype=torch.float32)

    if context_scale != 1.0:
        context_latent = context_latent * context_scale

    return context_latent, context_t


def staircase_sf_context(
    prev_context: torch.Tensor,
    history_pred: torch.Tensor,
    sf_depth: int,
    chunk_size: int,
) -> torch.Tensor:
    """Build the next resampling-forcing context as a per-chunk depth staircase.

    Plain self-forcing replaces the WHOLE history with a single self-prediction
    depth, but inference compounds error UNEVENLY across context chunks: counting
    the chunk-0 image anchor as clean, context chunk c carries only ~c rounds of
    self-prediction by the time it is attended to. This builds that gradient.

    Each SF round (``sf_depth`` = the depth of the step that just ran, 0 for a
    fresh step) finalizes exactly one more chunk::

        O_{d+1} = concat( prev_context[:, :(d+1)*chunk_size],   # chunks 0..d frozen
                          history_pred[:, (d+1)*chunk_size:] )   # rest bumped to d+1

    where ``prev_context`` is the (un-damped) context the just-finished step
    consumed — clean GT for a fresh step, the prior staircase for an SF step.
    Unrolling from O_0 = clean GT gives context chunk c a self-prediction depth
    of ``min(c, depth_reached)`` with chunk 0 forever clean, so chunk 1 is forced
    at most once, chunk 2 at most twice, ... matching inference's per-chunk error
    accumulation. Once the boundary reaches the context length every chunk has hit
    its cap and the staircase is the steady-state (deep-context) distribution.

    Args:
        prev_context:  [B, context_size, ...] un-damped context the step consumed.
        history_pred:  [B, context_size, ...] that step's x0 prediction over the
                       history region.
        sf_depth:      depth of the step that just ran (0 = fresh).
        chunk_size:    latent frames per AR chunk.

    Returns:
        [B, context_size, ...] next step's self-forcing context.
    """
    context_size = prev_context.shape[1]
    boundary = min((sf_depth + 1) * chunk_size, context_size)
    return torch.cat([prev_context[:, :boundary], history_pred[:, boundary:]], dim=1)


def ramp_noise_level(
    history_len: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    sigma_anchor: float = 0.0,   # First frame: Absolutely clean (Anchor)
    sigma_base: float = 0.05,    # Middle history: Tiny noise (Long-term memory)
    sigma_peak: float = 0.15,    # Recent history: Moderate noise (Prevent copying)
    ramp_len: int = 5            # Number of frames for the noise ramp-up
) -> torch.Tensor:
    """
    Generates noise standard deviations following a "Hockey Stick" curve.
    Shape:
    [Anchor (0.0), Base (0.05)..., Base (0.05), Ramp_Start, ..., Peak (0.15)]
    """
    # 1. Initialize everything with the Base noise level
    sigmas = torch.full((history_len,), sigma_base, device=device, dtype=dtype)
    # 2. Handle the Anchor (The very first frame)
    if history_len > 0:
        sigmas[0] = sigma_anchor
    # 3. Handle the Ramp (The most recent K frames)
    # We only ramp up if we have history beyond the anchor
    # The ramp length cannot exceed available history minus the anchor
    actual_ramp_len = min(history_len - 1, ramp_len)
    if actual_ramp_len > 0:
        # Generate a linear ramp from Base to Peak
        # steps = actual_ramp_len + 1 because linspace includes the start point (Base)
        # which we effectively want to transition FROM.
        ramp_values = torch.linspace(sigma_base, sigma_peak, steps=actual_ramp_len + 1, device=device)
        # Assign to the last N frames
        # We slice [1:] to exclude the starting 'Base' value to ensure smooth transition
        sigmas[-actual_ramp_len:] = ramp_values[1:]
    return sigmas


def prepare_pack(pack: Dict[str, torch.Tensor]) -> Dict[str, List[int]]:
    """
    Unified preprocessing function to extract metadata from the pack dictionary.
    Returns a dictionary of Python-native types (lists, ints, floats) compatible with torch.compile.
    Call this once at the start of each batch to avoid repetitive CPU-GPU syncs.

    Coordinate convention (see PackFactory in dataset/mvgame.py): the raw
    pack `xs`/`ys` are fractions of the PRIMARY view's pixel size (e.g.
    xs=[0.0, 1.0, 1.5, ...] means "0, 1×, 1.5× the primary view's width"),
    and `rs` is each view's resolution ratio vs the primary view. So
    multiplying by (w, h) — the primary view's pixel dims — turns them into
    absolute pixel top-left coords (xs, ys) and pixel view sizes (ws, hs).
    Forcing the lists to Python ints/floats here keeps them as compile-time
    constants (not GPU SymInts) for every downstream torch.compile'd helper.
    """
    # 1. Extract base grid dimensions and core parameters
    mv = int(pack['mv'])
    h = int(pack['height'])
    w = int(pack['width'])

    # 2. Prepare coordinate and ratio lists as standard Python types.
    #    xs/ys: pixel top-left of each view's tile; ws/hs: pixel w/h of each
    #    tile (= ratio × primary-view pixel size). One .cpu() sync per list.
    xs = (pack['xs'].cpu() * w).int().tolist()
    ys = (pack['ys'].cpu() * h).int().tolist()
    ws = (pack['rs'].cpu() * w).int().tolist()
    hs = (pack['rs'].cpu() * h).int().tolist()

    rs = pack['rs'].cpu().tolist()  # per-view resolution ratios (e.g., [1.0, 0.5, 0.5, 0.5, 0.25, ...])

    return {
        'mv': mv,
        'xs': xs, 'ys': ys, 'ws': ws, 'hs': hs,
        'rs': rs,
    }


def unpack_encode_pack(
    frames: torch.Tensor,
    vae: nn.Module,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    mv: int,
    **kwargs
):
    # Unpack, encode and repack
    # Note: HARDCODED VAE OUTPUT SIZES
    # hd/wd = 8: WanVAE spatial downsample stride (pixel → latent grid).
    # te/td = 1/4: WanVAE temporal layout — the first frame is encoded on its
    #   own (te=1) and every following td=4 pixel frames collapse to 1 latent
    #   frame, so F pixel frames → (F-1)//4 + 1 latent frames.
    hd, wd = 8, 8
    te, td = 1, 4

    device, dtype = frames.device, frames.dtype
    B, F, C, HP, WP = frames.shape  # packed sizes (HP/WP = full packed canvas)
    F_1444 = (F - te) // td + te    # latent frame count (the "1,4,4,4..." cadence)
    HP_8 = HP // hd
    WP_8 = WP // wd

    latent = torch.empty((B, F_1444, 16, HP_8, WP_8), device=device, dtype=dtype)
    # Encode each view's tile separately: a single VAE pass over the whole
    # packed canvas would let conv receptive fields bleed across tile borders
    # (different cameras), corrupting the latents near seams.
    # for x, y, w, h in zip(xs, ys, ws, hs):
    for i in range(mv):
        x, y, w, h = xs[i], ys[i], ws[i], hs[i]
        x_8, y_8, w_8, h_8 = x // wd, y // hd, w // wd, h // hd
        frames_v = frames[:, :, :, y:y + h, x:x + w]
        # Crossing vae boundary: pixel-space crop (x,y,w,h) maps to latent-space
        # crop (x_8,y_8,w_8,h_8) — pack tiles are multiples of 8 so this is exact.
        latent[:, :, :, y_8:y_8 + h_8, x_8:x_8 + w_8] = vae(frames_v)

    return latent


def unpack_decode_pack(
    latent: torch.Tensor,
    vae: nn.Module,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    mv: int,
    **kwargs
):
    # Unpack, decode and repack
    # Note: HARDCODED VAE OUTPUT SIZES (see unpack_encode_pack for the meaning
    # of hd/wd=8 spatial stride and te/td=1/4 temporal cadence). This is the
    # exact inverse: latent grid → pixel canvas, decoded one view tile at a time
    # to avoid cross-tile conv bleed at the seams.
    hd, wd = 8, 8
    te, td = 1, 4

    device, dtype = latent.device, latent.dtype
    B, F_1444, dim, HP_8, WP_8 = latent.shape  # packed sizes (latent grid)
    F = (F_1444 - te) * td + te                # inverse of the latent-frame count formula
    HP, WP = HP_8 * hd, WP_8 * wd

    frames = torch.empty((B, F, 3, HP, WP), device=device, dtype=dtype)
    # for x, y, w, h in zip(xs, ys, ws, hs):
    for i in range(mv):
        x, y, w, h = xs[i], ys[i], ws[i], hs[i]
        x_8, y_8, w_8, h_8 = x // wd, y // hd, w // wd, h // hd
        # Crossing vae boundary
        latent_v = latent[:, :, :, y_8:y_8 + h_8, x_8:x_8 + w_8]
        frames[:, :, :, y:y + h, x:x + w] = vae(latent_v, decode=True)

    return frames


def unpack_decode_pack_to_cpu_uint8(
    latent: torch.Tensor,
    vae: nn.Module,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    mv: int,
    **kwargs,
) -> np.ndarray:
    """Decode packed latent view-by-view, offloading each view's pixels to CPU
    immediately as channel-last uint8.

    Functionally equivalent to ``unpack_decode_pack`` followed by the
    fp32 → uint8 → CPU → numpy conversion in
    ``add_labels_and_convert_to_np``, but with much lower GPU peak:

    Legacy path peak (per call):
        full unpacked fp32 [B,F,3,HP,WP]   (allocated up-front)
        + current view's fp32 decode output
        + VAE-internal cat-grow buffer
      → ~3× full-pixel-fp32 transient

    This path peak (per call):
        current view's fp32 decode output
        + current view's uint8 channel-last copy
      → ~2× per-view-pixel; the persistent buffer is on CPU and uint8.

    Returns numpy ``[B, F_pix, HP, WP, 3] uint8`` (channel-last) ready for cv2 /
    PIL overlays without further conversion.
    """
    hd, wd = 8, 8
    te, td = 1, 4

    B, F_1444, _, HP_8, WP_8 = latent.shape
    F = (F_1444 - te) * td + te
    HP, WP = HP_8 * hd, WP_8 * wd

    # CPU buffer in the channel-last uint8 layout the downstream label /
    # pose-panel code already expects. Pinned for faster d2h copies; falls back
    # silently to non-pinned if pinning is unavailable in the host environment.
    try:
        frames_cpu = torch.empty((B, F, HP, WP, 3), dtype=torch.uint8, pin_memory=True)
    except RuntimeError:
        frames_cpu = torch.empty((B, F, HP, WP, 3), dtype=torch.uint8)

    for i in range(mv):
        x, y, w, h = xs[i], ys[i], ws[i], hs[i]
        x_8, y_8, w_8, h_8 = x // wd, y // hd, w // wd, h // hd

        latent_v = latent[:, :, :, y_8:y_8 + h_8, x_8:x_8 + w_8]
        # vae(latent_v, decode=True) returns fp32 in [0, 1] range, [B, F, 3, h, w].
        pixels_v = vae(latent_v, decode=True)
        pixels_v.clamp_(0, 1).mul_(255)
        # Cast fp32 → uint8 (allocates a 4× smaller tensor; fp32 is freed once
        # we drop the pixels_v reference). Then permute channel-last + contig
        # so the d2h copy lands in row-major order matching frames_cpu.
        pixels_v_u8 = pixels_v.to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()
        del pixels_v

        # Per-view d2h transfer. Slice-assign goes through copy_(); when the
        # CPU target is pinned this can overlap with the next iteration's
        # decode kernel. We sync at the end before handing back to numpy.
        frames_cpu[:, :, y:y + h, x:x + w].copy_(pixels_v_u8, non_blocking=True)
        del pixels_v_u8

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return frames_cpu.numpy()


def unpack_apply_pack(
    x: torch.Tensor,
    r: torch.Tensor,
    apply_fn: Callable,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    mv: int, f: int, h: int, w: int,
    p: Optional[torch.Tensor] = None,
    **kwargs
):
    # Note: HARDCODED VAE OUTPUT SIZES
    # p: b, f, 4, 4: f / 4 / 8 = number of latents
    # r: b, s, 1, d/2 (Complex)
    # x: b, s, n, d
    #
    # Stride is 16 here (NOT 8 as in unpack_encode_pack): x/r are token-grid
    # tensors, already downsampled by the VAE (/8) AND the model's 2×2 patch
    # embedding (/2) → /16 from pixels. So pixel coords (xs/ys/ws/hs) // 16
    # give patch-grid coords. h/w passed in are likewise patch-grid dims.
    b, s, n, d = x.shape  # sequence
    d_2 = d // 2
    hd, wd = 16, 16

    # Using fhw directly as ints
    x_grid = x.view(b, -1, h, w, n, d)
    if r.ndim == 4:
        r_grid = r.view(b, -1, h, w, 1, d_2)
    else:
        r_grid = r.view(b, mv, -1, h, w, 1, d_2)
    o = torch.empty_like(x_grid)  # will fill this one by one

    # Loop over unbound CPU scalars
    for i in range(mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd
        # seq_len = f * h_16 * w_16

        x_v = x_grid[:, :, y_16:y_16 + h_16, x_16:x_16 + w_16].reshape(b, -1, n, d)  # reshape to sequence
        # RoPE freqs are sliced from the grid ORIGIN (:h_16, :w_16), not from
        # the view's tile offset (y_16, x_16). So each view's tokens get the
        # SAME positional grid as if it sat at the canvas top-left — RoPE is
        # per-view-local (position relative to the view's own corner), which is
        # what we want since each tile is an independent image. r.ndim==4 means
        # one shared freq grid for all views; ndim==5 means per-view freqs
        # (extra `mv` axis), indexed by [:, i].
        if r.ndim == 4:
            r_v = r_grid[:, :, :h_16, :w_16].reshape(b, -1, 1, d_2)  # rope freqs from grid origin (per-view-local)
        else:
            r_v = r_grid[:, i, :, :h_16, :w_16].reshape(b, -1, 1, d_2)  # rope freqs from grid origin (per-view-local)
        # j_r = hs[0] // h_val  # rope params jump ratio
        # if r.ndim == 4:
        #     r_v = r_grid[:, :, :h_16 * j_r:j_r, :w_16 * j_r:j_r].reshape(b, -1, 1, d_2)  # using rope freqs from the start of the sequence to match the image shapes
        # else:
        #     r_v = r_grid[:, i, :, :h_16 * j_r:j_r, :w_16 * j_r:j_r].reshape(b, -1, 1, d_2)  # using rope freqs from the start of the sequence to match the image shapes

        if p is not None:
            # p is ordered [..., frame, view] flattened on dim 1 as
            # (f0_v0, f0_v1, ..., f0_v{mv-1}, f1_v0, ...). Stride-mv from offset
            # i picks exactly view i's projection matrix at every frame.
            x_v = apply_fn(x_v, r_v, p[:, i::mv])  # select this view's projection matrices
        else:
            x_v = apply_fn(x_v, r_v)  # apply rope with unpacking too
        o[:, :, y_16:y_16 + h_16, x_16:x_16 + w_16] = x_v.reshape(b, -1, h_16, w_16, n, d)
    return o.view(b, s, n, d)


def get_viewwise_rope(  # BUG: this function will make compiler bug out
    r: torch.Tensor,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """Overwrite each non-primary view's RoPE-freq tile with the freqs from the
    grid origin, so every view gets per-view-local positional encoding (its own
    corner = position 0) rather than its absolute canvas position. View 0 starts
    at (0,0) with full resolution, so its tile already equals the source slice
    (:h_16,:w_16) — the loop starts at 1 to skip that no-op self-copy."""
    # r: b, s, 1, d/2 (Complex)
    b, s = r.shape[:2]  # sequence
    d_2 = r.shape[-1]

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    nc = f // c  # number of chunks
    hd, wd = 16, 16  # VAE(/8) × patch(/2): pixel coords // 16 → patch-grid coords

    # Using fhw directly as ints
    r_grid = r.view(b, nc, c, h, w, 1, d_2)

    # Loop over unbound CPU scalars
    for i in range(1, mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd
        # Copy origin freqs into view i's tile. Source is sliced to the view's
        # own (h_16, w_16) extent, so a half/quarter-res view gets exactly the
        # freqs of an image of that size starting at position 0.
        r_grid[:, :, :, y_16:y_16 + h_16, x_16:x_16 + w_16] = r_grid[:, :, :, :h_16, :w_16]  # using rope freqs from the start of the sequence to match the image shapes
        # j_r = hs[0] // h_val  # rope params jump ratio
        # r_grid[:, :, :, y_16:y_16 + h_16, x_16:x_16 + w_16] = r_grid[:, :, :, :h_16 * j_r:j_r, :w_16 * j_r:j_r]  # using rope freqs from the start of the sequence to match the image shapes

    return r_grid.view(r.shape)


def get_viewwise_prope(  # BUG: this function will make compiler bug out
    p: torch.Tensor,  # b, f8, n, d
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """Broadcast each view's per-frame camera matrix (a 4x4 projection or a 3x3
    rotation, used by PRoPE) over its spatial tile in the packed patch grid.
    Unlike RoPE, PRoPE is the
    SAME for every token of a view at a given frame (the camera is per-view,
    not per-pixel), so each view's matrix `p[:, i::mv]` is replicated across
    that view's h_16×w_16 region. All mv views are written (loop from 0) — there
    is no origin-broadcast shortcut as in get_viewwise_rope."""
    b, f8, n, d, d = p.shape  # sequence
    b, f8, n, d, d = int(b), int(f8), int(n), int(d), int(d)  # type hint the compiler
    p = p.view(b, f8, n, d, d)

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    hd, wd = 16, 16  # VAE(/8) × patch(/2): pixel coords // 16 → patch-grid coords

    # Using fhw directly as ints
    # TODO: Use float32 for camera projection injection to avoid large numbers
    po = p.new_empty(b, f, h, w, n, d, d)

    # Loop over unbound CPU scalars
    for i in range(0, mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd
        # p[:, i::mv] = view i's matrices over all frames (ordered [frame, view]);
        # [None, None] inserts the h,w axes so it broadcasts to the whole tile.
        po[:, :, y_16:y_16 + h_16, x_16:x_16 + w_16] = p[:, i::mv, None, None]  # bfhwndd <- bf11ndd

    return po.view(b, -1, n, d, d)


@torch.compile
def varlen_shuffle_x_and_rope(
    x: torch.Tensor,
    r: torch.Tensor,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """Same chunk-major per-view reshuffle as `varlen_shuffle` (see its
    docstring for the output layout / rationale), but also reorders the RoPE
    freqs `r` the same way so they stay aligned with the shuffled tokens.
    Each view's freqs are sliced from the grid origin (:h_16, :w_16) so RoPE
    stays per-view-local."""
    # Shuffle the packed views
    # x: B, S, D
    # b, c, f, h, w2, n, d
    # We want to arrange them to
    # b, c, f, v, h, w, n, d and then pack it up afterwards
    # So we don't need to change the chunk-wise causal attn mask for controlnet

    # r: b, s, 1, d/2 (Complex)
    # x: b, s, nd
    b, s, nd = x.shape  # sequence
    d_2 = r.shape[-1]

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    nc = f // c  # number of chunks
    sc = c * h * w  # sequence per chunk
    hd, wd = 16, 16

    # Using fhw directly as ints
    x_grid = x.view(b, nc, c, h, w, nd)
    if r.ndim == 4:
        r_grid = r.view(b, nc, c, h, w, 1, d_2)
    else:
        r_grid = r.view(b, mv, nc, c, h, w, 1, d_2)

    # Output buffer for the actual sequence
    x_parts = []
    # Output buffer for the rope freqs
    r_parts = []
    seq_lens = []

    # Loop over unbound CPU scalars
    for i in range(mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd
        x_v = x_grid[:, :, :, y_16:y_16 + h_16, x_16:x_16 + w_16]  # reshape to sequence
        if r.ndim == 4:
            r_v = r_grid[:, :, :, :h_16, :w_16]  # using rope freqs from the start of the sequence to match the image shapes
        else:
            r_v = r_grid[:, i, :, :, :h_16, :w_16]  # using rope freqs from the start of the sequence to match the image shapes

        seq_len = c * h_16 * w_16
        x_parts.append(x_v.reshape(b, nc, seq_len, nd))
        r_parts.append(r_v.reshape(b, nc, seq_len, 1, d_2))
        seq_lens.append(seq_len)

    # Concatenate along sequence dimension
    xo = torch.cat(x_parts, dim=2)
    ro = torch.cat(r_parts, dim=2)

    xo = xo.view(b, s, nd)
    ro = ro.view(b, s, 1, d_2)

    # seq_lens = torch.as_tensor(seq_lens, dtype=torch.int32).to(device, non_blocking=True)
    # Use it as a cpu tensor for now
    # seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=x.device)
    # seq_lens = torch.as_tensor([sum(seq_lens)], dtype=torch.int32)  # chunk-wise causal
    return xo, ro, seq_lens


@torch.compile
def varlen_shuffle(
    x: torch.Tensor,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """Reorder packed multi-view tokens from spatial-canvas order to
    chunk-major per-view-segment order.

    Input `x` is laid out as the packed 2D canvas per frame (all views tiled
    together: row-major over the [h, w] patch grid). This regroups it so that
    each (chunk × view) tile becomes a contiguous run, with the views of a
    chunk laid out back to back before moving to the next chunk:

        [c0_v0, c0_v1, ..., c0_v{mv-1}, c1_v0, ..., c1_v{mv-1}, ...]

    Why: with this layout the per-chunk token count is sum(seq_lens) and the
    SAME chunk-causal attention mask works unchanged across views — no need to
    rebuild a custom mask. The returned `seq_lens` (per-view token counts
    c*h_16*w_16) also let flash attention treat each view as an isolated
    varlen segment when desired. `varlen_restore` is the exact inverse.
    """
    # Shuffle the packed views
    # x: B, S, D
    # b, c, f, h, w2, n, d
    # We want to arrange them to
    # b, c, f, v, h, w, n, d and then pack it up afterwards
    # So we don't need to change the chunk-wise causal attn mask for controlnet

    # x: b, s, nd
    (b, s), nd = x.shape[:2], x.shape[2:]  # sequence

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    nc = f // c  # number of chunks
    hd, wd = 16, 16

    # Using fhw directly as ints
    x_grid = x.view(b, nc, c, h, w, *nd)

    # Output buffer for the actual sequence
    x_parts = []
    seq_lens = []

    # Loop over unbound CPU scalars
    for i in range(mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd
        x_v = x_grid[:, :, :, y_16:y_16 + h_16, x_16:x_16 + w_16]  # reshape to sequence
        seq_len = c * h_16 * w_16
        x_parts.append(x_v.reshape(b, nc, seq_len, *nd))
        seq_lens.append(seq_len)

    # Concatenate along sequence dimension
    xo = torch.cat(x_parts, dim=2)
    xo = xo.view(b, s, *nd)

    return xo, seq_lens


@torch.compile
def varlen_restore(
    x: torch.Tensor,
    xs: List[int], ys: List[int], ws: List[int], hs: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """Inverse of `varlen_shuffle`: scatter chunk-major per-view segments back
    into the packed spatial canvas. `cu_seq_len` walks the concatenated
    segments in the SAME view order the shuffle used, so the loop must iterate
    views identically (xs/ys/ws/hs unchanged) for the round-trip to be exact."""
    # This is basically the reverse process of the varlen_shuffle function

    # Shuffle the packed views
    device, dtype = x.device, x.dtype

    # x: b, s, nd
    (b, s), nd = x.shape[:2], x.shape[2:]  # sequence

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    nc = f // c  # number of chunks
    sc = c * h * w  # sequence length in each chunk

    hd, wd = 16, 16

    # Using fhw directly as ints
    x_in = x.view(b, nc, sc, *nd)

    # Output buffer for the actual sequence
    xo_grid = torch.empty((b, nc, c, h, w, *nd), device=device, dtype=dtype)

    cu_seq_len = 0
    # Loop over unbound CPU scalars
    for i in range(mv):
        x_val, y_val, w_val, h_val = xs[i], ys[i], ws[i], hs[i]
        x_16, y_16, w_16, h_16 = x_val // wd, y_val // hd, w_val // wd, h_val // hd

        seq_len = c * h_16 * w_16
        x_v = x_in[:, :, cu_seq_len:cu_seq_len + seq_len].reshape(b, nc, c, h_16, w_16, *nd)
        xo_grid[:, :, :, y_16:y_16 + h_16, x_16:x_16 + w_16] = x_v
        cu_seq_len += seq_len

    return xo_grid.view(b, s, *nd)


def varlen_apply(
    x: torch.Tensor,
    r: torch.Tensor,
    apply_fn: Callable,
    p: torch.Tensor,
    hs: List[int], ws: List[int],
    f: int, h: int, w: int, c: int, mv: int,
    **kwargs
):
    """
    x: b, s, n, d
    r: b, s, 1, d/2 (Complex)
    p: b, mv * 4 * f, 4, 4

    x and r already converted to view major format
    b, nc * c * h * w, n, d -> b, nc, (c*h0*w0)+(c*h1*w1)+(c*h2*w2)+..., n, d

    Note that if you don't pass in the projection matrices, there's no need for this shifting, just call apply_fn directly
    """
    # r: b, s, 1, d/2 (Complex)
    # x: b, s, n， d
    b, s, n, d = x.shape
    d_2 = r.shape[-1]

    # Use int() to ensure Dynamo treats these as SymInts/ints for view(), avoiding _local_scalar_dense errors
    nc = f // c  # number of chunks
    sc = c * h * w  # sequence length in each chunk

    hd, wd = 16, 16

    # Using fhw directly as ints
    x_in = x.view(b, nc, sc, n, d)
    r_in = r.view(b, nc, sc, 1, d_2)

    # Output buffer for the actual sequence
    xo_parts = []

    cu_seq_len = 0
    # Loop over unbound CPU scalars
    for i in range(mv):
        h_val, w_val = hs[i], ws[i]
        h_16, w_16 = h_val // hd, w_val // wd
        seq_len = c * h_16 * w_16

        x_v = x_in[:, :, cu_seq_len:cu_seq_len + seq_len].reshape(b, -1, n, d)  # b, nc*seq_len, n, d
        r_v = r_in[:, :, cu_seq_len:cu_seq_len + seq_len].reshape(b, -1, 1, d_2)  # b, nc*seq_len, 1, d_2

        p_v = p[:, i::mv]  # select projection matrices corresponding to this view
        x_v = apply_fn(x_v, r_v, p_v)   # b, sv, n, d

        xo_parts.append(x_v.reshape(b, nc, seq_len, n, d))
        cu_seq_len += seq_len

    xo = torch.cat(xo_parts, dim=2)
    return xo.view(b, s, n, d)


varlen_apply_train = torch.compile(varlen_apply)
# FIXME: SLOWER INFERENCE SPEED DUE TO THIS
# Strange bug: if we do training first, then inference in the same process, the compilation would take hours
# if we do inference first, compilation would only take a few minutes, but then training would get stuck at the compilation stage for a few hours
# only way for now is to separate the function calls
# but this would also come at the sacrifice of inference speed if all we do is inference
varlen_apply_inference = torch.compiler.disable(varlen_apply)
