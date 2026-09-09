"""Shared pose-visualization helpers.

Builds a 3-panel orthographic view (BEV / Side / Front) with data-driven axes
from camera-to-world matrices. Used by:
- pipeline/ar_lb_inference.py::ARLBInferencePipeline.render_pose_panel
- scripts/data/camera/vis_parquet_poses.py::render_combined_video

All callers use **RGB** canvas order. cv2 drawing functions just write
(ch0, ch1, ch2) bytes — there is no implicit BGR conversion. Colors below
are defined in RGB so they match regardless of whether the canvas ends up
in ffmpeg (rgb24) or an imageio/PIL path.

All distances are in input world units. Callers that work in a scaled
pose-space (e.g. anything divided by pose_stable_factor) are responsible for
restoring true world scale before passing positions in.
"""

import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# RGB channel order — matches both PIL and ffmpeg -pix_fmt rgb24.
CAM_COLORS = (
    (0, 255, 0), (255, 200, 0), (0, 200, 255), (255, 0, 255),
    (0, 255, 255), (255, 100, 100), (100, 100, 255), (200, 255, 100),
)

DEFAULT_FONT_PATH = "assets/fonts/maple/maplecn-SemiBold.woff2"

from functools import lru_cache


@lru_cache(maxsize=128)  # load each (path, size) font ONCE and reuse it: re-loading
def load_font(font_path, size):  # the font per text draw was 77% of render time
    try:                         # (PIL getfont + font-dir scandir, ~397 calls/sample)
        return ImageFont.truetype(font_path or DEFAULT_FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def blend_stamp(canvas, stamp, x, y):
    """Alpha-blend an RGBA stamp onto an RGB canvas at (x, y)."""
    sh, sw = stamp.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + sw), min(ch, y + sh)
    sx0, sy0 = x0 - x, y0 - y
    if x1 <= x0 or y1 <= y0:
        return
    s = stamp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    a = s[:, :, 3:4] / 255.0
    roi = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = (s[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def _wrap_text(text, font, max_w):
    """Greedy word-wrap so each line fits within max_w pixels."""
    if not text:
        return []
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        cur = ""
        for w in words:
            test = (cur + " " + w) if cur else w
            bb = font.getbbox(test)
            # `or not cur`: a single word wider than max_w still gets placed
            # on its own line (overflow tolerated) — otherwise the line would
            # never accept it and the loop would emit an empty line.
            if bb[2] - bb[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


# Caption-strip label coloring (parallels META_KEY_COLORS for the metadata strip):
# the leading field label of each line gets a type color, every [...] bracket
# (timestamps / view tags / severity) gets a structural color, QUALITY +
# violations are red so defects/flags pop. Rest stays white.
_CAP_LABEL_RULES = [
    (re.compile(r'^scene:'),            (100, 200, 255)),  # cyan
    (re.compile(r'^motion:'),           (140, 255, 160)),  # green
    (re.compile(r'^C\d+'),              (255, 200, 100)),  # amber (chunks)
    (re.compile(r'^event'),             (255, 160,  70)),  # orange (events)
    (re.compile(r'^camera\.\w+'),       (200, 160, 255)),  # violet (camera)
    (re.compile(r'^\s+(?:rig|v\d+:)'),  (200, 160, 255)),  # violet (rig/view_motion sub-lines)
    (re.compile(r'^QUALITY'),           (255,  80,  80)),  # RED (defect / non-static flag)
    (re.compile(r'^quality: clean'),    (140, 255, 160)),  # green (verified-clean status line)
    (re.compile(r'^\([^)]*violation'),  (255, 110, 110)),  # red (violation count)
]
_CAP_BRACKET_COLOR = (120, 210, 210)  # [...] = teal (structural: t-spans, tags, severity)


def _caption_color_spans(line):
    """(start, end, color) spans for the label prefix + every [...] in the line."""
    spans = []
    for rx, col in _CAP_LABEL_RULES:
        m = rx.match(line)
        if m:
            spans.append((m.start(), m.end(), col)); break
    for m in re.finditer(r'\[[^\]]*\]', line):
        spans.append((m.start(), m.end(), _CAP_BRACKET_COLOR))
    return spans


def make_caption_panel(caption, target_w, font_path=None, font_size=20, pad=10,
                       return_chunk_spans=False):
    """Render a caption strip: black bg, word-wrapped to target_w, with field
    labels (scene/motion/Cn/event/camera/QUALITY) and [...] brackets colored for
    scanability (white body text). Returns (h, target_w, 3) uint8 or None.

    return_chunk_spans=True -> returns (panel, {k: (y0, y1)}) where each entry is
    the vertical pixel span (covering wrapped rows) of the 'C{k} [...]' chunk
    line, for per-frame active-chunk highlighting. {} when there are no chunks."""
    if not caption:
        return (None, {}) if return_chunk_spans else None
    font = load_font(font_path, font_size)
    line_h = int(font_size * 1.35)
    maxw = target_w - pad * 2

    def color_at(i, spans):
        for s, e, c in spans:
            if s <= i < e:
                return c
        return (255, 255, 255)

    placements, y = [], pad  # (x, y, word, color)
    chunk_spans = {}
    for logical in str(caption).split('\n'):
        spans = _caption_color_spans(logical)
        cm = re.match(r'^C(\d+)\b', logical)  # 'C{k} [..]:' chunk line
        y_line_start = y
        x, idx = pad, 0
        for word in logical.split(' '):
            if word:
                w = int(font.getlength(word))
                if x > pad and x + w > pad + maxw:  # wrap before this word
                    y += line_h; x = pad
                placements.append((x, y, word, color_at(idx, spans)))
                x += int(font.getlength(word + ' '))
            idx += len(word) + 1
        y += line_h
        if cm:
            chunk_spans[int(cm.group(1))] = (y_line_start, y)
    total_h = y + pad
    img = Image.new('RGB', (target_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for x, yy, word, col in placements:
        draw.text((x, yy), word, font=font, fill=col)
    arr = np.array(img)
    return (arr, chunk_spans) if return_chunk_spans else arr


# Timeline / playhead colors (RGB), aligned with the caption-strip palette so the
# bar reads as part of the same annotation: ticks reuse the [t-span] bracket teal,
# the active band reuses the C{k} chunk amber, cursor is white.
TIMELINE_TRACK = (70, 70, 70)
TIMELINE_TICK = _CAP_BRACKET_COLOR          # teal (120, 210, 210)
TIMELINE_ACTIVE = (255, 200, 100)           # amber == C{k} caption label
TIMELINE_CURSOR = (255, 255, 255)           # white playhead


def draw_timeline_bar(canvas, x0, y0, w, h, progress_frac,
                      bounds_frac=None, active_chunk=None):
    """Progress track + chunk-boundary ticks + active-chunk band + moving playhead
    cursor, drawn in place on an RGB canvas. cv2-only (no PIL) so it is cheap
    enough to call once per frame inside a render loop. Shared by the caption
    renderer (scripts/data/camera/vis_parquet_poses.py) and the AR-LB inference
    vis (pipeline/ar_lb_inference.py) so both look identical.

    progress_frac : current position in [0, 1] -> cursor x = x0 + progress_frac*w.
    bounds_frac   : monotone chunk boundaries in [0, 1], len = num_chunks + 1;
                    interior values become ticks. None -> no ticks/band.
    active_chunk  : chunk index k -> dim amber band over [bounds_frac[k], k+1].
                    None -> no band (e.g. AR-LB boot/Memory region)."""
    x0, y0, w, h = int(x0), int(y0), int(w), int(h)
    x1 = x0 + w
    cv2.rectangle(canvas, (x0, y0), (x1, y0 + h), TIMELINE_TRACK, -1)
    if (active_chunk is not None and bounds_frac is not None
            and 0 <= active_chunk < len(bounds_frac) - 1):
        bx0 = x0 + int(bounds_frac[active_chunk] * w)
        bx1 = x0 + int(bounds_frac[active_chunk + 1] * w)
        band = tuple(c // 3 for c in TIMELINE_ACTIVE)
        cv2.rectangle(canvas, (bx0, y0), (bx1, y0 + h), band, -1)
    if bounds_frac is not None:
        for b in bounds_frac[1:-1]:  # interior boundaries only
            tx = x0 + int(b * w)
            cv2.line(canvas, (tx, y0 - 1), (tx, y0 + h + 1), TIMELINE_TICK, 1)
    xc = x0 + int(max(0.0, min(1.0, progress_frac)) * w)
    cv2.line(canvas, (xc, y0 - 2), (xc, y0 + h + 2), TIMELINE_CURSOR, 2)
    cv2.circle(canvas, (xc, y0), 3, TIMELINE_CURSOR, -1)


# Color palette for metadata key labels (RGB). Values are always white.
META_KEY_COLORS = {
    'exp':     (255, 255, 255),  # white (experiment / model name — run identity)
    'step':    (255, 235, 130),  # gold  (training step)
    'run':     (130, 220, 255),  # azure (wandb run id)
    'dataset': (100, 200, 255),  # cyan
    'row':     (255, 200, 100),  # amber
    'frames':  (140, 255, 160),  # green
    'parquet': (210, 170, 255),  # violet
    'view':    (255, 160, 200),  # pink
    'segment': (255, 200, 100),  # amber (shared w/ row, used for static_seg)
    'psf':     (255, 120, 120),  # salmon (pose_stable_factor divisor)
    'maxt':    (120, 220, 220),  # teal (post-division max|T|)
    'static':  (255, 235, 130),  # gold  (static-prior flag: world expected motionless)
    'rig':     (180, 160, 255),  # violet (cameras rigidly co-mounted)
}


def make_metadata_panel(meta, target_w, font_path=None, font_size=18, pad=8,
                        sep_color=(120, 120, 120), value_color=(245, 245, 245)):
    """Horizontal `[key]: value  •  [key]: value …` strip, wraps at target_w.

    `meta`: ordered dict {key: value-string}. Each key gets its
    META_KEY_COLORS color; values are white; entries separated by `•`.
    Returns (h, target_w, 3) uint8 RGB, or None when meta is empty.
    """
    if not meta:
        return None
    font = load_font(font_path, font_size)
    sep = '  •  '
    sep_w = font.getbbox(sep)[2]
    inner_w = target_w - pad * 2

    entries = [(k, str(v), font.getbbox(f'[{k}]: ')[2] + font.getbbox(str(v))[2])
               for k, v in meta.items()]
    lines, cur, cur_w = [], [], 0
    for e in entries:
        gap = sep_w if cur else 0
        if cur and cur_w + gap + e[2] > inner_w:
            lines.append(cur); cur, cur_w = [], 0; gap = 0
        cur.append(e); cur_w += gap + e[2]
    if cur:
        lines.append(cur)

    line_h = int(font_size * 1.35)
    img = Image.new('RGB', (target_w, len(lines) * line_h + pad * 2), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for li, entries in enumerate(lines):
        x, y = pad, pad + li * line_h
        for ei, (k, v, _) in enumerate(entries):
            if ei:
                draw.text((x, y), sep, font=font, fill=sep_color); x += sep_w
            pre = f'[{k}]: '
            draw.text((x, y), pre, font=font,
                      fill=META_KEY_COLORS.get(k, (180, 180, 180)))
            x += font.getbbox(pre)[2]
            draw.text((x, y), v, font=font, fill=value_color)
            x += font.getbbox(v)[2]
    return np.array(img)


def make_label(text, font, color=(255, 255, 255),
               bg_alpha=0, pad=0, radius=5):
    """Render text as RGBA numpy array. When bg_alpha > 0, draws a
    semi-transparent black rounded-rect background (matches GT/C0 label
    style in pipeline/ar_lb_inference.py)."""
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0]) + pad * 2
    h = max(1, bbox[3] - bbox[1]) + pad * 2
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if bg_alpha > 0:
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                               fill=(0, 0, 0, bg_alpha))
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font,
              fill=color + (255,))
    return np.array(img)

# Legacy fixed-shape frustum offsets — NOT used by any current code path.
# The live frustum geometry is computed pinhole-correctly per camera in
# frustum_corners_camframe() (half-width/height from fx/fy/img_size), which
# supersedes this constant. Kept only for reference; the [s*aspect, s, s]
# recipe below no longer matches how frustums are drawn.
# Camera-frame frustum offsets: center then 4 far-plane corners (TR, TL, BL, BR).
# Multiply by [s*aspect, s, s] then by R.T and translate by p to get world points.
FRUSTUM_OFFSETS = np.array([
    [0, 0, 0],
    [+1, +1, +1.6], [-1, +1, +1.6],
    [-1, -1, +1.6], [+1, -1, +1.6],
])


def compute_view_axes(c2ws):
    """Gravity-aligned orthographic axes from c2w matrices.

    Algorithm:
    1. cam_X (camera right) is ~parallel to ground for upright cameras.
       Compute pairwise cross(cam_X[i], cam_X[j]) — each is perpendicular
       to both, i.e. the ground-plane NORMAL. Align signs (using -cam_Y
       mean as a coarse "up" reference) and average.
    2. right = cam_Z[0] × up    (camera forward at v0f0 × up)
    3. fwd   = up × right       (right-handed completion)

    c2ws: numpy array of shape (..., 4, 4).
    Returns (up, right, fwd, centroid), each shape (3,).
    """
    flat = np.asarray(c2ws).reshape(-1, 4, 4)

    # Step 1: ground-plane normal from pairwise cam_X cross products.
    # Subsample for O(N) pairs (N rather than N^2). Pair each frame with
    # frame k+N/2 to maximize angular separation → larger cross magnitude
    # → less sensitive to noise.
    cam_x = flat[:, :3, 0].astype(np.float64)
    N = len(cam_x)
    half = N // 2
    if half > 0:
        a = cam_x[:half]
        b = cam_x[half:half * 2]
        normals = np.cross(a, b)            # (half, 3)
    else:
        normals = cam_x.copy()              # degenerate fallback

    # Filter near-zero cross products (parallel cam_X pairs).
    mags = np.linalg.norm(normals, axis=1)
    keep = mags > 1e-4
    if keep.sum() > 0:
        normals = normals[keep]

    # Sign alignment: -mean(cam_Y) gives a coarse "up" reference for an
    # upright camera. Flip any normal pointing opposite to it.
    ref_up = -flat[:, :3, 1].mean(axis=0).astype(np.float64)
    ref_up = ref_up / (np.linalg.norm(ref_up) + 1e-8)
    signs = np.sign(normals @ ref_up)
    signs[signs == 0] = 1
    normals = normals * signs[:, None]

    # The cross-product ground normal is only reliable when cam_X actually
    # ROTATES between the paired frames — i.e. the rig changes heading (yaws).
    # Pure-translation clips (a forward dolly, or "all-translation +
    # augmentation roll" data) keep cam_X nearly constant, so every cross
    # product is noise: their sign-aligned mean cancels to ~0 and, once
    # normalised, points in an arbitrary direction — typically picking a
    # HORIZONTAL world axis as "up" (measured: |mean normal| ~0.005 for pure
    # translation vs ~0.36 for a real yaw sweep; the failure shows up as a
    # 90deg-wrong BEV plane on level forward-moving shots). `coherence` =
    # the pre-normalisation magnitude (≈ mean sin(inter-frame yaw)); when it
    # is below a few-degrees-of-yaw floor, fall back to the gravity proxy
    # -mean(cam_Y), which is correct for the upright cameras these clips use.
    # (A clip that is BOTH heading-locked AND consistently pitched defeats
    # both estimators — rare, and the cross-product method was already wrong
    # there too, so this is no regression. Datasets with real yaw — nymeria
    # head-cam, orbiting rigs — keep using the pitch-robust cross-product up.)
    up = normals.mean(axis=0)
    coherence = float(np.linalg.norm(up))
    if coherence < 0.05:
        up = ref_up
    up = up / (np.linalg.norm(up) + 1e-8)

    # Step 2: right = cam_Z[0] × up
    cam_z0 = flat[0, :3, 2].astype(np.float64)
    right = np.cross(cam_z0, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Step 3: fwd = up × right  (right-handed)
    fwd = np.cross(up, right)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

    centroid = np.median(flat[:, :3, 3], axis=0)
    return up, right, fwd, centroid


# Minimum span (meters) used when picking the pose-panel display scale. A
# near-stationary rig has a tiny true extent; without a floor the auto-scale
# zooms in so far that sub-mm pose jitter fills the panel and looks shaky.
# Flooring the *scale* extent (not the reported span) caps the zoom-in while the
# span label still shows the true extent. Overridable via vis_parquet_poses.py
# --min_span; ar_lb_inference.py render_pose_panel uses this default.
MIN_PANEL_SPAN_M = 0.1


def compute_extent_and_scale(positions, right, fwd, up, view_w, view_h, margin=0.15,
                             min_span=None):
    """Project positions onto the (right, fwd, up) basis and pick a display scale.

    positions: (N, 3) world points.
    Returns (max_extent, scale, medians) where medians = (mr, mf, mu) — the
    median of each projected axis, used to recenter sub-views. `max_extent` is
    the TRUE span; the display scale is floored at `min_span` meters (default
    MIN_PANEL_SPAN_M) so a near-stationary rig is not over-zoomed (which would
    amplify sub-mm pose jitter).
    """
    if min_span is None:
        min_span = MIN_PANEL_SPAN_M
    positions = np.asarray(positions)
    centered = positions - np.median(positions, axis=0)
    pr, pf, pu = centered @ right, centered @ fwd, centered @ up
    max_extent = max(np.ptp(pr), np.ptp(pf), np.ptp(pu)) + 1e-6
    scale = min(view_w, view_h) * (1.0 - 2 * margin) / max(max_extent, min_span)
    medians = (float(np.median(pr)), float(np.median(pf)), float(np.median(pu)))
    return max_extent, scale, medians


def make_view_defs(right, fwd, up, medians):
    """Standard 3 sub-views. Returns list of (name, ax_x, ax_y, mx, my).

    BEV:   前后 = 上下, 左右 = 左右. screen_x = right, screen_y = fwd.
    Side:  前后 = 左右, 上下 = 上下. screen_x = fwd,   screen_y = up.
    Front: 上下 = 上下, 左右 = 左右. screen_x = right, screen_y = up.
    """
    mr, mf, mu = medians
    return [
        ('BEV',   right, fwd, mr, mf),
        ('Side',  fwd,   up,  mf, mu),
        ('Front', right, up,  mr, mu),
    ]


def make_projector(centroid, scale, view_w, view_h):
    """Return project(pts, ax_x, ax_y, mx, my, vx0) -> (N, 2) int pixel coords."""
    def project(pts, ax_x, ax_y, mx, my, vx0):
        pts = np.asarray(pts)
        sx = (pts - centroid) @ ax_x
        sy = (pts - centroid) @ ax_y
        px = ((sx - mx) * scale + view_w / 2).astype(np.int32) + vx0
        py = (view_h / 2 - (sy - my) * scale).astype(np.int32)
        return np.stack([px, py], axis=-1)
    return project


def nice_scale_bar_length(view_w, scale, target_frac=0.18):
    """Smallest 1/2/5 × 10^k value whose pixel length covers target_frac of view_w."""
    target_m = view_w * target_frac / max(scale, 1e-6)
    mag = 10 ** np.floor(np.log10(max(target_m, 1e-3)))
    return next(m * mag for m in (1, 2, 5, 10) if m * mag >= target_m)


def draw_scale_bar(canvas, vx0, view_w, view_h, scale, text_scale=0.84, unit='m',
                   color=(240, 240, 240), y_offset_from_bottom=20):
    """Bottom-right scale bar inside one sub-view starting at (vx0, 0)."""
    nice = nice_scale_bar_length(view_w, scale)
    bar_px = int(nice * scale)
    bx1, by = vx0 + view_w - 12, view_h - y_offset_from_bottom
    bx0 = bx1 - bar_px
    cv2.line(canvas, (bx0, by), (bx1, by), color, 2, cv2.LINE_AA)
    for x in (bx0, bx1):
        cv2.line(canvas, (x, by - 4), (x, by + 4), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f'{nice:g}{unit}', (bx0, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 2, cv2.LINE_AA)


def draw_view_label(canvas, vx0, name, max_extent, text_scale, unit='m',
                    y=None, color=(200, 200, 200), extras=''):
    """Top-left view label ('BEV  span=12.3m')."""
    if y is None:
        y = max(16, int(text_scale * 28)) + 4
    extras = f'  {extras}' if extras else ''
    cv2.putText(canvas, f'{name}  span={max_extent:.1f}{unit}{extras}', (vx0 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 2, cv2.LINE_AA)


def frustum_corners_camframe(intrinsics, img_size, scale, depth_pixels=36.0):
    """Pinhole-correct frustum corners in CAMERA frame.

    Picks a far-plane depth `d` such that the displayed depth equals
    `depth_pixels` on screen. Then computes half-width/half-height of the
    far plane via the pinhole projection: a 3D point at depth d projects
    to image pixel x = fx * X / d + cx, so the image edge x ∈ [0, W]
    corresponds to X ∈ [-cx * d / fx, (W - cx) * d / fx]. For an
    approximately-centered camera (cx ≈ W/2), this simplifies to
    ±(W/2) * d / fx. Same for Y vs fy.

    Result: frustum aspect = image aspect (when fx=fy); larger fx →
    narrower frustum (more elongated); larger fy → flatter frustum.

    Returns (5, 3) array: [center, TR, TL, BL, BR] at depth d.
    Camera convention: +X right, +Y up, +Z forward.
    """
    fx, fy, cx, cy = float(intrinsics[0]), float(intrinsics[1]), float(intrinsics[2]), float(intrinsics[3])
    if img_size is None:
        # Infer image size from the principal point, per axis: cx > 1 means
        # pixel-space intrinsics (cx ≈ W/2 → W ≈ 2*cx) so W = 2*cx; cx <= 1
        # means normalized intrinsics → unit width. Same for cy/H.
        # (format_intrinsics_legend infers the same pixel-vs-normalized split
        # but gates on fx > 1 for both axes, not per-axis cx/cy as here.)
        img_w = max(1.0, 2 * cx) if cx > 1 else 1.0
        img_h = max(1.0, 2 * cy) if cy > 1 else 1.0
    else:
        img_w, img_h = float(img_size[0]), float(img_size[1])
    # `d` is the far-plane depth in WORLD units chosen so the frustum spans
    # `depth_pixels` on screen (scale = pixels per world unit), keeping all
    # frustums the same on-screen size regardless of true camera distances.
    d = depth_pixels / max(scale, 1e-9)
    hw = d * img_w / (2 * max(fx, 1e-6))
    hh = d * img_h / (2 * max(fy, 1e-6))
    return np.array([
        [0, 0, 0],
        [+hw, +hh, +d], [-hw, +hh, +d], [-hw, -hh, +d], [+hw, -hh, +d],
    ], dtype=np.float64), d


def draw_frustum(canvas, c2w, intrinsics, img_size, project, ax_x, ax_y, mx, my, vx0,
                 scale, color, draw_axes=False, depth_pixels=36.0):
    """Draw a camera frustum (4 side edges + far rect + center dot).

    Frustum geometry matches the actual pinhole projection — image aspect
    and per-axis FoV (fx, fy, img_w, img_h) determine the shape; ``scale``
    keeps overall size visually consistent across cameras at depth_pixels px.
    Frustum lines = 2 px, axes = 3 px.
    """
    R = np.asarray(c2w)[:3, :3]
    p = np.asarray(c2w)[:3, 3]
    offsets, d = frustum_corners_camframe(intrinsics, img_size, scale, depth_pixels)
    pts = p + offsets @ R.T
    px = project(pts, ax_x, ax_y, mx, my, vx0)
    pp = (int(px[0, 0]), int(px[0, 1]))
    for i in range(1, 5):
        cv2.line(canvas, pp, (int(px[i, 0]), int(px[i, 1])), color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [px[1:]], True, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, pp, 3, color, -1)
    if draw_axes:
        # Axes drawn 0.6 × the far-plane depth `d` so their on-screen length
        # tracks the frustum size at the current display scale (d already
        # encodes depth_pixels/scale), staying readable but shorter than the
        # frustum body.
        axis_len = d * 0.6
        # Per-axis colors in RGB (canvas is RGB — see module docstring):
        # X=red, Y=green, Z=blue. R[:, ai] is camera-frame axis ai (rotation
        # column). Z uses (50,50,255) not pure (0,0,255) — dark blue is nearly
        # invisible on the black panel, so the channel is lifted to stay legible.
        axis_colors = [(255, 0, 0), (0, 255, 0), (50, 50, 255)]
        for ai in range(3):
            tip = p + R[:, ai] * axis_len
            tip_px = project(tip[np.newaxis], ax_x, ax_y, mx, my, vx0)
            cv2.line(canvas, pp, (int(tip_px[0, 0]), int(tip_px[0, 1])),
                     axis_colors[ai], 3, cv2.LINE_AA)


# ─── Camera intrinsics legend ──────────────────────────────────────────────

def format_intrinsics_legend(fx, fy, cx, cy, img_w=None, img_h=None):
    """Format FOV and principal-point as two compact legend strings.

    When img_w/img_h are given, (fx, fy, cx, cy) are pixel-space.
    When omitted and fx > 1, infers image size from principal point
    (w ~ 2*cx, h ~ 2*cy). Otherwise treats intrinsics as normalized.

    wfov = width (horizontal) FOV  = 2*atan(w / 2fx)
    hfov = height (vertical) FOV   = 2*atan(h / 2fy)

    Returns (fov_str, pp_str):
        fov_str = 'wfov=90 hfov=60'
        pp_str  = 'cx/w=0.50 cy/h=0.50'
    """
    fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    if img_w is not None and img_h is not None:
        w, h = float(img_w), float(img_h)
    elif fx > 1.0:
        w, h = cx * 2, cy * 2
    else:
        w, h = 1.0, 1.0
    wfov = math.degrees(2 * math.atan2(w, 2 * fx)) if fx > 0 else 0
    hfov = math.degrees(2 * math.atan2(h, 2 * fy)) if fy > 0 else 0
    cx_w = cx / w if w > 0 else 0
    cy_h = cy / h if h > 0 else 0
    return f"wfov={wfov:.0f} hfov={hfov:.0f}", f"cx/w={cx_w:.2f} cy/h={cy_h:.2f}"


def draw_cam_legend(canvas, x, y, mv, intrinsics_list, img_sizes=None,
                    cam_names=None, font_path=None, font_size=12,
                    max_height=0, cams_per_col=4):
    """Draw per-camera legend with truetype font, matching GT/C0 label style.

    Renders one RGBA image (semi-transparent dark background) then
    alpha-blends onto *canvas* at (x, y).

    Per camera, 3 lines::

        [dot] cam0
              hfov=90 vfov=60
              cx/w=0.50 cy/h=0.50

    Cameras fill columns top-to-bottom (cams_per_col per column). Extra
    columns are added to the LEFT so the right edge of the legend stays
    anchored at (x + total_w). max_height auto-shrinks font_size.
    """
    if cam_names is None:
        cam_names = [f"cam{i}" for i in range(mv)]

    lines_per_cam = 3
    pad = 6
    col_gap = 12
    cams_per_col = max(1, cams_per_col)
    n_cols = (mv + cams_per_col - 1) // cams_per_col
    col_count = min(cams_per_col, mv)  # rows in the tallest column

    # Auto-fit: shrink font_size until legend fits within max_height.
    if max_height > 0:
        while font_size > 5:
            line_h = int(font_size * 1.35)
            needed = col_count * line_h * lines_per_cam + pad * 2
            if needed <= max_height:
                break
            font_size -= 1

    font = load_font(font_path, font_size)
    info_font = load_font(font_path, max(5, int(font_size * 0.8)))
    line_h = int(font_size * 1.35)
    dot_w = int(font_size * 1.2)

    info_color = (180, 180, 180, 255)
    entries = []
    max_tw = 0
    for ci in range(mv):
        color = CAM_COLORS[ci % len(CAM_COLORS)]
        name = cam_names[ci] if ci < len(cam_names) else f"cam{ci}"
        iw, ih = (None, None)
        if img_sizes and ci < len(img_sizes) and img_sizes[ci]:
            iw, ih = img_sizes[ci]
        intr = intrinsics_list[ci]
        fov_str, pp_str = format_intrinsics_legend(
            intr[0], intr[1], intr[2], intr[3], iw, ih)
        entries.append((name, fov_str, pp_str, color))
        for txt, f in [(name, font), (fov_str, info_font), (pp_str, info_font)]:
            bb = f.getbbox(txt)
            max_tw = max(max_tw, bb[2] - bb[0])

    col_w = max_tw + dot_w
    total_w = n_cols * col_w + (n_cols - 1) * col_gap + pad * 2
    total_h = col_count * line_h * lines_per_cam + pad * 2
    legend = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(legend)
    draw.rounded_rectangle([0, 0, total_w - 1, total_h - 1],
                           radius=6, fill=(0, 0, 0, 140))

    for ci, (name, fov_str, pp_str, color) in enumerate(entries):
        col = ci // cams_per_col
        row = ci % cams_per_col
        # Extra columns extend LEFTward (col 0 = rightmost).
        col_from_right = (n_cols - 1) - col
        col_x = pad + col_from_right * (col_w + col_gap)
        cy_pos = pad + row * line_h * lines_per_cam
        dot_cy = cy_pos + font_size // 2
        r = max(2, font_size // 4)
        draw.ellipse([col_x, dot_cy - r, col_x + r * 2, dot_cy + r],
                     fill=color + (255,))
        draw.text((col_x + dot_w, cy_pos), name,
                  font=font, fill=color + (255,))
        draw.text((col_x + dot_w, cy_pos + line_h), fov_str,
                  font=info_font, fill=info_color)
        draw.text((col_x + dot_w, cy_pos + line_h * 2), pp_str,
                  font=info_font, fill=info_color)

    # Right-anchor: shift the blend position so the right edge stays at
    # the original `x` (callers pass the right edge as `x`).
    blend_stamp(canvas, np.array(legend), x - total_w, y)
