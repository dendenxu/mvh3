"""Per-(model_fps, source_fps) frame-stride remap factory.

Why this exists: when model_fps != source_fps and ratio = src/model is non-
integer, the per-step frame stride round(i*ratio) produces Δf ∈ {⌊r⌋, ⌈r⌉}
alternation. For ratios with large q in p/q (e.g. 25/16 → q=16), the Δf
pattern is irregular over a 16-step period and visually "jumpy" — adjacent
training samples cover 1× vs 2× of the source's per-frame motion.

The factory only needs entries for source fps where the natural ratio
src/model_fps has q>2 (chaotic pattern). Sources whose natural ratio is
already small-q (1, 3/2, 2, 4, ...) fall through to the trivial mapping
(model_fps, source_fps) without any warning. e.g. under model_fps=16 the
sources 16, 24, 32, 48, 64 all pass through cleanly; only 25, 30, 60, etc.
need explicit table entries.

Chaotic sources that don't hit a key exactly (decord avg_fps garbage like
26.84 / 27.58 / 52 on VFR / bad-duration videos) are snapped ONLINE to the
proportionally-nearest registered rate within a ±FPS_MATCH_TOLERANCE_FRAC band
(see resolve_fps_remap). No parquet is ever modified — the snap is a pure
load-time lookup, so it can't desync pose (pose is indexed by frame number, not
fps).

Outer key: outer model_fps (the global config setting).
Inner key: row's source fps (parquet 'fps' column, may be float for drop-frame).
Value: (effective_model_fps, snapped_source_fps).
  ratio = snapped_source_fps / effective_model_fps
  effective_fps reported to model = effective_model_fps
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

from utils.console import warn_once


# Format: outer model_fps → inner source_fps → (effective_model_fps, snapped_src_fps).
# Float keys handle broadcast/drop-frame fps (e.g. NTSC 29.97 / 59.94 / 23.976)
# which match within 0.05 of an entered key.
#
# Only register sources where natural src/model_fps has q>2 (chaotic pattern).
# Trivially-clean sources (q≤2 already) pass through and are not warned about.
FPS_REMAP_FACTORY: dict[int, dict[float, Tuple[int, int]]] = {
    16: {
        # source: (effective_model_fps, snapped_src_fps) → ratio
        15: (16, 16),   # snap up to 16 → ratio 1.0 (downsample-only, no resample needed)
        23.976: (16, 24),   # NTSC film 24*1000/1001 → snap to 24 (ratio 3/2)
        25: (16, 24),       # 3/2  q=2  alternating (was 25/16 = 1.5625, q=16 chaotic)
        29.97: (15, 30),    # NTSC drop-frame → snap to 30 (ratio 2)
        29.997: (15, 30),   # NTSC broadcast → snap to 30 (ratio 2)
        29.916666666666668: (15, 30),   # NTSC broadcast → snap to 30 (ratio 2)
        30: (15, 30),       # 2.0  q=1  perfect    (was 30/16 = 1.875,  q=8 chaotic)
        47.952047952047955: (16, 48),   # NTSC 48*1000/1001 → snap to 48 (ratio 3/1)
        50: (15, 50),       # 10/3 q=3  (drop model_fps 16→15; was 50/16 = 25/8, q=8 chaotic)
        59.94: (15, 60),    # NTSC drop-frame → snap to 60 (ratio 4)
        60: (15, 60),       # 4.0  q=1  perfect    (was 60/16 = 3.75,   q=4 chaotic)
        60.08010680907877: (15, 60),       # 4.0  q=1  perfect    (was 60/16 = 3.75,   q=4 chaotic)
    },
}


def is_clean_ratio(num: float, denom: float, max_q: int = 2) -> bool:
    """True if num/denom = p/q with q ≤ max_q (after rationalizing). Used to
    suppress fallthrough warnings when the natural ratio is already regular.
    """
    try:
        # limit_denominator(1024) rationalizes float fps (e.g. NTSC 29.97 →
        # 2997/100) before dividing, so the q test reflects the snapped rate
        # rather than the float's full binary expansion (which would always
        # report a huge denominator). 1024 is a generous cap: real fps rates
        # rationalize well below it (1001 is the largest denom seen, NTSC).
        frac = Fraction(num).limit_denominator(1024) / Fraction(denom).limit_denominator(1024)
        return frac.denominator <= max_q
    except (ZeroDivisionError, OverflowError):
        return True


# Proportional snap band: a source fps within ±FPS_MATCH_TOLERANCE_FRAC of a
# registered rate snaps to it. 0.10 → ±3 fps at 30, ±2.5 at 25, ±5 at 50, ±6 at
# 60. Chosen over a flat ±N fps gap because the same 2 fps miss is a big stride
# change at 24 but negligible at 60 — a percentage band is uniform across rates.
FPS_MATCH_TOLERANCE_FRAC = 0.10


def resolve_fps_remap(model_fps: int, source_fps) -> Tuple[int, int]:
    """Look up (effective_model_fps, snapped_source_fps) for this (model, source).

    Resolution order, all done ONLINE at load time (no parquet rewrite):
      1. Exact table key → its curated mapping.
      2. Clean source (natural ratio src/model_fps has q≤2, e.g. 16/24/32/48/64)
         → pass through untouched. These never warned and never need a remap.
      3. Chaotic source (q>2) → snap to the nearest registered rate BY RATIO
         (proportional distance |src/key − 1|, not absolute fps gap) when it lands
         inside that rate's ±FPS_MATCH_TOLERANCE_FRAC band. This pulls decord
         avg_fps garbage (26.8 / 27.6 / 52 / 58.8 …) onto the nearest clean
         standard, giving a small-q stride. Covers NTSC drift / FP noise too.
      4. Chaotic source whose nearest standard is still outside the band (e.g.
         20, 12, 10) → fall through and warn once.
    """
    table = FPS_REMAP_FACTORY.get(model_fps)
    if table is None:
        return model_fps, source_fps
    if source_fps in table:
        return table[source_fps]
    src = float(source_fps)
    # Clean small-q sources need no remap — pass through (preserves 16/24/32/48/64).
    if is_clean_ratio(src, float(model_fps)):
        return model_fps, source_fps
    # Chaotic source: snap to the proportionally-nearest registered rate, if in band.
    nearest_key = min(table.keys(), key=lambda k: abs(src / float(k) - 1.0))
    if abs(src / float(nearest_key) - 1.0) <= FPS_MATCH_TOLERANCE_FRAC:
        return table[nearest_key]
    # Fallthrough: nearest standard is >FRAC away and ratio is chaotic → warn once.
    warn_once(f'[fps_remap] unknown (model_fps={model_fps}, '
              f'source_fps={source_fps}): nearest standard >'
              f'{FPS_MATCH_TOLERANCE_FRAC:.0%} away, will be jumpy. '
              f'Register in FPS_REMAP_FACTORY.')
    return model_fps, source_fps
