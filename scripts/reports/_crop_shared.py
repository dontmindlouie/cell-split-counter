"""Shared crop/centroid/frame-sampling helpers for the review_crops-based report
tools (researcher_browser.py, spot_check_review.py). Extracted 2026-07-10 after this
exact logic was hand-duplicated between both files -- the same duplication pattern
that let src/review_gpt.py silently drift from src/review.py's prompt-building logic
for a full day (see _build_review_content in src/review.py). death_shape_browser.py
is NOT included here: it generates its own from-scratch crops with a baked-in marker
rather than reading review_crops/, so it doesn't share this code path.
"""

import re
import struct
from pathlib import Path

CROP_RADIUS = 192  # must match src/review.py's _CROP_RADIUS
FRAME_STRIDE = 3   # must match src/review.py's _FRAME_STRIDE

# Matches _save_debug_crops's "{pos:02d}_{before|split|after}_{idx:05d}.png" naming.
CROP_NAME_RE = re.compile(r"^\d+_(?:before|split|after)_(\d+)\.png$")


def png_size(path: Path) -> tuple[int, int]:
    """Width/height from a PNG's IHDR chunk -- no imaging library needed."""
    with open(path, "rb") as f:
        header = f.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def centroid_in_crop_pct(img_path: Path, cx: float, cy: float) -> tuple[float, float]:
    """Where the tracked centroid sits within this crop, as a 0-100% (left, top) pair.

    src/review.py crops [cy-R:cy+R, cx-R:cx+R], clamped to the frame at 0 on the low
    side (`max(0, cx - R)`). So the centroid's distance from the crop's own left edge
    is exactly `cx - max(0, cx - R)` = `min(cx, R)` -- R (dead center) when the left
    side wasn't clamped, or less than R (shifted toward that edge) when it was. Same
    logic for y. Dividing by the crop's *actual* saved width/height (read from the
    PNG itself) turns that pixel offset into a percentage CSS can position against,
    correct for interior, edge-clamped, and corner-clamped crops alike.
    """
    w, h = png_size(img_path)
    offset_x = min(cx, CROP_RADIUS)
    offset_y = min(cy, CROP_RADIUS)
    return (offset_x / w) * 100, (offset_y / h) * 100


def frame_idx_from_name(name: str) -> int | None:
    m = CROP_NAME_RE.match(name)
    return int(m.group(1)) if m else None


def sampled_only(imgs: list[Path], peak_frame: int) -> list[Path]:
    """review_crops/ holds every consecutive frame (src/review.py's
    _build_dense_debug_window, added 2026-07-10 for spot_check_review.py's
    frame-by-frame QC view) -- filter back down to the stride-sampled subset the AI
    actually reviewed. Falls back to showing everything if names don't match
    (older runs predating the dense window)."""
    out = [p for p in imgs if (idx := frame_idx_from_name(p.name)) is not None
           and (idx - peak_frame) % FRAME_STRIDE == 0]
    return out or imgs
