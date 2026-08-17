"""Rendering: display LUT, crops/tiles, scale bars, PNG encoding, and the
mask-following filmstrip/family-membership machinery shared across tools.

Split out of the original single-file cell_mcp.py's "render" section, plus the
filmstrip-rendering helpers (walk_positions, filmstrip_frames, family_filmstrip_frames)
that were interleaved with the tool functions but are used by several of them
(follow_cells_over_time, watch_location_over_time, list_nearby_tracks, show_cells_in_browser).
"""

import base64
from typing import NamedTuple

import cv2
import numpy as np
from mcp.types import ImageContent

from .server import MAX_IMAGES, _WINDOW_BEFORE_MIN, _WINDOW_AFTER_MIN, _STRIDE_MIN, _UPSCALE_TO, _HDR_SEP
from .io import _frame_at_offset_min, _minutes_between, _pick_frames

import cell_mcp_server as _cm

# BUNDLE, _manifest, _tracks, _frame_png, and _hours below go through `_cm.` rather
# than a direct import -- see the note at the top of io.py. Tests monkeypatch these
# on the `cell_mcp_server` package, and only a call routed back through
# `cell_mcp_server` at call time will observe the patch.


def _colorize(grey: np.ndarray, well: str, color: bool) -> np.ndarray:
    """Apply the acquisition's own display LUT, so renders match what the
    researcher sees in Fiji rather than a colormap we invented.

    Multi-channel wells' frame is already a pre-tinted multi-color composite
    (io._frame_png prefers frames_display/ when it exists, see src/ingest.py,
    2026-08-05) -- it arrives here 3-channel BGR, already carrying both markers'
    real colors, so it passes through untouched rather than being tinted again.
    """
    if grey.ndim == 3:
        return grey if color else cv2.cvtColor(cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if not color:
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    rgb = _cm._manifest(well).get("display_color_rgb") or [255, 255, 255]
    r, g, b = [v / 255.0 for v in rgb]
    return cv2.merge([(grey * b).astype(np.uint8),
                      (grey * g).astype(np.uint8),
                      (grey * r).astype(np.uint8)])


# Every displayed frame was contrast-stretched to its OWN 0.5/99.5 percentiles when the
# ND2 was exported to 8-bit (src/ingest.py::_rescale_to_uint8), field-wide and before any
# cropping. So the images carry morphology faithfully and brightness only relatively:
# two frames rendered side by side have had different windows applied, and a real
# field-wide trend -- photobleaching over 40-70 h is the obvious one -- is flattened
# frame by frame into looking constant. The numbers do not have this problem;
# build_bundle measures intensity against the raw 16-bit ND2 for exactly this reason.
# Nothing in the image can reveal this, so every tool that returns one says it.
_DISPLAY_NOTE_BASE = (
    " Brightness across frames is NOT comparable: each frame was separately stretched "
    "to its own 0.5/99.5 percentiles on export, which flattens real trends like "
    "photobleaching. Judge shape, position and texture from these images; take any "
    "brightness claim from measure() or get_track_profile, which read the raw ND2."
)


def _display_note(well: str) -> str:
    """The brightness caveat, and whether this particular bundle can undo it."""
    try:
        w = (_cm._manifest(well).get("display_window") or {})
    except Exception:
        w = {}
    if w.get("recorded"):
        return _DISPLAY_NOTE_BASE + (
            " This bundle DOES record the per-frame window (manifest.display_window), "
            "so the stretch is reversible if you need frames on a common scale: "
            "raw ~= lo + png/255 * (hi - lo)."
        )
    return _DISPLAY_NOTE_BASE + (
        " This bundle predates window recording, so the stretch cannot be undone from "
        "the images at all -- the numbers are the only route to brightness here."
    )


def _scale_bar(img: np.ndarray, um_per_px: float, target_um: float = 20.0, k: float = 1.0) -> np.ndarray:
    """Burn a labelled scale bar into the bottom-right corner, label to its left.

    This is the calibration check: the researcher compares it once against her
    own measurement of the same cell, rather than trusting a number in a file.

    Label and bar sit side by side in one row (not stacked) so the burned-in
    footer is a single text-line tall instead of two -- both because that is a
    smaller bite out of the image and because it makes the crop that hides this
    band for figures (see tools_output.py) a lot cheaper. Width is measured with
    getTextSize and the whole thing is skipped, never truncated, if it wouldn't
    fit -- a partially-drawn "10 u" reads as a rendering bug, not a narrow tile.

    `k` scales the text/bar/margins up with the tile's own resolution (see
    _stamp_tile) -- at the base 312px render this is 1.0 (unchanged), but a
    900px figure-mode render has ~3x the pixels and the same absolute-pixel
    text/bar was reading as tiny relative to the image (2026-08-13 feedback).
    """
    h, w = img.shape[:2]
    px = int(round(target_um / um_per_px / 4))
    if px < 5 or px > w - 20:
        return img
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.4 * k
    thickness = max(1, round(k))
    text = f"{target_um:g} um"
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin, gap = round(6 * k), round(5 * k)
    if w - margin - px - gap - tw < 2:
        return img
    x1, y_base = w - margin, h - margin
    bar_cy = y_base - th // 2
    bar_half_h = round(2 * k)
    cv2.rectangle(img, (x1 - px, bar_cy - bar_half_h), (x1, bar_cy + bar_half_h), (255, 255, 255), -1)
    cv2.putText(img, text, (x1 - px - gap - tw, y_base), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return img


class _Tile(NamedTuple):
    """One rendered crop, plus what an overlay needs to draw on it.

    `scale` and `x0`/`y0` are the crop's own geometry: a full-frame point (fx, fy)
    lands at ((fx - x0) * scale, (fy - y0) * scale). `crop_h` is the PRE-upscale
    height, which is the only thing that converts the upscaled image back to
    micrometres per pixel -- see _stamp_tile.
    """
    img: np.ndarray
    cx: float          # subject's position within the crop, post-upscale
    cy: float
    scale: float
    crop_h: int
    x0: int
    y0: int


def _crop_tile(well: str, frame: int, cx: float, cy: float, half: int,
               color: bool, *, upscale_to: int = _UPSCALE_TO) -> "_Tile | None":
    """Cut a 2*half crop around (cx, cy), apply the display LUT, upscale small crops.

    Returns None when the box misses the field entirely, which every caller treats
    as "skip this frame" rather than as an error -- a crop centred on a held
    position can legitimately fall outside the image.

    The upscale is here rather than in each caller because the coordinate rescale
    that goes with it was the part getting copied: an overlay drawn at pre-upscale
    coordinates lands at a quarter of the way into the image and looks like a
    tracking error.

    upscale_to defaults to _UPSCALE_TO (the size tuned for MCP tools that return
    ImageContent inline, where every pixel is a token in this conversation) but is
    overridable per call -- show_cells_in_browser passes a much larger target
    (_FIGURE_UPSCALE_TO in tools_output.py) because that tool writes an HTML file
    to disk instead of returning images inline, so a bigger render costs disk space
    and browser paint time, not context tokens.
    """
    grey = _cm._frame_png(well, int(frame))
    h, w = grey.shape[:2]  # grey may be (h, w) grayscale or (h, w, 3) multi-channel composite
    cxi, cyi = int(round(cx)), int(round(cy))
    x0, x1 = max(0, cxi - half), min(w, cxi + half)
    y0, y1 = max(0, cyi - half), min(h, cyi + half)
    crop = grey[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    img = _colorize(crop, well, color)
    cx_crop, cy_crop = float(cxi - x0), float(cyi - y0)
    s = 1.0
    if img.shape[0] < upscale_to:
        s = upscale_to / img.shape[0]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
        cx_crop, cy_crop = cx_crop * s, cy_crop * s
    return _Tile(img, cx_crop, cy_crop, s, int(crop.shape[0]), int(x0), int(y0))


def _stamp_tile(tile: _Tile, label: str, um_px: float, scale_bar: bool,
                corner: str | None = None,
                corner_color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Burn the per-frame label in, optionally a bottom-left note, then the bar.

    The scale bar must be told the um/px of the UPSCALED image, not of the source
    frame -- hence crop_h / current height. Getting that ratio wrong mislabels the
    bar by the upscale factor, and the bar is the calibration check a researcher
    trusts over the numbers, so it is computed in exactly one place.

    All burned-in text/bar sizing scales with the tile's own resolution, not a fixed
    pixel size -- 1.0 at the base 312px render (unchanged from before). A 900px
    figure-mode render (show_cells_in_browser's lightbox) has ~2.9x the linear
    resolution; scaling the full ratio read as too big (2026-08-13 feedback, right
    after the first fix -- fixed-size overlays had read as too SMALL against it
    before that), so only half the excess is applied (~1.9x at 900px, closer to the
    "twice as big" originally asked for) -- text needs to grow with the image, but
    not linearly, or it starts to dominate a crop meant to show the cell.
    """
    img = tile.img
    ratio = img.shape[0] / _UPSCALE_TO
    k = 1.0 + (ratio - 1.0) * 0.5
    thickness = max(1, round(k))
    cv2.putText(img, label, (round(4 * k), round(14 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * k, (255, 255, 255), thickness, cv2.LINE_AA)
    if corner:
        cv2.putText(img, corner, (round(4 * k), img.shape[0] - round(6 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * k, corner_color, thickness, cv2.LINE_AA)
    if scale_bar:
        img = _scale_bar(img, um_px * (tile.crop_h / img.shape[0]), target_um=10.0, k=k)
    return img


def _encode(img: np.ndarray) -> ImageContent:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode image")
    return ImageContent(type="image", mime_type="image/png",
                        data=base64.b64encode(buf.tobytes()).decode())



# Ported from the trajectory-features branch's walk_trajectory() (never merged --
# that copy walked the pipeline's raw tracked_masks.dat memmap; this one walks the
# bundle's own labels/*.png, so it needs nothing beyond what already ships). Follows
# the nearest detected blob frame-to-frame, tolerant of a few consecutive misses
# (Cellpose mask flicker) before giving up and freezing -- replaces OFF-TRACK's old
# behaviour of freezing immediately at the track's last known position, which could
# not distinguish "the cell moved out of a static crop" from "it vanished".
_WALK_MAX_GAP_DIST = 60.0  # px; matches src/track.py's _bridge_track_gaps convention
_WALK_MAX_GAP_FRAMES = 4   # consecutive unresolved frames tolerated before freezing


def _label_img(well: str, frame: int) -> np.ndarray | None:
    p = _cm.BUNDLE / well / "labels" / f"frame_{frame:05d}.png"
    if not p.is_file():
        return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def _blob_centroids(label_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (x, y) of every distinct nonzero label value in one label map.

    No regionprops/skimage needed -- the label map is already a per-pixel id
    assignment, so this is a groupby-mean, not a connected-components pass.
    """
    ys, xs = np.nonzero(label_img)
    if len(ys) == 0:
        return np.array([]), np.array([])
    vals = label_img[ys, xs].astype(np.int64)
    order = np.argsort(vals, kind="stable")
    vals, ys, xs = vals[order], ys[order].astype(np.float64), xs[order].astype(np.float64)
    _, start_idx = np.unique(vals, return_index=True)
    counts = np.diff(np.append(start_idx, len(vals)))
    cxs = np.add.reduceat(xs, start_idx) / counts
    cys = np.add.reduceat(ys, start_idx) / counts
    return cxs, cys


_WALK_ESTABLISHED_TOL_PX = 3.0  # blob-to-track-centroid match tolerance for identifying a tracked blob


def _walk_positions(
    well: str, frames: list[int], seed_cx: float, seed_cy: float,
    *, boundary_frame: int | None = None, df=None,
) -> dict[int, tuple[float, float, bool]]:
    """Nearest-centroid walk across `frames` (already ordered outward from the
    track's boundary), starting adjacent to (seed_cx, seed_cy).

    Returns {frame: (cx, cy, resolved)}. `resolved=False` means the walk lost the
    cell here (too far, or nothing detected) and the position is carried over from
    the last frame it WAS resolved at -- a real "last known position", same as the
    old behaviour, just reached only after actually trying rather than immediately.

    boundary_frame/df (optional): if given, a candidate blob that matches an
    EXISTING track already alive before `boundary_frame` is refused, same as if
    nothing had been detected there -- 2026-08-16 field feedback found the walk
    locking onto a stable, long-established neighbour in 4/4 tries (tracks 51, 9,
    21, 12 on 20251016_ACTB_M1), never the real outcome, because nearest-by-
    distance alone cannot tell "the same cell re-acquired" from "a calm neighbour
    that happened to be closest." A blob with no matching track (untracked
    debris/a genuinely fresh detection) or one whose own track started at or
    after `boundary_frame` is still fair game -- only a track that predates the
    gap being walked through is refused. Without these two args the walk behaves
    exactly as before (matches list_nearby_tracks/watch_location_over_time
    callers that don't have a tracks table handy).
    """
    starts = None
    if boundary_frame is not None and df is not None:
        starts = df.sort_values("frame").groupby("track_id").frame.first()

    out: dict[int, tuple[float, float, bool]] = {}
    cx, cy = seed_cx, seed_cy
    misses = 0
    for f in frames:
        img = _label_img(well, f)
        cxs, cys = _blob_centroids(img) if img is not None else (np.array([]), np.array([]))
        if len(cxs) == 0:
            misses += 1
            out[f] = (cx, cy, False)
            continue
        d = np.hypot(cxs - cx, cys - cy)
        i = int(np.argmin(d))
        if d[i] > _WALK_MAX_GAP_DIST * (misses + 1):
            misses += 1
            out[f] = (cx, cy, False)
            continue
        if starts is not None:
            frame_rows = df[df.frame == f]
            if not frame_rows.empty:
                td = np.hypot(frame_rows.cx.to_numpy() - cxs[i], frame_rows.cy.to_numpy() - cys[i])
                j = int(np.argmin(td))
                if td[j] <= _WALK_ESTABLISHED_TOL_PX:
                    cand_track = int(frame_rows.track_id.iloc[j])
                    cand_start = int(starts.get(cand_track, boundary_frame))
                    if cand_start < boundary_frame:
                        misses += 1
                        out[f] = (cx, cy, False)
                        continue
        misses = 0
        cx, cy = float(cxs[i]), float(cys[i])
        out[f] = (cx, cy, True)
    return out


def _filmstrip_frames(
    well: str, track_id: int,
    start_frame: int | None, end_frame: int | None,
    *,
    max_images: int | None, crop_um: float,
    color: bool, scale_bar: bool, marker: bool,
    stride_min: float = _STRIDE_MIN, cap: int = MAX_IMAGES,
    upscale_to: int = _UPSCALE_TO,
) -> tuple[str, list[np.ndarray]]:
    """Shared by follow_cells_over_time (MCP images) and show_cells_in_browser (HTML page).

    Returns (header text, rendered crop images) -- see follow_cells_over_time's docstring
    for the semantics; this is that function's body with the ImageContent
    encoding split off so a second caller can embed the same pixels differently.
    """
    df = _cm._tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    t_lo, t_hi = int(t.frame.min()), int(t.frame.max())

    # Deliberately NOT clamped to the track's lifetime. The clamp this replaces was
    # silent, and since a track typically terminates at the very event the caller is
    # asking about, it reliably withheld the only frames that answered the question.
    # The range is still clamped to frames that exist on disk, which is a real limit
    # rather than a bookkeeping one.
    lo = max(0, t_lo if start_frame is None else int(start_frame))
    hi = min(n_frames - 1, t_hi if end_frame is None else int(end_frame))
    if hi < lo:
        raise ValueError(
            f"empty range: start_frame={start_frame} end_frame={end_frame} resolves to "
            f"{lo}-{hi}. {well} has frames 0-{n_frames - 1}; track {track_id} is tracked "
            f"in {t_lo}-{t_hi}."
        )

    avail = list(range(lo, hi + 1))
    picks, pick_note = _pick_frames(well, avail, max_images, cap, stride_min)
    n = len(picks)

    # Anchor for frames outside the track's lifetime: its first or last known position.
    by_frame = {int(r.frame): r for r in t.itertuples()}
    first_row, last_row = by_frame[t_lo], by_frame[t_hi]

    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))
    suspect = set(m.get("track_multiplicity", {}).get("suspect_tracks", []))

    n_off = sum(1 for f in picks if f not in by_frame)
    # Walk outward from each boundary across every frame in range (not just the
    # sampled picks) so the miss-tolerance state stays meaningful -- a walk that
    # skipped straight to a far pick would have no idea how many frames it "missed".
    off_before = [f for f in picks if f not in by_frame and f < t_lo]
    off_after = [f for f in picks if f not in by_frame and f > t_hi]
    walked: dict[int, tuple[float, float, bool]] = {}
    if off_before:
        walked.update(_walk_positions(
            well, list(range(t_lo - 1, min(off_before) - 1, -1)), first_row.cx, first_row.cy,
            boundary_frame=t_lo, df=df))
    if off_after:
        walked.update(_walk_positions(
            well, list(range(t_hi + 1, max(off_after) + 1)), last_row.cx, last_row.cy,
            boundary_frame=t_hi, df=df))
    n_walked = sum(1 for f in (off_before + off_after) if walked.get(f, (0, 0, False))[2])

    # The ring is drawn clear of the nucleus, never over it: the chromatin's shape is
    # the evidence being judged, so an overlay across it would destroy the thing the
    # image exists to show.
    header = (f"{well} track {track_id}: frames {lo}-{hi} "
              f"({_minutes_between(well, lo, hi):.0f} min), {pick_note}. "
              f"Crop {crop_um:g} um wide, re-centred on the tracked cell each frame -- "
              f"the cell of interest is the one at the CENTRE of every image; others are "
              f"neighbours. Time is elapsed hours from the start of the recording."
              + _display_note(well))
    if n_off:
        header += (
            f" NOTE: {n_off} of these {n} frames fall outside the track's own lifetime "
            f"({t_lo}-{t_hi}) and are labelled OFF-TRACK. {n_walked} of them were "
            f"re-centred by following the nearest detected blob frame-to-frame (solid "
            f"orange ring); the rest could not be resolved that way and are frozen at "
            f"the last position that WAS resolved (dashed blue ring). Either way nothing "
            f"is confirmed to be this cell rather than its daughters or a neighbour that "
            f"drifted in -- judge those frames on the pixels alone."
        )
    if track_id in suspect:
        header += (" WARNING: the tracker merged two different cells under this id -- "
                   "expect the crop to jump between them; do not measure it.")

    images: list[np.ndarray] = []
    for f in picks:
        on_track = f in by_frame
        row = by_frame[f] if on_track else (first_row if f < t_lo else last_row)
        walk_resolved = False
        if on_track:
            cx, cy = float(row.cx), float(row.cy)
        else:
            cx, cy, walk_resolved = walked.get(f, (row.cx, row.cy, False))
        # Where the cell lands in the crop is not always its centre pixel: the crop
        # is clipped at the field edge.
        tile = _crop_tile(well, int(f), cx, cy, half, color, upscale_to=upscale_to)
        if tile is None:
            continue
        img, cx_crop, cy_crop, s = tile.img, tile.cx, tile.cy, tile.scale
        if marker or not on_track:
            # Ring radius from the cell's own area, pushed out far enough to clear it.
            # Solid white when on-track (a real detection); solid orange when
            # OFF-TRACK but the nearest-centroid walk resolved a position (likely
            # real, but never confirmed to be THIS cell); dashed blue when the walk
            # lost it and the position is frozen at the last resolved point, so
            # that case can never be mistaken for an actual detection.
            r_px = float(np.sqrt(max(float(row.area_px), 1.0) / np.pi)) * 1.9 * s
            r_px = float(np.clip(r_px, 10, min(img.shape[:2]) / 2 - 2))
            if on_track:
                ring = (255, 255, 255)
                cv2.circle(img, (int(cx_crop), int(cy_crop)), int(r_px), ring, 1, cv2.LINE_AA)
            elif walk_resolved:
                ring = (0, 165, 255)
                cv2.circle(img, (int(cx_crop), int(cy_crop)), int(r_px), ring, 1, cv2.LINE_AA)
            else:
                ring = (80, 160, 255)
                for a in range(0, 360, 30):  # dashed
                    cv2.ellipse(img, (int(cx_crop), int(cy_crop)), (int(r_px), int(r_px)),
                                0, a, a + 15, ring, 1, cv2.LINE_AA)
        label = f"f{int(f)} t={_cm._elapsed_str(well, int(f))} @({cx:.0f}, {cy:.0f})"
        if on_track:
            if row.n_masks_in_frame > 1:
                label += f" [{int(row.n_masks_in_frame)} masks]"
        elif walk_resolved:
            label += " OFF-TRACK (walked)"
        else:
            label += f" OFF-TRACK (held @f{t_lo if f < t_lo else t_hi})"
        images.append(_stamp_tile(tile, label, um_px, scale_bar))
    return header, images


def _nearest_detection(well: str, frame: int, x: float, y: float,
                       exclude: int | None = None) -> tuple[int, float] | None:
    """(track_id, distance_um) of the closest tracked cell to a point in one frame.

    `exclude` drops one id from the search -- in anchor mode the anchor is always
    nearest to itself at 0.0 um, which says nothing; the useful answer is what ELSE
    is near the place being watched.
    """
    df = _cm._tracks(well)
    f = df[df.frame == frame]
    if exclude is not None:
        f = f[f.track_id != exclude]
    if f.empty:
        return None
    d = np.hypot(f.cx.to_numpy() - x, f.cy.to_numpy() - y)
    i = int(np.argmin(d))
    return int(f.track_id.iloc[i]), float(d[i]) * _cm._manifest(well)["pixel_size_um"]


def _fixed_point_frames(
    well: str,
    start_frame: int, end_frame: int,
    x: float | None, y: float | None,
    anchor_track_id: int | None,
    *,
    max_images: int | None, crop_um: float,
    color: bool, scale_bar: bool,
    stride_min: float = _STRIDE_MIN, cap: int = MAX_IMAGES,
    upscale_to: int = _UPSCALE_TO, crosshair: bool = True,
) -> tuple[str, list[np.ndarray]]:
    """Shared by watch_location_over_time (MCP images) and show_cells_in_browser (HTML page).

    Crop on a PLACE (or a neighbour's track, as a stable vantage point) rather than
    a mask -- see watch_location_over_time's docstring for the semantics; this is
    that function's body with the ImageContent encoding split off so a second caller
    can embed the same pixels differently, same split as _filmstrip_frames above.
    """
    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    lo, hi = max(0, int(start_frame)), min(n_frames - 1, int(end_frame))
    if hi < lo:
        raise ValueError(f"empty range {start_frame}-{end_frame}; {well} has 0-{n_frames - 1}.")

    anchor_pos: dict[int, tuple[float, float]] = {}
    if anchor_track_id is not None:
        t = _cm._tracks(well)
        t = t[t.track_id == anchor_track_id]
        if t.empty:
            raise ValueError(f"anchor track {anchor_track_id} not found in {well}.")
        anchor_pos = {int(r.frame): (float(r.cx), float(r.cy)) for r in t.itertuples()}
        known = sorted(anchor_pos)
    elif x is None or y is None:
        raise ValueError("give either x and y (full-frame pixels), or anchor_track_id.")

    def _centre(f: int) -> tuple[float, float]:
        if anchor_track_id is None:
            return float(x), float(y)
        if f in anchor_pos:
            return anchor_pos[f]
        nf = min(known, key=lambda k: abs(k - f))
        return anchor_pos[nf]

    avail = list(range(lo, hi + 1))
    picks, pick_note = _pick_frames(well, avail, max_images, cap, stride_min)

    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))

    where = (f"anchored on track {anchor_track_id}" if anchor_track_id is not None
             else f"fixed at ({float(x):.0f}, {float(y):.0f}) px")
    spec = (f"{well}: frames {lo}-{hi} ({_minutes_between(well, lo, hi):.0f} min), "
            f"{pick_note}, {where}. Crop {crop_um:g} um wide.")
    where_word = "crosshair" if crosshair else "crop centre"
    gen = (
        "This is a PLACE, not a tracked object -- nothing is ringed, because a ring "
        "would claim a detection that was never made."
        + (" The yellow crosshair marks WHERE YOU ASKED to look." if crosshair else
           " No crosshair is drawn on this page -- it would read as a detection ring "
           "to someone opening it cold, so the report view omits it; the crop is "
           "centred on the requested point regardless.")
        + f" Each frame's label also names the NEAREST tracked cell and how far its "
        f"centre sits from the {where_word} (only when closer than the crop is "
        "wide) -- that is the nearest cell, not necessarily the thing at the "
        f"{where_word}, which may be untracked or nothing at all. A nearest cell many "
        "microns away means the thing at this position is not tracked, which is the "
        "usual reason to be here. Distances are centre-to-centre, so a large nucleus "
        "can read several microns away while still overlapping the point. Time is "
        "elapsed hours from the start of the recording." + _display_note(well)
    )
    header = spec + _HDR_SEP + gen

    images: list[np.ndarray] = []
    for f in picks:
        ccx, ccy = _centre(int(f))
        tile = _crop_tile(well, int(f), ccx, ccy, half, color, upscale_to=upscale_to)
        if tile is None:
            continue
        img = tile.img
        cxp, cyp = int(tile.cx), int(tile.cy)
        if crosshair:
            for dx0, dx1 in ((-12, -5), (5, 12)):
                cv2.line(img, (cxp + dx0, cyp), (cxp + dx1, cyp), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(img, (cxp, cyp + dx0), (cxp, cyp + dx1), (0, 255, 255), 1, cv2.LINE_AA)

        near = _nearest_detection(well, int(f), ccx, ccy, exclude=anchor_track_id)
        corner = (f"~{near[0]} @{near[1]:.0f}um"
                  if near is not None and near[1] < crop_um else None)
        label = f"f{int(f)} t={_cm._elapsed_str(well, int(f))} @({ccx:.0f}, {ccy:.0f})"
        images.append(_stamp_tile(tile, label, um_px, scale_bar, corner=corner))
    return header, images


_FAMILY_MAX_MEMBERS = 6

# How far past a fully-known member set's own last frame to keep rendering, once
# there's nothing left in the set to show -- a couple of confirmation frames of the
# settled daughters, not the full forward-discovery horizon (_WINDOW_AFTER_MIN,
# ~180 min) that exists to search for a daughter that might not have appeared yet.
# 2026-08-16 (researcher feedback on ACTB track 2->387+388, daughters end f100/f104
# at ~3 min/frame): "holding for too long, the mitosis is long done -- I'd cap it at
# frame 104, could go 105 106", i.e. essentially zero slack, a frame or two at most.
_FAMILY_TAIL_BUFFER_MIN = 6.0

# The mirror problem on the FRONT of the window: before_min (a fixed default, ~120
# min) can stop well short of an anchor member's own real tracked history even
# though that history is right there and cheap to show -- on the same ACTB case,
# the mother (track 2) is tracked all the way back to f0 (the very start of the
# recording), 264 min before the transition, but before_min=120 only reached back
# to f47, silently cutting ~140 min of real interphase lead-in. Extending to the
# earliest known member's own first frame is capped by this constant so a mother
# alive for days doesn't reopen "hundreds of frames of nothing" (the failure mode
# _WINDOW_BEFORE_MIN's fixed default exists to avoid in the first place) --
# generous enough to reach this case's 264 min gap, bounded everywhere else.
_FAMILY_LEAD_IN_CAP_MIN = 300.0


def _resolve_family_centres(
    win, pos: dict[int, list], picks: list[int], *, hold_after_end: bool = False,
) -> tuple[dict[int, tuple[float, float, int, int]], set[int], dict[int, list[int]]]:
    """One crop centre per sampled frame, plus which frames are gap-filled or held.

    Returns {frame: (cx, cy, n_seen, n_gap)}, the HELD frames, and {frame: [gap ids]}.

    A member missing for a frame or two in the MIDDLE of its own span is a
    segmentation dropout, not a departure. Dropping it from the mean swings the centre
    onto whoever is left, and the crop pans with nothing in the image saying so -- on
    M12 that shift was read as a cell having moved, and then as evidence of a lagging
    chromosome. So a member inside its span keeps contributing its last measured
    position while it is missing; only a member whose span has genuinely ended (the
    mother, after the handoff) leaves the mean, which is what keeps the handoff
    working with no mode switch.

    Strictly inside its span: a member never contributes before its first appearance
    or after its last, because there is nothing measured to hold there.

    hold_after_end changes that last rule: once True, a member whose span has
    genuinely ended keeps contributing its LAST measured position forever after,
    same as a mid-span gap. 2026-08-16 field feedback: with several members in a
    family, a member's span ending (not just a mid-span gap) silently re-centres
    the crop onto whichever members remain, which can be a large jump if the
    vanished member was spatially far from the survivors -- a real correction cost
    on three separate events in one session. Off by default because most callers
    DO want the mean to move on once a member is confirmed gone (that's the whole
    point of the handoff, mother to daughters); this is for the case where the
    caller wants the framing to hold still on a spot a member left rather than
    snap onto whoever's left.
    """
    mspan = {int(t): (int(g.frame.min()), int(g.frame.max()))
             for t, g in win.groupby("track_id")}
    seen_at: dict[int, dict[int, tuple[float, float]]] = {}
    for r in win.itertuples():
        seen_at.setdefault(int(r.track_id), {})[int(r.frame)] = (float(r.cx), float(r.cy))

    centres: dict[int, tuple[float, float, int, int]] = {}
    held: set[int] = set()
    gapped: dict[int, list[int]] = {}
    last: tuple[float, float] | None = None
    for f in picks:
        rows_f = pos.get(f, [])
        present = [(float(r.cx), float(r.cy)) for r in rows_f]
        here = {int(r.track_id) for r in rows_f}
        gap_ids, gap_pts = [], []
        for t, (a, b) in sorted(mspan.items()):
            if t in here:
                continue
            in_span = a < f < b
            past_end = hold_after_end and f >= b
            if not (in_span or past_end):
                continue
            earlier = [g for g in seen_at[t] if g < f]
            if earlier:
                gap_ids.append(t)
                gap_pts.append(seen_at[t][max(earlier)])
        pts = present + gap_pts
        if pts:
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            last = (cx, cy)
            centres[f] = (cx, cy, len(present), len(gap_pts))
            if gap_ids:
                gapped[f] = gap_ids
        elif last is not None:
            centres[f] = (*last, 0, 0)
            held.add(f)
        else:
            # No member present yet and nothing to hold -- fall back to the earliest
            # member's first known position rather than guessing.
            first_row = win.sort_values("frame").iloc[0]
            centres[f] = (float(first_row.cx), float(first_row.cy), 0, 0)
            held.add(f)
    return centres, held, gapped


def _family_filmstrip_frames(
    well: str, track_ids: list[int],
    start_frame: int | None, end_frame: int | None,
    *,
    max_images: int | None, crop_um: float | None,
    color: bool, scale_bar: bool, marker: bool,
    before_min: float = _WINDOW_BEFORE_MIN, after_min: float = _WINDOW_AFTER_MIN,
    stride_min: float = _STRIDE_MIN, cap: int = MAX_IMAGES,
    added: list[int] | None = None, centre_frame: int | None = None,
    upscale_to: int = _UPSCALE_TO, min_crop_um: float = 90.0,
    hold_centre_after_member_end: bool = False,
) -> tuple[str, list[np.ndarray]]:
    """Crop centred on a SET of tracks, resolved per frame from whoever is present.

    This is the "before, during, after" strip. Give it a mother and her daughters and
    the crop follows the mother while only she exists, then the daughters' midpoint
    once they appear -- with no mode switch, because the centre is just the mean of
    the members present in that frame and membership does the switching.

    Why a member set rather than a fixed (x, y): a static point only works when the
    subject happens not to migrate. On one M12 division it would have been fine (the
    midpoint sat within ~5 px for 30 frames); in general cells move, and you would be
    back to hand-computing a position per frame. Members are already measured every
    frame in tracks.csv, so the set costs nothing and cannot drift off its subject.

    Four calls worth knowing about, each of which could reasonably have gone the
    other way:

    1. PLAIN MEAN, not area-weighted. Weighting drags the centre onto the big object,
       which is exactly wrong in the case you most need to see -- a micronucleus
       logged as a daughter is tiny, and weighting pushes it toward the crop edge or
       out of frame. The fragment is the evidence.
    2. ONE crop size for the whole strip, auto-fitted (crop_um=None). Sizing per
       frame would rescale every image, the nuclei would appear to breathe, and a
       rendering artifact would read as biology.
    3. Gaps are HELD, never interpolated -- at two levels. A member missing inside its
       own span is a segmentation dropout, so it keeps contributing its last measured
       position and the frame is labelled 'gap'; without that the mean swings onto
       whoever remains and the crop pans with nothing in the image to explain it (on
       M12 that shift was read as a cell moving, then as a lagging chromosome). A frame
       where NO member is present reuses the whole last centre and is labelled HELD.
       Neither invents a position: interpolating would render as a measured one.
    4. Members are capped (default 6), chosen ONCE by median area over the window.
       A cell undergoing necrosis can shatter into many ids; re-picking per frame
       would make the centre lurch as membership churned. The strip stays jagged --
       that is honest -- but it stays on the same objects.
    """
    df = _cm._tracks(well)
    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    um_px = m["pixel_size_um"]

    ids = [int(t) for t in dict.fromkeys(track_ids)]
    sub = df[df.track_id.isin(ids)]
    missing = [t for t in ids if t not in set(sub.track_id.unique().tolist())]
    if sub.empty:
        raise ValueError(f"none of {ids} are tracks in {well}. Use list_tracks().")

    spans = {int(t): (int(g.frame.min()), int(g.frame.max()))
             for t, g in sub.groupby("track_id")}

    # Auto-window on the MEMBERSHIP TRANSITION -- the frame where a member other than
    # the earliest-starting one first appears. For a division that is the handoff
    # frame, so the window lands either side of it. Using the members' full span
    # instead would open the mother's entire lifetime, hundreds of frames of nothing.
    #
    # The window is measured in MINUTES, and it reaches much further forward than
    # back, because the transition is where the TRACKER gave up and not where the
    # cell divided -- the mitotic figure can be ~20 min past it. See the comment on
    # _WINDOW_AFTER_MIN: rendering to the transition is what made real divisions
    # look like artifacts.
    # centre_frame wins when given, because inferring the transition from member spans
    # only means anything for a mother-plus-daughters set. Hand-pick the members from
    # list_nearby_tracks and the "second-earliest member's first frame" rule starts
    # sliding the window AWAY from the event -- adding the intermediate tracks of BeWo
    # 1824 moved it from f470-510 to f463-503, off the mitosis it was opened for.
    starts = sorted(spans.items(), key=lambda kv: kv[1][0])
    transition = (int(centre_frame) if centre_frame is not None
                  else (starts[1][1][0] if len(starts) > 1 else starts[0][1][0]))
    known = [t for t in ids if t in spans]
    lo = (_frame_at_offset_min(well, transition, -before_min)
          if start_frame is None else max(0, int(start_frame)))
    hi = (_frame_at_offset_min(well, transition, after_min)
          if end_frame is None else min(n_frames - 1, int(end_frame)))
    # Don't stop short of real, already-tracked history that before_min's fixed
    # default happens not to reach -- extend lo back to the earliest known member's
    # own first frame (typically the mother) when that's before the auto lo,
    # capped so a long-lived mother doesn't reopen an unbounded lead-in. See
    # _FAMILY_LEAD_IN_CAP_MIN.
    if start_frame is None and known:
        first_start = min(spans[t][0] for t in known)
        if first_start < lo and _minutes_between(well, first_start, transition) <= _FAMILY_LEAD_IN_CAP_MIN:
            lo = first_start
    # Once the full cast is already known (2+ members -- not a lone mother still
    # being searched for a daughter, which is exactly the case the forward-heavy
    # after_min above exists for) and every member's own span already ends well
    # before the auto window's forward edge, the frames past that point are pure
    # HELD -- a frozen crop of nothing. On ACTB track 2->387+388 (2026-08-16),
    # 23 of 52 rendered frames (44%) were this empty tail past f104, while the
    # actual division (f88-104) got only 8; shrinking the window here instead
    # gives every one of the (now fewer) picks below meaningful content, at the
    # SAME target stride_min, rather than diluting them across a mostly-empty
    # window. Only touches the auto path -- pass end_frame explicitly to look
    # further than a known member's own lifetime on purpose.
    if end_frame is None and len(known) > 1:
        last_end = max(spans[t][1] for t in known)
        hi = min(hi, _frame_at_offset_min(well, last_end, _FAMILY_TAIL_BUFFER_MIN))
    if hi < lo:
        raise ValueError(f"empty range: resolves to {lo}-{hi}; {well} has 0-{n_frames - 1}.")

    avail = list(range(lo, hi + 1))
    picks, pick_note = _pick_frames(well, avail, max_images, cap, stride_min)
    n = len(picks)

    win = sub[(sub.frame >= lo) & (sub.frame <= hi)]
    # A member with no rows inside the window was not "dropped" -- it simply is not
    # there, which is a different fact and needs saying differently. Conflating the two
    # produced "6 members were dropped to keep the centre stable" for a window in which
    # six of the seven members had already ended.
    present = set(int(t) for t in win.track_id.unique())
    absent = [t for t in ids if t not in present]
    ids = [t for t in ids if t in present]
    kept = ids
    dropped: list[int] = []
    if len(ids) > _FAMILY_MAX_MEMBERS:
        # LONGEST-LIVED first, size only as a tie-break. Ranking by median area threw
        # out BeWo 1893 -- f480-497, the surviving daughter and the only member that
        # carried the outcome -- for being the smallest object in the set. The member
        # you cannot afford to lose is the one that is still there at the end.
        # cx/cy medians ride along on this same groupby so the far-member spatial
        # check below (guarded by `if dropped`, so only reached when this branch
        # ran) does not pay for a second full groupby over `win` just for them.
        rank = (win.groupby("track_id")
                .agg(n=("frame", "size"), a=("area_px", "median"),
                     cx=("cx", "median"), cy=("cy", "median"))
                .sort_values(["n", "a"], ascending=False))
        kept = [int(t) for t in rank.index[:_FAMILY_MAX_MEMBERS]]
        dropped = [t for t in ids if t not in kept]
        win = win[win.track_id.isin(kept)]

    pos: dict[int, list] = {}
    for r in win.itertuples():
        pos.setdefault(int(r.frame), []).append(r)

    centres, held, gapped = _resolve_family_centres(
        win, pos, picks, hold_after_end=hold_centre_after_member_end)

    # Auto-fit ONE crop width: the 90th percentile, over EVERY frame in the window
    # (not just the sampled `picks`), of the radius needed to contain every present
    # member (centroid distance plus that member's own radius). A percentile, not
    # the max, because one fragment drifting away would otherwise zoom the whole
    # strip out to the size of the field. Sampling only `picks` used to miss the
    # widest-separation frame entirely when it fell between stride-sampled picks --
    # 2026-08-16 field feedback: a family of two daughters 70+ um apart auto-fit to
    # a crop that only showed one, because the frame where they were furthest apart
    # wasn't one of the ~12 rendered. `pos` already covers every frame in [lo, hi]
    # (it's the same dict _resolve_family_centres uses), so this costs nothing extra
    # to compute over the full range instead of the subsample.
    #
    # min_crop_um is the floor of that clip -- 90.0 by default, the same value
    # watch_location_over_time was bumped to on 2026-08-15 after identical
    # feedback ("crop too tight, hides neighbour nuclei that cause mix-ups").
    # That fix only touched watch_location_over_time; this function's own floor
    # (previously 25.0, ~1.2x a typical ACTB nucleus's ~20um diameter) was left
    # untouched, so every family/division view kept the tight crop regardless --
    # "still way too zoomed in... over multiple iterations" (2026-08-16), because
    # every prior fix corrected COVERAGE (does the crop contain every tracked
    # member) without ever touching CONTEXT (room to see neighbours), a
    # different axis. 90um is ~4.5x nucleus diameter, matching the precedent.
    # The upper clip (was 120.0) is raised to 250.0 for the same reason: a
    # genuinely wide-spread family should not be clamped back down below what
    # its real separation needs.
    auto = crop_um is None
    if auto:
        radii = []
        for rows_f in pos.values():
            if not rows_f:
                continue
            cx = float(np.mean([r.cx for r in rows_f]))
            cy = float(np.mean([r.cy for r in rows_f]))
            radii.append(max(
                float(np.hypot(r.cx - cx, r.cy - cy))
                + float(np.sqrt(max(float(r.area_px), 1.0) / np.pi))
                for r in rows_f))
        r_px = float(np.percentile(radii, 90)) if radii else 20.0
        crop_um = float(np.clip(2 * r_px * um_px * 1.15, min_crop_um, 250.0))

    half = max(8, int(round(crop_um / um_px / 2)))

    who = ", ".join(str(t) for t in kept)

    # The header is built in two halves, separated by _HDR_SEP: what is true of THIS
    # strip, then the standing explanation of how the tool renders. A page of 14 cases
    # repeated the standing half 14 times, and the reviewer stopped reading it after
    # the first -- which means the warnings inside it stopped working. show_cells_in_browser
    # hoists the second half to the top of the page and prints it once.
    spec = [
        f"{well} tracks [{who}]: frames {lo}-{hi} "
        f"({_minutes_between(well, lo, hi):.0f} min), {pick_note}. "
        f"Crop {crop_um:g} um wide{' (auto-fit)' if auto else ''}. "
        "Member spans: " + "; ".join(
            f"{t}:f{spans[t][0]}-{spans[t][1]}" for t in kept if t in spans) + "."
    ]
    if start_frame is None and end_frame is None:
        spec.append(
            f"Window auto-chosen around "
            + (f"the frame you centred on, f{transition}"
               if centre_frame is not None
               else f"the membership transition at f{transition}")
            + f": {before_min:g} min before, {after_min:g} min after.")
    if gapped:
        ids_g = sorted({t for v in gapped.values() for t in v})
        spec.append(f"{len(gapped)} frame(s) labelled 'gap' (member "
                    f"{', '.join(str(t) for t in ids_g)} not segmented there).")
    # A HELD frame is a frozen crop of a PLACE, and it costs exactly what a real one
    # costs. When most of the strip is HELD the member set does not cover the window
    # that was asked for, and the reader finds out only by spending the images and
    # looking at them -- 8 of 12 in one call on 2026-08-01, a third of that session's
    # whole budget. Say it up front, and hand back the call that fixes it: the real
    # daughters are usually unlinked ids that begin inside the window, because the
    # tracker breaks AT the division rather than through it.
    severe = held and len(held) >= max(2, len(picks) // 3)
    if held and not severe:
        spec.append(f"{len(held)} frame(s) labelled HELD -- no member present.")
    elif severe:
        cand: dict[int, int] = {}
        for r in df[df.frame.isin(held) & ~df.track_id.isin(kept)].itertuples():
            cx_f, cy_f, _, _ = centres[int(r.frame)]
            if float(np.hypot(r.cx - cx_f, r.cy - cy_f)) * um_px <= crop_um / 2.0:
                cand[int(r.track_id)] = cand.get(int(r.track_id), 0) + 1
        top = sorted(cand, key=lambda t: -cand[t])[:4]
        msg = (f"WARNING -- the members cover only {len(picks) - len(held)} of "
               f"{len(picks)} sampled frames. The other {len(held)} are HELD: a frozen "
               f"crop of a place, showing whatever happens to sit there.")
        if top:
            msg += (f" Segmented inside the crop on those frames but NOT in this set: "
                    f"{', '.join(f'{t} ({cand[t]}f)' for t in top)}. Add them -- "
                    f'follow_cells_over_time(well="{well}", '
                    f"track_ids={sorted(set(kept) | set(top))}, centre_frame={transition}) "
                    f"-- or list_nearby_tracks(well, track_id={kept[0] if kept else ids[0]}) "
                    f"to see everything there. Unlinked ids beginning LATE are the "
                    f"expected shape of a real division here, not an anomaly.")
        else:
            msg += (" Nothing else was segmented inside the crop on those frames either, "
                    "so the window most likely reaches past the event: re-centre with "
                    "centre_frame= on the condensation peak, or cut after_min.")
        spec.insert(0, msg)
    if dropped:
        spec.append(f"{len(dropped)} further member(s) were dropped to keep the centre "
                    f"stable ({', '.join(str(t) for t in dropped)}); the {len(kept)} "
                    f"kept are the longest-lived over this window.")
        # 2026-08-15 feedback: "8 further member(s) were dropped..." retained a
        # long-persisting but spatially-unrelated track while dropping the short
        # connecting fragments that actually told the story -- a crop that visibly
        # jumps between unrelated objects, with nothing above explaining why. If a
        # KEPT member sits farther from every other kept member than the crop is
        # wide, it is not sharing a scene with the rest of the set; say so rather
        # than let the reader discover it by watching the crop jump.
        if len(kept) > 1:
            med_pos = {t: (float(rank.loc[t, "cx"]), float(rank.loc[t, "cy"])) for t in kept}
            far = [t for t in kept if min(
                float(np.hypot(med_pos[t][0] - med_pos[u][0],
                               med_pos[t][1] - med_pos[u][1])) * um_px
                for u in kept if u != t) > crop_um]
            if far:
                spec.append(
                    f"WARNING: {', '.join(str(t) for t in far)} "
                    f"{'sits' if len(far) == 1 else 'sit'} farther from every other "
                    f"kept member than the crop is wide -- the crop below will jump "
                    f"between unrelated objects rather than show one coherent scene. "
                    f"Consider dropping {'it' if len(far) == 1 else 'them'} by hand "
                    f"instead: pass a track_ids list that leaves "
                    f"{'it' if len(far) == 1 else 'them'} out."
                )
    if absent:
        spec.append(f"{', '.join(str(t) for t in absent)} "
                    f"{'is' if len(absent) == 1 else 'are'} not present anywhere in "
                    f"this window and contribute nothing to the centre.")
    if missing:
        spec.append(f"NOT FOUND in {well} and ignored: "
                    f"{', '.join(str(t) for t in missing)}.")
    shown_added = [t for t in (added or []) if t in kept]
    if shown_added:
        spec.append(
            f"{', '.join(str(t) for t in shown_added)} "
            f"{'is' if len(shown_added) == 1 else 'are'} in this strip by POSITION, "
            f"not by lineage: {'it' if len(shown_added) == 1 else 'they'} began next "
            f"to the mother within a few frames of the link, but nothing links "
            f"{'it' if len(shown_added) == 1 else 'them'} to her. Included so the crop "
            f"holds the whole event rather than the half the tracker recorded -- "
            f"treat as an object worth seeing, not as a recorded daughter.")

    gen = [
        "Each frame is centred on the MEAN position of the members present in it, so "
        "the crop follows whoever exists: the mother alone before the handoff, the "
        "daughters' midpoint after. The centre moves with membership, so the field CAN "
        "pan between frames without anything in the image having moved -- read the "
        "per-frame label before reading the scene as a change. One crop size for the "
        "whole strip, so nothing rescales between frames. Time is elapsed hours from "
        "the start of the recording.",
        "The auto window reaches much further FORWARD than back on purpose: the "
        "membership transition is where the tracker's link ENDS, which is not where "
        "the cell divides -- on BeWo the mitotic figure has been seen ~20 min later. "
        "Judging a division on the frames up to the transition systematically calls "
        "real mitoses artifacts.",
        "A 'gap' frame is a segmentation dropout: the member is missing there but "
        "present on both sides, so it keeps contributing its last measured position "
        "and is NOT ringed -- its pixels are usually still visible. A HELD frame has "
        "no member at all and reuses the previous centre. Neither is interpolated.",
        _display_note(well).strip(),
    ]
    if not marker:
        gen.append("Nothing is ringed: with several members in frame, one ring would "
                   "be ambiguous about what it is claiming. Pass marker=true to ring "
                   "them all.")
    header = " ".join(spec) + _HDR_SEP + " ".join(g for g in gen if g)

    images: list[np.ndarray] = []
    for f in picks:
        cx_f, cy_f, n_present, n_gap = centres[f]
        tile = _crop_tile(well, int(f), cx_f, cy_f, half, color, upscale_to=upscale_to)
        if tile is None:
            continue
        img, s = tile.img, tile.scale
        if marker:
            # Ring ALL present members or none. One ring among several says nothing
            # about which cell the claim is about.
            for r in pos.get(f, []):
                rx, ry = (r.cx - tile.x0) * s, (r.cy - tile.y0) * s
                rad = float(np.sqrt(max(float(r.area_px), 1.0) / np.pi)) * 1.9 * s
                rad = float(np.clip(rad, 8, min(img.shape[:2]) / 2 - 2))
                cv2.circle(img, (int(rx), int(ry)), int(rad), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"f{int(f)} t={_cm._elapsed_str(well, int(f))} @({cx_f:.0f}, {cy_f:.0f})"
        if f in held:
            label += " HELD"
        elif n_gap:
            # Name the missing ones: "1 seen" alone would read as a cell having gone,
            # which is the misreading this whole mechanism exists to prevent.
            label += (f" [{n_present} seen, gap {','.join(str(t) for t in gapped[f])}]")
        images.append(_stamp_tile(tile, label, um_px, scale_bar))
    return header, images


# How far either side of the mother's last frame to look for the mitotic figure.
# Forward-heavy for the same reason the filmstrip window is: the link is where the
# tracker gave up, and the figure can be ~20 min past it.
# How far from the mother's own edge, and how far either side of the link IN MINUTES,
# to look for a daughter the tracker never linked.
#
# Minutes and forward-heavy, for the third time in this file and for the same reason:
# the link is where the tracker gave up, not where the cell divided. A first cut used
# +/- 4 FRAMES and found nothing on BeWo 969, whose anaphase is 23 frames past its
# link -- the very case that prompted this. Real sisters appear late.

__all__ = [
    "_colorize", "_DISPLAY_NOTE_BASE", "_display_note", "_scale_bar", "_Tile",
    "_crop_tile", "_stamp_tile", "_encode",
    "_WALK_MAX_GAP_DIST", "_WALK_MAX_GAP_FRAMES", "_label_img", "_blob_centroids",
    "_walk_positions", "_filmstrip_frames", "_FAMILY_MAX_MEMBERS",
    "_resolve_family_centres", "_family_filmstrip_frames",
    "_nearest_detection", "_fixed_point_frames",
]
