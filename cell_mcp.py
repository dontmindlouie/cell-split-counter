"""Local stdio MCP server: a filesystem over time-lapse microscopy pixels.

Read-only except for annotate(), which appends to a CSV of its own. Serves a bundle
built by scripts/build_bundle.py -- indexed frame PNGs, PNG-16 label maps, a
per-frame track table, and a manifest carrying calibration read from the ND2 at
build time. Nothing here touches an ND2, a GPU, torch, or Cellpose, so the install
stays pure-python.

Point it at a bundle with the CELL_BUNDLE_DIR environment variable.

Docstrings and type hints ARE the schema the model sees, so they are written for
someone who has never used a microscope.
"""

import base64
import io
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mcp.server import MCPServer
from mcp.types import ImageContent

server = MCPServer(
    "cell-microscopy",
    version="0.1.0",
    instructions=(
        "Read-only access to time-lapse microscopy of dividing cells. Start with "
        "list_wells(), then list_tracks() to find cells. Before spending images on a "
        "track, call get_track_profile() -- it's free (no images) and often shows where "
        "to look: a sparkline of solidity, area, and brightness across the track's "
        "frames. Solidity (area / convex-hull area) is the strongest single signal -- it "
        "dips as a mask rounds up during mitosis, even when area barely moves. Then use "
        "get_filmstrip() to watch "
        "the flagged frames closely and measure() for real units. Expect a track to END "
        "at the moment its cell divides, with the daughters carrying new track_ids -- so "
        "if a cell's filmstrip stops abruptly, the event you want is just past it: use "
        "get_lineage() for the daughter ids, and pass start_frame/end_frame beyond the "
        "track's own lifetime, which get_filmstrip renders rather than truncating. The "
        "same thing happens in reverse: a track can just as easily BEGIN mid-division, "
        "so a track that looks already mid-event on its very first frame may need frames "
        "from BEFORE first_frame (via the mother, from get_lineage) to see the lead-up. "
        "Two more rules that matter: the interval between frames is NOT constant, so "
        "never compute durations from frame counts -- use measure() or time_ms; and some "
        "track_ids are flagged as merged cells, which must not be measured. Images show "
        "chromatin only (H2B-mCherry), so the shapes are nuclei rather than whole cells. "
        "If a cell looks dim or unusual and you can't tell whether that's the cell itself "
        "or the whole field, call get_neighbourhood_stats() before spending more images on "
        "it -- it's free and separates a cell-autonomous change from bleaching/defocus."
    ),
)

BUNDLE = Path(os.environ.get("CELL_BUNDLE_DIR", "data/bundle")).expanduser()

# A hard cap, not a default. Every image costs the model a large amount of
# context, and a filmstrip of 40 frames reliably exhausts it mid-task.
MAX_IMAGES = 12

# Filmstrip crops are tiny in absolute pixels -- a 60 um crop is 104 px here, and a
# nucleus inside it is about 21 px across. That 21 px is the microscope's limit, not
# ours (the ND2 is natively 1024x1024, and nothing upstream downsamples), so upscaling
# adds no information. It does add legibility: the previous 2x INTER_NEAREST to 160 px
# turned soft chromatin into hard blocks, and chromatin texture is the entire evidence
# for calling a stage. LANCZOS at 3x is visibly better on the same pixels. Ringing is
# not a concern on diffraction-limited fluorescence, which has no sharp edges to ring.
# Cost is negligible: an image this size is ~130 tokens.
_UPSCALE_TO = 312


# --------------------------------------------------------------------------- io

@lru_cache(maxsize=64)
def _manifest(well: str) -> dict:
    import json
    p = BUNDLE / well / "manifest.json"
    if not p.is_file():
        raise ValueError(f"unknown well {well!r}. Call list_wells() for valid names.")
    return json.loads(p.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _tracks(well: str) -> pd.DataFrame:
    """Per-frame track table. Cached -- these are 100k-500k rows each."""
    return pd.read_csv(BUNDLE / well / "tracks.csv")


def _frame_png(well: str, frame: int) -> np.ndarray:
    p = BUNDLE / well / "frames" / f"frame_{frame:05d}.png"
    if not p.is_file():
        n = _manifest(well)["n_frames"]
        raise ValueError(f"frame {frame} not in {well} (has frames 0-{n - 1})")
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _hours(well: str, frame: int) -> float:
    ts = _manifest(well)["frame_timestamps_ms"]
    return (ts[frame] - ts[0]) / 3.6e6


# ---------------------------------------------------------------------- render

def _colorize(grey: np.ndarray, well: str, color: bool) -> np.ndarray:
    """Apply the acquisition's own display LUT, so renders match what the
    researcher sees in Fiji rather than a colormap we invented."""
    if not color:
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    rgb = _manifest(well).get("display_color_rgb") or [255, 255, 255]
    r, g, b = [v / 255.0 for v in rgb]
    return cv2.merge([(grey * b).astype(np.uint8),
                      (grey * g).astype(np.uint8),
                      (grey * r).astype(np.uint8)])


def _scale_bar(img: np.ndarray, um_per_px: float, target_um: float = 20.0) -> np.ndarray:
    """Burn a labelled scale bar into the bottom-right corner.

    This is the calibration check: the researcher compares it once against her
    own measurement of the same cell, rather than trusting a number in a file.
    """
    h, w = img.shape[:2]
    px = int(round(target_um / um_per_px))
    if px < 5 or px > w - 20:
        return img
    x1, y1 = w - 12, h - 14
    cv2.rectangle(img, (x1 - px, y1), (x1, y1 + 5), (255, 255, 255), -1)
    cv2.putText(img, f"{target_um:g} um", (x1 - px, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _encode(img: np.ndarray) -> ImageContent:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode image")
    return ImageContent(type="image", mime_type="image/png",
                        data=base64.b64encode(buf.tobytes()).decode())


# ----------------------------------------------------------------------- tools

@server.tool()
def list_wells() -> str:
    """List every imaged well available, with its cell line and how long it ran.

    Each well is one field of view filmed over 1-3 days. Start here: the well
    name is the first argument to every other tool.
    """
    if not BUNDLE.is_dir():
        return f"No bundle at {BUNDLE}. Set CELL_BUNDLE_DIR to a built bundle directory."
    out = ["well | cell_line | frames | hours | tracks | um/px | interval_min"]
    for d in sorted(BUNDLE.iterdir()):
        if not (d / "manifest.json").is_file():
            continue
        m = _manifest(d.name)
        out.append(
            f"{d.name} | {m.get('cell_line') or '?'} | {m['n_frames']} | "
            f"{m['duration_hours']:.1f} | {m['n_tracks']} | {m['pixel_size_um']:.4f} | "
            f"{m['interval_ms']['median'] / 60000:.1f}"
        )
    out.append(
        "\nNote: the time between frames is NOT constant. Never assume a fixed interval -- "
        "use measure() or the per-frame timestamps, which are exact."
    )
    return "\n".join(out)


@server.tool()
def list_tracks(
    well: str,
    min_frames: int = 1,
    present_at_frame: int | None = None,
    sort_by: str = "duration",
    limit: int = 30,
) -> str:
    """List tracked cells in a well, most interesting first.

    A "track" is one cell followed over time, identified by track_id. Use this to
    find cells worth looking at before calling get_filmstrip on them.

    Args:
        well: well name from list_wells().
        min_frames: ignore tracks seen in fewer frames than this. Defaults to 1, i.e.
            nothing is hidden -- many real divisions and deaths produce 1-2 frame
            tracks, so a higher floor silently deletes the events you are looking
            for. Raise it only to suppress segmentation noise while browsing.
        present_at_frame: only tracks visible at this frame number.
        sort_by: "duration" (longest-lived first), "area" (largest first), or
            "start" (earliest first).
        limit: max rows returned. Keep small; this is a browsing tool.
    """
    df = _tracks(well)
    if present_at_frame is not None:
        keep = set(df.loc[df.frame == present_at_frame, "track_id"])
        df = df[df.track_id.isin(keep)]

    df = df.sort_values("frame")
    g = df.groupby("track_id").agg(
        first_frame=("frame", "min"), last_frame=("frame", "max"),
        n_frames=("frame", "nunique"), mean_area_um2=("area_um2", "mean"),
        max_masks=("n_masks_in_frame", "max"),
        # Position at birth and at death. Enough to find a track's neighbours, or to
        # spot the daughters of a division near where the mother was last seen,
        # without a measure() call per candidate.
        first_x=("cx", "first"), first_y=("cy", "first"),
        last_x=("cx", "last"), last_y=("cy", "last"),
    ).reset_index()
    g = g[g.n_frames >= min_frames]
    if g.empty:
        return f"No tracks in {well} with >= {min_frames} frames."

    key = {"duration": "n_frames", "area": "mean_area_um2", "start": "first_frame"}.get(sort_by, "n_frames")
    g = g.sort_values(key, ascending=(sort_by == "start")).head(limit)

    suspect = set(_manifest(well).get("track_multiplicity", {}).get("suspect_tracks", []))
    lines = ["track_id | frames | first | last | mean_area_um2 | xy_at_first | xy_at_last | flags"]
    for r in g.itertuples():
        flags = []
        if r.track_id in suspect:
            flags.append("UNRELIABLE-merged-cells")
        elif r.max_masks > 1:
            flags.append("sometimes-2-masks")
        lines.append(f"{r.track_id} | {r.n_frames} | {r.first_frame} | {r.last_frame} | "
                     f"{r.mean_area_um2:.0f} | {r.first_x:.0f},{r.first_y:.0f} | "
                     f"{r.last_x:.0f},{r.last_y:.0f} | {','.join(flags) or '-'}")
    lines.append(
        "\nUNRELIABLE-merged-cells: the tracker merged two different cells under one id; "
        "do not measure these. sometimes-2-masks: occasionally covers 2 shapes -- check visually."
    )
    return "\n".join(lines)


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """Render a sequence of numbers as one line of unicode block characters.

    Scaled to this sequence's own min/max, not any global range -- the point is
    to show shape (a dip, a spike, a plateau), not to compare absolute levels
    against another track's sparkline.
    """
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return _SPARK_CHARS[0] * len(values)
    n = len(_SPARK_CHARS)
    return "".join(
        _SPARK_CHARS[min(int((v - lo) / (hi - lo) * n), n - 1)] for v in values
    )


@server.tool()
def get_track_profile(well: str, track_id: int) -> str:
    """See how a cell's size, shape, and brightness change over its whole track, with no images.

    Free to call -- reads numbers already measured from the video, not pixels --
    so use this BEFORE get_filmstrip to decide which frames are worth spending
    images on. `solidity` (area / convex-hull area) is the strongest single
    signal: it dips when a mask rounds up or briefly fragments during mitosis,
    even in cases where area barely moves, and recovers over a few frames once
    the division resolves -- a dip-then-recover shape is a better division tell
    than either channel's raw min/max. Area's own useful shape is different:
    a division HALVES it and it stays down (the tracker follows one daughter),
    while a transient area dip that fully bounces back is more often noise than
    a real event. None of this is certain from numbers alone -- it narrows
    where to look, it doesn't replace looking.

    Args:
        well: well name from list_wells().
        track_id: the cell to profile, from list_tracks().
    """
    df = _tracks(well)
    t = df[df.track_id == track_id].sort_values("frame")
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")
    # One row per frame even where n_masks_in_frame > 1 -- profiling a multiplexed
    # id would blend two cells' numbers, which is the same reason measure() warns.
    t = t.drop_duplicates("frame", keep="first")

    frames = t.frame.tolist()
    area = t.area_um2.tolist()
    inten = t.intensity_integrated.tolist()
    has_solidity = "solidity" in t.columns
    solidity = t.solidity.tolist() if has_solidity else []

    def _biggest_jump(vals: list[float]) -> str:
        if len(vals) < 2:
            return "n/a (single frame)"
        deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        i = max(range(len(deltas)), key=lambda k: abs(deltas[k]))
        pct = (deltas[i] / vals[i] * 100) if vals[i] else float("inf")
        return (f"frame {frames[i]}->{frames[i + 1]}: {vals[i]:.0f} -> {vals[i + 1]:.0f} "
                f"({pct:+.0f}%)")

    def _lowest_point(vals: list[float]) -> str:
        i = min(range(len(vals)), key=lambda k: vals[k])
        return f"frame {frames[i]}: {vals[i]:.3f} (track range {min(vals):.3f}-{max(vals):.3f})"

    suspect = track_id in set(_manifest(well).get("track_multiplicity", {}).get("suspect_tracks", []))
    warn = ("\nWARNING: this track is flagged as merged cells (see list_tracks) -- "
            "these numbers mix two cells and any jump may be an id-swap, not a real "
            "event.\n") if suspect else ""

    lines = [
        f"{well} track {track_id}: frames {frames[0]}-{frames[-1]} ({len(frames)} points)" + warn,
    ]
    if has_solidity:
        lines.append(f"  solidity   {_sparkline(solidity)}  (min {min(solidity):.3f}, max {max(solidity):.3f})")
    lines.append(f"  area_um2   {_sparkline(area)}  (min {min(area):.0f}, max {max(area):.0f})")
    lines.append(f"  brightness {_sparkline(inten)}  (min {min(inten):.0f}, max {max(inten):.0f})")
    if has_solidity:
        lines.append(f"  lowest solidity (best mitosis candidate)  {_lowest_point(solidity)}")
    else:
        lines.append("  (no solidity column -- this bundle predates 2026-07-30; rebuild to get it)")
    lines.append(f"  biggest area jump        {_biggest_jump(area)}")
    lines.append(f"  biggest brightness jump  {_biggest_jump(inten)}")
    lines.append(
        "\nEach sparkline is scaled to this track's own min/max, so shape (a dip, a "
        "spike, a plateau) is meaningful but the two bars are not on the same scale as "
        "each other or as another track's. These are candidate frames to spend a "
        "get_filmstrip call on -- not a verdict; none of these numbers alone can tell "
        "a division from a death from a tracking artifact."
    )
    return "\n".join(lines)


# structured_output=False is load-bearing, not tidiness. ImageContent is a pydantic
# model, so annotating it as the return type makes the SDK build a structured-output
# model for the tool and emit the result TWICE -- once as an image content block, and
# again as JSON in structured_content, base64 payload and all. At downscale=1 that
# second copy is ~550k characters of base64 and blows the tool-output limit on its own.
@server.tool(structured_output=False)
def get_frame(well: str, frame: int, downscale: int = 2,
              color: bool = True, scale_bar: bool = True) -> ImageContent:
    """Show one whole field of view, to get oriented.

    Use this to see the overall layout and pick a region or cell, then use
    get_filmstrip to follow a specific cell closely over time.

    Args:
        well: well name from list_wells().
        frame: frame number, 0-based. See list_wells() for how many exist.
        downscale: shrink by this factor to save space. 2 is usually plenty. At 1 a
            single frame costs a large fraction of the context window, and a nucleus
            is still only ~20 px across -- too coarse to judge chromatin either way.
            Use get_filmstrip for anything that depends on a cell's shape.
        color: apply the microscope's own display colour (matches Fiji).
        scale_bar: burn in a labelled scale bar.
    """
    grey = _frame_png(well, frame)
    if downscale > 1:
        grey = cv2.resize(grey, None, fx=1 / downscale, fy=1 / downscale,
                          interpolation=cv2.INTER_AREA)
    img = _colorize(grey, well, color)
    if scale_bar:
        img = _scale_bar(img, _manifest(well)["pixel_size_um"] * downscale)
    return _encode(img)


def _filmstrip_frames(
    well: str, track_id: int,
    start_frame: int | None, end_frame: int | None,
    max_images: int, crop_um: float,
    color: bool, scale_bar: bool, marker: bool,
) -> tuple[str, list[np.ndarray]]:
    """Shared by get_filmstrip (MCP images) and render_browser (HTML gallery).

    Returns (header text, rendered crop images) -- see get_filmstrip's docstring
    for the semantics; this is that function's body with the ImageContent
    encoding split off so a second caller can embed the same pixels differently.
    """
    df = _tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    m = _manifest(well)
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
    n = min(max_images, MAX_IMAGES, len(avail))
    picks = [avail[i] for i in np.linspace(0, len(avail) - 1, n).astype(int)]

    # Anchor for frames outside the track's lifetime: its first or last known position.
    by_frame = {int(r.frame): r for r in t.itertuples()}
    first_row, last_row = by_frame[t_lo], by_frame[t_hi]

    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))
    suspect = set(m.get("track_multiplicity", {}).get("suspect_tracks", []))

    n_off = sum(1 for f in picks if f not in by_frame)
    # The ring is drawn clear of the nucleus, never over it: the chromatin's shape is
    # the evidence being judged, so an overlay across it would destroy the thing the
    # image exists to show.
    header = (f"{well} track {track_id}: frames {lo}-{hi}, showing {n} of {len(avail)}. "
              f"Crop {crop_um:g} um wide, re-centred on the tracked cell each frame -- "
              f"the cell of interest is the one at the CENTRE of every image; others are "
              f"neighbours. Time is elapsed hours from the start of the recording.")
    if n_off:
        header += (
            f" NOTE: {n_off} of these {n} frames fall outside the track's own lifetime "
            f"({t_lo}-{t_hi}) and are labelled OFF-TRACK. There the crop is frozen at the "
            f"cell's nearest known position and ringed: nothing is tracked or centred, so "
            f"whatever sits there may be this cell, its daughters, or an unrelated "
            f"neighbour that drifted in. Judge those frames on the pixels alone."
        )
    if track_id in suspect:
        header += (" WARNING: the tracker merged two different cells under this id -- "
                   "expect the crop to jump between them; do not measure it.")

    images: list[np.ndarray] = []
    for f in picks:
        on_track = f in by_frame
        row = by_frame[f] if on_track else (first_row if f < t_lo else last_row)
        grey = _frame_png(well, int(f))
        h, w = grey.shape
        cx, cy = int(round(row.cx)), int(round(row.cy))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        crop = grey[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        img = _colorize(crop, well, color)
        # Where the cell actually landed in the crop, before any upscaling -- the crop
        # is clipped at the field edge, so this is not always the centre pixel.
        cx_crop, cy_crop = cx - x0, cy - y0
        if img.shape[0] < _UPSCALE_TO:
            s = _UPSCALE_TO / img.shape[0]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
            cx_crop, cy_crop = cx_crop * s, cy_crop * s
        else:
            s = 1.0
        if marker or not on_track:
            # Ring radius from the cell's own area, pushed out far enough to clear it.
            # Dashed and off-colour when OFF-TRACK, so a held position can never be
            # mistaken for a detected cell.
            r_px = float(np.sqrt(max(float(row.area_px), 1.0) / np.pi)) * 1.9 * s
            r_px = float(np.clip(r_px, 10, min(img.shape[:2]) / 2 - 2))
            ring = (255, 255, 255) if on_track else (80, 160, 255)
            if on_track:
                cv2.circle(img, (int(cx_crop), int(cy_crop)), int(r_px), ring, 1, cv2.LINE_AA)
            else:
                for a in range(0, 360, 30):  # dashed
                    cv2.ellipse(img, (int(cx_crop), int(cy_crop)), (int(r_px), int(r_px)),
                                0, a, a + 15, ring, 1, cv2.LINE_AA)
        label = f"f{int(f)} t={_hours(well, int(f)):.1f}h"
        if on_track:
            if row.n_masks_in_frame > 1:
                label += f" [{int(row.n_masks_in_frame)} masks]"
        else:
            label += f" OFF-TRACK (held @f{t_lo if f < t_lo else t_hi})"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if scale_bar:
            img = _scale_bar(img, um_px * (crop.shape[0] / img.shape[0]), target_um=10.0)
        images.append(img)
    return header, images


@server.tool()
def get_filmstrip(
    well: str, track_id: int,
    start_frame: int | None = None, end_frame: int | None = None,
    max_images: int = 8, crop_um: float = 60.0,
    color: bool = True, scale_bar: bool = True, marker: bool = False,
) -> list:
    """Follow one cell over time as a series of close-up images.

    The crop re-centres on the cell in every frame, so a moving cell stays in
    view. This is the main tool for judging what a cell is actually doing --
    dividing, dying, or sitting still.

    You may request frames outside the track's own lifetime, and you often should:
    a track usually ENDS as its cell divides, with the daughters carrying new
    track_ids, so the division itself lies just past the last tracked frame. Those
    frames are rendered with the crop held at the cell's nearest known position and
    are labelled OFF-TRACK -- nothing is centred or identified for you there, so
    read them as "this patch of the field", not "this cell".

    Frames are sampled evenly across the requested range, so a wide range gives a
    coarse overview and a narrow range gives frame-by-frame detail. Each image is
    labelled with its frame number and elapsed time.

    Args:
        well: well name from list_wells().
        track_id: the cell to follow, from list_tracks().
        start_frame: defaults to when the cell first appears. May precede it.
        end_frame: defaults to when it was last seen. May follow it.
        max_images: how many frames to show (hard capped at 12).
        crop_um: width of the crop in micrometres. 60 shows a cell and its
            immediate neighbours; lower it to zoom in.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
        marker: draw a thin ring around the tracked cell, sized to sit clear of it.
            Off by default because the ring is one more shape in an image whose
            shapes are the evidence. Turn it on for wide crops, where "the one in
            the middle" stops being obvious. Forced on for OFF-TRACK frames, where
            the ring marks the held position rather than a detected cell.
    """
    header, images = _filmstrip_frames(
        well, track_id, start_frame, end_frame, max_images, crop_um, color, scale_bar, marker
    )
    from mcp.types import TextContent
    out: list = [TextContent(type="text", text=header)]
    out.extend(_encode(img) for img in images)
    return out


@lru_cache(maxsize=8)
def _lineage(well: str) -> dict[int, dict]:
    p = BUNDLE / well / "lineage.csv"
    if not p.is_file():
        return {}
    import csv
    out = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        out[int(r["track_id"])] = {
            "parent": int(r["parent_id"]) if r.get("parent_id") not in (None, "") else None,
            "daughters": [int(x) for x in (r.get("daughter_ids") or "").split() if x],
        }
    return out


@server.tool()
def get_lineage(well: str, track_id: int) -> str:
    """Find a cell's mother and daughters, to follow it across a division.

    Essential because a track usually ENDS when its cell divides: the two daughters
    are given new track_ids, so a mother's own filmstrip stops just as the
    interesting part begins. This tells you which ids to look at next.

    This is tracking bookkeeping, not a judgement. "Daughters" means the tracker
    linked these ids across a division; whether that division was real, normal, or
    even a division at all is still yours to decide from the images.

    Args:
        well: well name from list_wells().
        track_id: the cell whose relatives you want.
    """
    lin = _lineage(well)
    if not lin:
        return (f"No lineage.csv in this bundle for {well}, so mother/daughter links are "
                f"unavailable. You can still follow a cell past the end of its track: "
                f"get_filmstrip accepts start_frame/end_frame outside the track's lifetime "
                f"and will render those frames as OFF-TRACK.")

    cov = _manifest(well).get("lineage", {}).get("coverage", "unknown")
    df = _tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")
    lo, hi = int(t.frame.min()), int(t.frame.max())

    rec = lin.get(track_id, {"parent": None, "daughters": []})
    lines = [f"{well} track {track_id}: tracked in frames {lo}-{hi}"]

    def _span(tid: int) -> str:
        s = df[df.track_id == tid]
        return f"frames {int(s.frame.min())}-{int(s.frame.max())}" if not s.empty else "not in tracks.csv"

    p = rec["parent"]
    lines.append(f"  mother    {p} ({_span(p)})" if p is not None else "  mother    none recorded")
    if rec["daughters"]:
        lines.append(f"  daughters {len(rec['daughters'])}")
        for d in rec["daughters"]:
            lines.append(f"    {d} ({_span(d)})")
    else:
        lines.append("  daughters none recorded")

    if cov == "partial":
        lines.append(
            "\nNOTE: this bundle's lineage is PARTIAL -- it was recovered from the "
            "pipeline's event list, which only covers tracks it flagged, so a missing "
            "mother or daughter here means UNKNOWN, not none. Absence is not evidence "
            "that the cell did not divide."
        )
    lines.append(
        "\nTo see the division itself, ask get_filmstrip for a range spanning the "
        "mother's last frames and the daughters' first frames -- it renders frames "
        "outside a track's own lifetime rather than truncating to it."
    )
    return "\n".join(lines)


@server.tool()
def measure(well: str, track_id: int, frame: int | None = None) -> str:
    """Measure a cell in real units (micrometres and hours), not pixels.

    Replaces measuring by hand. Calibration comes from the microscope file itself,
    so sizes are in um^2 and times account for the uneven gap between frames.

    Args:
        well: well name from list_wells().
        track_id: the cell to measure.
        frame: a specific frame, or leave empty for a summary over the cell's whole
            lifetime.
    """
    df = _tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}.")
    m = _manifest(well)
    suspect = track_id in set(m.get("track_multiplicity", {}).get("suspect_tracks", []))
    warn = ("\nWARNING: the tracker merged two different cells under this id. "
            "These numbers mix two cells and should not be used.\n") if suspect else ""

    if frame is not None:
        r = t[t.frame == frame]
        if r.empty:
            raise ValueError(f"track {track_id} not visible at frame {frame} "
                             f"(present in {t.frame.min()}-{t.frame.max()})")
        r = r.iloc[0]
        extra = ""
        if r.n_masks_in_frame > 1:
            extra = (f"\nNOTE: {int(r.n_masks_in_frame)} separate shapes share this id at this "
                     "frame; the values below describe only one of them.")
        return (f"{well} track {track_id}, frame {frame} ({_hours(well, frame):.2f} h)\n"
                f"  area           {r.area_um2:.1f} um^2  ({r.area_px:.0f} px)\n"
                f"  position       x={r.cx:.1f} y={r.cy:.1f} px\n"
                f"  brightness     mean {r.intensity_mean:.0f}, total {r.intensity_integrated:,.0f}\n"
                f"  (total brightness tracks DNA content -- the marker is a labelled histone)"
                + extra + warn)

    lo, hi = int(t.frame.min()), int(t.frame.max())
    return (f"{well} track {track_id} summary{warn}\n"
            f"  seen in        {t.frame.nunique()} frames, {lo}-{hi}\n"
            f"  elapsed        {_hours(well, hi) - _hours(well, lo):.2f} h "
            f"({_hours(well, lo):.2f} -> {_hours(well, hi):.2f} h)\n"
            f"  area um^2      mean {t.area_um2.mean():.1f}, min {t.area_um2.min():.1f}, "
            f"max {t.area_um2.max():.1f}\n"
            f"  total bright.  mean {t.intensity_integrated.mean():,.0f}, "
            f"first {t.intensity_integrated.iloc[0]:,.0f}, last {t.intensity_integrated.iloc[-1]:,.0f}\n"
            f"  frames w/ >1 shape sharing this id: {(t.n_masks_in_frame > 1).sum()}")


@server.tool()
def get_neighbourhood_stats(well: str, track_id: int, frame: int, n_neighbours: int = 5) -> str:
    """Compare one cell against its nearest neighbours at a single frame, in real units.

    Free to call -- no images. Use this whenever you're guessing at *why* a cell
    looks dim or unusual: a falling area/brightness z-score relative to neighbours
    right now means something is happening to this specific cell (death, focal
    drift out of this cell alone); a flat z-score with everything else also low
    means the whole field is dim (bleaching, defocus, an optical effect), not this
    cell dying. Neighbours are the N nearest OTHER tracked cells present in the
    same frame, by centre-to-centre distance -- not a fixed radius, so a sparse
    region still returns whatever is nearest, however far.

    Args:
        well: well name from list_wells().
        track_id: the cell to evaluate, from list_tracks().
        frame: the frame to compare at. Must be a frame where this track is present.
        n_neighbours: how many nearest other cells to compare against. Default 5.
    """
    df = _tracks(well)
    f = df[df.frame == frame].drop_duplicates("track_id", keep="first")
    if f.empty:
        raise ValueError(f"no tracks present in {well} at frame {frame}.")
    mine = f[f.track_id == track_id]
    if mine.empty:
        raise ValueError(f"track {track_id} not present in {well} at frame {frame}. "
                          "Use measure() to find frames where it is.")
    mine = mine.iloc[0]

    others = f[f.track_id != track_id].copy()
    if others.empty:
        return f"{well} frame {frame}: track {track_id} is the only tracked cell -- no neighbours to compare."
    others["dist_um"] = np.hypot(others.cx - mine.cx, others.cy - mine.cy) * _manifest(well)["pixel_size_um"]
    nearest = others.sort_values("dist_um").head(n_neighbours)

    def _z(value: float, pool: pd.Series) -> float:
        std = pool.std()
        return (value - pool.mean()) / std if std > 1e-9 else 0.0

    lines = [
        f"{well} frame {frame} ({_hours(well, frame):.2f} h), track {track_id} vs its "
        f"{len(nearest)} nearest neighbours (of {len(others)} other cells in this frame):",
        f"  this cell      area {mine.area_um2:.0f} um^2, mean brightness {mine.intensity_mean:.0f}",
        f"  nearest {len(nearest)}   area median {nearest.area_um2.median():.0f} um^2, "
        f"mean brightness median {nearest.intensity_mean.median():.0f}",
        f"  field ({len(others) + 1})  area median {f.area_um2.median():.0f} um^2, "
        f"mean brightness median {f.intensity_mean.median():.0f}",
        f"  z-score vs nearest neighbours   area {_z(mine.area_um2, nearest.area_um2):+.2f}, "
        f"brightness {_z(mine.intensity_mean, nearest.intensity_mean):+.2f}",
        f"  z-score vs whole field          area {_z(mine.area_um2, f.area_um2):+.2f}, "
        f"brightness {_z(mine.intensity_mean, f.intensity_mean):+.2f}",
        "",
        "  nearest cells: track_id | dist_um | area_um2 | mean_brightness",
    ]
    for r in nearest.itertuples():
        lines.append(f"    {r.track_id} | {r.dist_um:.1f} | {r.area_um2:.0f} | {r.intensity_mean:.0f}")
    lines.append(
        "\nReading it: a negative brightness z-score against neighbours but ~0 against the "
        "field means this cell alone is dim (cell-autonomous). A negative z-score against "
        "BOTH means the whole neighbourhood/field is dim (bleaching or defocus, not this "
        "cell's own state)."
    )
    return "\n".join(lines)


@server.tool()
def render_browser(well: str, events: list[dict]) -> str:
    """Write a self-contained HTML gallery of specific cells, to hand to a person.

    Use this once you've already worked out what's interesting via get_track_profile /
    get_filmstrip / get_lineage / get_neighbourhood_stats, and want to show it rather
    than describe it -- e.g. after answering "what happens to these 12 tracks", write
    one browser covering all of them instead of pasting filmstrips into chat one at a
    time. Renders each event with the exact same crop logic as get_filmstrip (colour
    LUT, scale bar, OFF-TRACK handling), so it looks like what you already reviewed.

    Args:
        well: well name from list_wells().
        events: one dict per cell to show, each with:
            track_id (required): the cell, from list_tracks().
            start_frame, end_frame (optional): as in get_filmstrip -- may fall outside
                the track's own lifetime, e.g. to show a division just past its end.
            label (optional): a short heading, e.g. "2036 -- divides, pro/meta/ana
                309/319/321". Defaults to "track <track_id>".
            max_images (optional, default 6): frames to render for this event.
            crop_um (optional, default 60.0): crop width in micrometres.
            marker (optional, default False): ring the tracked cell.

    Returns the path to the written .html file. Images are embedded as base64, so the
    file is portable on its own -- open it directly, or serve it
    (`python -m http.server`) if file:// is blocked in the browser being used.
    """
    if not events:
        raise ValueError("events is empty -- nothing to render.")

    sections = []
    for i, ev in enumerate(events):
        if "track_id" not in ev:
            raise ValueError(f"events[{i}] is missing required key 'track_id': {ev!r}")
        track_id = int(ev["track_id"])
        header, images = _filmstrip_frames(
            well, track_id,
            ev.get("start_frame"), ev.get("end_frame"),
            int(ev.get("max_images", 6)), float(ev.get("crop_um", 60.0)),
            True, True, bool(ev.get("marker", False)),
        )
        label = ev.get("label") or f"track {track_id}"
        tiles = []
        for img in images:
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            tiles.append(f'<img src="data:image/png;base64,{b64}" loading="lazy">')
        sections.append(
            f"<section><h2>{i + 1}. {label}</h2>"
            f"<p class=hdr>{header}</p>"
            f"<div class=filmstrip>{''.join(tiles)}</div></section>"
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{well} browser</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }}
h1 {{ font-weight: 400; }}
section {{ margin-bottom: 2.5rem; border-top: 1px solid #333; padding-top: 1rem; }}
h2 {{ font-size: 1.1rem; font-weight: 600; }}
p.hdr {{ color: #999; font-size: 0.85rem; max-width: 60rem; }}
div.filmstrip {{ display: flex; flex-wrap: wrap; gap: 4px; }}
div.filmstrip img {{ image-rendering: pixelated; max-height: 260px; border: 1px solid #333; }}
</style></head><body>
<h1>{well}</h1>
{''.join(sections)}
</body></html>"""

    out_dir = BUNDLE / well / "browsers"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    out_path = out_dir / f"browser_{time.strftime('%Y%m%d_%H%M%S')}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


_ANNOTATION_FIELDS = [
    "timestamp", "annotator", "well", "cell_line", "condition", "track_id",
    "event_id", "outcome_class",
    "condensation_frame", "metaphase_frame", "anaphase_frame", "exit_frame",
    "parent_id", "daughter_ids", "notes",
]


@server.tool()
def annotate(
    well: str, track_id: int, outcome_class: str,
    condensation_frame: int | None = None,
    metaphase_frame: int | None = None,
    anaphase_frame: int | None = None,
    exit_frame: int | None = None,
    parent_id: int | None = None,
    daughter_ids: list[int] | None = None,
    event_id: str | None = None,
    notes: str | None = None,
    annotator: str | None = None,
) -> str:
    """Record a human-verified verdict for a cell. Appends a new row -- never edits
    or overwrites a previous one, so nothing is ever silently lost or replaced.

    This is the actual payoff of everything else here: browsing produces nothing
    durable on its own, this is what turns a review session into a labeled dataset.
    Written to a SEPARATE file (<bundle>/<well>/annotations.csv), never mixed into
    events.csv -- that file is machine-generated, gets overwritten on every pipeline
    re-run, and only has a row for events the detector already found, so writing
    human verdicts onto it would silently cap what can ever be recorded at the
    detector's own recall. The most valuable annotation is often one where nothing
    in events.csv corresponds to it at all.

    Only call this after you (or the person you're working with) actually looked at
    the pixels via get_filmstrip -- this is a verdict, not a guess from get_track_profile
    numbers alone.

    Args:
        well: well name from list_wells().
        track_id: the cell being annotated.
        outcome_class: e.g. "divides", "dies", "neither" -- free text, but stay
            consistent within a well so later rollups can group on it.
        condensation_frame, metaphase_frame, anaphase_frame, exit_frame: the four
            stage marks (chromatin condensation onset -> metaphase alignment ->
            anaphase separation -> mitotic exit), as frame numbers. Leave any that
            don't apply or weren't determined as None -- durations between whichever
            marks ARE set can still be computed later from manifest.frame_timestamps_ms.
        parent_id: the track this cell was born from, if relevant and known (may
            differ from lineage.csv's own record, if you determined it was wrong).
        daughter_ids: track_ids of the cells born from this one, if it divided.
        event_id: the events.csv row this corresponds to, if any (e.g. "peak_frame"
            value or similar identifier) -- leave None if you found this yourself and
            nothing in events.csv flagged it. Never invent one.
        notes: free text for anything the typed fields don't capture.
        annotator: who determined this. Defaults to the CELL_MCP_ANNOTATOR
            environment variable if set, else "unspecified" -- set that env var once
            rather than passing this every call.
    """
    from datetime import datetime, timezone
    import csv as _csv

    m = _manifest(well)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotator": annotator or os.environ.get("CELL_MCP_ANNOTATOR", "unspecified"),
        "well": well,
        "cell_line": m.get("cell_line") or "",
        "condition": m.get("condition") or "",
        "track_id": track_id,
        "event_id": event_id or "",
        "outcome_class": outcome_class,
        "condensation_frame": condensation_frame if condensation_frame is not None else "",
        "metaphase_frame": metaphase_frame if metaphase_frame is not None else "",
        "anaphase_frame": anaphase_frame if anaphase_frame is not None else "",
        "exit_frame": exit_frame if exit_frame is not None else "",
        "parent_id": parent_id if parent_id is not None else "",
        "daughter_ids": " ".join(str(d) for d in daughter_ids) if daughter_ids else "",
        "notes": notes or "",
    }

    out_path = BUNDLE / well / "annotations.csv"
    is_new = not out_path.is_file()
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=_ANNOTATION_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)

    n = sum(1 for _ in open(out_path, encoding="utf-8")) - 1
    return f"Recorded. {out_path} now has {n} annotation(s) for {well}."


if __name__ == "__main__":
    if not BUNDLE.is_dir():
        print(f"warning: CELL_BUNDLE_DIR={BUNDLE} does not exist", file=sys.stderr)
    server.run(transport="stdio")
