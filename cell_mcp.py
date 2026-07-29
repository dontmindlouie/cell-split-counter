"""Local stdio MCP server: a filesystem over time-lapse microscopy pixels.

Read-only. Serves a bundle built by scripts/build_bundle.py -- indexed frame
PNGs, PNG-16 label maps, a per-frame track table, and a manifest carrying
calibration read from the ND2 at build time. Nothing here touches an ND2, a GPU,
torch, or Cellpose, so the install stays pure-python.

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
        "list_wells(), then list_tracks() to find cells, then get_filmstrip() to watch "
        "one over time and measure() for real units. Two rules that matter: the interval "
        "between frames is NOT constant, so never compute durations from frame counts -- "
        "use measure() or time_ms; and some track_ids are flagged as merged cells, which "
        "must not be measured. Images show chromatin only (H2B-mCherry), so the shapes are "
        "nuclei rather than whole cells."
    ),
)

BUNDLE = Path(os.environ.get("CELL_BUNDLE_DIR", "data/bundle")).expanduser()

# A hard cap, not a default. Every image costs the model a large amount of
# context, and a filmstrip of 40 frames reliably exhausts it mid-task.
MAX_IMAGES = 12


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
    min_frames: int = 20,
    present_at_frame: int | None = None,
    sort_by: str = "duration",
    limit: int = 30,
) -> str:
    """List tracked cells in a well, most interesting first.

    A "track" is one cell followed over time, identified by track_id. Use this to
    find cells worth looking at before calling get_filmstrip on them.

    Args:
        well: well name from list_wells().
        min_frames: ignore tracks seen in fewer frames than this. Short tracks are
            usually segmentation noise rather than real cells.
        present_at_frame: only tracks visible at this frame number.
        sort_by: "duration" (longest-lived first), "area" (largest first), or
            "start" (earliest first).
        limit: max rows returned. Keep small; this is a browsing tool.
    """
    df = _tracks(well)
    if present_at_frame is not None:
        keep = set(df.loc[df.frame == present_at_frame, "track_id"])
        df = df[df.track_id.isin(keep)]

    g = df.groupby("track_id").agg(
        first_frame=("frame", "min"), last_frame=("frame", "max"),
        n_frames=("frame", "nunique"), mean_area_um2=("area_um2", "mean"),
        max_masks=("n_masks_in_frame", "max"),
    ).reset_index()
    g = g[g.n_frames >= min_frames]
    if g.empty:
        return f"No tracks in {well} with >= {min_frames} frames."

    key = {"duration": "n_frames", "area": "mean_area_um2", "start": "first_frame"}.get(sort_by, "n_frames")
    g = g.sort_values(key, ascending=(sort_by == "start")).head(limit)

    suspect = set(_manifest(well).get("track_multiplicity", {}).get("suspect_tracks", []))
    lines = ["track_id | frames | first | last | mean_area_um2 | flags"]
    for r in g.itertuples():
        flags = []
        if r.track_id in suspect:
            flags.append("UNRELIABLE-merged-cells")
        elif r.max_masks > 1:
            flags.append("sometimes-2-masks")
        lines.append(f"{r.track_id} | {r.n_frames} | {r.first_frame} | {r.last_frame} | "
                     f"{r.mean_area_um2:.0f} | {','.join(flags) or '-'}")
    lines.append(
        "\nUNRELIABLE-merged-cells: the tracker merged two different cells under one id; "
        "do not measure these. sometimes-2-masks: occasionally covers 2 shapes -- check visually."
    )
    return "\n".join(lines)


@server.tool()
def get_frame(well: str, frame: int, downscale: int = 2,
              color: bool = True, scale_bar: bool = True) -> ImageContent:
    """Show one whole field of view, to get oriented.

    Use this to see the overall layout and pick a region or cell, then use
    get_filmstrip to follow a specific cell closely over time.

    Args:
        well: well name from list_wells().
        frame: frame number, 0-based. See list_wells() for how many exist.
        downscale: shrink by this factor to save space. 2 is usually plenty.
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


@server.tool()
def get_filmstrip(
    well: str, track_id: int,
    start_frame: int | None = None, end_frame: int | None = None,
    max_images: int = 8, crop_um: float = 60.0,
    color: bool = True, scale_bar: bool = True,
) -> list:
    """Follow one cell over time as a series of close-up images.

    The crop re-centres on the cell in every frame, so a moving cell stays in
    view. This is the main tool for judging what a cell is actually doing --
    dividing, dying, or sitting still.

    Frames are sampled evenly across the requested range, so a wide range gives a
    coarse overview and a narrow range gives frame-by-frame detail. Each image is
    labelled with its frame number and elapsed time.

    Args:
        well: well name from list_wells().
        track_id: the cell to follow, from list_tracks().
        start_frame: defaults to when the cell first appears.
        end_frame: defaults to when it was last seen.
        max_images: how many frames to show (hard capped at 12).
        crop_um: width of the crop in micrometres. 60 shows a cell and its
            immediate neighbours; lower it to zoom in.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
    """
    df = _tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    lo = int(t.frame.min() if start_frame is None else max(start_frame, t.frame.min()))
    hi = int(t.frame.max() if end_frame is None else min(end_frame, t.frame.max()))
    if hi < lo:
        raise ValueError(f"track {track_id} is only present in frames {t.frame.min()}-{t.frame.max()}")

    avail = sorted(set(t[(t.frame >= lo) & (t.frame <= hi)].frame))
    n = min(max_images, MAX_IMAGES, len(avail))
    picks = [avail[i] for i in np.linspace(0, len(avail) - 1, n).astype(int)]

    m = _manifest(well)
    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))
    suspect = set(m.get("track_multiplicity", {}).get("suspect_tracks", []))

    out: list = []
    # No marker is drawn on the cell deliberately: the subject is centred by
    # construction, and an overlay would sit on top of the chromatin whose shape
    # is the thing being judged.
    header = (f"{well} track {track_id}: frames {lo}-{hi}, showing {n} of {len(avail)}. "
              f"Crop {crop_um:g} um wide, re-centred on the tracked cell each frame -- "
              f"the cell of interest is the one at the CENTRE of every image; others are "
              f"neighbours. Time is elapsed hours from the start of the recording.")
    if track_id in suspect:
        header += (" WARNING: the tracker merged two different cells under this id -- "
                   "expect the crop to jump between them; do not measure it.")
    from mcp.types import TextContent
    out.append(TextContent(type="text", text=header))

    for f in picks:
        row = t[t.frame == f].iloc[0]
        grey = _frame_png(well, int(f))
        h, w = grey.shape
        cx, cy = int(round(row.cx)), int(round(row.cy))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        crop = grey[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        img = _colorize(crop, well, color)
        if img.shape[0] < 160:  # upscale small crops so the model can actually see them
            s = int(np.ceil(160 / img.shape[0]))
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        label = f"f{int(f)} t={_hours(well, int(f)):.1f}h"
        if row.n_masks_in_frame > 1:
            label += f" [{int(row.n_masks_in_frame)} masks]"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if scale_bar:
            img = _scale_bar(img, um_px * (crop.shape[0] / img.shape[0]), target_um=10.0)
        out.append(_encode(img))
    return out


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


if __name__ == "__main__":
    if not BUNDLE.is_dir():
        print(f"warning: CELL_BUNDLE_DIR={BUNDLE} does not exist", file=sys.stderr)
    server.run(transport="stdio")
