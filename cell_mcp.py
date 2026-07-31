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
import json
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
        "list_wells(). To find cells worth looking at in a well of thousands, call "
        "find_candidates() -- it is free, scans the whole well, and ranks what the "
        "data already records (recorded splits by how fragment-like they look, tracks "
        "that stop early). list_tracks() is the raw listing when you already know what "
        "you want. Before spending images on a "
        "track, call get_track_profile() -- it's free (no images) and often shows where "
        "to look: a sparkline of solidity, area, and brightness across the track's "
        "frames, plus how close the cell gets to the frame edge (a clipped nucleus has "
        "understated area and brightness, so check that before spending images). "
        "Solidity (area / convex-hull area) dips as a mask rounds up during mitosis, but "
        "how much it is worth depends on the cell line -- strong on compact RUES2 nuclei, "
        "near-useless on the large lobed nuclei of a WGD line, where it has no headroom. "
        "Then use get_filmstrip() to watch "
        "the flagged frames closely and measure() for real units. Expect a track to END "
        "at the moment its cell divides, with the daughters carrying new track_ids -- so "
        "if a cell's filmstrip stops abruptly, the event you want is just past it: use "
        "get_lineage() for the daughter ids, and pass start_frame/end_frame beyond the "
        "track's own lifetime, which get_filmstrip renders rather than truncating. The "
        "same thing happens in reverse: a track can just as easily BEGIN mid-division, "
        "so a track that looks already mid-event on its very first frame may need frames "
        "from BEFORE first_frame (via the mother, from get_lineage) to see the lead-up. "
        "When a mask-following crop keeps losing the cell, or the object you care about "
        "was never segmented at all and so has no track_id to ask about, switch to "
        "get_filmstrip_at() -- it watches a POSITION over time instead of a mask, and "
        "reports the nearest tracked cell per frame so you can tell what you are seeing. "
        "Two more rules that matter: the interval between frames is NOT constant, so "
        "never compute durations from frame counts -- use measure() or time_ms; and some "
        "track_ids are flagged as merged cells, which must not be measured. Images show "
        "chromatin only (H2B-mCherry), so the shapes are nuclei rather than whole cells. "
        "If a cell looks dim or unusual and you can't tell whether that's the cell itself "
        "or the whole field, call get_neighbourhood_stats() before spending more images on "
        "it -- it's free and separates a cell-autonomous change from bleaching/defocus. "
        "When the user asks to SEE something rather than be told about it -- 'show me', "
        "'let me see', 'send me' -- answer with show_cells(), which writes a page of "
        "labelled filmstrips they can open, rather than describing frames in prose. "
        "Record human verdicts with annotate(); it is the only file here a human owns."
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


def _edge_um(well: str, cx: float, cy: float) -> float:
    """Distance from a centroid to the nearest frame edge, in microns.

    Derived at read time from cx/cy rather than stored as a bundle column. A stored
    column only describes bundles built after it was added -- exactly the trap
    `solidity` fell into, where its absence means "not yet rebuilt" and not "zero".
    This is two subtractions, so there is no reason to persist it.

    Worth surfacing before any image is spent: a nucleus near the edge is clipped,
    so its area and intensity are wrong and it can walk out of frame entirely. On
    M4_nTSC this was a perfect off-screen detector, and in the M14 session a
    filmstrip decision was burned on a track at cx=1010 of 1024 that turned out to
    be unviewable.
    """
    m = _manifest(well)
    w, h = m.get("width_px"), m.get("height_px")
    if not w or not h:
        return float("nan")
    return min(cx, cy, w - cx, h - cy) * m["pixel_size_um"]


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
    lines = ["track_id | frames | first | last | mean_area_um2 | xy_at_first | xy_at_last | "
             "edge_um | flags"]
    for r in g.itertuples():
        flags = []
        if r.track_id in suspect:
            flags.append("UNRELIABLE-merged-cells")
        elif r.max_masks > 1:
            flags.append("sometimes-2-masks")
        # Closest approach to the edge over the track's life, not its mean: a cell
        # that spends one frame clipped has a clipped measurement in that frame.
        edge = min(_edge_um(well, r.first_x, r.first_y), _edge_um(well, r.last_x, r.last_y))
        if edge == edge and edge < 15:
            flags.append("NEAR-EDGE")
        lines.append(f"{r.track_id} | {r.n_frames} | {r.first_frame} | {r.last_frame} | "
                     f"{r.mean_area_um2:.0f} | {r.first_x:.0f},{r.first_y:.0f} | "
                     f"{r.last_x:.0f},{r.last_y:.0f} | "
                     f"{'?' if edge != edge else f'{edge:.0f}'} | {','.join(flags) or '-'}")
    lines.append(
        "\nUNRELIABLE-merged-cells: the tracker merged two different cells under one id; "
        "do not measure these. sometimes-2-masks: occasionally covers 2 shapes -- check visually."
        "\nedge_um: how close the cell gets to the frame boundary (nearest of its first and "
        "last position). NEAR-EDGE marks under 15 um, where a nucleus is likely clipped -- "
        "its area and brightness are then understated, and it may simply leave the field. "
        "Check this BEFORE spending a filmstrip on a track."
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
    images on.

    `solidity` (area / convex-hull area) dips when a mask rounds up or briefly
    fragments during mitosis, even where area barely moves, and recovers over a
    few frames once the division resolves. **How much it is worth depends on the
    cell line.** It was measured as the strongest single shape signal on RUES2,
    whose nuclei are compact; on a genome-doubled (WGD) line it was the WEAKEST
    thing available, because those nuclei are large and lobed at baseline so the
    statistic has no headroom -- a real division there reached only 0.918 while
    noise on a non-dividing cell dipped to 0.937. Check the well's cell_line in
    list_wells() before leaning on it, and read the dip RELATIVE to this track's
    own range rather than against an absolute cutoff.

    Area's useful shape is different: a division HALVES it and it stays down (the
    tracker follows one daughter), while a transient dip that fully bounces back
    is more often noise than a real event.

    None of this is certain from numbers alone -- it narrows where to look, it
    doesn't replace looking.

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
    mean_int = t.intensity_mean.tolist()
    has_solidity = "solidity" in t.columns
    solidity = t.solidity.tolist() if has_solidity else []

    # Mean brightness relative to every other cell in the same frame. Integrated
    # brightness alone hid a real death in eval v2 run 3 (track 849): its total
    # looked unremarkable while its mean sat ~4x below its neighbours for the
    # track's whole life. Per-frame z against the field separates "this cell is
    # going dark" from "the whole frame is going dark" without spending an image.
    fstats = (df.drop_duplicates(["frame", "track_id"])
                .groupby("frame")["intensity_mean"].agg(["mean", "std"]))
    zs: list[float] = []
    for f_i, v in zip(frames, mean_int):
        if f_i in fstats.index:
            mu, sd = fstats.loc[f_i, "mean"], fstats.loc[f_i, "std"]
            zs.append((v - mu) / sd if sd and sd > 1e-9 else 0.0)
        else:
            zs.append(0.0)

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
    edge = min(_edge_um(well, t.cx.iloc[0], t.cy.iloc[0]),
               _edge_um(well, t.cx.iloc[-1], t.cy.iloc[-1]))
    if edge == edge:
        lines.append(f"  closest to frame edge  {edge:.0f} um"
                     + ("   <-- NEAR EDGE: likely clipped, so area and brightness are "
                        "understated and the cell may leave the field" if edge < 15 else ""))

    # What happens at the track's boundary is usually the question being asked, and
    # it lives in a different file -- so surface it here rather than making the
    # reader call get_lineage to find out whether this track ends in a division.
    rec = _lineage(well).get(track_id, {})
    if rec.get("daughters") or rec.get("parent") is not None:
        bits = []
        if rec.get("parent") is not None:
            bits.append(f"born from {rec['parent']}")
        if rec.get("daughters"):
            bits.append(f"ends in a recorded split into {' + '.join(str(d) for d in rec['daughters'])}")
        line = "  lineage   " + "; ".join(bits)
        # DNA conservation across the boundary: the free check that told a real
        # division from a segmentation artifact on track 919. Paired with size,
        # because DNA alone passes a micronucleus -- 6425 reads DNA 0.94 and size
        # 0.10, and only the second number gives it away.
        kid = (rec.get("daughters") or [None])[0]
        kr = _lineage(well).get(kid, {}) if kid is not None else {}
        if kr.get("dna_ratio") is not None:
            line += (f"  [DNA {kr['dna_ratio']:.2f}, size {kr['size_ratio']:.2f}"
                     f"{' -- low size ratio means one is a fragment, not a daughter' if (kr.get('size_ratio') or 1) < 0.25 else ''}]")
        lines.append(line)
        lines.append("  (get_lineage has the link scores and any contested alternatives)")
    if has_solidity:
        lines.append(f"  solidity   {_sparkline(solidity)}  (min {min(solidity):.3f}, max {max(solidity):.3f})")
    lines.append(f"  area_um2   {_sparkline(area)}  (min {min(area):.0f}, max {max(area):.0f})")
    lines.append(f"  total bri. {_sparkline(inten)}  (min {min(inten):.0f}, max {max(inten):.0f})")
    lines.append(f"  mean bri.  {_sparkline(mean_int)}  (min {min(mean_int):.0f}, max {max(mean_int):.0f})")
    lines.append(f"  mean bri. z vs field  {_sparkline(zs)}  "
                 f"(min {min(zs):+.2f}, max {max(zs):+.2f}, median {sorted(zs)[len(zs) // 2]:+.2f})")
    if has_solidity:
        lines.append(f"  lowest solidity (best mitosis candidate)  {_lowest_point(solidity)}")
    else:
        lines.append("  (no solidity column -- this bundle predates 2026-07-30; rebuild to get it)")
    lines.append(f"  biggest area jump        {_biggest_jump(area)}")
    lines.append(f"  biggest total-bri. jump  {_biggest_jump(inten)}")
    lines.append(f"  biggest mean-bri. jump   {_biggest_jump(mean_int)}")
    lines.append(
        "\nEach sparkline is scaled to this track's own min/max, so shape (a dip, a "
        "spike, a plateau) is meaningful but the bars are not on the same scale as "
        "each other or as another track's. The one exception is the z row, which is "
        "already an absolute cross-cell number: read its printed min/max/median, not "
        "just its shape."
    )
    lines.append(
        "total vs mean brightness are different measurements and answer different "
        "questions. TOTAL (integrated) tracks DNA content, so it roughly doubles "
        "before a division and halves across one. MEAN (per-pixel) tracks how bright "
        "the chromatin is, so it is the one that falls when a nucleus dies or drifts "
        "out of focus -- a dying cell can hold its total while its mean collapses. "
        "The z row puts mean brightness on an absolute scale against every other cell "
        "in the same frame: persistently around -2 or lower is a cell that is dark "
        "relative to its peers throughout (suspect death or out-of-plane), while a z "
        "that stays near 0 as the raw mean falls means the whole frame is dimming "
        "(bleaching or defocus), not this cell. Use get_neighbourhood_stats to "
        "separate a dim local patch from a dim field."
    )
    lines.append(
        "These are candidate frames to spend a get_filmstrip call on -- not a verdict; "
        "none of these numbers alone can tell a division from a death from a tracking "
        "artifact."
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
    p = BUNDLE / well / "labels" / f"frame_{frame:05d}.png"
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


def _walk_positions(
    well: str, frames: list[int], seed_cx: float, seed_cy: float
) -> dict[int, tuple[float, float, bool]]:
    """Nearest-centroid walk across `frames` (already ordered outward from the
    track's boundary), starting adjacent to (seed_cx, seed_cy).

    Returns {frame: (cx, cy, resolved)}. `resolved=False` means the walk lost the
    cell here (too far, or nothing detected) and the position is carried over from
    the last frame it WAS resolved at -- a real "last known position", same as the
    old behaviour, just reached only after actually trying rather than immediately.
    """
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
        misses = 0
        cx, cy = float(cxs[i]), float(cys[i])
        out[f] = (cx, cy, True)
    return out


def _filmstrip_frames(
    well: str, track_id: int,
    start_frame: int | None, end_frame: int | None,
    max_images: int, crop_um: float,
    color: bool, scale_bar: bool, marker: bool,
) -> tuple[str, list[np.ndarray]]:
    """Shared by get_filmstrip (MCP images) and show_cells (HTML page).

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
    # Walk outward from each boundary across every frame in range (not just the
    # sampled picks) so the miss-tolerance state stays meaningful -- a walk that
    # skipped straight to a far pick would have no idea how many frames it "missed".
    off_before = [f for f in picks if f not in by_frame and f < t_lo]
    off_after = [f for f in picks if f not in by_frame and f > t_hi]
    walked: dict[int, tuple[float, float, bool]] = {}
    if off_before:
        walked.update(_walk_positions(
            well, list(range(t_lo - 1, min(off_before) - 1, -1)), first_row.cx, first_row.cy))
    if off_after:
        walked.update(_walk_positions(
            well, list(range(t_hi + 1, max(off_after) + 1)), last_row.cx, last_row.cy))
    n_walked = sum(1 for f in (off_before + off_after) if walked.get(f, (0, 0, False))[2])

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
        grey = _frame_png(well, int(f))
        h, w = grey.shape
        if on_track:
            cx, cy = int(round(row.cx)), int(round(row.cy))
        else:
            wcx, wcy, walk_resolved = walked.get(f, (row.cx, row.cy, False))
            cx, cy = int(round(wcx)), int(round(wcy))
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
        label = f"f{int(f)} t={_hours(well, int(f)):.1f}h"
        if on_track:
            if row.n_masks_in_frame > 1:
                label += f" [{int(row.n_masks_in_frame)} masks]"
        elif walk_resolved:
            label += " OFF-TRACK (walked)"
        else:
            label += f" OFF-TRACK (held @f{t_lo if f < t_lo else t_hi})"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if scale_bar:
            img = _scale_bar(img, um_px * (crop.shape[0] / img.shape[0]), target_um=10.0)
        images.append(img)
    return header, images


@server.tool()
def find_candidates(
    well: str,
    pool: str = "division",
    sort_by: str = "fragment_like",
    limit: int = 20,
    exclude_near_edge: bool = True,
) -> str:
    """Scan a WHOLE WELL and rank what is worth looking at. Free -- no images.

    Every other tool here answers a question about one track you already picked.
    This is the one that answers "where do I even start" on a well with thousands
    of cells, which is the first question anyone actually has.

    It reports what the data ALREADY RECORDS, ranked -- it is not a detector and
    infers nothing new. A row here means "the tracker linked these ids" or "this
    track stops early", not "a division happened" or "this cell died".

    Pools:
      division   mothers with 2+ recorded daughters. The link scores come from
                 lineage.csv: dna_ratio ~1.0 when DNA is conserved across the
                 split, size_ratio ~1.0 when the daughters are comparable objects.
      track_end  tracks that stop before the recording does WITHOUT recorded
                 daughters. Deliberately one pool, not "deaths": a real death and
                 a division whose daughters were never linked look identical from
                 topology, and calling it either would be inventing a verdict.
      contested  links where another mother was equally close, so parent_id was a
                 tie-break. Only exists on geometry-sourced lineage.

    Sorts:
      fragment_like  lowest size_ratio first -- one "daughter" much smaller than
                     the other is the micronucleus/fragment signature, not a
                     division. This is the ranking that finds real problems.
      dna_anomaly    furthest from DNA conservation in EITHER direction; below 1
                     means signal went missing, above means the pair carries more
                     DNA than the mother had.
      duration       longest-lived first (division pool: the mother's lifetime).
      frame          earliest first.

    Args:
        well: well name from list_wells().
        pool: "division", "track_end", or "contested".
        sort_by: "fragment_like", "dna_anomaly", "duration", or "frame".
        limit: max rows. Keep small; this is for triage, not export.
        exclude_near_edge: drop cells within 15 um of the frame boundary, whose
            area and brightness are understated because the nucleus is clipped.
    """
    import pandas as pd

    p = BUNDLE / well / "lineage.csv"
    if not p.is_file():
        return f"No lineage.csv for {well}, so there is nothing to rank."
    lin = pd.read_csv(p)
    m = _manifest(well)
    n_frames = int(m["n_frames"])
    tracks = _tracks(well)

    pos = tracks.sort_values("frame").groupby("track_id")[["cx", "cy"]].last()

    def _edge(tid: int) -> float:
        if tid not in pos.index:
            return float("nan")
        return _edge_um(well, float(pos.loc[tid, "cx"]), float(pos.loc[tid, "cy"]))

    if pool == "division":
        rows = lin[lin.n_daughters.fillna(0) >= 2].copy()
        # Scores live on the DAUGHTER rows; lift the first daughter's onto the mother.
        by_id = lin.set_index("track_id")
        first_kid = rows.daughter_ids.astype(str).str.split().str[0]
        for col in ("dna_ratio", "size_ratio", "link_distance_px"):
            rows[col] = [
                by_id[col].get(int(k)) if str(k).strip().lstrip("-").isdigit() else None
                for k in first_kid
            ]
    elif pool == "track_end":
        rows = lin[(lin.last_frame < n_frames - 1) & (lin.n_daughters.fillna(0) == 0)].copy()
    elif pool == "contested":
        if "alt_parents" not in lin.columns:
            return (f"{well}'s lineage came from {m.get('lineage', {}).get('source', '?')}, "
                    f"which records no alternatives, so there is no contested pool. "
                    f"alt_parents only exists on geometry-sourced lineage.")
        rows = lin[lin.alt_parents.astype(str).str.strip().replace("nan", "") != ""].copy()
    else:
        raise ValueError(f"unknown pool {pool!r}; use division, track_end, or contested.")

    if rows.empty:
        return f"{well}: nothing in the {pool} pool."

    rows["edge_um"] = [_edge(int(t)) for t in rows.track_id]
    n_before = len(rows)
    if exclude_near_edge:
        rows = rows[~(rows.edge_um < 15)]
    rows["duration_f"] = rows.last_frame - rows.first_frame

    # A score-based sort on a pool that carries no scores would silently return the
    # rows in file order while the header claimed otherwise, so fall back explicitly
    # and say so. track_end rows have no link scores by definition -- they have no link.
    def _has(col: str) -> bool:
        return col in rows.columns and bool(rows[col].notna().any())

    applied = sort_by
    if sort_by == "fragment_like" and _has("size_ratio"):
        rows = rows.sort_values("size_ratio", na_position="last")
    elif sort_by == "dna_anomaly" and _has("dna_ratio"):
        rows = rows.reindex((rows.dna_ratio - 1.0).abs().sort_values(ascending=False).index)
    elif sort_by == "duration":
        rows = rows.sort_values("duration_f", ascending=False)
    elif sort_by == "frame":
        rows = rows.sort_values("first_frame")
    else:
        applied = "duration" if pool == "track_end" else "frame"
        rows = (rows.sort_values("duration_f", ascending=False) if applied == "duration"
                else rows.sort_values("first_frame"))

    shown = rows.head(limit)
    note = "" if applied == sort_by else (
        f" (asked for {sort_by}, but this pool carries no link scores -- "
        f"sorted by {applied} instead)")
    out = [f"{well}: {pool} pool, {n_before} total"
           + (f" ({n_before - len(rows)} dropped as near-edge)" if exclude_near_edge else "")
           + f", showing {len(shown)} sorted by {applied}.{note}"]
    if pool == "division":
        out.append("track_id | frames | daughters | dna | size | link_px | edge_um")
        for r in shown.itertuples():
            f = lambda v: "-" if v is None or v != v else f"{v:.2f}"  # noqa: E731
            out.append(f"{int(r.track_id)} | {int(r.first_frame)}-{int(r.last_frame)} | "
                       f"{r.daughter_ids} | {f(r.dna_ratio)} | {f(r.size_ratio)} | "
                       f"{'-' if r.link_distance_px != r.link_distance_px else f'{r.link_distance_px:.0f}'} | "
                       f"{'?' if r.edge_um != r.edge_um else f'{r.edge_um:.0f}'}")
        out.append(
            "\nA LOW size ratio with a healthy dna ratio is the fragment signature -- the "
            "big object carries the DNA and the small one is a micronucleus, so the 'split' "
            "is not a division. Confirm on the pixels before believing either way."
        )
    else:
        # These scores describe the track's OWN BIRTH link, not an ending -- a track
        # here has no daughters by definition. Shown because the combination is
        # genuinely informative: something born as a tiny fragment and then vanishing
        # is a different story from a full-sized nucleus that stops.
        out.append("track_id | frames | duration | born_dna | born_size | edge_um")
        for r in shown.itertuples():
            f = lambda v: "-" if v is None or v != v else f"{v:.2f}"  # noqa: E731
            out.append(f"{int(r.track_id)} | {int(r.first_frame)}-{int(r.last_frame)} | "
                       f"{int(r.duration_f)} | {f(getattr(r, 'dna_ratio', None))} | "
                       f"{f(getattr(r, 'size_ratio', None))} | "
                       f"{'?' if r.edge_um != r.edge_um else f'{r.edge_um:.0f}'}")
        if pool == "track_end":
            out.append(
                "\nborn_dna/born_size describe how this track was BORN, not how it ends -- "
                "it has no daughters by definition. A low born_size that then vanishes is "
                "most likely a fragment appearing and going, not a cell dying."
            )
            out.append(
                "This pool is deliberately NOT called deaths. A cell that died and a "
                "division whose daughters were never linked both look like a track that "
                "stops -- topology cannot tell them apart, and 96% of one hand-checked "
                "sample of 'deaths' in this project turned out to still be alive."
            )
    out.append("Next: get_track_profile (free) on anything here, then get_filmstrip to look.")
    return "\n".join(out)


def _nearest_detection(well: str, frame: int, x: float, y: float,
                       exclude: int | None = None) -> tuple[int, float] | None:
    """(track_id, distance_um) of the closest tracked cell to a point in one frame.

    `exclude` drops one id from the search -- in anchor mode the anchor is always
    nearest to itself at 0.0 um, which says nothing; the useful answer is what ELSE
    is near the place being watched.
    """
    df = _tracks(well)
    f = df[df.frame == frame]
    if exclude is not None:
        f = f[f.track_id != exclude]
    if f.empty:
        return None
    d = np.hypot(f.cx.to_numpy() - x, f.cy.to_numpy() - y)
    i = int(np.argmin(d))
    return int(f.track_id.iloc[i]), float(d[i]) * _manifest(well)["pixel_size_um"]


@server.tool(structured_output=False)
def get_filmstrip_at(
    well: str,
    start_frame: int,
    end_frame: int,
    x: float | None = None,
    y: float | None = None,
    anchor_track_id: int | None = None,
    max_images: int = 8,
    crop_um: float = 45.0,
    color: bool = True,
    scale_bar: bool = True,
) -> list:
    """Watch a PLACE over time, instead of following a cell's own mask.

    get_filmstrip answers "what happened to track N". This answers "what happened
    HERE", which is a different question and often the one you actually have.

    Reach for this when:

    - **The thing you care about has no track.** Every other tool addresses cells by
      track_id, so an object the segmenter never caught is otherwise impossible to
      ask about at all -- a small bright body beside a nucleus, debris, anything
      below the detector's threshold. Give its coordinates and look at it.
    - **You are not sure the track is on the object you mean.** A track and its own
      OFF-TRACK walk can sit on two different cells in one crop; a fixed position is
      not confused by that, because it never claims to be following anything.
    - **The mask breaks exactly when the event happens.** Chromatin rounding during
      mitosis and death is precisely when segmentation fails, so a mask-following
      crop tends to lose the cell at the moment of interest. A place does not move.
    - **You want a stable stage.** Pass anchor_track_id to ride a calm neighbouring
      cell's position while the cell you are watching does something violent.

    Coordinates are PIXELS in the full frame, matching cx/cy in tracks.csv and the
    xy columns of list_tracks -- not microns, and not crop-relative.

    Nothing is ringed, because nothing here is claimed to be a detection. Instead
    every frame reports the nearest tracked cell and how far away it is, so you can
    tell "this is track 2036" from "there is nothing tracked within 12 um of here".

    Args:
        well: well name from list_wells().
        start_frame, end_frame: inclusive frame range.
        x, y: the position to watch, in full-frame pixels. Required unless
            anchor_track_id is given.
        anchor_track_id: instead of a fixed point, follow THIS track's centroid --
            useful as a stable vantage on a neighbour. Where the anchor is missing
            from a frame, its closest known position is used.
        max_images: how many frames to show (hard capped at 12).
        crop_um: crop width in micrometres.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
    """
    from mcp.types import TextContent

    m = _manifest(well)
    n_frames = int(m["n_frames"])
    lo, hi = max(0, int(start_frame)), min(n_frames - 1, int(end_frame))
    if hi < lo:
        raise ValueError(f"empty range {start_frame}-{end_frame}; {well} has 0-{n_frames - 1}.")

    anchor_pos: dict[int, tuple[float, float]] = {}
    if anchor_track_id is not None:
        t = _tracks(well)
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
        # Nearest frame the anchor was actually seen in.
        nf = min(known, key=lambda k: abs(k - f))
        return anchor_pos[nf]

    avail = list(range(lo, hi + 1))
    n = min(max_images, MAX_IMAGES, len(avail))
    picks = [avail[i] for i in np.linspace(0, len(avail) - 1, n).astype(int)]

    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))

    where = (f"anchored on track {anchor_track_id}" if anchor_track_id is not None
             else f"fixed at ({float(x):.0f}, {float(y):.0f}) px")
    lines = [f"{well}: frames {lo}-{hi}, showing {n} of {len(avail)}, {where}. "
             f"Crop {crop_um:g} um wide. Nothing is ringed -- this is a PLACE, not a "
             f"tracked object. Nearest tracked cell per frame:"]

    images: list[np.ndarray] = []
    for f in picks:
        ccx, ccy = _centre(int(f))
        grey = _frame_png(well, int(f))
        h, w = grey.shape
        cxi, cyi = int(round(ccx)), int(round(ccy))
        x0, x1 = max(0, cxi - half), min(w, cxi + half)
        y0, y1 = max(0, cyi - half), min(h, cyi + half)
        crop = grey[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        img = _colorize(crop, well, color)
        cx_crop, cy_crop = cxi - x0, cyi - y0
        if img.shape[0] < _UPSCALE_TO:
            s = _UPSCALE_TO / img.shape[0]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
            cx_crop, cy_crop = cx_crop * s, cy_crop * s
        # A crosshair, not a ring: it marks where you asked to look, and must not
        # imply that something was detected there. Gapped in the middle so it never
        # covers the chromatin being judged.
        cxp, cyp = int(cx_crop), int(cy_crop)
        for dx0, dx1 in ((-12, -5), (5, 12)):
            cv2.line(img, (cxp + dx0, cyp), (cxp + dx1, cyp), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (cxp, cyp + dx0), (cxp, cyp + dx1), (0, 255, 255), 1, cv2.LINE_AA)

        near = _nearest_detection(well, int(f), ccx, ccy, exclude=anchor_track_id)
        if near is None:
            note = "no tracked cell in this frame"
        else:
            tid, dum = near
            note = f"track {tid}, {dum:.1f} um away"
        lines.append(f"  f{int(f)} ({_hours(well, int(f)):.2f} h): {note}")
        cv2.putText(img, f"f{int(f)} t={_hours(well, int(f)):.1f}h", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if near is not None and near[1] < crop_um:
            cv2.putText(img, f"~{near[0]} @{near[1]:.0f}um", (4, img.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        if scale_bar:
            img = _scale_bar(img, um_px * (crop.shape[0] / img.shape[0]), target_um=10.0)
        images.append(img)

    lines.append(
        "\nA nearest cell many microns away means the thing at this position is NOT "
        "tracked -- which is the usual reason to be here. Distances are centre-to-centre, "
        "so a large nucleus can read several microns away while still overlapping the point."
    )
    out: list = [TextContent(type="text", text="\n".join(lines))]
    out.extend(_encode(i) for i in images)
    return out


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
    frames are labelled OFF-TRACK and rendered by following the nearest detected
    blob frame-to-frame from the track's boundary (solid orange ring) -- usually
    keeps the crop on the same physical object even though its track_id changed,
    but is never confirmed to be THIS cell rather than a neighbour. Where that walk
    loses the trail (nothing nearby, or a multi-frame gap), the crop freezes at the
    last position it WAS resolved (dashed blue ring) instead of guessing further.
    Either way, nothing there is centred or identified the way an on-track frame
    is -- read OFF-TRACK frames as "this patch of the field", not "this cell".

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
        def _f(key: str) -> float | None:
            v = r.get(key)
            try:
                return float(v) if v not in (None, "") else None
            except ValueError:
                return None

        out[int(r["track_id"])] = {
            # float() first: an int column with blanks round-trips through pandas as
            # "127.0", and a bare int() on that raises.
            "parent": (int(float(r["parent_id"]))
                       if str(r.get("parent_id") or "").strip() not in ("", "nan") else None),
            "daughters": [int(x) for x in (r.get("daughter_ids") or "").split() if x],
            # Only present on geometry-sourced lineage; None on CTC-sourced.
            "dna_ratio": _f("dna_ratio"),
            "size_ratio": _f("size_ratio"),
            "link_px": _f("link_distance_px"),
            "alt_parents": (r.get("alt_parents") or "").strip(),
        }

    # Percentile a link against THIS well's own links rather than a fixed cutoff.
    # An absolute threshold does not survive a change of cell line -- WGD nuclei are
    # large and lobed where RUES2 are compact, so "size_ratio < 0.25 is a fragment"
    # is calibrated to whichever well it was written against and silently mis-fires
    # elsewhere. Same lesson as the max-delta feature, where normalisation was the
    # whole game and the raw statistic did not transfer between wells.
    for key in ("dna_ratio", "size_ratio"):
        vals = sorted(v[key] for v in out.values() if v.get(key) is not None)
        for v in out.values():
            x = v.get(key)
            if x is None or not vals:
                v[key + "_pct"] = None
            else:
                lo = sum(1 for y in vals if y < x)
                v[key + "_pct"] = 100.0 * lo / len(vals)
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

    def _quality(r: dict) -> list[str]:
        """Strength readout for a link, when the source scored it.

        Each number is given with its percentile against this well's own links, so
        the comparison recalibrates per cell line instead of assuming one cutoff
        fits RUES2, WGD and Bewo alike.
        """
        if r.get("dna_ratio") is None and r.get("size_ratio") is None:
            return []
        bits = []
        if r.get("link_px") is not None:
            bits.append(f"{r['link_px']:.0f}px apart")
        # Which tail is suspicious differs by measure. A LOW size ratio means one
        # "daughter" is a fragment; a high one is just a symmetric division, which
        # is normal. DNA is conserved across a real division, so BOTH tails are
        # wrong -- far below 1 means signal went missing, far above means the pair
        # carries more DNA than the mother had, i.e. something else got included.
        for key, label, both in (("dna_ratio", "DNA", True), ("size_ratio", "size", False)):
            if r.get(key) is None:
                bits.append("")
                continue
            pct = r.get(key + "_pct")
            tail = ""
            if pct is not None:
                if pct <= 25:
                    tail = f" (bottom {pct:.0f}% in this well)"
                elif both and pct >= 75:
                    tail = f" (top {100 - pct:.0f}% in this well)"
            bits.append(f"{label} {r[key]:.2f}{tail}")
        bits = [b for b in bits if b]
        out = ["      " + ", ".join(bits)]
        if r.get("alt_parents"):
            out.append(f"      CONTESTED: {r['alt_parents']} were also in range "
                       f"(id:px). Nearest won, which is a tie-break, not a finding.")
        return out

    p = rec["parent"]
    lines.append(f"  mother    {p} ({_span(p)})" if p is not None else "  mother    none recorded")
    if p is not None:
        lines.extend(_quality(rec))
    if rec["daughters"]:
        lines.append(f"  daughters {len(rec['daughters'])}")
        for d in rec["daughters"]:
            lines.append(f"    {d} ({_span(d)})")
            lines.extend(_quality(lin.get(d, {})))
    else:
        lines.append("  daughters none recorded")

    if cov == "partial":
        lines.append(
            "\nNOTE: this bundle's lineage is PARTIAL -- it was recovered from the "
            "pipeline's event list, which only covers tracks it flagged, so a missing "
            "mother or daughter here means UNKNOWN, not none. Absence is not evidence "
            "that the cell did not divide."
        )
    scored = any(rec.get(k) is not None for k in ("dna_ratio", "size_ratio"))
    lines.append(
        "\nPRESENCE IS NOT EVIDENCE. A recorded link means two ids were joined, not "
        "that a division happened. Verified failures in both directions in this data: "
        "three textbook anaphases with NO daughters recorded, a 'daughter' that was a "
        "~6 um^2 micronucleus budding off, and a recorded 'mother' that was a healthy "
        "neighbour a few microns away."
        + (
            "\nThe numbers on each link above are there so you can weigh it without "
            "spending an image. DNA near 1.00 means the mother's signal is accounted "
            "for by the daughters; size near 1.00 means they are comparable objects. "
            "A low size ratio with a healthy DNA ratio is the classic fragment case -- "
            "the big object carries the DNA and the small one is a micronucleus. These "
            "are not filters and nothing was withheld: weak links are still shown, "
            "labelled, and yours to judge."
            if scored else
            "\nThis bundle's lineage carries no link scores, so check by eye: a real "
            "division roughly halves the mother's area and it STAYS halved, and the "
            "daughters' areas should sum to about the mother's."
        )
        + " Check the frame spans too -- a 'daughter' starting long after the mother "
        "ends, or overlapping it heavily, is an id-linking artifact."
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
def get_neighbourhood_stats(well: str, track_id: int, frame: int, n_neighbours: int = 5,
                            region_um: float = 40.0) -> str:
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
        region_um: radius of the fixed-radius local patch, in microns. Default 40.
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
    # A fixed-radius patch, distinct from the k-NN pool above. k-NN always returns
    # N cells however far away they are, so in a sparse or locally-dark region it
    # silently reaches across the field and the comparison stops being local. The
    # radius pool can legitimately come back empty, which is itself informative.
    region = others[others.dist_um <= region_um]

    def _z(value: float, pool: pd.Series) -> float:
        std = pool.std()
        return (value - pool.mean()) / std if std > 1e-9 else 0.0

    # Is the PATCH dim, or is the CELL dim? This is the comparison that was missing:
    # eval v2 run 3 read z=+1.15 vs neighbours on track 5432 as "this cell is fine"
    # when the cell and all five neighbours had gone dark together.
    field_med = f.intensity_mean.median()
    pool_for_patch = region if not region.empty else nearest
    patch_label = (f"{region_um:.0f} um patch" if not region.empty
                   else f"nearest {len(nearest)} (no cell within {region_um:.0f} um)")
    patch_med = pool_for_patch.intensity_mean.median()
    patch_delta = ((patch_med - field_med) / field_med * 100) if field_med else 0.0

    lines = [
        f"{well} frame {frame} ({_hours(well, frame):.2f} h), track {track_id} vs its "
        f"{len(nearest)} nearest neighbours (of {len(others)} other cells in this frame):",
        f"  this cell      area {mine.area_um2:.0f} um^2, mean brightness {mine.intensity_mean:.0f}",
        f"  nearest {len(nearest)}   area median {nearest.area_um2.median():.0f} um^2, "
        f"mean brightness median {nearest.intensity_mean.median():.0f}",
        f"  {region_um:.0f}um patch ({len(region)})  " + (
            f"area median {region.area_um2.median():.0f} um^2, "
            f"mean brightness median {region.intensity_mean.median():.0f}"
            if not region.empty else "no other cell within this radius"),
        f"  field ({len(others) + 1})  area median {f.area_um2.median():.0f} um^2, "
        f"mean brightness median {field_med:.0f}",
        "",
        f"  IS THE PATCH ITSELF DIM?  {patch_label} median brightness {patch_med:.0f} "
        f"vs field {field_med:.0f} ({patch_delta:+.0f}%)",
        "",
        f"  z-score vs nearest neighbours   area {_z(mine.area_um2, nearest.area_um2):+.2f}, "
        f"brightness {_z(mine.intensity_mean, nearest.intensity_mean):+.2f}",
        (f"  z-score vs {region_um:.0f}um patch          area {_z(mine.area_um2, region.area_um2):+.2f}, "
         f"brightness {_z(mine.intensity_mean, region.intensity_mean):+.2f}"
         if len(region) > 1 else
         f"  z-score vs {region_um:.0f}um patch          n/a (fewer than 2 cells in radius)"),
        f"  z-score vs whole field          area {_z(mine.area_um2, f.area_um2):+.2f}, "
        f"brightness {_z(mine.intensity_mean, f.intensity_mean):+.2f}",
        "",
        "  nearest cells: track_id | dist_um | area_um2 | mean_brightness",
    ]
    for r in nearest.itertuples():
        lines.append(f"    {r.track_id} | {r.dist_um:.1f} | {r.area_um2:.0f} | {r.intensity_mean:.0f}")
    lines.append(
        "\nReading it -- check the PATCH line FIRST, then the z-scores. A z-score is "
        "relative to whatever pool it names, so a high z against a pool that has itself "
        "gone dark does NOT mean the cell is healthy:\n"
        "  patch ~= field, cell z negative     -> this cell alone is dim (cell-autonomous: "
        "death, or this one nucleus out of plane). The interesting case.\n"
        "  patch << field, cell z >= 0         -> the local patch is dark and this cell is "
        "merely typical FOR that dark patch. Regional defocus or shading, not a verdict "
        "about this cell -- and its own raw brightness may still have collapsed. Compare "
        "the cell's mean against its OWN earlier frames (get_track_profile) before "
        "concluding nothing is happening.\n"
        "  patch ~= field, cell z >= 0         -> nothing anomalous about this cell here.\n"
        "  patch << field and cell z negative  -> dim patch AND dim within it; strongest "
        "case for something happening to this cell, but confirm with images.\n"
        "A cell can be small-but-bright (area z very negative, brightness z positive) -- "
        "that is the signature of condensed mitotic chromatin or a micronucleus, not of a "
        "dying cell, which is usually the opposite."
    )
    return "\n".join(lines)


@server.tool()
def show_cells(well: str, events: list[dict]) -> str:
    """Show the user cells -- writes a page of labelled filmstrips they can open.

    CALL THIS WHENEVER THE USER SAYS "show me", "let me see", "send me", "put
    together", "can I look at" -- anything meaning they want to LOOK at cells rather
    than read a description of them. Describing twelve frames in prose when the
    person asked to see them is the wrong answer, and it is the most common way to
    get this wrong: the words "show me" should end in this call.

    Also the right tool when you have finished an investigation and want to hand over
    the evidence -- after answering "what happens to these 12 tracks", write one page
    covering all of them instead of pasting filmstrips into chat one at a time. Each
    cell renders with the exact same crop logic as get_filmstrip (colour LUT, scale
    bar, OFF-TRACK handling), so the page matches what you already reviewed.

    Pair each entry with a `label` that says what you concluded and why -- the page
    outlives the conversation, and a bare track id tells the reader nothing about
    what they are looking for.

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
    lb_data = []
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
        b64_list = []
        for img in images:
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                continue
            b64_list.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        tiles = [
            f'<img src="data:image/png;base64,{b64}" loading="lazy" '
            f'onclick="openLightbox({i},{j})">'
            for j, b64 in enumerate(b64_list)
        ]
        sections.append(
            f"<section><h2>{i + 1}. {label}</h2>"
            f"<p class=hdr>{header}</p>"
            f"<div class=filmstrip>{''.join(tiles)}</div></section>"
        )
        lb_data.append({"label": label, "images": b64_list})

    lb_json = json.dumps(lb_data)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{well} browser</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }}
h1 {{ font-weight: 400; }}
section {{ margin-bottom: 2.5rem; border-top: 1px solid #333; padding-top: 1rem; }}
h2 {{ font-size: 1.1rem; font-weight: 600; }}
p.hdr {{ color: #999; font-size: 0.85rem; max-width: 60rem; }}
div.filmstrip {{ display: flex; flex-wrap: wrap; gap: 4px; }}
div.filmstrip img {{ image-rendering: pixelated; max-height: 260px; border: 1px solid #333; cursor: zoom-in; }}
.lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); align-items: center; justify-content: center; z-index: 20; }}
.lightbox.open {{ display: flex; }}
.lightbox img {{ max-width: 92vw; max-height: 88vh; }}
.lightbox-caption {{ position: absolute; bottom: 22px; left: 50%; transform: translateX(-50%); color: #ccc; font-size: 12.5px; }}
.lightbox-close {{ position: absolute; top: 14px; right: 20px; color: #fff; font-size: 28px; line-height: 1; cursor: pointer; background: none; border: none; padding: 6px 10px; }}
.lightbox-nav {{ position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.12); color: #fff; border: none; font-size: 28px; line-height: 1; width: 52px; height: 64px; cursor: pointer; border-radius: 8px; }}
.lightbox-nav:hover {{ background: rgba(255,255,255,0.24); }}
.lightbox-nav:disabled {{ opacity: 0.25; cursor: default; }}
.lightbox-nav.prev {{ left: 16px; }}
.lightbox-nav.next {{ right: 16px; }}
</style></head><body>
<h1>{well}</h1>
{''.join(sections)}
<div class="lightbox" id="lightbox">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <button class="lightbox-nav prev" id="lb-prev" onclick="stepLightbox(-1)">&lsaquo;</button>
  <img id="lb-img" alt="">
  <div class="lightbox-caption" id="lb-caption"></div>
  <button class="lightbox-nav next" id="lb-next" onclick="stepLightbox(1)">&rsaquo;</button>
</div>
<script>
var LB_DATA = {lb_json};
var lbSection = 0, lbIdx = 0;
function showLbFrame() {{
  var imgs = LB_DATA[lbSection].images;
  document.getElementById('lb-img').src = 'data:image/png;base64,' + imgs[lbIdx];
  document.getElementById('lb-caption').textContent =
    LB_DATA[lbSection].label + ' -- frame ' + (lbIdx + 1) + ' / ' + imgs.length;
  document.getElementById('lb-prev').disabled = lbIdx === 0;
  document.getElementById('lb-next').disabled = lbIdx === imgs.length - 1;
}}
function openLightbox(section, idx) {{
  lbSection = section; lbIdx = idx;
  showLbFrame();
  document.getElementById('lightbox').classList.add('open');
}}
function closeLightbox() {{ document.getElementById('lightbox').classList.remove('open'); }}
function stepLightbox(delta) {{
  var imgs = LB_DATA[lbSection].images;
  var next = lbIdx + delta;
  if (next < 0 || next >= imgs.length) return;
  lbIdx = next;
  showLbFrame();
}}
document.getElementById('lightbox').addEventListener('click', function(e) {{
  if (e.target.id === 'lightbox') closeLightbox();
}});
document.addEventListener('keydown', function(e) {{
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') stepLightbox(-1);
  if (e.key === 'ArrowRight') stepLightbox(1);
}});
</script>
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
