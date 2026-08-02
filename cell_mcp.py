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
import functools
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

# The same cap for an HTML page, where the images land on disk and in a human's
# browser instead of in the model's context. Nothing is spent per frame there, and
# the thing a researcher asks for over and over is EVERY frame around the event --
# so the token budget has no business shrinking a page it never pays for.
MAX_IMAGES_PAGE = 60

# Auto-window around a membership transition, in MINUTES of real time.
#
# It used to be +/- a fixed number of FRAMES, and that was wrong in a way that
# inverted a whole census. On BeWo M2 the mitotic figure appears up to ~7 frames
# AFTER the frame where lineage.csv records the mother->daughter link (verified on
# track 802: link ends f771, prometaphase f778, two objects by f782) -- so a window
# that stops near the link renders the lead-up and hides the outcome, and every real
# division scored off it reads as an artifact. Blind scoring on 2026-07-31 put 4/5
# `vanishing_daughter` cases as real mitoses and 0/4 `clean` ones.
#
# Frames made it worse: +10 frames is 49 min on RUES2 (4.9 min/frame) and only 30 min
# on BeWo (3.0 min/frame), so the line whose tracker fails hardest got the SHORTEST
# real-time look. Minutes are the units the biology is in -- mitosis runs ~30-60 min
# start to finish -- so the window is stated in minutes and converted per well from
# its own timestamps. Asymmetric on purpose: the interesting half is after.
_WINDOW_BEFORE_MIN = 30.0
_WINDOW_AFTER_MIN = 90.0

# Frames are sampled at this spacing when the caller does not pin max_images, so a
# strip's time resolution stays the same whether the well runs at 3.0 or 4.9 min per
# frame. Roughly anaphase-scale; below this the extra frames mostly repeat.
_STRIDE_MIN = 6.0

# Filmstrip crops are tiny in absolute pixels -- a 60 um crop is 104 px here, and a
# nucleus inside it is about 21 px across. That 21 px is the microscope's limit, not
# ours (the ND2 is natively 1024x1024, and nothing upstream downsamples), so upscaling
# adds no information. It does add legibility: the previous 2x INTER_NEAREST to 160 px
# turned soft chromatin into hard blocks, and chromatin texture is the entire evidence
# for calling a stage. LANCZOS at 3x is visibly better on the same pixels. Ringing is
# not a concern on diffraction-limited fluorescence, which has no sharp edges to ring.
# Cost is negligible: an image this size is ~130 tokens.
_UPSCALE_TO = 312

# Splits a filmstrip header into "what is true of this strip" and "how this tool
# renders". show_cells prints the second half once per PAGE instead of once per
# case: a reviewer reading 14 identical paragraphs reads none of them, and the
# caveats live in that half.
_HDR_SEP = "\n\n"


# --------------------------------------------------------------------------- io

def _server_stamp() -> str:
    """What code is answering, and since when.

    A bundle says what built it; nothing said what was READING it. A session on
    2026-07-31 asked "is the MCP up to date with the repo?" and had to shell out to
    git status, git log and a process listing to find out -- three escapes from the
    toolset to answer a question about the toolset. It matters because this server is
    a long-lived process: the code is whatever was on disk when it started, which can
    be many commits ago, and an MCP server does not hot-reload.
    """
    import datetime as dt
    import subprocess
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit, dirty = "", ""
    started = dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M%z")
    return (f"code {commit or 'unknown'}{' +uncommitted' if dirty else ''}, "
            f"loaded {started}")


_SERVER_STAMP = _server_stamp()


def _fresh(path_of, maxsize: int = 8):
    """Cache a per-well loader, keyed on the source file's mtime.

    These were `lru_cache`d on the well name alone, which meant the cache had no way
    to notice a rebuild. `list_wells()` loads every manifest on its first call, so one
    early call pinned all 21 wells for the life of the process. On 2026-07-31 a session
    quoted BeWo M3 at 11,053 tracks more than an hour after that well had been rebuilt
    to 11,291 -- and concluded from it that the BeWo bundles still needed rebuilding.
    Nothing was wrong with the bundle on disk; the server simply never looked again.

    Same failure as the canonical-label cache in src/lineage.py: a key that cannot see
    the thing that changes. Rebuilds are routine now, so the key has to be the file.
    """
    def deco(fn):
        cache: dict[str, tuple] = {}

        @functools.wraps(fn)
        def wrapper(well: str):
            p = path_of(well)
            stamp = p.stat().st_mtime_ns if p.is_file() else None
            hit = cache.get(well)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            val = fn(well)
            cache[well] = (stamp, val)
            while len(cache) > maxsize:           # insertion-ordered: drop the oldest
                cache.pop(next(iter(cache)))
            return val

        wrapper.cache_clear = cache.clear
        wrapper.cache_size = lambda: len(cache)
        return wrapper
    return deco


@_fresh(lambda w: BUNDLE / w / "manifest.json", maxsize=64)
def _manifest(well: str) -> dict:
    import json
    p = BUNDLE / well / "manifest.json"
    if not p.is_file():
        raise ValueError(f"unknown well {well!r}. Call list_wells() for valid names.")
    return json.loads(p.read_text(encoding="utf-8"))


@_fresh(lambda w: BUNDLE / w / "tracks.csv")
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


def _frame_at_offset_min(well: str, frame: int, minutes: float) -> int:
    """The frame `minutes` of REAL TIME away from `frame` (negative = earlier).

    Walks the recorded timestamps rather than dividing by a nominal interval,
    because the interval is not constant: M12's median is 4.9 min but it ranges
    2.9 -> 14.4, so "10 frames" is anywhere from 29 to 144 minutes depending on
    where in the recording you stand. Converting through the median would put the
    window in the right place on average and the wrong place exactly where the
    acquisition hiccupped.

    Rounds outward -- the returned frame is at least `minutes` away, never less --
    so a window asked for in minutes is never quietly shorter than requested.
    """
    ts = _manifest(well)["frame_timestamps_ms"]
    target = ts[max(0, min(frame, len(ts) - 1))] + minutes * 60_000.0
    if minutes >= 0:
        for f in range(frame, len(ts)):
            if ts[f] >= target:
                return f
        return len(ts) - 1
    for f in range(frame, -1, -1):
        if ts[f] <= target:
            return f
    return 0


def _minutes_between(well: str, a: int, b: int) -> float:
    ts = _manifest(well)["frame_timestamps_ms"]
    n = len(ts)
    return abs(ts[max(0, min(b, n - 1))] - ts[max(0, min(a, n - 1))]) / 60_000.0


def _pick_frames(well: str, avail: list[int], max_images: int | None,
                 cap: int, stride_min: float) -> tuple[list[int], str]:
    """Choose which frames of `avail` to render, and say how the choice was made.

    Three cases, and the default one is the point:

    - `max_images=None` (the default) means "sample by TIME": take frames about
      `stride_min` apart. If the whole range fits under the cap it is rendered
      GAP-FREE, because a researcher who names a frame range wants that range, not
      a sample of it -- their cost is eyes on images, and the gap is where the
      evidence hides. This is the behaviour that a fixed max_images=6 kept
      overriding: 6 images across a 17-frame window is a 3-frame stride, and
      anaphase is 1-2 frames long.
    - `max_images=N` pins the count, evenly spaced, for a caller who is budgeting.
    - Either way `cap` is the hard ceiling -- MAX_IMAGES for images that land in
      the model's context, MAX_IMAGES_PAGE for an HTML page that costs nothing.
    """
    n_avail = len(avail)
    if max_images is not None:
        n = max(1, min(int(max_images), cap, n_avail))
        picks = [avail[i] for i in np.linspace(0, n_avail - 1, n).astype(int)]
        return picks, f"showing {n} of {n_avail} frames, evenly spaced (max_images={max_images})"

    if n_avail <= cap:
        return avail, f"showing all {n_avail} frames, no sampling gaps"

    span_min = _minutes_between(well, avail[0], avail[-1])
    want = int(round(span_min / stride_min)) + 1
    n = max(2, min(want, cap, n_avail))
    picks = [avail[i] for i in np.linspace(0, n_avail - 1, n).astype(int)]
    got = span_min / max(n - 1, 1)
    note = (f"showing {n} of {n_avail} frames, ~{got:.1f} min apart"
            + (f" (asked for ~{stride_min:g} min, capped at {cap} images)"
               if want > n else ""))
    return picks, note


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
        w = (_manifest(well).get("display_window") or {})
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
    out = ["well | cell_line | frames | hours | tracks | um/px | interval_min | built"]
    unstamped = []
    for d in sorted(BUNDLE.iterdir()):
        if not (d / "manifest.json").is_file():
            continue
        m = _manifest(d.name)
        prov = m.get("provenance") or {}
        built = (prov.get("built_at") or "")[:10] or "UNSTAMPED"
        if built == "UNSTAMPED":
            unstamped.append(d.name)
        out.append(
            f"{d.name} | {m.get('cell_line') or '?'} | {m['n_frames']} | "
            f"{m['duration_hours']:.1f} | {m['n_tracks']} | {m['pixel_size_um']:.4f} | "
            f"{m['interval_ms']['median'] / 60000:.1f} | {built}"
        )
    out.append(
        "\nNote: the time between frames is NOT constant. Never assume a fixed interval -- "
        "use measure() or the per-frame timestamps, which are exact."
    )
    out.append(
        f"Server: {_SERVER_STAMP}. Bundles are re-read whenever their files change, so a "
        f"rebuild mid-session is picked up; the CODE is not -- it is whatever was on disk "
        f"when this process started, and restarting the session is the only way to reload it."
    )
    if unstamped:
        out.append(
            f"\nWARNING -- {len(unstamped)} well(s) carry no provenance block, so there is "
            f"no way to tell what code built them or how old the tracking underneath is: "
            f"{', '.join(unstamped[:6])}{' ...' if len(unstamped) > 6 else ''}. "
            "Track ids are not stable across re-tracks, so numbers computed from an "
            "unstamped bundle cannot be compared with anything, including themselves at "
            "a later date. Rebuild before quoting counts from these."
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
        "\nNeither flag should appear at all on a bundle rebuilt after 2026-07-31: merging "
        "was a gap-bridging bug, not a property of the imaging, and across all 21 wells and "
        "four cell lines every track is now single-masked. If you see one, either the bundle "
        "predates the fix (check manifest.provenance) or bridging has regressed."
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
        # Point at the other path when the recorded one is threadbare. A cold session
        # reads "ends in a recorded split into 1846 + 1847", believes it, and never
        # learns that both are 1-2 frame stubs and the real daughters are unlinked
        # tracks a few frames later. Cellpose segmented them; only the tracker
        # declined to connect them.
        kid_lives = [len(df[df.track_id == int(d)])
                     for d in (rec.get("daughters") or [])]
        if kid_lives and max(kid_lives) < 5:
            lines.append(
                f"  ** the recorded daughters last "
                f"{', '.join(str(n) for n in kid_lives)} frame(s) -- too short to be "
                f"the outcome. The real daughters are usually unlinked tracks that "
                f"appear LATER, since the tracker breaks at the division rather than "
                f"through it. Call list_nearby_tracks(well, track_id={track_id}) to "
                f"see every object segmented there, and centre a filmstrip with "
                f"centre_frame= rather than trusting this link. **")
    elif not rec.get("daughters"):
        lines.append(
            f"  lineage   no recorded daughters. That is not evidence the cell did not "
            f"divide -- it more often means the tracker lost it at the division. "
            f"list_nearby_tracks(well, track_id={track_id}) lists what was actually "
            f"segmented where this track ended.")
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
    max_images: int | None, crop_um: float,
    color: bool, scale_bar: bool, marker: bool,
    stride_min: float = _STRIDE_MIN, cap: int = MAX_IMAGES,
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
            well, list(range(t_lo - 1, min(off_before) - 1, -1)), first_row.cx, first_row.cy))
    if off_after:
        walked.update(_walk_positions(
            well, list(range(t_hi + 1, max(off_after) + 1)), last_row.cx, last_row.cy))
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


_FAMILY_MAX_MEMBERS = 6


def _resolve_family_centres(
    win, pos: dict[int, list], picks: list[int],
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
            if t in here or not (a < f < b):
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
    max_images: int | None, crop_um: float | None,
    color: bool, scale_bar: bool, marker: bool,
    before_min: float = _WINDOW_BEFORE_MIN, after_min: float = _WINDOW_AFTER_MIN,
    stride_min: float = _STRIDE_MIN, cap: int = MAX_IMAGES,
    added: list[int] | None = None, centre_frame: int | None = None,
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
    df = _tracks(well)
    m = _manifest(well)
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
    lo = (_frame_at_offset_min(well, transition, -before_min)
          if start_frame is None else max(0, int(start_frame)))
    hi = (_frame_at_offset_min(well, transition, after_min)
          if end_frame is None else min(n_frames - 1, int(end_frame)))
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
        rank = (win.groupby("track_id")
                .agg(n=("frame", "size"), a=("area_px", "median"))
                .sort_values(["n", "a"], ascending=False))
        kept = [int(t) for t in rank.index[:_FAMILY_MAX_MEMBERS]]
        dropped = [t for t in ids if t not in kept]
        win = win[win.track_id.isin(kept)]

    pos: dict[int, list] = {}
    for r in win.itertuples():
        pos.setdefault(int(r.frame), []).append(r)

    centres, held, gapped = _resolve_family_centres(win, pos, picks)

    # Auto-fit ONE crop width: the 90th percentile over sampled frames of the radius
    # needed to contain every present member (centroid distance plus that member's own
    # radius). A percentile, not the max, because one fragment drifting away would
    # otherwise zoom the whole strip out to the size of the field.
    auto = crop_um is None
    if auto:
        radii = []
        for f in picks:
            rows_f = pos.get(f, [])
            if not rows_f:
                continue
            cx, cy = centres[f][0], centres[f][1]
            radii.append(max(
                float(np.hypot(r.cx - cx, r.cy - cy))
                + float(np.sqrt(max(float(r.area_px), 1.0) / np.pi))
                for r in rows_f))
        r_px = float(np.percentile(radii, 90)) if radii else 20.0
        crop_um = float(np.clip(2 * r_px * um_px * 1.15, 25.0, 120.0))

    half = max(8, int(round(crop_um / um_px / 2)))

    who = ", ".join(str(t) for t in kept)

    # The header is built in two halves, separated by _HDR_SEP: what is true of THIS
    # strip, then the standing explanation of how the tool renders. A page of 14 cases
    # repeated the standing half 14 times, and the reviewer stopped reading it after
    # the first -- which means the warnings inside it stopped working. show_cells
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
    if held:
        spec.append(f"{len(held)} frame(s) labelled HELD -- no member present.")
    if dropped:
        spec.append(f"{len(dropped)} further member(s) were dropped to keep the centre "
                    f"stable ({', '.join(str(t) for t in dropped)}); the {len(kept)} "
                    f"kept are the longest-lived over this window.")
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
        grey = _frame_png(well, int(f))
        h, w = grey.shape
        cx, cy = int(round(cx_f)), int(round(cy_f))
        x0, x1 = max(0, cx - half), min(w, cx + half)
        y0, y1 = max(0, cy - half), min(h, cy + half)
        crop = grey[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        img = _colorize(crop, well, color)
        s = 1.0
        if img.shape[0] < _UPSCALE_TO:
            s = _UPSCALE_TO / img.shape[0]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LANCZOS4)
        if marker:
            # Ring ALL present members or none. One ring among several says nothing
            # about which cell the claim is about.
            for r in pos.get(f, []):
                rx, ry = (r.cx - x0) * s, (r.cy - y0) * s
                rad = float(np.sqrt(max(float(r.area_px), 1.0) / np.pi)) * 1.9 * s
                rad = float(np.clip(rad, 8, min(img.shape[:2]) / 2 - 2))
                cv2.circle(img, (int(rx), int(ry)), int(rad), (255, 255, 255), 1, cv2.LINE_AA)
        label = f"f{int(f)} t={_hours(well, int(f)):.1f}h"
        if f in held:
            label += " HELD"
        elif n_gap:
            # Name the missing ones: "1 seen" alone would read as a cell having gone,
            # which is the misreading this whole mechanism exists to prevent.
            label += (f" [{n_present} seen, gap {','.join(str(t) for t in gapped[f])}]")
        else:
            label += f" [{n_present} member{'s' if n_present != 1 else ''}]"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if scale_bar:
            img = _scale_bar(img, um_px * (crop.shape[0] / img.shape[0]), target_um=10.0)
        images.append(img)
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
_FAMILY_NEARBY_UM = 14.0
_FAMILY_NEARBY_BEFORE_MIN = 15.0
_FAMILY_NEARBY_AFTER_MIN = 75.0

_COND_BEFORE_MIN = 20.0
_COND_AFTER_MIN = 60.0

# The conservation gate: a frame only counts as a condensation candidate if the family
# still holds roughly the DNA the mother had. Asymmetric because the two ends fail
# differently -- below the floor, signal has genuinely gone missing and whatever is
# left is a fragment, not a compacted nucleus; above the ceiling, a neighbour has been
# swept into the family. The ceiling is loose on purpose: once both daughters are
# segmented the family legitimately reads more signal than the mother alone did.
# Margin added to the mother's own radius to make the measuring disc.
_COND_MARGIN_UM = 8.0

# How much of the mother's own recent history forms the baseline. An hour, not her
# whole track: raw signal bleaches over 72 h, so a lifetime median makes a late
# division look like it lost DNA it never lost.
_COND_BASE_MIN = 60.0
_COND_DNA_MIN = 0.75
_COND_DNA_MAX = 2.5


def _condensation(well: str, rows, lin, tracks, um_px: float) -> tuple[list, list, list, list]:
    """Score how strongly each recorded division shows CONDENSING CHROMATIN.

    Every other signal in this file is topological -- who the tracker linked to whom,
    whether a daughter persisted. That is what the 2026-07-31 blind scoring showed to
    be the wrong question on BeWo: where the tracker fails THROUGH the division, the
    recorded daughters are pre-mitotic debris and "daughter persists" is anti-correlated
    with a division having happened. A human scored the same events on morphology --
    is there a metaphase plate -- and disagreed with topology on 7 of 11 BeWo cases.
    This is that morphology question, asked of numbers the bundle already has.

    Mitosis packs the same DNA into a smaller object. So across the transition:

        area          FALLS      chromatin compacts
        mean intensity RISES     same signal, fewer pixels
        integrated     ~FLAT     no DNA was created or destroyed

    The SCORE is the brightness rise, and the conservation is a GATE on it. That
    split is the whole design, and the obvious alternative is wrong: scoring
    (brightness up) x (area down) ranked M12's fragments first, at cond 93 with only
    65% of the DNA still present. Both fragmentation and condensation shrink the
    area, so any factor of a0/area rewards a mask falling apart -- and a fragment
    shrinks the total signal in step with the area, leaving brightness per pixel
    FLAT. That is precisely what tells the two apart:

        condensation  area down, brightness UP, total conserved
        fragment      area down, brightness flat, total DOWN
        death/bleach  brightness DOWN

    So: look only at frames where the family still holds its DNA, and among those
    take the brightest. Area is reported, never multiplied in.

    Brightness is measured RELATIVE TO THE FIELD in the same frame, because the
    signal bleaches over 72 h -- an absolute rise late in a recording is a bigger
    deal than the same rise early, and a mother whose baseline spans hours would
    otherwise be compared against her own brighter past.

    Scored against the mother's OWN history, never an absolute cutoff -- the lesson
    from `solidity`, which is the strongest shape signal on compact RUES2 nuclei and
    the weakest on lobed WGD ones. A ratio to her own median says the same thing on
    both.

    Measured over a NEIGHBOURHOOD, not over the recorded family. That is the second
    thing this got wrong and the more important one. Summing the mother and her
    recorded daughters cannot see the figure at all in the cases that matter: on BeWo
    track 802 the family has rows in only 10 of the 28 window frames, because the
    tracker lost the cell at the link and the condensed object at f778 carries a track
    id nobody linked to anybody. Scoring the family returned NaN there -- on the exact
    event a human called an unmistakable prometaphase figure.

    So the window is a disc around the mother's last known position, and everything
    segmented inside it is summed. That is what makes the score independent of the
    tracking, which is the entire reason to want a morphology signal: if it needed the
    link to be right, it would fail wherever topology already fails, and those are the
    same events. The cost is neighbours drifting into the disc, so the count of objects
    summed is reported rather than hidden.

    Returns three parallel lists (peak score, the frame it peaked at, DNA conservation
    at that frame). NaN where the mother has too little history to have a baseline --
    a one-frame mother has no "normal" to be compared against, and inventing one would
    manufacture the signal this is supposed to measure.

    HOW WELL IT ACTUALLY WORKS, measured against the only human labels this project
    has (the maintainer, 2026-07-31, 26 blind-scored divisions across two lines):

        RUES2 M12   AUC 0.63   (3 real / 9 not)
        BeWo M2     AUC 0.75   (4 real / 10 not)
        combined    AUC 0.68

    That is a weak ranking, and it is stated rather than hidden because the number
    is the point: on BeWo, topology scored BELOW 0.5 on the same events -- the
    `clean` stratum held 0/4 real divisions and `vanishing_daughter` held 4/5. A 0.75
    that is independent of the tracker beats a ranking that is confidently backwards.
    It changes where a reviewer spends images; it does not decide anything.

    n=26 is far too small to tune against, so it has NOT been tuned against them --
    the thresholds are the first physically-argued values. Fitting them to 26 points
    would produce a better-looking number and a worse tool.

    It is a ranking, not a verdict: a condensed-looking object can be a dying cell
    whose chromatin clumped, which is a real and known confusion this cannot resolve.
    Confirm on the pixels.
    """
    need = {"area_um2", "area_px", "intensity_mean", "intensity_integrated"}
    if not need.issubset(tracks.columns):
        n = len(rows)
        return ([float("nan")] * n, [-1] * n, [float("nan")] * n,
                [float("nan")] * n)

    cols = ["track_id", "frame", "cx", "cy", "area_um2", "area_px",
            "intensity_mean", "intensity_integrated"]
    sub = tracks[cols].copy()
    # Bleach correction: every cell's brightness is expressed against the median cell
    # in ITS OWN frame, so a 72 h decay in the illumination or the dye cancels out.
    field = sub.groupby("frame").intensity_mean.median().replace(0, np.nan)
    sub["rel"] = sub.intensity_mean / sub.frame.map(field)
    by_track = {int(t): g for t, g in sub.groupby("track_id")}

    # Per-frame arrays, built once. The inner loop asks "what is near (x, y) in frame
    # f" ~30 times per division across ~1,300 divisions, and a pandas filter per
    # question turns seconds into minutes.
    per_frame: dict[int, tuple] = {}
    for f, g in sub.groupby("frame"):
        per_frame[int(f)] = (
            g.cx.to_numpy(), g.cy.to_numpy(), g.area_um2.to_numpy(),
            g.area_px.to_numpy(), g.rel.to_numpy(),
            g.intensity_integrated.to_numpy(),
        )

    scores, peaks, dnas, areas = [], [], [], []

    def _blank():
        scores.append(float("nan")); peaks.append(-1)
        dnas.append(float("nan")); areas.append(float("nan"))

    for r in rows.itertuples():
        mid = int(r.track_id)
        mrows = by_track.get(mid)
        # Fewer than 5 frames of mother is not a baseline, it is a guess.
        if mrows is None or len(mrows) < 5:
            _blank(); continue
        mrows = mrows.sort_values("frame")
        last = int(mrows.frame.iloc[-1])
        x0, y0 = float(mrows.cx.iloc[-1]), float(mrows.cy.iloc[-1])
        lo = _frame_at_offset_min(well, last, -_COND_BEFORE_MIN)
        hi = _frame_at_offset_min(well, last, _COND_AFTER_MIN)

        # Baseline is LOCAL -- the hour of this mother's own life immediately before
        # the window, not her whole track. It must be, because `intensity_integrated`
        # is raw and the signal bleaches: on BeWo track 793 the whole-track median put
        # every window frame at 0.60 of "baseline", the conservation gate rejected all
        # 28 of them, and a case a human read as prophase scored NaN. A cell is not
        # losing DNA because the recording started brighter than it ended. It also
        # excludes the tail, which is the very thing being measured against it.
        base = mrows[(mrows.frame < lo)
                     & (mrows.frame >= _frame_at_offset_min(well, lo, -_COND_BASE_MIN))]
        if len(base) < 3:
            base = mrows[mrows.frame < lo].tail(10)
        if len(base) < 3:
            base = mrows.iloc[:-3]
        a0 = float(base.area_um2.median())
        i0 = float(base.rel.median())
        s0 = float(base.intensity_integrated.median())
        if not (a0 > 0 and i0 > 0 and s0 > 0):
            _blank(); continue

        # Disc radius: the mother's own equivalent radius plus a fixed margin, so it
        # scales with the cell line. A BeWo nucleus is 2-3x a RUES2 one, and a fixed
        # radius would either clip the daughters apart on one line or sweep in the
        # neighbours on the other.
        r_um = float(np.sqrt(a0 / np.pi)) + _COND_MARGIN_UM
        r_px = r_um / um_px

        best = (float("-inf"), -1, float("nan"), float("nan"))
        for f in range(lo, hi + 1):
            pf = per_frame.get(f)
            if pf is None:
                continue
            cx, cy, a_um, a_px, rel, integ = pf
            near = ((cx - x0) ** 2 + (cy - y0) ** 2) <= r_px * r_px
            if not near.any():
                continue
            area_f = float(a_um[near].sum())
            sig_f = float(integ[near].sum())
            wpx = float(a_px[near].sum())
            if area_f <= 0 or wpx <= 0:
                continue
            # Area-weighted: an unweighted mean would let a 10 px fragment count as
            # much as the nucleus it broke off.
            imean_f = float((rel[near] * a_px[near]).sum()) / wpx
            dna = sig_f / s0
            # The gate. Outside this band the disc no longer holds the same DNA, so
            # whatever its brightness does is not condensation.
            if not (_COND_DNA_MIN <= dna <= _COND_DNA_MAX):
                continue
            val = imean_f / i0
            if val > best[0]:
                best = (val, f, dna, area_f / a0)

        if best[1] < 0:
            _blank(); continue
        scores.append(best[0]); peaks.append(best[1])
        dnas.append(best[2]); areas.append(best[3])

    return scores, peaks, dnas, areas


def _collapse_sites(well: str, rows, tracks) -> dict[int, list[int]]:
    """Group recorded divisions that are the SAME physical event.

    Returns {representative_mother: [the others]}.

    A tracker that fails through a division does not fail once. On BeWo M2, tracks
    1824 (f468-472), 1860 (f475-479) and 1883 (f479-496) are one cell at one spot,
    each recorded as its own division with its own daughters -- and they took three of
    the top five rows of a `cond`-ranked sample. A reviewer asked for five candidates
    and got three copies of one event: the review budget is spent three times on one
    answer, and any rate computed from the pool counts it three times.

    The rule is the one that settled the daughter question, applied to mothers: a
    mother whose track BEGINS where another mother's ENDED, within a short window, is
    the same cell re-acquired. Non-overlapping spans at one place is a broken track,
    not two cells -- BeWo does not divide twice in 84 minutes.

    But proximity alone folds real divisions too, because a genuine daughter also
    begins where her mother ended. What separates them is DURATION: a re-acquisition
    is over in minutes, a lineage takes hours. So a merge is allowed only while the
    whole group stays inside _SITE_MAX_SPAN_MIN.

    Union-find, because the relation chains: 1824 links 1860, 1860 links 1883, and all
    three must land in one group even though 1824 and 1883 are 24 frames apart.

    Nothing is discarded -- the members are returned so the caller can list them. A
    dropped row is a row nobody can audit, which is the failure that made events.csv
    unusable.
    """
    m = _manifest(well)
    # Needs geometry, size and real timestamps. A bundle missing any of them gets no
    # fold rather than a silently wrong one -- every row stays its own site.
    ids0 = [int(t) for t in rows.track_id]
    if (not {"cx", "cy", "area_um2"}.issubset(tracks.columns)
            or not m.get("frame_timestamps_ms")):
        return {t: [] for t in ids0}
    um_px = m["pixel_size_um"]
    srt = tracks.sort_values("frame")
    first = srt.groupby("track_id").first()
    last = srt.groupby("track_id").last()
    med = srt.groupby("track_id").area_um2.median()

    ids = [int(t) for t in rows.track_id if t in last.index]
    parent = {t: t for t in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairs first, then a SPAN-CAPPED merge. Two knobs, and each was forced by a
    # failure:
    #
    # Proximity alone chained f140 to f697 through a whole lineage and folded 81% of
    # the pool, because a genuine daughter also begins where her mother ended -- so
    # A->B->C walks straight through real divisions. Gating on "do two new objects
    # coexist here" then folded almost NOTHING (3 rows of 1253): in a crowded BeWo
    # field some pair always coexists, so the gate fires everywhere.
    #
    # What actually separates the two is DURATION. A re-acquisition of one cell is
    # over in minutes; a lineage takes hours. So the merge is allowed only while the
    # whole group stays inside _SITE_MAX_SPAN_MIN, which is far shorter than any cell
    # cycle and long enough for a tracker to drop and regain an id several times.
    starts = first.loc[ids]
    sx, sy, sf = starts.cx.to_numpy(), starts.cy.to_numpy(), starts.frame.to_numpy()
    sid = np.array(ids)
    links = []
    for t in ids:
        fe = int(last.loc[t, "frame"])
        x0, y0 = float(last.loc[t, "cx"]), float(last.loc[t, "cy"])
        r_px = (float(np.sqrt(max(float(med.loc[t]), 1.0) / np.pi))
                + _SITE_RADIUS_UM) / um_px
        hi = _frame_at_offset_min(well, fe, _SITE_GAP_MIN)
        sel = ((sf >= fe) & (sf <= hi) & (sid != t)
               & (((sx - x0) ** 2 + (sy - y0) ** 2) <= r_px * r_px))
        for u in sid[sel]:
            links.append((fe, t, int(u)))

    span = {t: (int(first.loc[t, "frame"]), int(last.loc[t, "frame"])) for t in ids}
    gspan = {t: span[t] for t in ids}
    for _, a, b in sorted(links):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lo_f = min(gspan[ra][0], gspan[rb][0])
        hi_f = max(gspan[ra][1], gspan[rb][1])
        if _minutes_between(well, lo_f, hi_f) > _SITE_MAX_SPAN_MIN:
            continue
        union(a, b)
        gspan[find(a)] = (lo_f, hi_f)

    groups: dict[int, list[int]] = {}
    for t in ids:
        groups.setdefault(find(t), []).append(t)
    # Representative: the mother with the most frames -- the recording of this event
    # that saw the most of it.
    out: dict[int, list[int]] = {}
    for members in groups.values():
        rep = max(members, key=lambda t: int(last.loc[t, "frame"])
                  - int(first.loc[t, "frame"]))
        out[rep] = [t for t in members if t != rep]
    return out


# How close, and how long after, a mother must re-appear to count as the same cell.
_SITE_RADIUS_UM = 14.0
_SITE_GAP_MIN = 30.0
# A group may not span more than this. Far shorter than any cell cycle, long enough
# for a tracker to drop and regain one cell's id several times over.
_SITE_MAX_SPAN_MIN = 60.0


_STRATA = [
    # (name, blurb) in PRIORITY order -- first match wins, so the counts partition the
    # pool and sum to its total. Measurement defects are tested before biological
    # ambiguity, on the grounds that a clipped or merged mask makes every other number
    # on the row untrustworthy, so it is the more informative label to carry.
    ("merged_id", "mother or a daughter is a track the manifest flags as multiplexed"),
    ("edge_clipped", "mother within 15 um of the frame boundary; area and brightness understated"),
    ("touching", "mother or a daughter shares a mask with another cell in some frame"),
    ("far_link", "daughter linked from >25 px away; most likely an id swap"),
    ("fragment_like", "one daughter tiny while DNA is conserved: micronucleus signature"),
    ("vanishing_daughter", "a daughter lasts <5 frames"),
    ("dim_daughter", "a daughter averages <60% of the well's median brightness"),
    ("clean", "trips none of the above"),
]


def _division_strata(rows, lin, tracks, m) -> list:
    """Label each recorded division with the FIRST artifact class it trips.

    Classifies; never discards. The reject rate per stratum is itself the finding,
    and a filter that silently drops rows is the exact failure that made events.csv
    unusable. Every signal here comes from tracks.csv + lineage.csv + manifest.json --
    no images, no model, seconds to run.

    The thresholds are UNVALIDATED. They were written from one session's look at one
    RUES2 well; treat each stratum as a hypothesis with a count attached, not as a
    verdict. In particular they are almost certainly wrong per cell line: a
    fragment/micronucleus is segmentation garbage on RUES2 and plausibly the real
    phenotype on a genome-doubled WGD line, so the same bucket can be noise in one
    well and the result in another.
    """
    suspect = {int(t) for t in (m.get("track_multiplicity", {}) or {}).get("suspect_tracks", [])}
    lin_by_id = lin.set_index("track_id")

    dur = (lin_by_id.last_frame - lin_by_id.first_frame).to_dict()
    touching = set(tracks.loc[tracks.get("n_masks_in_frame", 0) > 1, "track_id"].unique().tolist()) \
        if "n_masks_in_frame" in tracks.columns else set()
    bright = tracks.groupby("track_id")["intensity_mean"].mean() \
        if "intensity_mean" in tracks.columns else None
    dim_cut = float(bright.median()) * 0.6 if bright is not None and len(bright) else None

    def _kids(s) -> list[int]:
        return [int(k) for k in str(s).split() if k.strip().lstrip("-").isdigit()]

    labels = []
    for r in rows.itertuples():
        kids = _kids(r.daughter_ids)
        fam = [int(r.track_id), *kids]
        if suspect & set(fam):
            labels.append("merged_id")
        elif r.edge_um == r.edge_um and r.edge_um < 15:
            labels.append("edge_clipped")
        elif touching & set(fam):
            labels.append("touching")
        elif r.link_distance_px == r.link_distance_px and r.link_distance_px > 25:
            labels.append("far_link")
        elif (r.size_ratio == r.size_ratio and r.size_ratio < 0.25
              and r.dna_ratio == r.dna_ratio and r.dna_ratio > 0.8):
            labels.append("fragment_like")
        elif kids and min((dur.get(k, np.nan) for k in kids), default=np.nan) < 5:
            labels.append("vanishing_daughter")
        elif (dim_cut is not None
              and any(k in bright.index and bright[k] < dim_cut for k in kids)):
            labels.append("dim_daughter")
        else:
            labels.append("clean")
    return labels


@server.tool()
def find_candidates(
    well: str,
    pool: str = "division",
    sort_by: str = "fragment_like",
    limit: int = 20,
    exclude_near_edge: bool = True,
    stratum: str | None = None,
    seed: int | None = None,
    collapse: bool = True,
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
      condensation   strongest chromatin compaction first -- smaller AND brighter
                     while total signal is conserved, which is what mitosis looks
                     like and what fragmentation, death and bleaching do not. The
                     ONLY morphology-based sort here; every other one ranks on
                     topology, which is what a human reading metaphase plates
                     disagreed with on 7 of 11 BeWo cases. Reach for it when the
                     question is "did this cell divide" rather than "is this row
                     trustworthy".
      duration       longest-lived first (division pool: the mother's lifetime).
      frame          earliest first.
      random         a seeded shuffle -- THE ONLY SORT THAT MAY BE USED TO ESTIMATE
                     ANYTHING. Every other sort answers "show me the worst", which
                     is triage; a rate, a share, or a per-stratum true-positive rate
                     needs a sample that represents its stratum. `duration` in
                     particular is not neutral: it is measured in FRAMES, so it
                     favours faster-sampled wells, and on a BeWo draw of 5
                     `vanishing_daughter` cases it plausibly oversampled long-lived
                     cells that never divided. Pass `seed` to make the draw
                     reproducible and quotable; without one it is still random but
                     nobody can redraw it.

    The division pool also comes with a CENSUS: every recorded division is labelled
    with the first artifact class it trips, and the counts partition the pool. That
    is what turns a well into a sampling frame -- review ~20 from a stratum, learn
    how often that class is real, and the whole pool gets a corrected count with an
    error bar for ~150 reviewed events instead of 1,400. Reviewing every division is
    not affordable and never will be; the estimate with a stated uncertainty is the
    better number anyway.

    Use `limit=0` for the census with no rows -- it is a few hundred tokens and is
    usually the right first call on a well. Then `stratum="fragment_like"` (etc.) to
    draw the sample from one class.

    Args:
        well: well name from list_wells().
        pool: "division", "track_end", or "contested".
        sort_by: "fragment_like", "dna_anomaly", "condensation", "duration",
            "frame", or "random".
            Use "random" with a seed for any sample you intend to draw a number
            from; the ranked sorts are for finding problems, not for measuring.
        limit: max rows. Keep small; this is for triage, not export. 0 = census only.
        exclude_near_edge: drop cells within 15 um of the frame boundary, whose
            area and brightness are understated because the nucleus is clipped. They
            are still COUNTED in the census, and asking for stratum="edge_clipped"
            turns this off automatically.
        collapse: fold recorded divisions that are one physical event into a single
            row (division pool only, ON by default). A tracker that fails through a
            division fails repeatedly, so one BeWo mitosis appeared as tracks 1824,
            1860 and 1883 -- three of the top five rows of one ranked sample. The
            folded ids are listed in the `also` column, never dropped. Turn it off to
            see the raw recorded pool.
        stratum: restrict rows to one census class (division pool only).
        seed: fixes the shuffle for sort_by="random", so the same call returns the
            same sample and a result can be checked by someone else. Recorded in
            the header for exactly that reason.
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

    # Strata are computed on the WHOLE pool, before any dropping, so that what
    # exclude_near_edge removes still appears as a counted class rather than
    # vanishing. A number you cannot see is the thing that made events.csv dangerous.
    census = []
    if pool == "division":
        rows["stratum"] = _division_strata(rows, lin, tracks, m)
        cs, cf, cd, ca = _condensation(well, rows, lin, tracks, m["pixel_size_um"])
        rows["cond"], rows["cond_frame"] = cs, cf
        rows["cond_dna"], rows["cond_area"] = cd, ca
        sites = _collapse_sites(well, rows, tracks) if collapse else {}
        if collapse:
            rows["also"] = [" ".join(str(u) for u in sites.get(int(t), []))
                            for t in rows.track_id]
            n_folded = len(rows) - len(sites)
            rows = rows[rows.track_id.astype(int).isin(sites)]
        else:
            rows["also"] = ""
            n_folded = 0
        counts = rows.stratum.value_counts()
        census = [f"{name} | {int(counts.get(name, 0))} | "
                  f"{100 * int(counts.get(name, 0)) / max(n_before, 1):.1f}% | {blurb}"
                  for name, blurb in _STRATA]
        if stratum:
            if stratum not in {n for n, _ in _STRATA}:
                raise ValueError(f"unknown stratum {stratum!r}; one of "
                                 f"{', '.join(n for n, _ in _STRATA)}")
            rows = rows[rows.stratum == stratum]
            # Asking for the edge stratum while the edge filter is on returns nothing
            # and looks like "there are none". Honour the explicit request instead.
            if stratum == "edge_clipped":
                exclude_near_edge = False
    elif stratum:
        return (f"stratum= only applies to the division pool; the {pool} pool has no "
                f"strata defined. Call again without it.")

    n_pool = len(rows)
    if exclude_near_edge:
        rows = rows[~(rows.edge_um < 15)]
    n_edge_dropped = n_pool - len(rows)
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
    elif sort_by == "condensation" and _has("cond"):
        rows = rows.sort_values("cond", ascending=False, na_position="last")
    elif sort_by == "random":
        # Shuffle the WHOLE surviving pool and let limit take the head, so the sample
        # is drawn from every row that qualifies rather than from the top of some
        # other ordering. Sorting by first_frame first makes the draw independent of
        # the row order lineage.csv happened to be written in.
        rows = rows.sort_values("first_frame").sample(frac=1.0, random_state=seed)
    else:
        applied = "duration" if pool == "track_end" else "frame"
        rows = (rows.sort_values("duration_f", ascending=False) if applied == "duration"
                else rows.sort_values("first_frame"))

    shown = rows.head(limit) if limit > 0 else rows.iloc[:0]
    note = "" if applied == sort_by else (
        f" (asked for {sort_by}, but this pool carries no link scores -- "
        f"sorted by {applied} instead)")
    draw = (f" Random draw of {len(shown)} from the {len(rows)} rows in this pool"
            + (f", seed={seed} (re-callable)." if seed is not None else
               ", UNSEEDED -- pass seed= if this sample will back a number.")
            ) if applied == "random" and limit > 0 else ""
    fold = (f" {n_before} recorded divisions fold into {n_before - n_folded} distinct "
            f"SITES ({n_folded} rows were the same cell re-acquired); the folded ids "
            f"are in the `also` column, not discarded."
            if pool == "division" and n_folded else "")
    out = [f"{well}: {pool} pool, {n_before} total"
           + (f", stratum={stratum}" if stratum else "")
           + (f", showing {len(shown)} sorted by {applied}.{note}{draw}" if limit > 0
              else ", census only (limit=0).") + fold]
    if n_edge_dropped and not census:
        out.append(f"({n_edge_dropped} near-edge rows not shown; "
                   f"exclude_near_edge=False to include them.)")

    if census:
        out.append("\nWhat the pool is made of -- stratum | n | share | why")
        out += census
        out.append(
            "First match wins, so these partition the pool and sum to its total. "
            "Priority order is a judgement call and the marginal counts move if you "
            "change it. THE THRESHOLDS ARE UNVALIDATED -- each row is a hypothesis "
            "with a count, not a verdict, and they are very likely wrong per cell "
            "line (a micronucleus is garbage on RUES2 and may be the phenotype on WGD). "
            "Nothing here is discarded: pass stratum= to pull rows from one class, "
            "which is how you sample a class to measure how often it is real."
        )
        out.append(
            "Every stratum here describes the RECORDED link. When a row's daughters "
            "look like stubs, list_nearby_tracks(well, track_id=...) shows every "
            "object segmented at that spot -- including the ones nothing links to -- "
            "and get_filmstrip_family(track_ids=[...], centre_frame=<cond_f>) renders "
            "whichever of them you decide are the daughters."
        )

    if limit <= 0:
        return "\n".join(out)

    if rows.empty:
        out.append("\nNo rows left after filtering. Widen the stratum or set "
                   "exclude_near_edge=False.")
        return "\n".join(out)

    if pool == "division":
        out.append("track_id | frames | daughters | dna | size | link_px | edge_um | "
                   "cond | cond_f | cond_dna | cond_area | also")
        for r in shown.itertuples():
            f = lambda v: "-" if v is None or v != v else f"{v:.2f}"  # noqa: E731
            out.append(f"{int(r.track_id)} | {int(r.first_frame)}-{int(r.last_frame)} | "
                       f"{r.daughter_ids} | {f(r.dna_ratio)} | {f(r.size_ratio)} | "
                       f"{'-' if r.link_distance_px != r.link_distance_px else f'{r.link_distance_px:.0f}'} | "
                       f"{'?' if r.edge_um != r.edge_um else f'{r.edge_um:.0f}'} | "
                       f"{f(getattr(r, 'cond', None))} | "
                       f"{'-' if int(getattr(r, 'cond_frame', -1)) < 0 else int(r.cond_frame)} | "
                       f"{f(getattr(r, 'cond_dna', None))} | "
                       f"{f(getattr(r, 'cond_area', None))} | "
                       f"{getattr(r, 'also', '') or '-'}")
        out.append(
            "\nA LOW size ratio with a healthy dna ratio is the fragment signature -- the "
            "big object carries the DNA and the small one is a micronucleus, so the 'split' "
            "is not a division. Confirm on the pixels before believing either way."
        )
        out.append(
            f"cond is the CONDENSATION peak -- brightness per pixel at its highest, "
            f"relative to this mother's own median and to the other cells in the same "
            f"frame (so bleaching cancels). 1.0 is her normal interphase state; mitosis "
            f"packs the same DNA into fewer pixels, so it reads above 1. cond_f is the "
            f"frame it peaked at, and it is usually LATER than last_frame -- the link "
            f"ends where the tracker loses the cell, not where the cell divides. "
            f"cond_area is the family's area there over her baseline (below 1 = compacted) "
            f"and cond_dna is total signal over her baseline. "
            f"Only frames holding {_COND_DNA_MIN}-{_COND_DNA_MAX}x the baseline signal "
            f"were eligible: below that the DNA has genuinely gone and the object is a "
            f"fragment, whatever its brightness does. '-' means no eligible frame, or too "
            f"short a mother to have a baseline. "
            f"This is the only MORPHOLOGY column here -- every other one describes who the "
            f"tracker linked to whom, and on BeWo that topology scored close to "
            f"anti-correlated with a human reading metaphase plates. Unvalidated, and it "
            f"cannot tell condensed chromatin from the clumped chromatin of a dying cell."
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
    # The corner label is the one mark in this toolset a reader cannot decode from the
    # image: a session once grepped cell_mcp.py's source to find out what "~2362 @6um"
    # meant. It is burned in so a frame stays identified if it gets separated from this
    # text, which only works if the text says how to read it.
    lines = [f"{well}: frames {lo}-{hi}, showing {n} of {len(avail)}, {where}. "
             f"Crop {crop_um:g} um wide. The yellow crosshair marks WHERE YOU ASKED to "
             f"look -- nothing is ringed, because this is a place, not a tracked object, "
             f"and a ring would imply something was detected there. The bottom-left label "
             f"reads \"~<track_id> @<distance>um\": the NEAREST tracked cell to the "
             f"crosshair and how far its centre sits from it -- not the thing at the "
             f"crosshair, which may be untracked or nothing at all. It is only drawn when "
             f"that cell is closer than the crop is wide, so a blank corner means nothing "
             f"tracked is even in view. Nearest tracked cell per frame:"]

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
        + _display_note(well)
    )
    out: list = [TextContent(type="text", text="\n".join(lines))]
    out.extend(_encode(i) for i in images)
    return out


@server.tool()
def get_filmstrip(
    well: str, track_id: int,
    start_frame: int | None = None, end_frame: int | None = None,
    max_images: int | None = None, stride_min: float = _STRIDE_MIN,
    crop_um: float = 60.0,
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
        max_images: pin the frame count. None (recommended) samples by TIME: a
            range that fits under the cap of 12 comes back GAP-FREE, and a longer
            one is thinned to ~stride_min spacing rather than to a frame count. A
            fixed count means different time resolution on different wells, and it
            was silently skipping frames inside ranges that were asked for
            explicitly -- which is exactly where the evidence is.
        stride_min: target spacing between rendered frames, in minutes.
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
        well, track_id, start_frame, end_frame, max_images, crop_um,
        color, scale_bar, marker, stride_min, MAX_IMAGES,
    )
    from mcp.types import TextContent
    out: list = [TextContent(type="text", text=header)]
    out.extend(_encode(img) for img in images)
    return out


def _resolve_family(well: str, track_ids: list[int],
                    include_nearby: bool = False) -> tuple[list[int], list[int]]:
    """Add a track's recorded daughters, then the ones nobody recorded.

    Returns (members, added_positionally).

    The daughters come from lineage.csv, which records what the TRACKER linked, not
    what happened -- and where the tracker fails through a division it links one
    daughter, or none, or a piece of debris. A strip built from that member set then
    follows the mother and one daughter, the real sister drifts out of frame, and the
    reviewer sees half an event. That is the exact complaint this exists to answer:
    "only tracking 1 daughter, would be nice to have midpoint."

    So after the recorded daughters, look for tracks that BEGIN near where the mother
    ended, within a few frames of the link. A sister the tracker never connected still
    has to appear as a new object next to her mother -- that is what makes this
    findable from geometry with no new data. Anything found this way is returned
    separately so the header can say the strip is showing objects the lineage does not
    vouch for, which is a different claim from a recorded daughter.

    OFF BY DEFAULT, because nearest-by-distance is not good enough to decide this and
    a silent wrong guess is worse than no guess. On BeWo 969 it picked tracks 3829 and
    3879 -- which start 2.3 um apart, six frames apart, and NEVER COEXIST. They are one
    object at (250, 412) losing and regaining its id, 22 um north of the mother, and
    the strip centred on a chain of re-acquisitions instead of on a pair of daughters.
    The reviewer spotted it immediately: "I don't think that's the right daughter, and
    it's also not a midpoint -- are there two tracks that instantiated near the same
    place?" There were four.

    Use list_nearby_tracks() instead and choose the members yourself. It reports which
    candidates COEXIST, which is the test this heuristic lacks: two daughters must be
    on screen at the same time, so a chain of non-overlapping tracks at one spot is one
    cell being re-acquired, never a sister pair.
    """
    ids = [int(t) for t in track_ids]
    if len(track_ids) == 1:
        lin = _lineage(well)
        kids = (lin.get(ids[0]) or {}).get("daughters") or []
        ids = [ids[0], *[int(k) for k in kids]]
    if not include_nearby:
        return ids, []

    df = _tracks(well)
    mother = df[df.track_id == ids[0]]
    if mother.empty:
        return ids, []
    mother = mother.sort_values("frame")
    link = int(mother.frame.iloc[-1])
    x0, y0 = float(mother.cx.iloc[-1]), float(mother.cy.iloc[-1])
    a0 = float(mother.area_um2.median()) if "area_um2" in mother else 0.0
    um_px = _manifest(well)["pixel_size_um"]
    r_px = (float(np.sqrt(max(a0, 1.0) / np.pi)) + _FAMILY_NEARBY_UM) / um_px

    # Candidates: tracks whose FIRST frame lands near the link. A sister the tracker
    # lost the mother into appears as a new id right there; a neighbour that merely
    # happens to be close has been on screen for a long time already.
    starts = df.sort_values("frame").groupby("track_id").first()
    lo = _frame_at_offset_min(well, link, -_FAMILY_NEARBY_BEFORE_MIN)
    hi = _frame_at_offset_min(well, link, _FAMILY_NEARBY_AFTER_MIN)
    win = starts[(starts.frame >= lo) & (starts.frame <= hi)]
    d2 = (win.cx - x0) ** 2 + (win.cy - y0) ** 2
    near = win[d2 <= r_px * r_px].assign(d2=d2[d2 <= r_px * r_px])
    extra = [int(t) for t in near.sort_values("d2").index if int(t) not in ids]
    extra = extra[:2]
    return [*ids, *extra], extra


@server.tool()
def list_nearby_tracks(
    well: str,
    track_id: int | None = None,
    x: float | None = None, y: float | None = None, frame: int | None = None,
    before_min: float = 15.0, after_min: float = 75.0,
    radius_um: float | None = None,
    new_only: bool = True,
) -> str:
    """Every object segmented near a place and time -- so YOU can work out who the
    daughters are. Free: no images.

    This is the tool to reach for when lineage.csv is wrong or empty about a division,
    which on BeWo is most of the time. Cellpose segments the daughters perfectly well;
    it is the TRACKER that declines to link them, so the objects you need already exist
    in tracks.csv with their own ids. Nothing here needs re-segmenting or a model.

    It deliberately does not pick for you. A first attempt did -- nearest two starts to
    the mother -- and on BeWo 969 it chose tracks 3829 and 3879, which begin 2.3 um and
    six frames apart and are one cell being re-acquired, not two daughters. A wrong
    guess made silently is worse than no guess, so this hands you the evidence instead.

    THE TEST THAT SETTLES IT IS COEXISTENCE. Two daughters must be on screen at the
    same time. A run of tracks at one spot whose spans do not overlap -- 3782 f776-777,
    3806 f778-780, 3829 f781-786, 3879 f787-813 -- is a single object losing and
    regaining its id, however close together they look. The `coexists_with` column is
    there to make that judgement without opening an image, and a candidate coexisting
    with nothing cannot be half of a pair.

    A real sister also has to be NEW: a cell that has been on screen for hours and
    merely happens to be nearby is a neighbour. Hence new_only, and hence the window
    reaching much further forward than back -- the sister appears when the cell divides,
    which on BeWo runs ~20 min past where the tracker's link ends.

    Anchor it either on a track (its last known position and frame, which is where a
    mother was lost) or on an explicit x/y/frame from get_filmstrip_at.

    Args:
        well: well name from list_wells().
        track_id: anchor on this track's LAST position and frame. Usually the mother.
        x, y, frame: anchor on an explicit point instead, in pixels.
        before_min, after_min: how far either side of the anchor frame to look, in
            minutes. Forward-heavy by default, for the reason above.
        radius_um: search radius. None = the anchor's own radius + 14 um, so it scales
            with the cell line rather than assuming RUES2-sized nuclei.
        new_only: only tracks that BEGIN inside the window. False also lists tracks
            already running through it, which is what you want when asking "what is
            this thing overlapping with".
    """
    df = _tracks(well)
    m = _manifest(well)
    um_px = m["pixel_size_um"]
    srt = df.sort_values("frame")
    last = srt.groupby("track_id").last()
    first = srt.groupby("track_id").first()
    med = srt.groupby("track_id").area_um2.median()

    if track_id is not None:
        tid = int(track_id)
        if tid not in last.index:
            raise ValueError(f"track {tid} not found in {well}. Use list_tracks().")
        x0, y0 = float(last.loc[tid, "cx"]), float(last.loc[tid, "cy"])
        f0 = int(last.loc[tid, "frame"])
        a0 = float(med.loc[tid])
        anchor = f"track {tid}'s last position, f{f0}"
    elif None not in (x, y, frame):
        x0, y0, f0 = float(x), float(y), int(frame)
        a0 = float(med.median())
        anchor = f"point ({x0:.0f}, {y0:.0f}) at f{f0}"
        tid = None
    else:
        raise ValueError("give either track_id, or all of x, y and frame.")

    r_um = radius_um if radius_um else float(np.sqrt(max(a0, 1.0) / np.pi)) + 14.0
    r_px = r_um / um_px
    lo = _frame_at_offset_min(well, f0, -before_min)
    hi = _frame_at_offset_min(well, f0, after_min)

    if new_only:
        pool = first[(first.frame >= lo) & (first.frame <= hi)]
        near = pool[((pool.cx - x0) ** 2 + (pool.cy - y0) ** 2) <= r_px * r_px]
        ids = [int(t) for t in near.index if t != tid]
    else:
        win = df[(df.frame >= lo) & (df.frame <= hi)]
        win = win[((win.cx - x0) ** 2 + (win.cy - y0) ** 2) <= r_px * r_px]
        ids = [int(t) for t in win.track_id.unique() if t != tid]

    if not ids:
        return (f"{well}: nothing segmented within {r_um:.0f} um of {anchor} "
                f"in f{lo}-{hi}"
                + (" that BEGINS there (new_only=True; pass new_only=False to include "
                   "tracks already running)" if new_only else "") + ".")

    spans = {t: (int(first.loc[t, "frame"]), int(last.loc[t, "frame"])) for t in ids}
    out = [
        f"{well}: {len(ids)} object(s) within {r_um:.0f} um of {anchor}, "
        f"frames {lo}-{hi} ({_minutes_between(well, lo, hi):.0f} min)"
        + (", counting only tracks that BEGIN in the window." if new_only else "."),
        "",
        "track_id | frames | n | area_um2 | dist_um | coexists_with",
    ]
    rows = []
    for t in ids:
        s, e = spans[t]
        co = [u for u in ids
              if u != t and spans[u][0] <= e and spans[u][1] >= s]
        fr = first.loc[t]
        d = float(np.hypot(fr.cx - x0, fr.cy - y0)) * um_px
        rows.append((d, t, s, e, int((df.track_id == t).sum()),
                     float(med.loc[t]), co))
    for d, t, s, e, n, a, co in sorted(rows):
        out.append(f"{t} | {s}-{e} | {n} | {a:.0f} | {d:.1f} | "
                   f"{', '.join(str(u) for u in co) if co else 'NOTHING'}")
    out.append(
        "\nTwo daughters must COEXIST. A candidate whose coexists_with is NOTHING "
        "cannot be half of a pair, and a run of such tracks at one spot with "
        "consecutive spans is one cell losing and regaining its id. Distance alone "
        "will not tell you apart -- on BeWo 969 the two nearest starts were 2.3 um and "
        "six frames apart, and were the same cell. Pick the members yourself and pass "
        "them to get_filmstrip_family(track_ids=[...]) to see the event; nothing here "
        "is a claim that a division happened."
    )
    return "\n".join(out)


@server.tool()
def get_filmstrip_family(
    well: str,
    track_ids: list[int],
    start_frame: int | None = None, end_frame: int | None = None,
    centre_frame: int | None = None,
    before_min: float = _WINDOW_BEFORE_MIN, after_min: float = _WINDOW_AFTER_MIN,
    max_images: int | None = None, stride_min: float = _STRIDE_MIN,
    crop_um: float | None = None,
    color: bool = True, scale_bar: bool = True, marker: bool = False,
) -> list:
    """Before, during and after a division -- one strip that follows mother THEN
    daughters, without losing either.

    This is the strip to reach for on any event where the cell of interest stops
    being one object. get_filmstrip follows a single mask, so it goes OFF-TRACK at
    exactly the moment the division happens; get_filmstrip_at watches a fixed point,
    which only holds still if the cells happen not to migrate. Here the crop is
    centred on the mean position of whichever members are present in each frame, so
    it rides the mother up to the handoff and the daughters' midpoint after it. No
    mode switch, because membership does the switching.

    Pass just the mother and the daughters recorded in lineage.csv are added for you.
    Pass every id yourself when you do not trust that link, or for anything that is
    not a division -- a cell fragmenting during necrosis is a member set too, and the
    strip will stay on the debris field rather than chase one shard.

    By default the WINDOW is chosen for you around the frame where membership
    changes, and it is measured in MINUTES: 30 min before, 90 min after. It is
    lopsided on purpose. The transition is the frame where the TRACKER stopped
    linking, and on BeWo the mitotic figure appears up to ~20 min LATER -- so a
    window that stops near the transition shows the lead-up and hides the outcome,
    and every real division scored off it reads as an artifact. Widen `after_min`
    before concluding a candidate is not a division. Give start_frame/end_frame to
    override with exact frames.

    Frames are sampled by TIME (~`stride_min` apart), not by a fixed count, so a
    strip means the same thing on a well shot every 3.0 min as on one shot every
    4.9. If the whole window fits under the image cap you get EVERY frame -- no
    gaps. Set max_images only to budget context deliberately.

    By default the CROP is auto-fitted once for the whole strip, wide enough to hold
    every member across the sampled frames. Do not set crop_um by hand unless you
    want a particular zoom -- guessing it is how you end up with a sibling drifting
    out of frame halfway along, since separation grows as the daughters move apart.

    What it will not do: interpolate. A frame where no member is present holds the
    previous centre and is labelled HELD, because a made-up position rendered like a
    measured one is the failure this whole tool set exists to avoid.

    Args:
        well: well name from list_wells().
        track_ids: the members. One id = that track plus its recorded daughters.
        start_frame, end_frame: inclusive override of the automatic window. Given
            both, the range is rendered gap-free up to the image cap.
        centre_frame: put the window around THIS frame instead of around the frame
            where membership changes. Pass it whenever you chose the members
            yourself -- the membership rule only means something for a mother plus
            her recorded daughters, and on a hand-picked set it drifts. find_candidates
            hands you `cond_f`, the frame the chromatin was most condensed, which is
            usually the right thing to centre on.
        before_min, after_min: MINUTES either side of the membership transition,
            when the window is automatic. Converted to frames from this well's own
            timestamps, which are not evenly spaced.
        max_images: pin the frame count. None (recommended) samples by time.
        stride_min: target spacing between rendered frames, in minutes.
        crop_um: width in micrometres. None (recommended) = auto-fit.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
        marker: ring EVERY member present in each frame, or none at all.
    """
    if not track_ids:
        raise ValueError("track_ids is empty; give at least one track.")
    members, added = _resolve_family(well, track_ids)
    header, images = _family_filmstrip_frames(
        well, members, start_frame, end_frame,
        max_images, crop_um, color, scale_bar, marker,
        before_min, after_min, stride_min, MAX_IMAGES, added, centre_frame,
    )
    from mcp.types import TextContent
    out: list = [TextContent(type="text", text=header)]
    out.extend(_encode(img) for img in images)
    return out


@_fresh(lambda w: BUNDLE / w / "lineage.csv")
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
def show_cells(well: str, events: list[dict], note: str = "") -> str:
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
        note: one or two sentences at the top of the page saying WHAT THE READER IS
            being asked to do -- "score each case Y/N/unsure for a real division",
            "pick which of these is the metaphase frame". Write it whenever the page
            is a task rather than a report. A reviewer opening 14 filmstrips a day
            later has the labels but not the question, and asking again costs them
            more than writing it costs you.
        events: one dict per cell to show, each with EITHER
            track_id: one cell, rendered as get_filmstrip does it, OR
            track_ids: a member set, rendered as get_filmstrip_family does it -- use
                this for divisions, so the strip follows the mother and then the
                daughters' midpoint in one row instead of losing them at the handoff.
                A single id here expands to that track plus its recorded daughters,
                the window defaults to 30 min before / 90 min after the membership
                transition (`before_min`/`after_min` keys), and crop_um defaults to
                auto-fit rather than 60.
            start_frame, end_frame (optional): as in get_filmstrip -- may fall outside
                the track's own lifetime, e.g. to show a division just past its end.
            label (optional): a short heading, e.g. "2036 -- divides, pro/meta/ana
                309/319/321". Defaults to "track <track_id>".
            max_images (optional): pin the frame count. Leave it OFF by default. A
                page costs no context -- the images go to disk and to a human's
                browser -- so frames here are sampled by time and a window under 60
                frames renders GAP-FREE. Capping it is how a researcher ends up
                looking at every third frame of the event they asked to see.
            centre_frame (optional): centre the window on this frame rather than on
                the membership transition. Use it for hand-picked member sets.
            before_min, after_min, stride_min (optional): window and sampling in
                MINUTES, as in get_filmstrip_family.
            crop_um (optional, default 60.0): crop width in micrometres.
            marker (optional, default False): ring the tracked cell -- with
                track_ids, rings every member present or none.

    Returns TWO lines: the absolute path, then the same file as a file:// URL. Give
    the user BOTH, verbatim, each on its own line -- terminals and chat clients
    linkify a URL but not a Windows path, and which one is clickable depends on their
    client. Do not shorten either, do not describe where the file is instead of naming
    it, and do not make them ask twice.
    Images are embedded as base64, so the
    file is portable on its own -- open it directly, or serve it
    (`python -m http.server`) if file:// is blocked in the browser being used.
    """
    if not events:
        raise ValueError("events is empty -- nothing to render.")

    sections = []
    shared: list[str] = []
    lb_data = []
    for i, ev in enumerate(events):
        if "track_id" not in ev and "track_ids" not in ev:
            raise ValueError(f"events[{i}] needs 'track_id' or 'track_ids': {ev!r}")
        if "track_ids" in ev:
            members, added = _resolve_family(well, [int(t) for t in ev["track_ids"]])
            crop = ev.get("crop_um")
            mx = ev.get("max_images")
            header, images = _family_filmstrip_frames(
                well, members,
                ev.get("start_frame"), ev.get("end_frame"),
                None if mx is None else int(mx),
                None if crop is None else float(crop),
                True, True, bool(ev.get("marker", False)),
                float(ev.get("before_min", _WINDOW_BEFORE_MIN)),
                float(ev.get("after_min", _WINDOW_AFTER_MIN)),
                float(ev.get("stride_min", _STRIDE_MIN)),
                MAX_IMAGES_PAGE, added,
                None if ev.get("centre_frame") is None else int(ev["centre_frame"]),
            )
            label = ev.get("label") or f"tracks {', '.join(str(t) for t in members)}"
        else:
            track_id = int(ev["track_id"])
            mx = ev.get("max_images")
            header, images = _filmstrip_frames(
                well, track_id,
                ev.get("start_frame"), ev.get("end_frame"),
                None if mx is None else int(mx), float(ev.get("crop_um", 60.0)),
                True, True, bool(ev.get("marker", False)),
                float(ev.get("stride_min", _STRIDE_MIN)), MAX_IMAGES_PAGE,
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
        # Split the per-case facts from the standing how-this-renders text and collect
        # the latter for ONE printing at the top. Repeating it under every case is what
        # made a reviewer stop reading it, and the caveats live in that half.
        spec, _, gen = header.partition(_HDR_SEP)
        if gen and gen not in shared:
            shared.append(gen)
        sections.append(
            f"<section><h2>{i + 1}. {label}</h2>"
            f"<p class=hdr>{spec}</p>"
            f"<div class=filmstrip>{''.join(tiles)}</div></section>"
        )
        lb_data.append({"label": label, "images": b64_list})

    lb_json = json.dumps(lb_data)
    # Collapsed by default: it is reference, not the task. Open once, then get out of
    # the way of the 14 cases the page actually exists for.
    note_html = f"<p class=task>{note}</p>" if note else ""
    shared_html = "".join(
        f"<details class=howto><summary>How to read these strips</summary>"
        f"<p>{g}</p></details>" for g in shared)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{well} browser</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }}
h1 {{ font-weight: 400; }}
section {{ margin-bottom: 2.5rem; border-top: 1px solid #333; padding-top: 1rem; }}
h2 {{ font-size: 1.1rem; font-weight: 600; }}
p.hdr {{ color: #999; font-size: 0.85rem; max-width: 60rem; }}
p.task {{ color: #eee; font-size: 1rem; max-width: 60rem; background: #1d2a35; border-left: 3px solid #7aa7d0; padding: 0.7rem 1rem; }}
details.howto {{ color: #888; font-size: 0.82rem; max-width: 60rem; margin-bottom: 1rem; }}
details.howto summary {{ cursor: pointer; color: #7aa7d0; }}
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
{note_html}
{shared_html}
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
    # ABSOLUTE, always, and as a file:// URL too. BUNDLE is usually relative, so this
    # used to return "data\bundle\<well>\browsers\browser_....html" -- openable only by
    # someone already sitting in the repo root. The reader is a person in a terminal or
    # a chat window, and they said so twice: first that they had to "hunt down the html
    # link", then that the absolute path still was not clickable and they were pasting
    # it into Chrome by hand. Terminals and chat clients linkify a URL, not a Windows
    # path, so give them both and let whichever one their client understands win.
    full = out_path.resolve()
    return f"{full}\n{full.as_uri()}"


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
