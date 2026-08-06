"""Bundle IO: manifest/track-table loading, frame timing, and the mtime-keyed cache.

Split out of the original single-file cell_mcp.py's "io" section.
"""

import functools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import cell_mcp_server as _cm

# Every reference to BUNDLE, _manifest, and _tracks below goes through `_cm.` rather
# than a direct import. Tests monkeypatch these on the `cell_mcp_server` package
# (e.g. `monkeypatch.setattr(cell_mcp_server, "_manifest", ...)`), which only
# rewrites cell_mcp_server's own namespace -- a function that captured `_manifest`
# via a normal `from .io import _manifest` would keep calling the ORIGINAL,
# unpatched one. Routing through `_cm.` re-reads the name from cell_mcp_server's
# namespace on every call, so a patch takes effect everywhere, matching the
# single-module behaviour this package was split out of.
#
# Self-importing `cell_mcp_server` here (rather than the flat top-level launcher
# `cell_mcp.py`) is what makes this safe regardless of how that launcher is run or
# renamed -- see the note at the top of cell_mcp_server/__init__.py.


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


@_fresh(lambda w: _cm.BUNDLE / w / "manifest.json", maxsize=64)
def _manifest(well: str) -> dict:
    import json
    p = _cm.BUNDLE / well / "manifest.json"
    if not p.is_file():
        raise ValueError(f"unknown well {well!r}. Call list_wells() for valid names.")
    return json.loads(p.read_text(encoding="utf-8"))


@_fresh(lambda w: _cm.BUNDLE / w / "tracks.csv")
def _tracks(well: str) -> pd.DataFrame:
    """Per-frame track table. Cached -- these are 100k-500k rows each."""
    return pd.read_csv(_cm.BUNDLE / well / "tracks.csv")


def _frame_png(well: str, frame: int) -> np.ndarray:
    p = _cm.BUNDLE / well / "frames" / f"frame_{frame:05d}.png"
    if not p.is_file():
        n = _cm._manifest(well)["n_frames"]
        raise ValueError(f"frame {frame} not in {well} (has frames 0-{n - 1})")
    # Multi-channel wells (src/ingest.py's display/ composite, 2026-08-05) carry a
    # separate true multi-color render alongside the grayscale segmentation frame
    # -- prefer it here so display shows both markers' real colors, not one flat
    # tint. Falls back to the grayscale frame for single-channel wells and any
    # bundle built before this composite existed.
    disp = _cm.BUNDLE / well / "frames_display" / f"frame_{frame:05d}.png"
    if disp.is_file():
        return cv2.imread(str(disp), cv2.IMREAD_COLOR)
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _hours(well: str, frame: int) -> float:
    ts = _cm._manifest(well)["frame_timestamps_ms"]
    return (ts[frame] - ts[0]) / 3.6e6


def _elapsed_str(well: str, frame: int) -> str:
    """Elapsed time as '34h 03m' -- minutes zero-padded so a bare '3m' next to '34h'

    never reads as a truncated number.
    """
    total_min = round(_hours(well, frame) * 60)
    h, m = divmod(int(total_min), 60)
    return f"{h}h {m:02d}m"


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
    ts = _cm._manifest(well)["frame_timestamps_ms"]
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
    ts = _cm._manifest(well)["frame_timestamps_ms"]
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
    m = _cm._manifest(well)
    w, h = m.get("width_px"), m.get("height_px")
    if not w or not h:
        return float("nan")
    return min(cx, cy, w - cx, h - cy) * m["pixel_size_um"]



__all__ = [
    "_server_stamp", "_SERVER_STAMP", "_fresh", "_manifest", "_tracks",
    "_frame_png", "_hours", "_elapsed_str", "_frame_at_offset_min", "_minutes_between",
    "_pick_frames", "_edge_um",
]
