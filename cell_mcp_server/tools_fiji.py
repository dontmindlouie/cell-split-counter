"""open_in_fiji: launch Fiji pointed at the exact frame/channel/location a report
page or filmstrip is already showing, so a researcher cross-checking a call
against the raw .nd2 doesn't have to manually scrub sliders to find it.

Split out as its own module (rather than folded into tools_output.py) because it
shells out to an external application and resolves paths outside the bundle --
a different failure mode (missing exe, missing raw file, OOM in Fiji itself)
than every other tool here, which only ever touches the bundle it was given.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .server import server

import cell_mcp_server as _cm
from .tools_filmstrip import _nearest_detection

# Same env-var-with-default pattern as server.py's BUNDLE / src/config.py's
# SHARED_DATA_DIR: a real default for this machine, overridable for anyone whose
# Fiji or raw-data layout differs.
FIJI_EXE = Path(os.environ.get("CELL_MCP_FIJI_EXE", r"G:\Fiji\fiji-windows-x64.exe"))
RAW_ND2_ROOT = Path(os.environ.get("CELL_MCP_RAW_ND2_ROOT", r"G:\Projects"))


def _find_raw_nd2(well: str) -> Path:
    """Locate the raw .nd2 for `well` under RAW_ND2_ROOT.

    Matched by prefix, not exact name -- wells are bundle/run names (e.g.
    "20251016_ACTB_M2"), but the raw file on disk sometimes carries a suffix the
    bundle name drops (e.g. "20251016_ACTB_M2_red.nd2"). Raw files live scattered
    across per-experiment subfolders under the root, not one flat directory, so
    this searches recursively.
    """
    matches = sorted(RAW_ND2_ROOT.rglob(f"{well}*.nd2"))
    if not matches:
        raise ValueError(
            f"no .nd2 found for well {well!r} under {RAW_ND2_ROOT} "
            f"(searched recursively for '{well}*.nd2'). If the raw file lives "
            f"elsewhere, set CELL_MCP_RAW_ND2_ROOT."
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple .nd2 files match well {well!r} under {RAW_ND2_ROOT}: "
            f"{[str(m) for m in matches]} -- ambiguous, narrow CELL_MCP_RAW_ND2_ROOT."
        )
    return matches[0]


@server.tool()
def open_in_fiji(well: str, frame: int, cx: int | None = None,
                  cy: int | None = None, crop_um: float = 60.0) -> str:
    """Launch Fiji, opened directly on a specific frame (and, if given, a specific
    location) of a well's raw .nd2 -- so a call from a report page or filmstrip can
    be cross-checked against the real file without hand-scrubbing sliders to find
    the same spot.

    Opens as a virtual stack (reads planes from disk on demand) rather than
    loading the whole series into RAM -- several of these files are 10GB+ and will
    hit a Java heap OutOfMemoryError otherwise (seen firsthand on ACTB_M3, a 13.8GB
    series). Opens in Composite color mode with all channels visible, matching
    what a multi-channel bundle's own composite render shows.

    Args:
        well: well name from list_wells() -- also used to find the matching raw
            .nd2 under CELL_MCP_RAW_ND2_ROOT (default G:\\Projects).
        frame: 0-indexed frame number, as shown in a filmstrip label ("f428").
            Converted to Fiji's 1-indexed T position internally.
        cx, cy: pixel coordinates, as shown in a filmstrip label ("@(1055, 427)").
            When given, draws a rectangular ROI of width `crop_um` centred there
            and zooms the view to it -- the same region a report page's crop shows.
            Omit both to just land on the right frame, full field of view.
        crop_um: width of the ROI/zoom box in micrometres, when cx/cy are given.
            Defaults to 60.0, matching show_cells_in_browser's report crop.

    Returns a confirmation line once Fiji has been launched (does not wait for the
    user to close it).
    """
    if not FIJI_EXE.is_file():
        raise ValueError(
            f"Fiji executable not found at {FIJI_EXE}. Set CELL_MCP_FIJI_EXE to "
            f"its actual path."
        )
    nd2_path = _find_raw_nd2(well)
    m = _cm._manifest(well)
    n_frames = m["n_frames"]
    if not (0 <= frame < n_frames):
        raise ValueError(f"frame {frame} not in {well} (has frames 0-{n_frames - 1})")
    t_pos = frame + 1  # Fiji/ImageJ stack positions are 1-indexed

    n_channels = m["nd2_sizes"]["C"]
    macro_lines = [
        f'open_path = "{nd2_path.as_posix()}";',
        'run("Bio-Formats Importer", "open=[" + open_path + "] autoscale color_mode=Composite '
        'view=Hyperstack stack_order=XYCZT use_virtual_stack");',
        f'Stack.setPosition(1, 1, {t_pos});',
        # Bio-Formats' own import-time autoscale is a single heuristic applied across
        # all channels -- fine for a bright one, but leaves a dim/sparse channel
        # (e.g. punctate signal against a mostly-black field) scaled so its real
        # signal sits below visible black, i.e. looks like the channel is off even
        # though its checkbox is ticked. Re-autoscale each channel individually,
        # same as clicking Auto in the B&C dialog once per channel by hand.
    ] + [
        f'Stack.setChannel({c}); run("Enhance Contrast", "saturated=0.35");'
        for c in range(1, n_channels + 1)
    ] + ['Stack.setChannel(1);']  # leave the slider back where a fresh open would land
    if cx is not None and cy is not None:
        um_px = m["pixel_size_um"]
        half_px = max(8, int(round(crop_um / um_px / 2)))
        x0 = max(0, int(cx) - half_px)
        y0 = max(0, int(cy) - half_px)
        w = h = half_px * 2
        macro_lines += [
            f'makeRectangle({x0}, {y0}, {w}, {h});',
            'run("To Selection");',  # Image > Zoom > To Selection -- actually zooms in,
                                      # unlike "To Bounding Box" (resizes the window only)
        ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ijm", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(macro_lines) + "\n")
        macro_path = f.name

    subprocess.Popen([str(FIJI_EXE), "-macro", macro_path])
    loc = f" at ({cx}, {cy})" if cx is not None else ""
    return (f"Launched Fiji on {nd2_path.name}, frame {frame}{loc}, "
            f"crop_um={crop_um}.")


_SNAP_UM = 30.0  # beyond this, the click probably missed its target entirely

# Concrete case (2026-08-15): a click snapped to track 454, whose recorded
# mother-link was ~120 frames (~well over an hour on ACTB's cadence) before the
# clicked frame -- it was an already-settled daughter, not the thing dividing at
# that frame. 45 min is comfortably past the tool set's own division-search windows
# (list_nearby_tracks/resolve_division's after_min=75 max) while still well short of
# "this track has clearly been idle for most of the recording".
_STALE_SNAP_MIN = 45.0


@server.tool()
def resolve_fiji_sighting(well: str, fiji_frame: int, x: float, y: float) -> str:
    """Turn one row of a Fiji AutoMeasure Results table into a track_id, so a
    researcher who spotted something interesting scanning the raw .nd2 by eye can
    hand it straight to you instead of hand-converting units and off-by-one frames
    themselves. Point this at each Results row (or each point you clicked) one at a
    time -- do not pre-convert anything, that's this tool's job:

    - fiji_frame: read STRAIGHT off the Results table's "Frame" (or "Slice")
      column, or the Fiji title bar / status bar. Fiji is 1-indexed; every other
      tool in this server (frame args, tracks.csv) is 0-indexed. This tool does
      that subtraction for you -- do not subtract 1 yourself first.
    - x, y: read STRAIGHT off the Results table's X/Y columns, in WHATEVER unit
      Fiji happened to report (this varies with the "Scaled units" checkbox in
      Analyze > Set Measurements, which is easy to leave on by accident since
      Bio-Formats calibrates these .nd2s automatically) -- this tool tries both the
      raw-pixel and the micron reading against the real tracked positions and picks
      whichever one actually lands on a cell, so you do not need to know which one
      Fiji gave you.

    THIS IS ONLY STEP ONE. Resolving a click to a track_id is not an investigation.
    A real mitosis event runs from chromatin condensation (prophase) through both
    daughter cells resolving as separate objects -- and prophase can start HOURS
    before the frame the researcher happened to click, often before this track's own
    ID even begins (the tracker only starts a new ID when the mask is clean enough
    to segment, not when the biology starts). Concluding anything from the single
    clicked frame alone is not enough. After this tool returns a track_id:
      1. Call find_prophase_onset(track_id) -- free, text-only, checks for an
         earlier condensation onset than the track's own start.
      2. Call follow_cells_over_time(track_ids=[track_id], centre_frame=<the
         pipeline frame this tool resolved to>) -- NOT track_id=track_id alone, and
         NOT track_ids=[track_id] without centre_frame. Both of those default their
         window to the track's own start/end (or, for a member set with no linked
         daughters yet, to the track's own START), which is very often NOT the
         frame the researcher actually clicked -- that mismatch is exactly what
         produces a filmstrip that only shows "metaphase onwards" or only
         "metaphase and before", never both. Passing centre_frame explicitly
         anchors the (generous, 120 min before / 180 min after) window on the real
         point of interest instead.
      3. If a division is real, use list_nearby_tracks(track_id=...) to find the
         daughter candidates (must COEXIST -- see that tool's own docstring), then
         follow_cells_over_time on track_ids=[mother, *daughters] to confirm the
         event actually completes.
    Do not report a verdict to the researcher after only looking at the one frame
    they clicked.

    Args:
        well: well name from list_wells().
        fiji_frame: the Fiji Frame/Slice number, 1-indexed, exactly as shown in
            Fiji -- do not adjust it first.
        x, y: the Fiji X/Y values, exactly as shown in the Results table or status
            bar -- do not convert units first.
    """
    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    frame = int(fiji_frame) - 1
    if not (0 <= frame < n_frames):
        raise ValueError(
            f"fiji_frame {fiji_frame} -> pipeline frame {frame}, out of range for "
            f"{well} (has frames 0-{n_frames - 1}, i.e. Fiji frames 1-{n_frames}). "
            f"Double check the well and the Frame column value."
        )

    um_px = m["pixel_size_um"]
    px_guess = _nearest_detection(well, frame, float(x), float(y))
    um_guess = _nearest_detection(well, frame, float(x) / um_px, float(y) / um_px)

    candidates = [(label, hit) for label, hit in
                  [("pixels", px_guess), ("microns", um_guess)] if hit is not None]
    if not candidates:
        return (f"{well} f{frame} (Fiji frame {fiji_frame}): nothing tracked in this "
                f"frame at all -- empty frame in tracks.csv, or wrong well/frame?")

    candidates.sort(key=lambda c: c[1][1])
    (best_label, (best_id, best_dist)), *rest = candidates
    same_track = len(rest) == 1 and rest[0][1][0] == best_id

    lines = [f"{well} f{frame} (Fiji frame {fiji_frame}, x={x}, y={y}):"]
    if best_dist > _SNAP_UM:
        lines.append(
            f"  NO GOOD MATCH -- closest is track {best_id}, {best_dist:.1f} um away "
            f"(read as {best_label}), farther than the {_SNAP_UM} um snap radius. "
            f"The click likely missed, or fiji_frame/units are off. Try "
            f"list_nearby_tracks(well={well!r}, x=..., y=..., frame={frame}) with a "
            f"few candidate readings before trusting this."
        )
    elif same_track:
        lines.append(f"  -> track {best_id}, {best_dist:.1f} um away "
                      f"(pixel and micron readings agree).")
    else:
        lines.append(f"  -> track {best_id}, {best_dist:.1f} um away, read as "
                      f"{best_label} (this well: 1 px = {um_px:.3f} um).")
        if rest and rest[0][1][1] <= _SNAP_UM:
            alt_label, (alt_id, alt_dist) = rest[0]
            lines.append(
                f"  NOTE: the OTHER unit reading also lands on a real track -- "
                f"{alt_id}, {alt_dist:.1f} um away (read as {alt_label}). Ambiguous; "
                f"if the investigation below doesn't look right, try track {alt_id}."
            )
    # A snap to a real, clean-looking track is not evidence THAT track is doing
    # anything at the clicked frame. Concrete case (2026-08-15): a click snapped to
    # track 454, a perfectly real long-lived track -- but get_lineage(454) showed it
    # was the already-settled daughter of a division ~120 frames earlier; it was just
    # idling nearby by the time of the click, and the real event at that frame
    # belonged to an unrelated track found only by widening the crop. When the
    # snapped track's own start (= when its mother-link, if any, would have fired) is
    # long before the clicked frame, that is a signal to check for a closer, more
    # currently-active candidate instead of trusting the snap.
    lin = _cm._lineage(well)
    if lin:
        t_rows = _cm._tracks(well)
        t_rows = t_rows[t_rows.track_id == best_id]
        if not t_rows.empty:
            t_start = int(t_rows.frame.min())
            if t_start < frame:
                age_min = _cm._minutes_between(well, t_start, frame)
                has_parent = (lin.get(best_id) or {}).get("parent") is not None
                if age_min > _STALE_SNAP_MIN:
                    lines.append(
                        f"  CHECK BEFORE TRUSTING THIS SNAP: track {best_id} has been "
                        f"running since f{t_start}, {age_min:.0f} min before the clicked "
                        f"frame"
                        + (" (its recorded mother-link is that old too)" if has_parent else "")
                        + f". A track this settled by the clicked frame may just be "
                        f"idling nearby rather than the thing that happened here -- "
                        f"widen watch_location_over_time(x={x}, y={y}, "
                        f"start_frame={max(0, frame - 10)}, end_frame={frame + 10}) to "
                        f"look for a closer, more recently-active candidate before "
                        f"investigating track {best_id} further."
                    )
    lines.append(
        f"\nNext: find_prophase_onset({best_id}), then "
        f"follow_cells_over_time(track_ids=[{best_id}], centre_frame={frame}) -- use "
        f"centre_frame={frame} explicitly, not track_id={best_id} alone and not "
        f"track_ids=[{best_id}] without it, or the window anchors on the track's own "
        f"start/end instead of the clicked frame and the strip shows only one side "
        f"of the event. See this tool's docstring for the full investigation steps."
    )
    return "\n".join(lines)


__all__ = ["FIJI_EXE", "RAW_ND2_ROOT", "open_in_fiji", "resolve_fiji_sighting"]
