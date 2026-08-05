"""Browsing tools: list_wells, list_tracks, get_track_profile, view_whole_field,
get_lineage, measure, get_neighbourhood_stats.

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import cv2
import numpy as np
import pandas as pd
from mcp.types import ImageContent

from .server import server
from .io import _fresh, _edge_um, _SERVER_STAMP
from .render import _colorize, _scale_bar, _encode

import cell_mcp as _cm

# BUNDLE, _manifest, _tracks, _frame_png, _hours, and _lineage below go through
# `_cm.` rather than a direct import -- see the note at the top of io.py.


@server.tool()
def list_wells() -> str:
    """List every imaged well available, with its cell line and how long it ran.

    Each well is one field of view filmed over 1-3 days. Start here: the well
    name is the first argument to every other tool.
    """
    if not _cm.BUNDLE.is_dir():
        return f"No bundle at {_cm.BUNDLE}. Set CELL_BUNDLE_DIR to a built bundle directory."
    out = ["well | cell_line | frames | hours | tracks | um/px | "
           "interval_min med/mean/max | built"]
    unstamped = []
    for d in sorted(_cm.BUNDLE.iterdir()):
        if not (d / "manifest.json").is_file():
            continue
        m = _cm._manifest(d.name)
        prov = m.get("provenance") or {}
        built = (prov.get("built_at") or "")[:10] or "UNSTAMPED"
        if built == "UNSTAMPED":
            unstamped.append(d.name)
        out.append(
            f"{d.name} | {m.get('cell_line') or '?'} | {m['n_frames']} | "
            f"{m['duration_hours']:.1f} | {m['n_tracks']} | {m['pixel_size_um']:.4f} | "
            f"{m['interval_ms']['median'] / 60000:.1f}/"
            f"{m['interval_ms']['mean'] / 60000:.1f}/"
            f"{m['interval_ms']['max'] / 60000:.1f} | {built}"
        )
    out.append(
        "\nNote: the time between frames is NOT constant. Never assume a fixed interval -- "
        "use measure() or the per-frame timestamps, which are exact. Three numbers are "
        "printed because one would be a trap: the median is the typical frame, but only "
        "the MEAN reproduces the hours column, and the max is how far apart the worst "
        "pair gets. Where they spread (nTSC: 4.9 median, 6.2 mean, 14.5 max) the interval "
        "DRIFTS through the run, so frames x any of them is wrong -- and wrong by more, "
        "the later in the movie you are."
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
    find cells worth looking at before calling follow_cells_over_time on them.

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
    df = _cm._tracks(well)
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

    suspect = set(_cm._manifest(well).get("track_multiplicity", {}).get("suspect_tracks", []))
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
    so use this BEFORE follow_cells_over_time to decide which frames are worth spending
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
    df = _cm._tracks(well)
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

    suspect = track_id in set(_cm._manifest(well).get("track_multiplicity", {}).get("suspect_tracks", []))
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
    rec = _cm._lineage(well).get(track_id, {})
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
        kr = _cm._lineage(well).get(kid, {}) if kid is not None else {}
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
        "These are candidate frames to spend a follow_cells_over_time call on -- not a verdict; "
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
def view_whole_field(well: str, frame: int, downscale: int = 2,
              color: bool = True, scale_bar: bool = True) -> ImageContent:
    """Show one whole field of view, to get oriented.

    Use this to see the overall layout and pick a region or cell, then use
    follow_cells_over_time to follow a specific cell closely over time.

    Args:
        well: well name from list_wells().
        frame: frame number, 0-based. See list_wells() for how many exist.
        downscale: shrink by this factor to save space. 2 is usually plenty. At 1 a
            single frame costs a large fraction of the context window, and a nucleus
            is still only ~20 px across -- too coarse to judge chromatin either way.
            Use follow_cells_over_time for anything that depends on a cell's shape.
        color: apply the microscope's own display colour (matches Fiji).
        scale_bar: burn in a labelled scale bar.
    """
    grey = _cm._frame_png(well, frame)
    if downscale > 1:
        grey = cv2.resize(grey, None, fx=1 / downscale, fy=1 / downscale,
                          interpolation=cv2.INTER_AREA)
    img = _colorize(grey, well, color)
    if scale_bar:
        img = _scale_bar(img, _cm._manifest(well)["pixel_size_um"] * downscale)
    return _encode(img)

@_fresh(lambda w: _cm.BUNDLE / w / "lineage.csv")
def _lineage(well: str) -> dict[int, dict]:
    p = _cm.BUNDLE / well / "lineage.csv"
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
    lin = _cm._lineage(well)
    if not lin:
        return (f"No lineage.csv in this bundle for {well}, so mother/daughter links are "
                f"unavailable. You can still follow a cell past the end of its track: "
                f"follow_cells_over_time accepts start_frame/end_frame outside the track's lifetime "
                f"and will render those frames as OFF-TRACK.")

    cov = _cm._manifest(well).get("lineage", {}).get("coverage", "unknown")
    df = _cm._tracks(well)
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
        "\nTo see the division itself, ask follow_cells_over_time for a range spanning the "
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
    df = _cm._tracks(well)
    t = df[df.track_id == track_id]
    if t.empty:
        raise ValueError(f"track {track_id} not found in {well}.")
    m = _cm._manifest(well)
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
        return (f"{well} track {track_id}, frame {frame} ({_cm._hours(well, frame):.2f} h)\n"
                f"  area           {r.area_um2:.1f} um^2  ({r.area_px:.0f} px)\n"
                f"  position       x={r.cx:.1f} y={r.cy:.1f} px\n"
                f"  brightness     mean {r.intensity_mean:.0f}, total {r.intensity_integrated:,.0f}\n"
                f"  (total brightness tracks DNA content -- the marker is a labelled histone)"
                + extra + warn)

    lo, hi = int(t.frame.min()), int(t.frame.max())
    return (f"{well} track {track_id} summary{warn}\n"
            f"  seen in        {t.frame.nunique()} frames, {lo}-{hi}\n"
            f"  elapsed        {_cm._hours(well, hi) - _cm._hours(well, lo):.2f} h "
            f"({_cm._hours(well, lo):.2f} -> {_cm._hours(well, hi):.2f} h)\n"
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
    df = _cm._tracks(well)
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
    others["dist_um"] = np.hypot(others.cx - mine.cx, others.cy - mine.cy) * _cm._manifest(well)["pixel_size_um"]
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
        f"{well} frame {frame} ({_cm._hours(well, frame):.2f} h), track {track_id} vs its "
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



__all__ = [
    "list_wells", "list_tracks", "_SPARK_CHARS", "_sparkline", "get_track_profile",
    "view_whole_field", "_lineage", "get_lineage", "measure", "get_neighbourhood_stats",
]
