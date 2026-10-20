"""measure_points: measure many cells at once, from a list of frame/x/y points.

Added 2026-10-20. A researcher scoring divisions by hand produces dozens of points
per movie (her Fiji clicks, or frames she corrected in her own scoring sheet), and
needs a size and brightness at each. measure() takes one track_id per call, and her
points aren't track_ids yet, so her Claude wrote a fresh measuring script for every
position instead. This tool does the whole list in one call and returns plain
numbers, one row per point. Whatever sheet or figure comes next is the caller's to
build, in whatever shape the researcher wants.

Kept apart from tools_fiji.py, whose other tools launch Fiji: this one only reads
the bundle, plus (optionally) the raw .nd2 for the frame check.
"""

import csv
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .server import server

import cell_mcp_server as _cm

# Same radius as resolve_fiji_sighting: inside a cell is the normal case, this is the
# fallback for a click that landed just off the edge of its mask.
_SNAP_UM = 5.0
# Fiji's own X/Y calibration has been seen to differ from the .nd2 metadata by up to
# ~0.2% (Bewo M5 X 0.566455 vs 0.56749; ZO1 X 0.863167 vs 0.86475, both 2026-10). The
# fit searches this wide a band around the manifest's pixel size, per axis.
_SCALE_BAND = 0.01
# A click within this many frames and px of another counts as a possible duplicate.
_DUP_FRAMES, _DUP_PX = 3, 6
# How far either side of the given frame the .nd2 Mean check looks for a match.
_MEAN_SHIFT_FRAMES = 3

_FRAME_COLS_0 = ("frame0",)
_FRAME_COLS_FIJI = ("Slice", "Slice_fiji", "Frame", "slice", "fiji_frame")
_FRAME_COLS_PLAIN = ("frame", "t")
_X_COLS = ("x", "X", "x_px", "X_um")
_Y_COLS = ("y", "Y", "y_px", "Y_um")
_MEAN_COLS = ("Mean", "mean")
_LABEL_COLS = ("label", "click", "id", "division", " ")

_raw_nd2_cache: dict[str, Path | None] = {}


def _pick(cols, options):
    return next((c for c in options if c in cols), None)


def _read_points(points, csv_path, fiji) -> tuple[list[dict], str]:
    """Normalise either input form to [{label, frame (0-based), x, y, mean}], plus a note
    on which columns were read, so a wrong guess is visible in the output."""
    if (points is None) == (csv_path is None):
        raise ValueError("pass exactly one of points=[...] or csv_path=...")
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(points)
    cols = list(df.columns)
    # Already-converted columns win over raw Fiji ones when both are present (a
    # formatted click CSV carries both), so nothing gets converted twice.
    xc = "x_px" if "x_px" in cols else _pick(cols, _X_COLS)
    yc = "y_px" if "y_px" in cols else _pick(cols, _Y_COLS)
    if xc is None or yc is None:
        raise ValueError(f"no x/y columns found; have {cols}. Name them x and y.")
    if (fc := _pick(cols, _FRAME_COLS_0)) is not None:
        offset = 0
    elif (fc := _pick(cols, _FRAME_COLS_FIJI)) is not None:
        offset = 1
    elif (fc := _pick(cols, _FRAME_COLS_PLAIN)) is not None:
        offset = 1 if fiji else 0
    else:
        raise ValueError(f"no frame column found; have {cols}. Name it frame.")
    mc = _pick(cols, _MEAN_COLS)
    lc = _pick(cols, _LABEL_COLS)
    def fmt(v):  # 56.0 -> "56": pandas turns an int column float when any row is blank
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

    out = []
    for i, r in df.iterrows():
        out.append({
            "label": fmt(r[lc]) if lc is not None else str(i + 1),
            "frame": int(r[fc]) - offset,
            "x_in": float(r[xc]), "y_in": float(r[yc]),
            "mean": float(r[mc]) if mc is not None and pd.notna(r[mc]) else None,
        })
    note = (f"read columns: frame={fc} ({'1-based, Fiji' if offset else '0-based'}), "
            f"x={xc}, y={yc}" + (f", Fiji Mean={mc}" if mc else "") + (f", label={lc}" if lc else ""))
    return out, note


def _whole_pixel_residual(v: np.ndarray, s: float) -> float:
    q = v / s
    return float(np.mean(np.abs(q - np.round(q))))


def _fit_axis(v: np.ndarray, um_px: float) -> tuple[float, float]:
    """The µm-per-pixel that puts every value on a whole pixel, and the mean residual.

    Fiji's point tool always reports a pixel centre, so with the right scale every
    click converts to a whole number; a scale off by 0.1% leaves fractions that grow
    with x. A fine grid over a narrow band is plenty for one parameter.
    """
    grid = np.linspace(um_px * (1 - _SCALE_BAND), um_px * (1 + _SCALE_BAND), 4001)
    res = np.array([_whole_pixel_residual(v, s) for s in grid])
    i = int(np.argmin(res))
    return float(grid[i]), float(res[i])


def _resolve_units(pts: list[dict], um_px: float, units: str) -> str:
    """Set x_px/y_px on every point; return a one-line note on how units were decided."""
    xs = np.array([p["x_in"] for p in pts])
    ys = np.array([p["y_in"] for p in pts])

    def apply(sx, sy):
        for p in pts:
            p["x_px"], p["y_px"] = p["x_in"] / sx, p["y_in"] / sy

    if units == "px":
        apply(1.0, 1.0)
        return "units: pixels (as given)"
    if units == "auto" and _whole_pixel_residual(xs, 1.0) < 0.02 and _whole_pixel_residual(ys, 1.0) < 0.02:
        apply(1.0, 1.0)
        return "units: pixels (values are already whole numbers)"
    # µm, either requested or the auto fallback. Three points is the least that can
    # pin a scale; with fewer, trust the file's calibration.
    if len(pts) >= 3:
        sx, rx = _fit_axis(xs, um_px)
        sy, ry = _fit_axis(ys, um_px)
        if rx < 0.02 and ry < 0.02:
            apply(sx, sy)
            note = f"units: µm, fitted to whole pixels (X {sx:.6f}, Y {sy:.6f} µm/px"
            if abs(sx / um_px - 1) > 1e-4 or abs(sy / um_px - 1) > 1e-4:
                note += f"; file says {um_px:.6f}, so Fiji's calibration differs slightly"
            return note + ")"
    apply(um_px, um_px)
    return (f"units: µm at the file's {um_px:.6f} µm/px (no scale puts these points on whole "
            f"pixels, so they are not raw point-tool clicks; positions are approximate)")


def _locate(well: str, frame: int, x: float, y: float, snap_px: float, lab_cache: dict):
    """(raw_label, distance_px) of the mask under or nearest to (x, y), or (None, None)."""
    if frame not in lab_cache:
        lab_cache[frame] = _cm._label_img(well, frame)
    lab = lab_cache[frame]
    if lab is None:
        return None, None
    h, w = lab.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None, None
    v = int(lab[yi, xi])
    if v:
        return v, 0.0
    r = int(math.ceil(snap_px))
    y0, y1, x0, x1 = max(0, yi - r), min(h, yi + r + 1), max(0, xi - r), min(w, xi + r + 1)
    sub = lab[y0:y1, x0:x1]
    yy, xx = np.nonzero(sub)
    if len(yy) == 0:
        return None, None
    d = np.hypot(yy + y0 - y, xx + x0 - x)
    k = int(np.argmin(d))
    if d[k] > snap_px:
        return None, None
    return int(sub[yy[k], xx[k]]), float(d[k])


def _find_nd2(well: str, nd2_path: str | None) -> Path | None:
    if nd2_path:
        return Path(nd2_path)
    if well not in _raw_nd2_cache:
        name = _cm._manifest(well).get("source_nd2")
        hits = sorted(_cm.RAW_ND2_ROOT.rglob(name)) if name else []
        _raw_nd2_cache[well] = hits[0] if len(hits) == 1 else None
    return _raw_nd2_cache[well]


def _mean_check(well: str, pts: list[dict], nd2_file: Path) -> None:
    """Compare Fiji's Mean with the raw pixel at each point; set p['mean_check'].

    A point-tool Mean is that single pixel's raw value, so it matches exactly at the
    right frame and position. A match a frame or two away means the click (or the
    frame typed into a sheet) is off by that much. Any channel may match, since the
    researcher could have been viewing any of them.
    """
    import nd2

    m = _cm._manifest(well)
    pos = m.get("position_index")
    with nd2.ND2File(str(nd2_file)) as f:
        n_t = f.sizes.get("T", 1)
        n_p = f.sizes.get("P")
        frames: dict[int, np.ndarray] = {}

        def pixel(t, y, x):
            if t not in frames:
                seq = t * n_p + pos if n_p and pos is not None else t
                a = np.asarray(f.read_frame(seq))
                frames[t] = a.reshape((-1,) + a.shape[-2:])  # (C, Y, X)
            return frames[t][:, y, x]

        for p in sorted(pts, key=lambda q: q["frame"]):
            if p["mean"] is None:
                continue
            x, y = int(round(p["x_px"])), int(round(p["y_px"]))
            for d in sorted(range(-_MEAN_SHIFT_FRAMES, _MEAN_SHIFT_FRAMES + 1), key=abs):
                t = p["frame"] + d
                if 0 <= t < n_t and np.any(pixel(t, y, x) == p["mean"]):
                    p["mean_check"] = "ok" if d == 0 else f"frame {d:+d}"
                    break
            else:
                p["mean_check"] = "no match"
            if len(frames) > 64:
                frames.clear()


@server.tool()
def measure_points(well: str, points: list[dict] | None = None, csv_path: str | None = None,
                   fiji: bool = False, units: str = "auto", frame_tolerance: int = 0,
                   check_frames: str = "auto", nd2_path: str | None = None,
                   out_csv: str | None = None) -> str:
    """Measure the cell at each of many frame/x/y points in one call, as a table of numbers.

    Use this whenever there is a LIST of places to measure: a researcher's Fiji clicks,
    or frames she typed into her own scoring sheet (e.g. "2 frames before the DNA
    condenses"). Do not write a measuring script and do not call measure() in a loop.
    This returns numbers only. Building the scoring sheet or figure from them is
    yours to do, in whatever layout the researcher already uses.

    Each point is looked up in the cell outlines (masks) themselves, not by nearest
    centre: "inside" means the point is on a cell. "snapped" means it was within 5 µm
    of one. Fiji units and frame numbering are handled for you; pass the numbers as
    Fiji shows them:
    - Frames: a column named Slice/Frame (Fiji) is read as 1-based. "frame0" is read
      as 0-based. A plain "frame" is 0-based unless fiji=True.
    - X/Y: auto-detected over the whole list. Whole numbers are read as pixels.
      Otherwise they're read as µm, with the µm-per-pixel fitted so every click lands
      on a whole pixel. Fiji's calibration can differ slightly from the file's, and a
      single click can't tell you that.
    - If Fiji's Mean column is present and the raw .nd2 can be found, each point's
      frame is checked against it. A match one or two frames away is flagged, so a
      mistyped or misclicked frame shows up instead of measuring the wrong moment.

    Per point: track_id, area (µm²), diameter (µm, of a circle with that area), DNA
    brightness, per-channel nucleus/ring brightness on multi-channel wells (raw and
    relative to the field median), and warnings: merged-cells track, several shapes
    sharing one id, possible duplicate clicks.

    Args:
        well: well name from list_wells().
        points: [{"frame":..., "x":..., "y":..., "label": optional, "mean": optional}].
            Use the label to carry the researcher's own numbering (e.g. "P2-14").
        csv_path: or a CSV with the same columns (Fiji's own names X, Y, Slice, Mean
            also work). Give exactly one of points / csv_path.
        fiji: treat a plain "frame" column as Fiji's 1-based numbering.
        units: "auto" (default), "px", or "um".
        frame_tolerance: if nothing is at the point in its frame, look up to this many
            frames either side (the mask can flicker out for a frame). 0 = exact frame.
        check_frames: "auto" (check when Mean and the .nd2 are both available), "off".
        nd2_path: the raw .nd2, if it can't be found automatically under RAW_ND2_ROOT.
        out_csv: where to write the full table. Defaults to <csv>_measured.csv beside
            csv_path. With points=[...] and no out_csv, the table is returned inline.
            Never written inside the bundle, which a rebuild replaces.
    """
    if units not in ("auto", "px", "um"):
        raise ValueError("units must be 'auto', 'px' or 'um'")
    m = _cm._manifest(well)
    um_px = float(m["pixel_size_um"])
    n_frames = int(m["n_frames"])
    pts, col_note = _read_points(points, csv_path, fiji)
    if not pts:
        raise ValueError("no points given")
    unit_note = _resolve_units(pts, um_px, units)
    notes = [col_note, unit_note]
    nd2_t = (m.get("nd2_sizes") or {}).get("T")
    if nd2_t and int(nd2_t) != n_frames:
        notes.append(f"WARNING: the bundle kept {n_frames} of the .nd2's {nd2_t} frames "
                     f"(blank frames dropped), so a Fiji frame number may not equal the "
                     f"bundle frame + 1 late in the movie.")

    df = _cm._tracks(well)
    by_key = df.drop_duplicates(["frame", "raw_label"]).set_index(["frame", "raw_label"])
    suspect = set(m.get("track_multiplicity", {}).get("suspect_tracks", []))
    chans = m.get("channels") or []
    ch_cols = [f"{c['column']}_{r}_mean" for c in chans for r in ("nuc", "ring")
               if f"{c['column']}_{r}_mean" in df.columns]
    field_med = (df.drop_duplicates(["frame", "track_id"]).groupby("frame")[ch_cols].median()
                 if ch_cols else None)

    snap_px = _SNAP_UM / um_px
    lab_cache: dict = {}
    for p in pts:
        p["mean_check"] = ""
        f = p["frame"]
        if not (0 <= f < n_frames):
            p["match"] = f"frame out of range (0-{n_frames - 1})"
            continue
        hit, used = None, f
        for d in sorted(range(-frame_tolerance, frame_tolerance + 1), key=abs):
            if 0 <= f + d < n_frames:
                v, dist = _locate(well, f + d, p["x_px"], p["y_px"], snap_px, lab_cache)
                if v is not None:
                    hit, used = (v, dist), f + d
                    break
        if hit is None:
            p["match"] = "none"
            continue
        v, dist = hit
        row = by_key.loc[(used, v)] if (used, v) in by_key.index else None
        if row is None:
            p["match"] = "mask not in tracks.csv"
            continue
        p["match"] = ("inside" if dist == 0 else f"snapped {dist * um_px:.1f} um") + (
            f" (frame {used - f:+d})" if used != f else "")
        tid = int(row["track_id"])
        p.update({
            "matched_frame": used, "track_id": tid,
            "area_um2": round(float(row["area_um2"]), 2),
            "diameter_um": round(2 * math.sqrt(float(row["area_um2"]) / math.pi), 2),
            "cx": round(float(row["cx"]), 1), "cy": round(float(row["cy"]), 1),
            "dna_mean": round(float(row["intensity_mean"]), 1),
            "dna_total": round(float(row["intensity_integrated"]), 0),
            "hours": round(_cm._hours(well, used), 3),
            "warn": "; ".join(w for w in (
                "merged-cells track" if tid in suspect else "",
                f"{int(row['n_masks_in_frame'])} shapes share this id" if row.get("n_masks_in_frame", 1) > 1 else "",
            ) if w),
        })
        for c in ch_cols:
            val = row[c]
            p[c] = round(float(val), 1) if pd.notna(val) else None
            med = field_med.at[used, c] if used in field_med.index else np.nan
            p[c + "_vs_field"] = round(float(val / med), 3) if pd.notna(val) and med else None

    # Frame check against the raw file.
    if check_frames == "auto" and any(p["mean"] is not None for p in pts):
        nd2_file = _find_nd2(well, nd2_path)
        if nd2_file is None or not nd2_file.is_file():
            notes.append(f"frame check skipped: raw .nd2 {m.get('source_nd2')!r} not found "
                         f"under {_cm.RAW_ND2_ROOT} (pass nd2_path=...)")
        else:
            _mean_check(well, pts, nd2_file)
            n_ok = sum(p["mean_check"] == "ok" for p in pts)
            n_mean = sum(p["mean"] is not None for p in pts)
            notes.append(f"frame check vs {nd2_file.name}: {n_ok}/{n_mean} exact")

    # Possible duplicates: same track within a few frames, or the same spot.
    dups = []
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            if abs(a["frame"] - b["frame"]) > _DUP_FRAMES:
                continue
            same_track = a.get("track_id") is not None and a.get("track_id") == b.get("track_id")
            near = math.hypot(a["x_px"] - b["x_px"], a["y_px"] - b["y_px"]) <= _DUP_PX
            if same_track or near:
                dups.append(f"{a['label']}/{b['label']}")

    cols = ["label", "frame", "x_px", "y_px", "match", "matched_frame", "track_id", "hours",
            "area_um2", "diameter_um", "cx", "cy", "dna_mean", "dna_total"]
    cols += [c for cc in ch_cols for c in (cc, cc + "_vs_field")]
    cols += ["mean_check", "warn"]
    table = [{c: p.get(c, "") for c in cols} for p in pts]
    for r, p in zip(table, pts):
        r["x_px"], r["y_px"] = round(p["x_px"], 2), round(p["y_px"], 2)
        r["frame"] = p["frame"]

    if out_csv is None and csv_path is not None:
        src = Path(csv_path)
        out_csv = str(src.with_name(src.stem + "_measured.csv"))
    if out_csv is not None:
        out = Path(out_csv)
        if _cm.BUNDLE.resolve() in out.resolve().parents:
            raise ValueError(f"out_csv {out} is inside the bundle, which a rebuild replaces; "
                             f"write it next to the researcher's own files instead.")
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(table)

    counts: dict[str, int] = {}
    for p in pts:
        k = p["match"].split(" ")[0]
        counts[k] = counts.get(k, 0) + 1
    lines = [f"{well}: {len(pts)} points -- " + ", ".join(f"{v} {k}" for k, v in counts.items())]
    lines += ["  " + n for n in notes]
    shifted = [f"{p['label']} ({p['mean_check']})" for p in pts if p["mean_check"] not in ("", "ok")]
    if shifted:
        lines.append("  frame check flags: " + ", ".join(shifted))
    missed = [p["label"] for p in pts if p["match"] in ("none",) or p["match"].startswith(("frame out", "mask not"))]
    if missed:
        lines.append("  no cell at: " + ", ".join(missed)
                     + (" (try frame_tolerance=2)" if frame_tolerance == 0 else ""))
    if dups:
        lines.append("  possible duplicate clicks: " + ", ".join(dups))
    if out_csv is not None:
        lines.append(f"  table written: {out_csv} ({len(cols)} columns)")
    else:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(table)
        lines.append(buf.getvalue().rstrip("\n"))
    return "\n".join(lines)


__all__ = ["measure_points", "_fit_axis", "_resolve_units", "_read_points", "_locate"]
