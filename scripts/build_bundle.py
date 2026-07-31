"""Assemble a shippable artifact bundle from an existing pipeline run.

Why this exists: a finished run directory is ~8GB, of which ~98% is scratch that
nobody downstream can use -- `frames/_memmap/` (Trackastra's float32 input copy, a
duplicate uint8 copy, and two raw uint16 label stacks) plus `review_crops/`, which
is hundreds of thousands of small JPEGs regenerable from the frames on demand.
Measured on TSC_batch2_M12_RUES2: 6.2GB of frames/ is 6.1GB memmap and 0.10GB of
actual PNGs, and review_crops/ is another 4.0GB across 110,200 files.

This script writes the ~5% that is actually needed to navigate the video later:
indexed raw frames (already exported by src.ingest, copied as-is), the tracked
label maps re-encoded as lossless 16-bit PNG (52x smaller than the raw memmap,
verified bit-exact), a per-frame track table, and a manifest carrying calibration
read from the ND2. Result is ~0.2GB per run instead of ~8GB.

The manifest is the reason this is a build step and not a query-time operation:
calibration lives only in the ND2, and the ND2 never ships. Anything reading the
bundle gets real units without needing an ND2 reader installed. Calibration is
HARD-FAILED here rather than defaulted -- a silently-assumed 1.0 px/um or a
nominal frame interval produces confidently wrong measurements that a reader has
no way to detect (see the frame-interval note below).

Frame interval is stored as per-frame timestamps, never a scalar. The ND2's
requested TimeLoop period is not what actually happened: on TSC batch2 (16 XY
positions per cycle) the requested 180s came out as a 294s median with a
175-867s range, because the stage could not finish a cycle in time. Bewo (6
positions) held 180.03s. Trusting the nominal 180s on TSC batch2 UNDERSTATES
real elapsed time by ~39% (equivalently, real intervals run ~64% longer than
nominal), and even the median misprices individual short intervals -- which is
exactly what metaphase dwell is.

Usage:
  python scripts/build_bundle.py data/output/TSC_batch2_M12_RUES2 \
      --nd2 "G:/Projects/TSC batch2/20260709_..._M12 RUES2.nd2" \
      --out data/bundle --cell-line RUES2
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lineage import _build_tracking_index, _open_tracked_masks  # noqa: E402

_FRAME_RE = re.compile(r"frame_(\d+)_raw(\d+)\.png$")
_LABEL_DTYPE = np.uint16


class CalibrationError(RuntimeError):
    """Raised when the ND2 lacks something the bundle must not fabricate."""


def frame_index_pairs(run_dir: Path) -> list[tuple[int, int]]:
    """[(kept_index, raw_nd2_index)] from the exported frame filenames.

    src.ingest names frames `frame_<kept>_raw<raw>.png`, where kept is the index
    into the label memmaps and raw is the index into the ND2's T axis. They
    diverge whenever a run used frame_step > 1, and the timestamp lookup needs
    the raw one, so the mapping is parsed rather than assumed to be identity.
    """
    pairs = []
    for p in sorted((run_dir / "frames").glob("frame_*_raw*.png")):
        m = _FRAME_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    if not pairs:
        raise CalibrationError(f"no frame_*_raw*.png files under {run_dir / 'frames'}")
    kept = [k for k, _ in pairs]
    if kept != list(range(len(kept))):
        raise CalibrationError(f"{run_dir.name}: kept frame indices are not contiguous from 0")
    return pairs


def read_calibration(nd2_path: Path, raw_indices: list[int]) -> dict:
    """Pull everything the bundle needs out of the ND2. Hard-fails on absence.

    Deliberately reads acquisition parameters (objective/NA/emission/bit depth)
    alongside the strictly-required calibration: they cost nothing to carry and
    they bound what the pixels can support. NA 0.8 at 610nm resolves ~0.38um
    while the sampling is 0.567um/px, so sub-nuclear features are ~3x
    undersampled -- a reader measuring chromosome-scale structure needs to know
    that, and it is not recoverable from the PNGs alone.
    """
    import nd2

    with nd2.ND2File(str(nd2_path)) as f:
        sizes = dict(f.sizes)
        T = sizes.get("T")
        if T is None:
            raise CalibrationError(f"{nd2_path.name}: no T axis; not a time-lapse")

        vs = f.voxel_size()
        if not vs.x or not np.isfinite(vs.x):
            raise CalibrationError(f"{nd2_path.name}: no usable pixel size in voxel_size()")

        bad = [r for r in raw_indices if r >= T]
        if bad:
            raise CalibrationError(
                f"{nd2_path.name}: run references raw frame {max(bad)} but ND2 has T={T}. "
                "Wrong ND2 for this run?"
            )

        timestamps = []
        for r in raw_indices:
            ts = f.frame_metadata(int(r)).channels[0].time.relativeTimeMs
            if ts is None or not np.isfinite(ts):
                raise CalibrationError(f"{nd2_path.name}: missing timestamp at raw frame {r}")
            timestamps.append(float(ts))
        if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
            raise CalibrationError(f"{nd2_path.name}: timestamps are not strictly increasing")

        ch = f.metadata.channels[0]
        color = getattr(ch.channel, "color", None)
        scope = ch.microscope
        info = f.text_info or {}
        desc = info.get("description", "")

        def _grab(pattern):
            m = re.search(pattern, desc)
            return m.group(1).strip() if m else None

        attrs = f.attributes
        cal = {
            "pixel_size_um": float(vs.x),
            "frame_timestamps_ms": timestamps,
            "nd2_sizes": sizes,
            "height_px": attrs.heightPx,
            "width_px": attrs.widthPx,
            "bits_significant": attrs.bitsPerComponentSignificant,
            "channel_name": ch.channel.name,
            # The ND2's own display LUT. NIS writes it from the acquisition preset;
            # reproducing it is what makes a rendered frame match what she sees in
            # Fiji, instead of us picking a colormap by taste.
            "display_color_rgb": [color.r, color.g, color.b] if color else None,
            "emission_range": _grab(r"Emission Range:\s*(.+)"),
            "excitation_nm": getattr(ch.channel, "excitationLambdaNm", None),
            "emission_nm": getattr(ch.channel, "emissionLambdaNm", None),
            "objective": scope.objectiveName,
            "numerical_aperture": scope.objectiveNumericalAperture,
            "objective_magnification": scope.objectiveMagnification,
            "zoom_magnification": scope.zoomMagnification,
            "modality": list(scope.modalityFlags or []),
            "position_name": f.frame_metadata(int(raw_indices[0])).channels[0].position.name,
            "acquisition_date": info.get("date"),
            "acquisition_dimensions": _grab(r"(Dimensions:.*)"),
            "source_nd2": nd2_path.name,
        }

    d = np.diff(timestamps) if len(timestamps) > 1 else np.array([0.0])
    cal["interval_ms"] = {
        "median": float(np.median(d)), "mean": float(d.mean()),
        "min": float(d.min()), "max": float(d.max()), "std": float(d.std()),
    }
    cal["duration_hours"] = (timestamps[-1] - timestamps[0]) / 3.6e6
    # Nyquist wants ~2.3 samples across the resolution limit; report the shortfall
    # rather than making the reader recompute it.
    em_nm = cal["emission_nm"] or 610.0
    na = cal["numerical_aperture"]
    if na:
        res_um = (em_nm / 1000.0) / (2.0 * na)
        cal["optical_resolution_um"] = res_um
        cal["sampling_shortfall_x"] = float(vs.x) / (res_um / 2.0)
    return cal


def write_lineage(run_dir: Path, out_run: Path) -> dict:
    """Emit lineage.csv -- who each track's mother was, and which tracks are its
    daughters -- so a track that ends mid-division can be followed into the next one.

    Preferred source is _memmap/ctc_lineage.csv, written by src.track from
    Trackastra's own CTC table: complete, every track, topology only.

    Runs tracked before 2026-07-29 have no such file, and Trackastra's own graph
    cannot be recovered from what they left on disk (canonical_labels.json preserves
    the label remap, but the parent table lived only in memory). Rather than re-run a
    GPU tracking pass per well, rebuild the topology from tracks.csv geometry --
    see src.lineage.build_lineage_from_tracks. That is complete for every track,
    costs nothing, and uses the same adjacency rule that generates division
    candidates, so lineage and candidates cannot disagree with each other.

    The events.csv fallback that used to sit here is GONE (2026-07-30). It was
    partial by construction -- a track appeared only if the pipeline emitted an
    event for it (2,364 of 5,163 tracks on M12_RUES2) -- and it welded pure topology
    to a file of AI verdicts, so anything wanting the graph had to read the answers.
    Both replacements are strictly better and neither depends on events.csv, which
    is what lets that file leave the bundle entirely.
    """
    full = run_dir / "frames" / "_memmap" / "ctc_lineage.csv"
    if full.is_file():
        shutil.copy2(full, out_run / "lineage.csv")
        n = sum(1 for _ in open(full, encoding="utf-8")) - 1
        print(f"  lineage.csv: {n:,} tracks (complete, from Trackastra CTC)")
        return {"coverage": "complete", "source": "ctc", "n_tracks": n}

    tracks_csv = out_run / "tracks.csv"
    if not tracks_csv.is_file():
        print("  lineage.csv: skipped (no ctc_lineage.csv and no tracks.csv)")
        return {"coverage": "none"}

    import pandas as pd

    from src.lineage import build_lineage_from_tracks

    lin = build_lineage_from_tracks(pd.read_csv(tracks_csv))
    lin.to_csv(out_run / "lineage.csv", index=False)
    n = len(lin)
    n_linked = int((lin.parent_id != "").sum())
    print(f"  lineage.csv: {n:,} tracks (complete, from tracks.csv geometry; "
          f"{n_linked:,} have a parent)")
    return {"coverage": "complete", "source": "geometry", "n_tracks": n,
            "n_linked": n_linked}


def write_labels_and_tracks(
    run_dir: Path, out_run: Path, pairs: list[tuple[int, int]],
    nd2_path: Path, cal: dict, bleach_curve: bool,
) -> tuple[dict, list[dict]]:
    """Re-encode label maps as PNG-16 and build the per-frame track table.

    One pass over the run: for each frame, read the tracked label map from the
    memmap, write it as a lossless 16-bit PNG, and measure every cell in it
    against the ND2's raw 16-bit pixels.

    Intensity is measured against the ND2, not the exported PNGs, because
    src.ingest rescales each frame to 8-bit using that frame's OWN 0.5/99.5
    percentiles -- so brightness in the PNGs is not comparable across frames and
    a real bleaching trend is normalised away frame by frame. With an H2B-mCherry
    marker (histone-fused, so signal is stoichiometric with nucleosomes)
    integrated intensity tracks DNA content, which makes it worth measuring
    properly: it should stay flat through mitosis while chromatin condenses, and
    should differ ~2x in a whole-genome-duplicated line.

    NOTE (track_id, frame) is NOT unique -- one row per MASK, not per track.
    _bridge_track_gaps can map several raw Trackastra labels onto one canonical
    track_id, and in ~2% of rows two or more are present in the same frame. The
    real key is (track_id, frame, raw_label).

    Measured on M12_RUES2, this is a TRACKING DEFECT, not a division: only 1.2%
    of multiplexed keys fall within +/-12 frames of a split of the same track,
    and the co-existing masks sit a median 17.9um apart -- wider than a whole
    nucleus (~12.2um). So _bridge_track_gaps is over-merging two genuinely
    distinct cells onto one id. 168 of 5,163 tracks (3.3%) are affected; 17 of
    those are multiplexed across >50% of their own lifetime.

    Rows are NOT collapsed, because averaging two distinct cells' centroids
    produces a position where no cell is. Instead `n_masks_in_frame` makes the
    condition queryable and `suspect_tracks` in the manifest lists the badly
    affected ids, so a reader can exclude them from measurements rather than
    silently averaging through a tracker error. Note events.csv inherits the
    same canonical ids, so those events carry the defect too.

    `solidity` (area / convex-hull area) is included because it's the strongest
    single geometric feature found for locating mitosis so far -- a mask that
    rounds up during chromatin condensation and briefly fragments visually dips
    solidity even where area alone doesn't move (see
    project-cell-split-counter-combo-lag-intensity-tests in Claude memory:
    solidity-alone AUC 0.689, area+solidity combined 0.751-0.783). It's free
    here since regionprops_table already computes area from the same mask --
    solidity just asks it for the convex hull too.
    """
    from skimage.measure import regionprops_table

    tracked = _open_tracked_masks(run_dir)
    canonical_of, _, _ = _build_tracking_index(tracked, run_dir)

    labels_dir = out_run / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    import nd2

    rows: list[dict] = []
    bleach: list[dict] = []
    ts = cal["frame_timestamps_ms"]
    t0 = ts[0]

    with nd2.ND2File(str(nd2_path)) as f:
        for i, (kept, raw_idx) in enumerate(pairs):
            lab = np.asarray(tracked[kept]).astype(_LABEL_DTYPE)
            ok, buf = cv2.imencode(".png", lab)
            if not ok:
                raise RuntimeError(f"failed to PNG-encode label map for frame {kept}")
            (labels_dir / f"frame_{kept:05d}.png").write_bytes(buf.tobytes())

            img = np.array(f.read_frame(int(raw_idx)), copy=True)

            if lab.max() == 0:
                if bleach_curve:
                    bleach.append({"frame": kept, "n_cells": 0,
                                   "frame_mean": round(float(img.mean()), 2),
                                   "cell_intensity_median": None})
                continue
            props = regionprops_table(
                lab, intensity_image=img,
                properties=("label", "centroid", "area", "bbox", "intensity_mean", "solidity"),
            )
            if bleach_curve:
                # frame_mean alone is NOT a bleaching curve: it scales with how many
                # nuclei are in the field, and in a proliferating line that dominates.
                # On M12_RUES2 frame_mean rises +108% over 72h purely because the cell
                # count goes 127 -> 353, while median per-cell intensity is flat
                # (535 -> 549). The per-cell median is the density-controlled signal;
                # confluent lines like BeWo, where count is stable, do show a real
                # decline in it. Both are stored so the confound stays visible.
                bleach.append({
                    "frame": kept,
                    "n_cells": int(len(props["label"])),
                    "frame_mean": round(float(img.mean()), 2),
                    "cell_intensity_median": round(float(np.median(props["intensity_mean"])), 1),
                })
            px2 = cal["pixel_size_um"] ** 2
            for j in range(len(props["label"])):
                rawlab = int(props["label"][j])
                area = float(props["area"][j])
                mean_i = float(props["intensity_mean"][j])
                rows.append({
                    "track_id": canonical_of.get(rawlab, rawlab),
                    "frame": kept,
                    "time_ms": round(ts[i] - t0, 1),
                    "cx": round(float(props["centroid-1"][j]), 2),
                    "cy": round(float(props["centroid-0"][j]), 2),
                    "area_px": round(area, 1),
                    "area_um2": round(area * px2, 2),
                    "bbox_y0": int(props["bbox-0"][j]), "bbox_x0": int(props["bbox-1"][j]),
                    "bbox_y1": int(props["bbox-2"][j]), "bbox_x1": int(props["bbox-3"][j]),
                    "intensity_mean": round(mean_i, 1),
                    "intensity_integrated": round(mean_i * area, 1),
                    "solidity": round(float(props["solidity"][j]), 4),
                    "raw_label": rawlab,
                })
            if i % 50 == 0:
                print(f"    frame {i+1}/{len(pairs)}", end="\r", flush=True)
    print(" " * 40, end="\r")

    # Annotate mask multiplicity per (track_id, frame), and flag tracks where it is
    # pervasive enough that the id should not be trusted as one cell.
    counts: dict[tuple[int, int], int] = {}
    for r in rows:
        k = (r["track_id"], r["frame"])
        counts[k] = counts.get(k, 0) + 1
    frames_seen: dict[int, set] = {}
    frames_multi: dict[int, set] = {}
    for r in rows:
        n = counts[(r["track_id"], r["frame"])]
        r["n_masks_in_frame"] = n
        frames_seen.setdefault(r["track_id"], set()).add(r["frame"])
        if n > 1:
            frames_multi.setdefault(r["track_id"], set()).add(r["frame"])

    suspect = sorted(
        tid for tid, mf in frames_multi.items()
        if len(mf) / max(len(frames_seen[tid]), 1) > 0.5
    )
    stats = {
        "n_multiplexed_rows": sum(1 for r in rows if r["n_masks_in_frame"] > 1),
        "n_tracks_affected": len(frames_multi),
        "suspect_tracks": suspect,
        "suspect_track_rule": "track multiplexed in >50% of the frames it appears in",
    }
    return {"intensity_curve": bleach, "track_multiplicity": stats}, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a pipeline output run directory")
    ap.add_argument("--nd2", type=Path, required=True, help="source ND2 (read for calibration only)")
    ap.add_argument("--out", type=Path, default=Path("data/bundle"), help="bundle root")
    ap.add_argument("--cell-line", default=None, help="e.g. RUES2, nTSC, pTSC, WGD, BeWo")
    ap.add_argument("--condition", default=None, help="perturbation/treatment label, if any")
    ap.add_argument("--no-intensity-curve", action="store_true", help="skip per-frame intensity stats")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"not a directory: {run_dir}")
    if not args.nd2.is_file():
        sys.exit(f"ND2 not found: {args.nd2}")

    out_run = args.out / run_dir.name
    if out_run.exists() and not args.overwrite:
        sys.exit(f"{out_run} already exists (use --overwrite)")
    out_run.mkdir(parents=True, exist_ok=True)

    print(f"[{run_dir.name}]")
    pairs = frame_index_pairs(run_dir)
    print(f"  frames: {len(pairs)}")

    try:
        cal = read_calibration(args.nd2, [r for _, r in pairs])
    except CalibrationError as exc:
        sys.exit(f"  CALIBRATION FAILED: {exc}\n  Refusing to write a bundle with fabricated units.")
    iv = cal["interval_ms"]
    print(f"  px size: {cal['pixel_size_um']:.4f} um/px | interval median {iv['median']/1000:.1f}s "
          f"(range {iv['min']/1000:.0f}-{iv['max']/1000:.0f}s) | {cal['duration_hours']:.1f}h")

    print("  encoding labels + measuring tracks...")
    extra, rows = write_labels_and_tracks(
        run_dir, out_run, pairs, args.nd2, cal, not args.no_intensity_curve
    )

    tracks_csv = out_run / "tracks.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(tracks_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_tracks = len({r["track_id"] for r in rows})
    print(f"  tracks.csv: {len(rows):,} rows / {n_tracks:,} tracks")

    frames_out = out_run / "frames"
    frames_out.mkdir(exist_ok=True)
    for p in sorted((run_dir / "frames").glob("frame_*_raw*.png")):
        m = _FRAME_RE.search(p.name)
        shutil.copy2(p, frames_out / f"frame_{int(m.group(1)):05d}.png")

    for name in ("events.csv", "summary.json"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, out_run / name)

    lineage_info = write_lineage(run_dir, out_run)

    # Ship the schema doc INSIDE the bundle -- it is routinely handed to someone
    # (or some agent) with no repo, no docs/, and no history to reverse-engineer
    # from. Same reasoning as scripts/generate_package_readme.py.
    doc = Path(__file__).resolve().parents[1] / "docs" / "bundle_README.md"
    if doc.is_file():
        shutil.copy2(doc, out_run.parent / "README.md")

    manifest = {
        "run": run_dir.name,
        "cell_line": args.cell_line,
        "condition": args.condition,
        "n_frames": len(pairs),
        "n_tracks": n_tracks,
        "lineage": lineage_info,
        **cal,
        **extra,
    }
    (out_run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in out_run.rglob("*") if p.is_file())
    print(f"  bundle: {total/1e9:.2f} GB -> {out_run}")


if __name__ == "__main__":
    main()
