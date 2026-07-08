"""Package confirmed division events for papers and presentations (Track 2).

For each real division event (confidence >= min_conf), writes a folder containing:
  - numbered PNG crops around the split frame (5 before + 5 after)
  - info.txt with all metadata including ACD classification fields

Folder naming: {acd_type}__frame_{pf:05d}_parent_{pid}__conf_{conf}
ACD type is "unclassified" if the event predates the combined verify+classify review
(older events.csv files from before that merge won't have acd_division_type populated).

Daughters from the same split share one folder (deduplicated by parent_id + peak_frame).

Usage:
  python scripts/package_events.py
  python scripts/package_events.py --min-conf 0.7 --output data/packages_v2
  python scripts/package_events.py --events data/output/events.csv --frames data/frames
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVENTS_CSV, FRAME_DIR, PACKAGES_DIR as OUTPUT_DIR
from src.review import _DIV_FRAMES_AFTER, _DIV_FRAMES_BEFORE, _crop_image, _find_frame


def _abn_tag(row: dict) -> str:
    """Short abnormality string for info.txt — empty if none detected."""
    flags = {
        "misaligned_chromosomes": "misaligned",
        "lagging_chromosome":     "lagging",
        "anaphase_bridge":        "bridge",
        "micronucleus":           "micronucleus",
    }
    active = [label for col, label in flags.items() if row.get(col) == "1"]
    return ", ".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events",   type=Path, default=EVENTS_CSV, help="path to events.csv")
    parser.add_argument("--frames",   type=Path, default=FRAME_DIR,  help="directory containing frame PNGs")
    parser.add_argument("--output",   type=Path, default=OUTPUT_DIR, help="output directory (cleared each run)")
    parser.add_argument("--min-conf", type=float, default=0.5,       help="minimum confidence to include (default 0.5)")
    args = parser.parse_args()

    with open(args.events, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep only real events above threshold; deduplicate by split point.
    seen: set[tuple] = set()
    targets = []
    for r in rows:
        # this tool packages divisions specifically -- death rows are track ends.
        if r.get("split_topology") not in ("normal_split", "multi_way_split"):
            continue
        try:
            conf = float(r["claude_confidence"])
        except (KeyError, ValueError):
            continue
        if conf < args.min_conf:
            continue
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        if key in seen:
            continue
        seen.add(key)
        targets.append(r)

    print(f"Packaging {len(targets)} events (confidence >= {args.min_conf}) from {args.events}")

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    saved = 0
    for r in targets:
        pf     = int(r["peak_frame"])
        pid    = r.get("parent_id", "?")
        conf   = r["claude_confidence"]
        src    = r.get("classification_source", "")
        acd    = r.get("acd_division_type", "") or "unclassified"
        br     = r.get("bleach_risk", "")
        notes  = r.get("claude_notes", "")
        tc     = r.get("tracker_persistence_score", "")

        cx = float(r["centroid_x"]) if r.get("centroid_x") else None
        cy = float(r["centroid_y"]) if r.get("centroid_y") else None
        centroid = (cx, cy) if cx is not None and cy is not None else None

        abn_flags = "_".join(
            short for col, short in [
                ("misaligned_chromosomes", "mis"),
                ("lagging_chromosome",     "lag"),
                ("anaphase_bridge",        "bridge"),
                ("micronucleus",           "mn"),
            ] if r.get(col) == "1"
        )
        acd_part = f"{acd}_{abn_flags}" if abn_flags else acd
        conf_pct = f"{int(float(conf) * 100):03d}"
        folder_name = f"{pf:05d}_{acd_part}__p{pid}__conf_{conf_pct}"
        event_dir   = args.output / folder_name
        event_dir.mkdir()

        indices = list(range(max(0, pf - _DIV_FRAMES_BEFORE), pf + _DIV_FRAMES_AFTER + 1))
        frame_count = 0
        for pos, idx in enumerate(indices):
            path = _find_frame(args.frames, idx)
            if path is None:
                continue
            tag  = "before" if idx < pf else ("split" if idx == pf else "after")
            crop = _crop_image(path, centroid)
            cv2.imwrite(str(event_dir / f"{pos:02d}_{tag}_{idx:05d}.png"), crop)
            frame_count += 1

        (event_dir / "info.txt").write_text(
            f"peak_frame:             {pf}\n"
            f"parent_id:              {pid}\n"
            f"claude_confidence:      {conf}\n"
            f"tracker_persistence_score: {tc}\n"
            f"source:                 {src}\n"
            f"bleach_risk:            {br}\n"
            f"claude_notes:           {notes}\n"
            f"acd_division_type:      {acd}\n"
            f"abnormalities:          {_abn_tag(r)}\n"
            f"centroid:               {centroid}\n"
            f"frames_saved:           {frame_count}\n"
        )
        abn_display = f"  [{abn_flags}]" if abn_flags else ""
        print(f"  {acd:<14} frame={pf:3d} conf={conf_pct}%{abn_display}")
        saved += 1

    print(f"\nSaved {saved} event folders to {args.output}")
    print(f"Open in Explorer: start {args.output.resolve()}")


if __name__ == "__main__":
    main()
