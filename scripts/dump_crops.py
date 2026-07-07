"""Dump the frame crops that would be sent to Claude for events in a frame range.

Reads an events.csv and a cached frame directory; saves cropped PNGs without making
any Claude API calls (frames must already be extracted/cached from a prior run --
this does not re-segment or re-track). Use this to inspect exactly what Claude saw,
or to backfill review_crops/ for a run that skipped them via --no-debug-crops.

Output folder naming matches src/review.py's own debug-crops convention exactly
(frame_<peak_frame, 5 digits>_parent_<parent_id>, with a verdict.txt) so crops
generated after the fact are indistinguishable from ones saved during a live review
call, and so scripts/generate_package_readme.py's review_crops detection picks them up.

Usage:
  python scripts/dump_crops.py [--start-frame N] [--end-frame N] [--source rule|claude|any]
  python scripts/dump_crops.py --events-csv data/output/<run>/events.csv \\
      --frame-dir data/frames_<run> --out-dir data/output/<run>/review_crops --confirmed-only
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

# Claude's notes can contain non-ASCII characters; Windows defaults redirected/piped
# stdout to the system codepage (cp1252), which crashes on them -- same issue fixed in
# main.py, needed again here since this script is its own entrypoint. See src/review.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEBUG_DIR, EVENTS_CSV, FRAME_DIR
from src.review import _FRAMES_BEFORE, _FRAMES_AFTER, _FRAME_STRIDE, _crop_image, _find_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", type=Path, default=EVENTS_CSV)
    parser.add_argument("--frame-dir", type=Path, default=FRAME_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEBUG_DIR,
                         help="default matches old behavior (data/debug/crops); pass an "
                              "output package's review_crops/ to backfill it")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--source", choices=["rule", "claude", "any"], default="any")
    parser.add_argument("--confirmed-only", action="store_true",
                         help="only dump events with claude_confidence > 0 (skip rejected false positives)")
    args = parser.parse_args()

    with open(args.events_csv, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    seen: set[tuple] = set()
    targets = []
    for r in rows:
        pf = int(r["peak_frame"])
        if pf < args.start_frame:
            continue
        if args.end_frame is not None and pf >= args.end_frame:
            continue
        if args.source != "any" and r["classification_source"] != args.source:
            continue
        if args.confirmed_only and not (r["claude_confidence"] and float(r["claude_confidence"]) > 0):
            continue
        key = (r["parent_id"], r["peak_frame"])
        if key in seen:
            continue
        seen.add(key)
        targets.append(r)

    print(f"Found {len(targets)} unique splits in frames {args.start_frame}–{args.end_frame or 'end'} "
          f"(source={args.source}, confirmed_only={args.confirmed_only})")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    import cv2

    saved = 0
    for r in targets:
        pf = int(r["peak_frame"])
        cx = float(r["centroid_x"]) if r.get("centroid_x") else None
        cy = float(r["centroid_y"]) if r.get("centroid_y") else None
        centroid = (cx, cy) if cx is not None and cy is not None else None

        conf = r["claude_confidence"]
        tracker_conf = r.get("tracker_persistence_score", "")
        src = r["classification_source"]
        label = "REAL" if float(conf) > 0 else "FP"
        event_dir = args.out_dir / f"frame_{pf:05d}_parent_{r['parent_id']}"
        event_dir.mkdir()

        before_indices = [pf - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
        after_indices = [pf + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
        indices = [i for i in before_indices if i >= 0] + [pf] + after_indices
        frame_count = 0
        for pos, idx in enumerate(indices):
            path = _find_frame(args.frame_dir, idx)
            if path is None:
                continue
            tag = "before" if idx < pf else ("split" if idx == pf else "after")
            crop = _crop_image(path, centroid)
            out = event_dir / f"{pos:02d}_{tag}_{idx:05d}.png"
            cv2.imwrite(str(out), crop)
            frame_count += 1

        notes = r.get("claude_notes", "")
        (event_dir / "verdict.txt").write_text(
            f"verdict:    {'real' if label == 'REAL' else 'false_positive'}\n"
            f"confidence: {conf}\n"
            f"notes:      {notes}\n"
            f"---\n"
            f"peak_frame:                {pf}\n"
            f"parent_id:                 {r['parent_id']}\n"
            f"tracker_persistence_score: {tracker_conf}\n"
            f"classification_source:     {src}\n"
            f"centroid:                  {centroid}\n"
            f"split_topology:            {r.get('split_topology', '')}\n"
            f"frames_saved:              {frame_count}\n",
            encoding="utf-8",
        )
        print(f"  [{label}] frame={pf:3d} tc={tracker_conf:>6s} conf={conf:>4s}  {notes[:60]}")
        saved += 1

    print(f"\nSaved {saved} event folders to {args.out_dir}")
    print(f"Open in Explorer: start {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
