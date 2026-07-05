"""Dump the frame crops that would be sent to Claude for events in a frame range.

Reads data/output/events.csv and data/frames/; saves cropped PNGs to
data/debug/crops/ without making any Claude API calls. Use this to inspect
exactly what Claude sees before deciding whether to re-run review.

Usage:
  python scripts/dump_crops.py [--start-frame N] [--end-frame N] [--source rule|claude|any]
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEBUG_DIR, EVENTS_CSV, FRAME_DIR
from src.review import _FRAMES_BEFORE, _FRAMES_AFTER, _FRAME_STRIDE, _crop_image, _find_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--source", choices=["rule", "claude", "any"], default="any")
    args = parser.parse_args()

    with open(EVENTS_CSV, newline="") as f:
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
        key = (r["parent_id"], r["peak_frame"])
        if key in seen:
            continue
        seen.add(key)
        targets.append(r)

    print(f"Found {len(targets)} unique splits in frames {args.start_frame}–{args.end_frame or 'end'} (source={args.source})")

    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True)

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
        label = f"REAL" if float(conf) > 0 else "FP"
        folder_name = f"{label}__frame_{pf:05d}_parent_{r['parent_id']}__src_{src}__conf_{conf}__tc_{tracker_conf}"
        event_dir = DEBUG_DIR / folder_name
        event_dir.mkdir()

        before_indices = [pf - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
        after_indices = [pf + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
        indices = [i for i in before_indices if i >= 0] + [pf] + after_indices
        frame_count = 0
        for pos, idx in enumerate(indices):
            path = _find_frame(FRAME_DIR, idx)
            if path is None:
                continue
            tag = "before" if idx < pf else ("split" if idx == pf else "after")
            crop = _crop_image(path, centroid)
            out = event_dir / f"{pos:02d}_{tag}_{idx:05d}.png"
            cv2.imwrite(str(out), crop)
            frame_count += 1

        # summary file
        notes = r.get("claude_notes", "")
        (event_dir / "info.txt").write_text(
            f"peak_frame:        {pf}\n"
            f"parent_id:         {r['parent_id']}\n"
            f"claude_confidence:        {conf}\n"
            f"tracker_persistence_score:{tracker_conf}\n"
            f"source:            {src}\n"
            f"claude_notes:      {notes}\n"
            f"centroid:          {centroid}\n"
            f"frames_saved:      {frame_count}\n"
        )
        print(f"  [{label}] frame={pf:3d} tc={tracker_conf:>6s} conf={conf:>4s}  {notes[:60]}")
        saved += 1

    print(f"\nSaved {saved} event folders to {DEBUG_DIR}")
    print(f"Open in Explorer: start {DEBUG_DIR.resolve()}")


if __name__ == "__main__":
    main()
