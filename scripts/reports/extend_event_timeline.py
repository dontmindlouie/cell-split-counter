"""Extend one event's saved review crops beyond the pipeline's fixed +/-24 frame
review window (src/review.py's _FRAMES_BEFORE/_FRAMES_AFTER, 8 frames each side at
stride 3), so a researcher who already spotted something interesting in
researcher_browser.py can rewind further to see the run-up to it, or fast-forward
further to see what happened after (did the cell die, keep dividing, stay alive?) --
2026-07-18, the researcher's real question after finding an interesting event.

No API calls, no re-running segmentation/tracking -- reads the same run_dir/frames/
raw PNGs already on disk and re-crops at the event's already-recorded (fixed)
centroid, same as scripts/regenerate_review_crops.py. Deletes and rewrites the
event's existing review_crops/ folder rather than appending to it: _save_debug_crops
numbers files by position-in-window (`{pos:02d}_...`), and a wider window shifts
every frame's position, so appending would leave stale duplicate copies of frames
that already existed under the old (smaller) window's pos numbers.

Fixed-centroid, same as every other crop in this codebase today -- if the cell has
drifted far from where it was at the split/death frame by the edge of the requested
window, it can end up off-center or clipped near the edge of the crop. See
[[project_cell_split_counter_marker_tracking_backlog]] for the (separate, not yet
built) per-frame crop-following backlog item -- more relevant than ever once windows
get wider, since more elapsed time means more drift.

After running this, regenerate researcher_browser.py's HTML for the run -- it already
shows every PNG found in an event's crop folder via the existing "Show every frame"
dense-mode toggle in the lightbox, no code changes needed there to pick up the wider
window. (researcher_browser.py prints a ready-to-copy command for this on each card.)

Usage:
    python scripts/reports/extend_event_timeline.py data/output/<run_dir> \\
        --track-id 1234 --peak-frame 567 [--before 40] [--after 40]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.classify import LineageEvent
from src.review import _build_dense_debug_window, _save_debug_crops, _write_verdict_txt


def _verdict_from_row(r: dict) -> dict:
    if r.get("split_topology") == "death":
        dropout = r.get("likely_division_dropout")
        return {
            "likely_division_dropout": None if dropout in (None, "") else dropout == "1",
            "confidence": float(r["ai_confidence"]) if r.get("ai_confidence") else 0.0,
            "description": r.get("ai_notes") or "",
            "anomaly_notes": r.get("anomaly_notes") or None,
        }
    conf = r.get("ai_confidence")
    return {
        "verdict": "real" if conf and float(conf) > 0 else "false_positive",
        "confidence": float(r["raw_ai_confidence"]) if r.get("raw_ai_confidence") else (float(conf) if conf else 0.0),
        "description": r.get("ai_notes") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--peak-frame", required=True, type=int)
    parser.add_argument("--before", type=int, default=40,
                         help="frames before the event to save (default 40, vs. the pipeline's fixed 24)")
    parser.add_argument("--after", type=int, default=40,
                         help="frames after the event to save (default 40, vs. the pipeline's fixed 24)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    frame_dir = run_dir / "frames"
    debug_dir = run_dir / "review_crops"

    rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))
    matches = [
        r for r in rows
        if r.get("track_id") == str(args.track_id) and r.get("peak_frame") == str(args.peak_frame)
    ]
    if not matches:
        raise SystemExit(f"no events.csv row with track_id={args.track_id} peak_frame={args.peak_frame}")
    row = matches[0]

    cx = float(row["centroid_x"]) if row.get("centroid_x") else None
    cy = float(row["centroid_y"]) if row.get("centroid_y") else None
    event = LineageEvent(
        track_id=args.track_id,
        parent_id=int(row["parent_id"]) if row.get("parent_id") else None,
        frame=args.peak_frame,
        event_type=None,
        classification_source=row.get("classification_source", ""),
        confidence=float(row["ai_confidence"]) if row.get("ai_confidence") else 0.0,
        centroid=(cx, cy) if cx is not None and cy is not None else None,
    )

    dense_paths = _build_dense_debug_window(event, frame_dir, frames_before=args.before, frames_after=args.after)
    if not dense_paths:
        raise SystemExit(f"no frame PNGs found under {frame_dir} for the requested window")

    event_debug_dir = debug_dir / f"frame_{args.peak_frame:05d}_track_{args.track_id}"
    if event_debug_dir.exists():
        for p in event_debug_dir.glob("*.png"):
            p.unlink()

    event_debug_dir = _save_debug_crops(dense_paths, event, debug_dir)
    _write_verdict_txt(event_debug_dir, _verdict_from_row(row))

    lo, hi = dense_paths[0][0], dense_paths[-1][0]
    print(f"extended track {args.track_id} @ frame {args.peak_frame}: frames {lo}-{hi} "
          f"({len(dense_paths)} crops) -> {event_debug_dir}")
    print("re-run scripts/reports/researcher_browser.py on this run dir to see the wider "
          "window (toggle 'Show every frame' on the card)")


if __name__ == "__main__":
    main()
