"""Run ACD division type + abnormality classification on confirmed events in an existing events.csv.

Reads events.csv, classifies events with confidence >= min_conf using Claude vision,
and writes an updated CSV with acd_division_type + abnormality flag columns appended.
Does not re-run segmentation or tracking.

Usage:
  python scripts/classify_divisions.py [--events PATH] [--frames PATH] [--min-conf F] [--max-reviews N]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import EventType, LineageEvent
from src.review import review_division_type

# Default to main worktree data since frames/events live there
_ROOT = Path(__file__).resolve().parents[2] / "cell-split-counter"
DEFAULT_EVENTS = _ROOT / "data/output/events.csv"
DEFAULT_FRAMES = _ROOT / "data/frames"


def _row_to_event(row: dict) -> LineageEvent:
    cx = float(row["centroid_x"]) if row.get("centroid_x") else None
    cy = float(row["centroid_y"]) if row.get("centroid_y") else None
    # handle both old column name (division_type) and new (split_topology)
    topology_val = row.get("split_topology") or row.get("division_type") or "normal_split"
    bleach = row.get("bleach_risk")
    return LineageEvent(
        track_id=int(row["track_id"]),
        parent_id=int(row["parent_id"]) if row.get("parent_id") else None,
        frame=int(row["peak_frame"]),
        event_type=EventType(topology_val),
        classification_source=row["classification_source"],
        confidence=float(row["confidence"]),
        centroid=(cx, cy) if cx is not None and cy is not None else None,
        claude_notes=row.get("claude_notes") or None,
        bleach_risk=float(bleach) if bleach else None,
    )


ACD_COLS = ["acd_division_type", "misaligned_chromosomes", "lagging_chromosome", "anaphase_bridge", "micronucleus"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--max-reviews", type=int, default=50)
    args = parser.parse_args()

    with open(args.events, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Ensure ACD columns exist in fieldnames
    for col in ACD_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        for col in ACD_COLS:
            row.setdefault(col, "")

    events = [_row_to_event(r) for r in rows]
    n_eligible = sum(1 for e in events if e.confidence >= args.min_conf)
    print(f"Found {len(events)} total events, {n_eligible} with confidence >= {args.min_conf}")

    classified = review_division_type(
        events,
        args.frames,
        min_confidence=args.min_conf,
        max_reviews=args.max_reviews,
    )

    # Build lookup by (track_id, frame) → classified event
    result_map = {(e.track_id, e.frame): e for e in classified}

    def _flag(v):
        return "" if v is None else ("1" if v else "0")

    updated = 0
    for row in rows:
        key = (int(row["track_id"]), int(row["peak_frame"]))
        e = result_map.get(key)
        if e and e.acd_division_type is not None:
            row["acd_division_type"] = e.acd_division_type or ""
            row["misaligned_chromosomes"] = _flag(e.misaligned_chromosomes)
            row["lagging_chromosome"] = _flag(e.lagging_chromosome)
            row["anaphase_bridge"] = _flag(e.anaphase_bridge)
            row["micronucleus"] = _flag(e.micronucleus)
            updated += 1

    out_path = args.events.parent / (args.events.stem + "_classified.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nClassified {updated} events. Written to {out_path}")


if __name__ == "__main__":
    main()
