"""Backfill ACD division type + abnormality classification onto an existing events.csv.

Older runs (or runs made before the combined verify+classify call existed) can end up
with ai_notes populated but the acd_division_type/misaligned/lagging/anaphase/
micronucleus columns empty, since classification used to be a separate opt-in pass
that defaulted off. This re-sends confirmed events (confidence >= min_conf) through
the same combined Claude call and fills in just the classification fields.
Does not re-run segmentation or tracking, and does not touch the existing verdict/notes.

Usage:
  python scripts/classify_divisions.py [--events PATH] [--frames PATH] [--min-conf F] [--max-reviews N]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv; load_dotenv()

import anthropic

from src.classify import EventType, LineageEvent
from src.config import CLAUDE_MODEL, HIGH_CONFIDENCE
from src.review import _review_and_classify

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
        confidence=float(row["ai_confidence"]),
        centroid=(cx, cy) if cx is not None and cy is not None else None,
        ai_notes=row.get("ai_notes") or None,
        bleach_risk=float(bleach) if bleach else None,
    )


ACD_COLS = ["acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
            "anaphase_bridge", "micronucleus", "anomaly_notes"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--min-conf", type=float, default=HIGH_CONFIDENCE)
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
    to_classify = [e for e in events if e.confidence >= args.min_conf]
    print(f"Found {len(events)} total events, {len(to_classify)} with confidence >= {args.min_conf}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    # One call per unique split point (parent_id, frame) -- daughters share a result.
    to_call: dict[tuple, LineageEvent] = {}
    for e in to_classify:
        key = (e.parent_id, e.frame)
        if key not in to_call and len(to_call) < args.max_reviews:
            to_call[key] = e

    def _flag(v):
        return "" if v is None else ("1" if v else "0")

    split_result: dict[tuple, dict] = {}
    for key, event in to_call.items():
        try:
            split_result[key] = _review_and_classify(client, event, args.frames, CLAUDE_MODEL)
        except Exception as exc:
            print(f"  frame={event.frame:3d} parent={event.parent_id} [ERROR] {exc}")
            continue
        r = split_result[key]
        print(
            f"  frame={event.frame:3d} parent={event.parent_id}"
            f" {r.get('acd_division_type', '?')}"
            f" mis={r.get('misaligned_chromosomes')} lag={r.get('lagging_chromosome')}"
            f" bridge={r.get('anaphase_bridge')} mn={r.get('micronucleus')}"
        )

    updated = 0
    for e in to_classify:
        key = (e.parent_id, e.frame)
        r = split_result.get(key)
        if r is None:
            continue
        row = next(row for row in rows if int(row["track_id"]) == e.track_id and int(row["peak_frame"]) == e.frame)
        row["acd_division_type"] = r.get("acd_division_type") or ""
        row["misaligned_chromosomes"] = _flag(r.get("misaligned_chromosomes"))
        row["lagging_chromosome"] = _flag(r.get("lagging_chromosome"))
        row["anaphase_bridge"] = _flag(r.get("anaphase_bridge"))
        row["micronucleus"] = _flag(r.get("micronucleus"))
        row["anomaly_notes"] = r.get("anomaly_notes") or ""
        updated += 1

    out_path = args.events.parent / (args.events.stem + "_classified.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nClassified {updated} events. Written to {out_path}")


if __name__ == "__main__":
    main()
