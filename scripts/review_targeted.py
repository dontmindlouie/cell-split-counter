"""Run Claude vision review on unreviewed mid-confidence events in an existing events.csv.

Reads data/output/events.csv, finds rule-classified events in [min_conf, max_conf),
calls Claude once per unique split point, and updates the CSV in place.

Usage:
  python scripts/review_targeted.py [--max-reviews N] [--min-conf F] [--max-conf F]
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv; load_dotenv()

import anthropic

from src.classify import EventType, LineageEvent
from src.config import CLAUDE_MODEL as MODEL, EVENTS_CSV, FRAME_DIR, HIGH_CONFIDENCE
from src.review import _review_and_classify


def _row_to_event(row: dict) -> LineageEvent:
    cx = float(row["centroid_x"]) if row.get("centroid_x") else None
    cy = float(row["centroid_y"]) if row.get("centroid_y") else None
    return LineageEvent(
        track_id=int(row["track_id"]),
        parent_id=int(row["parent_id"]) if row.get("parent_id") else None,
        frame=int(row["peak_frame"]),
        event_type=EventType(row.get("split_topology") or row.get("division_type") or "normal_split"),
        classification_source=row["classification_source"],
        confidence=float(row["claude_confidence"]),
        centroid=(cx, cy) if cx is not None and cy is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reviews", type=int, default=200)
    parser.add_argument("--min-conf", type=float, default=HIGH_CONFIDENCE)
    parser.add_argument("--max-conf", type=float, default=1.0)
    parser.add_argument("--debug-crops", action="store_true", help="save review crops to data/debug/crops/")
    args = parser.parse_args()

    with open(EVENTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    targets = [
        r for r in rows
        if r["classification_source"] == "rule"
        and args.min_conf <= float(r["claude_confidence"]) < args.max_conf
    ]

    # Deduplicate by (parent_id, peak_frame) — daughters share one API call
    seen: set[tuple] = set()
    unique_targets: list[dict] = []
    for r in targets:
        key = (r["parent_id"], r["peak_frame"])
        if key not in seen:
            seen.add(key)
            unique_targets.append(r)

    total = min(len(unique_targets), args.max_reviews)
    print(f"Targeting {total} unique splits for Claude review (conf {args.min_conf}–{args.max_conf})")

    debug_dir: Path | None = None
    if args.debug_crops:
        debug_dir = FRAME_DIR.parent / "debug" / "crops"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True)
        print(f"Saving crops to {debug_dir}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    results: dict[tuple, dict] = {}
    real_count = fp_count = 0
    for row in unique_targets[:args.max_reviews]:
        key = (row["parent_id"], row["peak_frame"])
        event = _row_to_event(row)
        try:
            result = _review_and_classify(client, event, FRAME_DIR, MODEL, debug_dir=debug_dir)
        except Exception as exc:
            print(f"  frame={event.frame:3d} parent={event.parent_id} [ERROR] {exc}")
            continue
        results[key] = result
        verdict = result.get("verdict", "real")
        tag = "real" if verdict == "real" else "false_positive"
        print(f"  frame={event.frame:3d} parent={event.parent_id} [{tag}] {result.get('description', '')[:100]}")
        if verdict == "real":
            real_count += 1
        else:
            fp_count += 1

    print(f"\nReviewed {len(results)}: {real_count} real, {fp_count} false positive")

    # Ensure classifier columns exist
    for col in ("acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
                "anaphase_bridge", "micronucleus", "anomaly_notes"):
        if col not in fieldnames:
            fieldnames.append(col)
    for row in rows:
        for col in ("acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
                    "anaphase_bridge", "micronucleus", "anomaly_notes"):
            row.setdefault(col, "")

    def _flag(v):
        return "" if v is None else ("1" if v else "0")

    # Update matching rows in CSV
    updated = 0
    for row in rows:
        key = (row["parent_id"], row["peak_frame"])
        if key in results:
            r = results[key]
            verdict = r.get("verdict", "real")
            confidence = float(r.get("confidence", 0.0))
            is_real = verdict == "real"
            split_type = r.get("split_type") or ""
            description = r.get("description", "")
            notes = f"{split_type}: {description}".strip(": ") if split_type else description
            # preserve original tracker score before overwriting confidence
            if not row.get("tracker_persistence_score"):
                row["tracker_persistence_score"] = row["claude_confidence"]
            row["classification_source"] = "claude"
            row["claude_confidence"] = str(confidence if is_real else 0.0)
            row["claude_notes"] = notes if is_real else ""
            row["acd_division_type"] = (r.get("acd_division_type") or "") if is_real else ""
            row["misaligned_chromosomes"] = _flag(r.get("misaligned_chromosomes")) if is_real else ""
            row["lagging_chromosome"] = _flag(r.get("lagging_chromosome")) if is_real else ""
            row["anaphase_bridge"] = _flag(r.get("anaphase_bridge")) if is_real else ""
            row["micronucleus"] = _flag(r.get("micronucleus")) if is_real else ""
            row["anomaly_notes"] = (r.get("anomaly_notes") or "") if is_real else ""
            updated += 1

    with open(EVENTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} rows in {EVENTS_CSV}")


if __name__ == "__main__":
    main()
