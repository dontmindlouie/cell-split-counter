"""Write lineage events and summary metadata to CSV/JSON."""

import csv
import json
from collections import Counter
from pathlib import Path

from src.classify import LineageEvent


def write_events_csv(events: list[LineageEvent], out_path: Path, source_video: str, frame_range_lookback: int = 10) -> None:
    """Write one row per detected split event.

    frame_range is approximated as [peak_frame - frame_range_lookback, peak_frame] since
    v1 only detects the frame a split is observed (peak_frame), not the metaphase-anaphase
    window the ground-truth sheet records -- that needs per-frame morphology classification,
    out of scope for v1. See docs/output_schema.md for full column definitions.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "source_video", "frame_range", "peak_frame", "split_topology",
            "track_id", "parent_id", "confidence", "tracker_confidence", "classification_source",
            "centroid_x", "centroid_y", "claude_notes", "bleach_risk",
            "acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
            "anaphase_bridge", "micronucleus", "anomaly_notes",
        ])
        for i, e in enumerate(events):
            range_start = max(e.frame - frame_range_lookback, 0)
            cx, cy = (e.centroid[0], e.centroid[1]) if e.centroid else ("", "")
            bleach = f"{e.bleach_risk:.3f}" if e.bleach_risk is not None else ""
            tracker_conf = f"{e.tracker_confidence:.4f}" if e.tracker_confidence is not None else ""
            def _flag(v): return "" if v is None else ("1" if v else "0")
            writer.writerow([
                i, source_video, f"{range_start}-{e.frame}", e.frame, e.event_type.value,
                e.track_id, e.parent_id, e.confidence, tracker_conf, e.classification_source,
                cx, cy, e.claude_notes or "", bleach,
                e.acd_division_type or "", _flag(e.misaligned_chromosomes),
                _flag(e.lagging_chromosome), _flag(e.anaphase_bridge), _flag(e.micronucleus),
                e.anomaly_notes or "",
            ])


def write_summary_json(events: list[LineageEvent], video_metadata: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(e.event_type.value for e in events)
    summary = {
        **video_metadata,
        "total_events": len(events),
        "event_counts": dict(counts),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
