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
    # Explicit utf-8: Claude's text can contain non-ASCII characters (en-dashes, arrows).
    # Without this, open() falls back to the OS default codepage on Windows (cp1252),
    # which either mangles those characters or, if this write itself throws, is a second
    # place the encoding bug from main.py could resurface (that fix only covers stdout).
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "source_video", "frame_range", "peak_frame",
            "centroid_x", "centroid_y", "near_edge", "cell_area_px", "cell_size_um2",
            "neighbor_distance_px", "neighbor_area_px", "eccentricity", "solidity",
            "split_topology", "split_type", "track_id", "parent_id", "classification_source",
            "ai_confidence", "raw_ai_confidence", "tracker_persistence_score",
            "ai_notes", "review_error", "bleach_risk",
            "acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
            "anaphase_bridge", "micronucleus", "binucleation", "anomaly_notes",
            "likely_division_dropout",
        ])
        for i, e in enumerate(events):
            range_start = max(e.frame - frame_range_lookback, 0)
            cx, cy = (e.centroid[0], e.centroid[1]) if e.centroid else ("", "")
            bleach = f"{e.bleach_risk:.3f}" if e.bleach_risk is not None else ""
            tracker_conf = f"{e.tracker_confidence:.4f}" if e.tracker_confidence is not None else ""
            area_px = f"{e.cell_area_px:.1f}" if e.cell_area_px is not None else ""
            size_um2 = f"{e.cell_size_um2:.2f}" if e.cell_size_um2 is not None else ""
            neighbor_dist = f"{e.neighbor_distance_px:.1f}" if e.neighbor_distance_px is not None else ""
            neighbor_area = f"{e.neighbor_area_px:.1f}" if e.neighbor_area_px is not None else ""
            eccentricity = f"{e.eccentricity:.4f}" if e.eccentricity is not None else ""
            solidity = f"{e.solidity:.4f}" if e.solidity is not None else ""
            def _flag(v): return "" if v is None else ("1" if v else "0")
            writer.writerow([
                i, source_video, f"{range_start}-{e.frame}", e.frame,
                cx, cy, _flag(e.near_edge), area_px, size_um2, neighbor_dist, neighbor_area,
                eccentricity, solidity,
                e.event_type.value, e.split_type or "", e.track_id, e.parent_id, e.classification_source,
                e.confidence,
                f"{e.raw_ai_confidence:.4f}" if e.raw_ai_confidence is not None else "",
                tracker_conf,
                e.ai_notes or "", _flag(e.review_error) if e.review_error else "0", bleach,
                e.acd_division_type or "", _flag(e.misaligned_chromosomes),
                _flag(e.lagging_chromosome), _flag(e.anaphase_bridge), _flag(e.micronucleus),
                _flag(e.binucleation), e.anomaly_notes or "",
                _flag(e.likely_division_dropout),
            ])


def write_summary_json(
    events: list[LineageEvent], video_metadata: dict, out_path: Path,
    vision_usage: dict | None = None, timing_sec: dict | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(e.event_type.value for e in events)
    summary = {
        **video_metadata,
        "total_events": len(events),
        "event_counts": dict(counts),
    }
    if vision_usage is not None:
        summary["vision_usage"] = vision_usage
    if timing_sec is not None:
        summary["timing_sec"] = timing_sec
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
