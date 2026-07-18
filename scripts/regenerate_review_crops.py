"""Regenerate review_crops/ for an existing run using the fixed track_id-keyed folder
naming (2026-07-12) -- no API calls, no re-review, just re-crops from cached frame PNGs
and rewrites verdict.txt from events.csv's already-stored verdict data.

Why this exists: _save_debug_crops used to key folders by parent_id, which collides
constantly between splits and deaths (parent_id means "dividing track" for a split but
"distant birth-ancestor, often None" for a death) -- found via the 2026-07-12
researcher_browser.py death-support work (211 split/death folder collisions in this run
alone). This rebuilds review_crops/ from scratch with the corrected track-id keying so
existing runs aren't stuck with silently-wrong/overwritten crops.

Usage:
    python scripts/regenerate_review_crops.py data/output/<run_dir>

Not committed -- throwaway script, matches the pattern of
scripts/regenerate_crops_from_csv.py (2026-07-12 neighbor-fix validation session).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.classify import LineageEvent
from src.review import _build_dense_debug_window, _save_debug_crops, _write_verdict_txt


def _event_from_row(r: dict) -> LineageEvent:
    cx = float(r["centroid_x"]) if r.get("centroid_x") else None
    cy = float(r["centroid_y"]) if r.get("centroid_y") else None
    return LineageEvent(
        track_id=int(r["track_id"]),
        parent_id=int(r["parent_id"]) if r.get("parent_id") else None,
        frame=int(r["peak_frame"]),
        event_type=None,
        classification_source=r.get("classification_source", ""),
        confidence=float(r["ai_confidence"]) if r.get("ai_confidence") else 0.0,
        centroid=(cx, cy) if cx is not None and cy is not None else None,
    )


def _split_verdict(r: dict) -> dict:
    conf = r.get("ai_confidence")
    verdict = "real" if conf and float(conf) > 0 else "false_positive"
    return {
        "verdict": verdict,
        "confidence": float(r["raw_ai_confidence"]) if r.get("raw_ai_confidence") else (float(conf) if conf else 0.0),
        "description": r.get("ai_notes") or "",
    }


def _death_verdict(r: dict) -> dict:
    dropout = r.get("likely_division_dropout")
    dropout_bool = None if dropout in (None, "") else dropout == "1"
    return {
        "likely_division_dropout": dropout_bool,
        "confidence": float(r["ai_confidence"]) if r.get("ai_confidence") else 0.0,
        "description": r.get("ai_notes") or "",
        "anomaly_notes": r.get("anomaly_notes") or None,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/regenerate_review_crops.py <run_dir>")
    run_dir = Path(sys.argv[1])
    frame_dir = run_dir / "frames"
    debug_dir = run_dir / "review_crops"

    rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))

    splits = [r for r in rows if r.get("split_topology") in ("normal_split", "multi_way_split", "failed_split")]
    by_split: dict[tuple, dict] = {}
    for r in splits:
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        by_split.setdefault(key, r)  # one representative row per split point, matches review_ambiguous

    deaths = [r for r in rows if r.get("split_topology") == "death"]

    debug_dir.mkdir(parents=True, exist_ok=True)
    n_splits = 0
    for row in by_split.values():
        event = _event_from_row(row)
        dense_paths = _build_dense_debug_window(event, frame_dir)
        event_debug_dir = _save_debug_crops(dense_paths, event, debug_dir)
        _write_verdict_txt(event_debug_dir, _split_verdict(row))
        n_splits += 1

    n_deaths = 0
    for row in deaths:
        event = _event_from_row(row)
        dense_paths = _build_dense_debug_window(event, frame_dir)
        event_debug_dir = _save_debug_crops(dense_paths, event, debug_dir)
        _write_verdict_txt(event_debug_dir, _death_verdict(row))
        n_deaths += 1

    print(f"regenerated {n_splits} split folders + {n_deaths} death folders into {debug_dir}")


if __name__ == "__main__":
    main()
