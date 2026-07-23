"""Trace one track's full lifetime -- birth, real per-frame-tracked crops across its
whole observed life (not the AI review's fixed +/-24 frame window), and its outcome
(who it divided into, or how it died) -- for a researcher's Claude Code session to
walk a lineage one hop at a time. 2026-07-18.

Why "one hop": a dividing population can branch for many generations. Rather than
eagerly rendering crops for an entire descendant tree up front (expensive, mostly
wasted on branches nobody looks at), this reports ONE node's own story plus its
children's/parent's track_id -- Claude decides which branch is worth a closer look,
then calls this again with that track_id. See
[[project_cell_split_counter_lineage_tracking]].

Crops are cropped at this track's OWN real per-frame centroid (src/lineage.py's
per_frame_centroids(), backed by the cached Trackastra tracked_masks.dat) rather
than one fixed point -- necessary here since a track's life can span 50-200+
frames, far beyond the ~24-frame span where a fixed centroid is usually tolerable.

No padding past birth/end: a track_id's raw labels in tracked_masks.dat only exist
while that track is actually alive, so asking for frames before its birth or after
its split/death finds nothing there (verified against real data -- the requested
padding was silently absorbed with no extra frames returned). Getting that
context IS the point of the "hop" design: call this again with `parent_id` for
what came before, or a `children[i].track_id` for what came after.

Usage:
    python scripts/reports/trace_lineage_node.py <run_dir> --track-id 1234

Writes crops to review_crops/lineage_track_<id>/ (separate from the AI-review
crop folders, never overwrites them) and prints a JSON manifest to stdout.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.lineage import per_frame_centroids, track_lifetime
from src.review import _crop_image, _find_frame

_SPLIT_TOPOLOGIES = ("normal_split", "multi_way_split", "failed_split")


def _conf_col(row: dict) -> str:
    return "ai_confidence" if "ai_confidence" in row else "claude_confidence"


def _notes_col(row: dict) -> str:
    return "ai_notes" if "ai_notes" in row else "claude_notes"


def _describe_children(rows: list[dict], track_id: int) -> list[dict]:
    children = []
    for r in rows:
        if r.get("split_topology") in _SPLIT_TOPOLOGIES and r.get("parent_id") == str(track_id):
            children.append({
                "track_id": int(r["track_id"]),
                "peak_frame": int(r["peak_frame"]),
                "split_type": r.get("split_type") or None,
                "confidence": float(r.get(_conf_col(r)) or 0),
                "ai_notes": r.get(_notes_col(r)) or "",
            })
    return children


def _describe_death(rows: list[dict], track_id: int) -> dict | None:
    for r in rows:
        if r.get("split_topology") == "death" and r.get("track_id") == str(track_id):
            return {
                "peak_frame": int(r["peak_frame"]),
                "confidence": float(r.get(_conf_col(r)) or 0),
                "ai_notes": r.get(_notes_col(r)) or "",
                "likely_division_dropout": r.get("likely_division_dropout") == "1",
                "classification_source": r.get("classification_source") or "rule",
            }
    return None


def _own_birth_row(rows: list[dict], track_id: int) -> dict | None:
    for r in rows:
        if r.get("split_topology") in _SPLIT_TOPOLOGIES and r.get("track_id") == str(track_id):
            return r
    return None


def trace(run_dir: Path, track_id: int) -> dict:
    rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))

    birth_row = _own_birth_row(rows, track_id)
    parent_id = int(birth_row["parent_id"]) if birth_row and birth_row.get("parent_id") else None
    children = _describe_children(rows, track_id)
    death = _describe_death(rows, track_id)

    lifetime = track_lifetime(run_dir, track_id)
    if lifetime is None:
        return {"track_id": track_id, "found": False,
                "error": "track_id never appears in tracked_masks.dat for this run"}
    birth_frame, end_frame = lifetime

    if children and death:
        outcome = "split_and_death_conflict"  # surface, don't silently pick one -- shouldn't happen
    elif children:
        outcome = "split"
    elif death:
        outcome = "death"
    else:
        outcome = "unresolved"  # alive at video end, or below classify_track_ends's min_track_frames

    centroids = per_frame_centroids(run_dir, track_id, frame_lo=birth_frame, frame_hi=end_frame)

    crop_dir = run_dir / "review_crops" / f"lineage_track_{track_id}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = run_dir / "frames"
    crops = []
    for pos, frame_idx in enumerate(sorted(centroids)):
        src_path = _find_frame(frame_dir, frame_idx)
        if src_path is None:
            continue
        crop = _crop_image(src_path, centroids[frame_idx])
        out_path = crop_dir / f"{pos:03d}_frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), crop)
        crops.append({"frame": frame_idx, "path": str(out_path.relative_to(run_dir))})

    requested_frames = set(range(birth_frame, end_frame + 1))
    missing_frames = sorted(requested_frames - set(centroids))

    return {
        "track_id": track_id,
        "found": True,
        "parent_id": parent_id,
        "birth_frame": birth_frame,
        "end_frame": end_frame,
        "outcome": outcome,
        "children": children,
        "death": death,
        "crop_dir": str(crop_dir.relative_to(run_dir)),
        "crops": crops,
        "missing_frames": missing_frames,  # requested but undetected (tracking gap) -- see
                                            # src.track._bridge_track_gaps's docstring
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--track-id", required=True, type=int)
    args = parser.parse_args()

    result = trace(Path(args.run_dir), args.track_id)
    print(json.dumps(result, indent=2))
    if not result.get("found"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
