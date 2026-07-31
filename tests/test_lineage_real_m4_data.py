"""Real-data regression check for src/lineage.py -- cross-checks per_frame_centroids()
against events.csv's own stored centroids on the actual M4 pipeline output, using
classify.py's documented frame/track offset (split events store the PARENT's
centroid one frame before peak_frame; death events store the track's own centroid
AT peak_frame -- see LineageEvent.centroid's docstring in src/classify.py).

Not in test_lineage.py: this depends on ~1.8GB of real pipeline output
(frames/_memmap/tracked_masks.dat) that lives under the gitignored data/ directory,
not something a fresh clone or CI has -- kept as a separate, skippable file rather
than baking a hard dependency into the main suite. Scoped to peak_frame < 200 (the
same source video the project's eval_harness "validation_200frame" sweep used,
though that sweep's own output directory doesn't retain the raw frames/tracked
masks needed here -- this re-derives an equivalent 200-frame-scoped check directly
from the one M4 run that still has them on disk) so it stays fast (~1s once the
canonical-label cache is warm) rather than re-scanning the full 848-frame video.

Run manually / locally where the data exists:
    pytest tests/test_lineage_real_m4_data.py -v
"""

import csv
from pathlib import Path

import pytest

from src.lineage import per_frame_centroids

_M4_RUN_DIR = Path(__file__).resolve().parents[1] / "data" / "output" / "202660629_Bewop920x_M4"
_MAX_PEAK_FRAME = 200

pytestmark = pytest.mark.skipif(
    not (_M4_RUN_DIR / "frames" / "_memmap" / "tracked_masks.dat").exists(),
    reason="real M4 pipeline output (frames/_memmap/tracked_masks.dat) not present on disk",
)


def _load_early_rows() -> list[dict]:
    rows = list(csv.DictReader(open(_M4_RUN_DIR / "events.csv", encoding="utf-8", errors="replace")))
    return [
        r for r in rows
        if r.get("split_topology") in ("normal_split", "multi_way_split", "failed_split", "death")
        and r.get("centroid_x") and r.get("centroid_y")
        and int(r["peak_frame"]) < _MAX_PEAK_FRAME
    ]


def test_early_frame_events_exist():
    """Sanity check the fixture itself has real events to test against."""
    assert len(_load_early_rows()) > 10


def test_per_frame_centroids_matches_events_csv_in_first_200_frames():
    """Geometry must match for every track id that still resolves.

    Deliberately NOT "every row resolves". events.csv is detector-era and was written
    under the pre-03df4b4 canonicalization, and that fix intentionally moves some
    canonical ids -- so a handful of its track_ids no longer name anything, which is
    the fix working rather than a regression. What must not change is where a cell
    WAS: any id that still resolves has to land on the same pixel.

    This distinction was invisible until 2026-07-31, because the canonical-label cache
    was keyed on tracked_masks.dat's file size alone and a re-track leaves that size
    identical. The stale cache pinned pre-fix ids, so this test kept passing against
    superseded tracking and would have gone on doing so indefinitely.

    The floor on resolvable rows is what stops the relaxation from hollowing the test
    out: if a future change strands most ids, that is a regression, not a fix.
    """
    rows = _load_early_rows()
    unresolved = []
    mismatches = []
    for r in rows:
        track_id = int(r["track_id"])
        peak_frame = int(r["peak_frame"])
        stored_cx, stored_cy = float(r["centroid_x"]), float(r["centroid_y"])

        is_split = r.get("split_topology") in ("normal_split", "multi_way_split", "failed_split")
        lookup_track_id = int(r["parent_id"]) if is_split else track_id
        lookup_frame = peak_frame - 1 if is_split else peak_frame

        traj = per_frame_centroids(_M4_RUN_DIR, lookup_track_id, frame_lo=lookup_frame, frame_hi=lookup_frame)
        if lookup_frame not in traj:
            unresolved.append((track_id, lookup_frame))
            continue
        got_cx, got_cy = traj[lookup_frame]
        dist = ((got_cx - stored_cx) ** 2 + (got_cy - stored_cy) ** 2) ** 0.5
        if dist >= 1.0:
            mismatches.append((track_id, lookup_frame, f"dist={dist:.2f}"))

    resolved = len(rows) - len(unresolved)
    assert not mismatches, (
        f"{len(mismatches)}/{resolved} resolvable rows moved: {mismatches[:10]}")
    assert resolved >= 0.95 * len(rows), (
        f"only {resolved}/{len(rows)} of events.csv's track ids still resolve. A few "
        f"retired ids are expected after a canonicalization change; most of them "
        f"vanishing means the mapping broke. Unresolved: {unresolved[:10]}")
