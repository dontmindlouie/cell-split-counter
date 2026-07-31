"""Tests for src/lineage.py (per-frame centroid lookup) and
scripts/reports/trace_lineage_node.py (topology walk + crop generation).

Coverage goals
--------------
1. per_frame_centroids  — basic correctness, gap-bridged track stitching, missing-frame
                          reporting, default lifetime-bounded range
2. track_lifetime       — normal + nonexistent-track cases
3. trace()              — split/death/unresolved outcome classification, parent/child
                          linkage, crop file generation, from a small fabricated events.csv

Fixture style matches tests/test_track_memmap.py -- small, hand-built (T,H,W) label
arrays and real tmp_path files (not mocks), so the real np.memmap/cv2 code paths run.

A separate real-data regression check (test_real_m4_data.py) cross-checks this module
against actual pipeline output and is skipped when that data isn't present on disk --
see that file's docstring for why it isn't here.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.lineage import per_frame_centroids, track_lifetime
from scripts.reports.trace_lineage_node import trace


def _write_run_dir(tmp_path: Path, video: np.ndarray) -> Path:
    """Write a minimal real run_dir: frames/frame_NNNNN_*.png (content irrelevant,
    just needs valid HxW) + frames/_memmap/tracked_masks.dat matching `video`."""
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(parents=True)
    T, H, W = video.shape
    for t in range(T):
        cv2.imwrite(str(frame_dir / f"frame_{t:05d}_raw{t:05d}.png"), np.full((H, W), 128, dtype=np.uint8))

    memmap_dir = frame_dir / "_memmap"
    memmap_dir.mkdir()
    masks = np.memmap(memmap_dir / "tracked_masks.dat", dtype=np.uint16, mode="w+", shape=(T, H, W))
    masks[:] = video
    masks.flush()
    return run_dir


def _labeled_video(T, H, W, placements):
    """(T,H,W) uint16 label array from (frame, label, y0, x0, h, w) tuples."""
    video = np.zeros((T, H, W), dtype=np.uint16)
    for f, label, y0, x0, h, w in placements:
        video[f, y0:y0 + h, x0:x0 + w] = label
    return video


def _write_events_csv(run_dir: Path, rows: list[dict]) -> None:
    fieldnames = [
        "track_id", "parent_id", "peak_frame", "split_topology", "split_type",
        "centroid_x", "centroid_y", "ai_confidence", "raw_ai_confidence", "ai_notes",
        "likely_division_dropout", "classification_source",
    ]
    with open(run_dir / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ── per_frame_centroids ───────────────────────────────────────────────────────

class TestPerFrameCentroids:
    def test_basic_moving_cell(self, tmp_path):
        video = _labeled_video(3, 30, 30, [
            (0, 5, 5, 5, 4, 4),
            (1, 5, 6, 6, 4, 4),
            (2, 5, 7, 7, 4, 4),
        ])
        run_dir = _write_run_dir(tmp_path, video)
        traj = per_frame_centroids(run_dir, track_id=5)
        assert set(traj.keys()) == {0, 1, 2}
        cx0, cy0 = traj[0]
        assert abs(cx0 - 6.5) < 0.1 and abs(cy0 - 6.5) < 0.1  # centre of 5:9,5:9
        cx2, cy2 = traj[2]
        assert abs(cx2 - 8.5) < 0.1 and abs(cy2 - 8.5) < 0.1  # centre of 7:11,7:11

    def test_nonexistent_track_returns_empty(self, tmp_path):
        video = _labeled_video(2, 20, 20, [(0, 1, 5, 5, 4, 4), (1, 1, 5, 5, 4, 4)])
        run_dir = _write_run_dir(tmp_path, video)
        assert per_frame_centroids(run_dir, track_id=999) == {}

    def test_gap_bridged_track_stitches_with_reported_gap(self, tmp_path):
        """Label 7 lives frames 0-2, vanishes at frame 3, resumes as a brand new raw
        label 9 at frame 4 nearby -- _bridge_track_gaps (imported, not reimplemented)
        should merge these into one canonical id, and per_frame_centroids should
        return both segments' real centroids with frame 3 simply absent."""
        video = _labeled_video(5, 40, 40, [
            (0, 7, 10, 10, 4, 4),
            (1, 7, 10, 10, 4, 4),
            (2, 7, 10, 10, 4, 4),
            # frame 3: nothing (the gap)
            (4, 9, 11, 11, 4, 4),  # resumes nearby under a new raw label
        ])
        run_dir = _write_run_dir(tmp_path, video)
        traj = per_frame_centroids(run_dir, track_id=7)
        assert set(traj.keys()) == {0, 1, 2, 4}
        assert 3 not in traj
        # frame 4's centroid should reflect label 9's actual (shifted) position
        cx4, cy4 = traj[4]
        assert abs(cx4 - 12.5) < 0.1 and abs(cy4 - 12.5) < 0.1

    def test_explicit_frame_range_overrides_lifetime(self, tmp_path):
        video = _labeled_video(4, 20, 20, [
            (0, 1, 2, 2, 3, 3), (1, 1, 2, 2, 3, 3), (2, 1, 2, 2, 3, 3), (3, 1, 2, 2, 3, 3),
        ])
        run_dir = _write_run_dir(tmp_path, video)
        traj = per_frame_centroids(run_dir, track_id=1, frame_lo=1, frame_hi=2)
        assert set(traj.keys()) == {1, 2}


# ── track_lifetime ────────────────────────────────────────────────────────────

class TestTrackLifetime:
    def test_normal_track(self, tmp_path):
        video = _labeled_video(5, 20, 20, [(1, 3, 2, 2, 3, 3), (2, 3, 2, 2, 3, 3), (3, 3, 2, 2, 3, 3)])
        run_dir = _write_run_dir(tmp_path, video)
        assert track_lifetime(run_dir, 3) == (1, 3)

    def test_nonexistent_track_returns_none(self, tmp_path):
        video = _labeled_video(2, 20, 20, [(0, 1, 2, 2, 3, 3), (1, 1, 2, 2, 3, 3)])
        run_dir = _write_run_dir(tmp_path, video)
        assert track_lifetime(run_dir, 42) is None


# ── trace() topology + crop generation ────────────────────────────────────────

class TestTrace:
    def _build_lineage_fixture(self, tmp_path):
        """track 1 (no parent) lives frames 0-3, splits at frame 4 into 2 and 3.
        Track 2 lives 4-6 then dies (peak_frame=6, matching classify_track_ends's
        convention of the LAST frame actually seen). Track 3 lives only 4-5 with
        no further events.csv row at all -- an "unresolved" track (e.g. dropped
        below classify_track_ends's min_track_frames)."""
        video = _labeled_video(7, 40, 40, [
            (0, 1, 10, 10, 4, 4), (1, 1, 10, 10, 4, 4), (2, 1, 10, 10, 4, 4), (3, 1, 10, 10, 4, 4),
            (4, 2, 10, 10, 4, 4), (5, 2, 10, 10, 4, 4), (6, 2, 10, 10, 4, 4),
            (4, 3, 20, 20, 4, 4), (5, 3, 20, 20, 4, 4),
        ])
        run_dir = _write_run_dir(tmp_path, video)
        _write_events_csv(run_dir, [
            {"track_id": 2, "parent_id": 1, "peak_frame": 4, "split_topology": "normal_split",
             "split_type": "symmetric", "ai_confidence": 0.9, "ai_notes": "child B"},
            {"track_id": 3, "parent_id": 1, "peak_frame": 4, "split_topology": "normal_split",
             "split_type": "symmetric", "ai_confidence": 0.9, "ai_notes": "child C"},
            {"track_id": 2, "parent_id": 1, "peak_frame": 6, "split_topology": "death",
             "ai_confidence": 0.7, "ai_notes": "B died", "likely_division_dropout": "0",
             "classification_source": "claude-haiku-4-5"},
        ])
        return run_dir

    def test_split_outcome_and_children(self, tmp_path):
        run_dir = self._build_lineage_fixture(tmp_path)
        result = trace(run_dir, 1)
        assert result["found"] is True
        assert result["parent_id"] is None
        assert result["outcome"] == "split"
        assert result["birth_frame"] == 0 and result["end_frame"] == 3
        child_ids = {c["track_id"] for c in result["children"]}
        assert child_ids == {2, 3}
        assert result["death"] is None
        assert len(result["crops"]) == 4  # frames 0-3
        assert result["missing_frames"] == []

    def test_death_outcome_and_parent_link(self, tmp_path):
        run_dir = self._build_lineage_fixture(tmp_path)
        result = trace(run_dir, 2)
        assert result["parent_id"] == 1
        assert result["outcome"] == "death"
        assert result["birth_frame"] == 4 and result["end_frame"] == 6
        assert result["children"] == []
        assert result["death"]["peak_frame"] == 6
        assert result["death"]["likely_division_dropout"] is False
        assert len(result["crops"]) == 3  # frames 4-6

    def test_unresolved_outcome(self, tmp_path):
        run_dir = self._build_lineage_fixture(tmp_path)
        result = trace(run_dir, 3)
        assert result["parent_id"] == 1
        assert result["outcome"] == "unresolved"
        assert result["children"] == []
        assert result["death"] is None
        assert result["birth_frame"] == 4 and result["end_frame"] == 5

    def test_crop_files_actually_written(self, tmp_path):
        run_dir = self._build_lineage_fixture(tmp_path)
        result = trace(run_dir, 1)
        for c in result["crops"]:
            assert (run_dir / c["path"]).exists()

    def test_track_not_in_tracked_masks_reports_not_found(self, tmp_path):
        run_dir = self._build_lineage_fixture(tmp_path)
        result = trace(run_dir, 999)
        assert result["found"] is False


# --------------------------------------------------- build_lineage_from_tracks

def _tracks_frame(rows):
    """rows: (track_id, frame, cx, cy, area_um2, intensity_integrated)."""
    import pandas as pd
    return pd.DataFrame(rows, columns=[
        "track_id", "frame", "cx", "cy", "area_um2", "intensity_integrated"])


def _lin(rows):
    from src.lineage import build_lineage_from_tracks
    return build_lineage_from_tracks(_tracks_frame(rows)).set_index("track_id")


def test_geometric_lineage_links_a_clean_division():
    # Mother ends f1; two comparable daughters born f2 right where it was.
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (1, 1, 100.0, 100.0, 200.0, 1000.0),
        (2, 2, 95.0, 100.0, 100.0, 500.0),
        (3, 2, 105.0, 100.0, 100.0, 500.0),
    ])
    assert out.loc[2, "parent_id"] == 1
    assert out.loc[3, "parent_id"] == 1
    assert out.loc[1, "n_daughters"] == 2
    assert out.loc[1, "daughter_ids"] == "2 3"
    assert out.loc[2, "dna_ratio"] == 1.0
    assert out.loc[2, "size_ratio"] == 1.0


def test_geometric_lineage_scores_a_fragment_without_dropping_it():
    """The track-6425 case: a micronucleus budding beside a nucleus is geometrically
    identical to a division, so the link is still made -- but size_ratio must expose
    it, because that is the column a reader discounts the link on."""
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (1, 1, 100.0, 100.0, 200.0, 1000.0),
        (2, 2, 100.0, 100.0, 190.0, 960.0),   # the real continuation
        (3, 2, 112.0, 100.0, 10.0, 40.0),     # a micronucleus
    ])
    assert out.loc[3, "parent_id"] == 1, "weak links are still reported, not filtered"
    assert out.loc[3, "size_ratio"] < 0.25
    assert out.loc[3, "dna_ratio"] > 0.9, "DNA alone would NOT have caught this"


def test_geometric_lineage_assigns_a_contested_daughter_to_its_nearest_mother():
    """Two mothers ending in the same frame near each other both reach the same
    births. Without nearest-mother resolution the winner depends on row order."""
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (2, 0, 130.0, 100.0, 200.0, 1000.0),
        (3, 1, 98.0, 100.0, 100.0, 500.0),
        (4, 1, 102.0, 100.0, 100.0, 500.0),
    ])
    assert out.loc[3, "parent_id"] == 1
    assert out.loc[4, "parent_id"] == 1
    assert out.loc[2, "n_daughters"] == 0


def test_geometric_lineage_ignores_single_successor_and_distant_births():
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (2, 1, 100.0, 100.0, 200.0, 1000.0),   # lone successor = continuation
        (3, 0, 500.0, 500.0, 200.0, 1000.0),
        (4, 1, 900.0, 900.0, 100.0, 500.0),    # far away
        (5, 1, 901.0, 901.0, 100.0, 500.0),
    ])
    assert out.loc[2, "parent_id"] == ""
    assert out.loc[4, "parent_id"] == ""
    assert out.loc[5, "parent_id"] == ""


def test_geometric_lineage_covers_every_track():
    """Coverage is the whole point of replacing the events-derived graph: a track
    with no parent must still get a row, or 'missing' and 'orphan' stay confused."""
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (7, 5, 300.0, 300.0, 150.0, 800.0),
    ])
    assert set(out.index) == {1, 7}
    assert out.loc[7, "parent_id"] == ""
    assert out.loc[7, "first_frame"] == 5 and out.loc[7, "last_frame"] == 5


def test_geometric_lineage_records_the_runner_up_mother():
    """Nearest-mother is a tie-break, not a fact -- the loser must ship alongside
    the winner so a reader can see the parent_id was a judgement call."""
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (2, 0, 118.0, 100.0, 200.0, 1000.0),
        (3, 1, 98.0, 100.0, 100.0, 500.0),
        (4, 1, 102.0, 100.0, 100.0, 500.0),
    ])
    assert out.loc[3, "parent_id"] == 1
    assert out.loc[3, "alt_parents"].startswith("2:")
    assert out.loc[4, "alt_parents"].startswith("2:")


def test_geometric_lineage_leaves_alt_parents_empty_when_unambiguous():
    out = _lin([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (2, 1, 98.0, 100.0, 100.0, 500.0),
        (3, 1, 102.0, 100.0, 100.0, 500.0),
    ])
    assert out.loc[2, "alt_parents"] == ""
    assert out.loc[3, "alt_parents"] == ""


# ------------------------------------------------------- score_lineage_links

def test_score_lineage_links_scores_a_graph_it_did_not_build():
    """Trackastra's CTC graph is topology with no scores, and it makes the same
    micronucleus mistake geometry does -- so scoring has to work on any source."""
    import pandas as pd

    from src.lineage import score_lineage_links
    tracks = _tracks_frame([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (1, 1, 100.0, 100.0, 200.0, 1000.0),
        (2, 2, 100.0, 100.0, 190.0, 960.0),
        (3, 2, 112.0, 100.0, 10.0, 40.0),
    ])
    ctc = pd.DataFrame([
        {"track_id": 1, "parent_id": "", "daughter_ids": "2 3"},
        {"track_id": 2, "parent_id": 1, "daughter_ids": ""},
        {"track_id": 3, "parent_id": 1, "daughter_ids": ""},
    ])
    out = score_lineage_links(ctc, tracks).set_index("track_id")
    assert out.loc[3, "size_ratio"] < 0.25, "fragment must be flagged whatever built the graph"
    assert out.loc[3, "dna_ratio"] > 0.9
    assert out.loc[1, "size_ratio"] == "", "a mother carries no link score of its own"


def test_score_lineage_links_survives_a_multi_frame_gap():
    """Unlike the geometric builder, this must not assume the daughter starts at
    the mother's last frame + 1 -- CTC links across gaps that geometry cannot."""
    import pandas as pd

    from src.lineage import score_lineage_links
    tracks = _tracks_frame([
        (1, 0, 100.0, 100.0, 200.0, 1000.0),
        (2, 5, 98.0, 100.0, 100.0, 500.0),   # four-frame gap
        (3, 5, 102.0, 100.0, 100.0, 500.0),
    ])
    ctc = pd.DataFrame([{"track_id": 1, "parent_id": "", "daughter_ids": "2 3"},
                        {"track_id": 2, "parent_id": 1, "daughter_ids": ""},
                        {"track_id": 3, "parent_id": 1, "daughter_ids": ""}])
    out = score_lineage_links(ctc, tracks).set_index("track_id")
    assert out.loc[2, "dna_ratio"] == 1.0
    assert out.loc[2, "size_ratio"] == 1.0


def test_score_lineage_links_normalises_float_parent_ids():
    """pandas reads an int column with blanks as float, so parent_id round-trips as
    '127.0' and every downstream int() raises."""
    import pandas as pd

    from src.lineage import score_lineage_links
    tracks = _tracks_frame([(1, 0, 10.0, 10.0, 5.0, 5.0), (2, 1, 10.0, 10.0, 5.0, 5.0)])
    ctc = pd.DataFrame([{"track_id": 1, "parent_id": float("nan"), "daughter_ids": ""},
                        {"track_id": 2, "parent_id": 1.0, "daughter_ids": ""}])
    out = score_lineage_links(ctc, tracks).set_index("track_id")
    assert out.loc[2, "parent_id"] == "1"
    assert out.loc[1, "parent_id"] == ""
