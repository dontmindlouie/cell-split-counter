"""Tests for scripts/build_bundle.write_candidates -- the events.csv -> candidates.csv
transform that fixes the split row-vs-event doubling and drops the columns that were
measured to be untrustworthy.

The doubling is the reason this file exists: two people who already knew about the bug
still mis-derived it within an hour of each other, so the invariant is pinned in a test
rather than left to whoever next writes a groupby.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_bundle import write_candidates  # noqa: E402

_COLS = ["event_id", "frame_range", "peak_frame", "centroid_x", "centroid_y",
         "split_topology", "track_id", "parent_id", "ai_confidence",
         "raw_ai_confidence", "split_type", "ai_notes"]


def _run_dir(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})
    return d


def _split_pair(**over):
    """A split as events.csv wrote it: TWO rows, one per daughter, sharing everything
    that identifies the event."""
    base = dict(frame_range="10-14", peak_frame="12", centroid_x="100",
                centroid_y="200", split_topology="normal_split", parent_id="7",
                ai_confidence="0.0", raw_ai_confidence="0.81", split_type="symmetric")
    base.update(over)
    return [dict(base, event_id="0", track_id="11"), dict(base, event_id="1", track_id="12")]


def _death(**over):
    base = dict(frame_range="30-34", peak_frame="32", centroid_x="300",
                centroid_y="400", split_topology="death", track_id="20",
                ai_confidence="0.77", raw_ai_confidence="0.77")
    base.update(over)
    return [dict(base, event_id="2")]


def test_two_daughter_rows_collapse_to_one_event(tmp_path):
    out = tmp_path / "cand"
    info = write_candidates(_run_dir(tmp_path, _split_pair()), out)
    assert info["n_source_rows"] == 2
    assert info["n_events"] == 1
    rows = list(csv.DictReader(open(out / "candidates.csv", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["daughter_ids"] == "11 12"
    assert rows[0]["n_daughters"] == "2"
    # track_id was arbitrarily one of the two daughters; it must not imply a single object
    assert rows[0]["track_id"] == ""


def test_deaths_are_not_collapsed_and_keep_their_track_id(tmp_path):
    out = tmp_path / "cand"
    info = write_candidates(_run_dir(tmp_path, _death()), out)
    assert info["n_events"] == 1
    rows = list(csv.DictReader(open(out / "candidates.csv", encoding="utf-8")))
    assert rows[0]["track_id"] == "20"
    assert rows[0]["n_daughters"] == "0"
    assert rows[0]["daughter_ids"] == ""


def test_the_ratio_only_the_splits_were_doubled(tmp_path):
    """The bug's whole sting: doubling ONE category does not cancel. Two splits and
    two deaths were written as 4+2=6 rows, reading 2.0 splits per death instead of 1.0."""
    rows = (_split_pair(peak_frame="12", frame_range="10-14")
            + _split_pair(peak_frame="50", frame_range="48-52", centroid_x="500")
            + _death(peak_frame="32") + _death(peak_frame="70", frame_range="68-72"))
    out = tmp_path / "cand"
    info = write_candidates(_run_dir(tmp_path, rows), out)
    assert info["n_source_rows"] == 6
    assert info["by_kind"] == {"normal_split": 2, "death": 2}
    got = list(csv.DictReader(open(out / "candidates.csv", encoding="utf-8")))
    n_sp = sum(1 for r in got if r["split_topology"] != "death")
    n_de = sum(1 for r in got if r["split_topology"] == "death")
    assert n_sp / n_de == 1.0, "row counting would have said 2.0"


def test_untrustworthy_columns_are_dropped_and_raw_confidence_survives(tmp_path):
    out = tmp_path / "cand"
    write_candidates(_run_dir(tmp_path, _split_pair()), out)
    cols = list(csv.DictReader(open(out / "candidates.csv", encoding="utf-8")).fieldnames)
    assert "ai_confidence" not in cols, "0.0 on 81% of split rows; filtering on it inverts the ratio"
    assert "split_type" not in cols, "~0% precision for a genuine failed division"
    assert "raw_ai_confidence" in cols, "the one confidence field that survives is kept"


def test_two_events_at_the_same_frame_stay_separate(tmp_path):
    """Dedupe must key on WHERE as well as when, or two cells dividing in the same
    frame collapse into one and the count under-reports."""
    rows = _split_pair(centroid_x="100") + _split_pair(centroid_x="900")
    out = tmp_path / "cand"
    info = write_candidates(_run_dir(tmp_path, rows), out)
    assert info["n_events"] == 2


def test_missing_events_csv_is_not_an_error(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    assert write_candidates(d, tmp_path / "cand")["n_events"] == 0
