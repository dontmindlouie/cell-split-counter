"""Tests for CSV output column shape and values."""

import csv
import io

from src.classify import EventType, LineageEvent
from src.output import write_events_csv


def make_event(track_id, frame=10, confidence=1.0, source="rule", bleach_risk=None, claude_notes=None):
    return LineageEvent(
        track_id=track_id,
        parent_id=1,
        frame=frame,
        event_type=EventType.NORMAL_SPLIT,
        classification_source=source,
        confidence=confidence,
        centroid=(50.0, 75.0),
        claude_notes=claude_notes,
        bleach_risk=bleach_risk,
    )


def _read_csv(tmp_path):
    out = tmp_path / "events.csv"
    return list(csv.DictReader(out.open()))


def test_csv_header_columns(tmp_path):
    write_events_csv([], tmp_path / "events.csv", source_video="test.avi")
    out = tmp_path / "events.csv"
    reader = csv.reader(out.open())
    header = next(reader)
    assert header == [
        "event_id", "source_video", "frame_range", "peak_frame", "split_topology",
        "track_id", "parent_id", "claude_confidence", "tracker_persistence_score", "classification_source",
        "centroid_x", "centroid_y", "claude_notes", "bleach_risk",
        "acd_division_type", "misaligned_chromosomes", "lagging_chromosome",
        "anaphase_bridge", "micronucleus", "anomaly_notes", "near_edge",
    ]


def test_csv_split_topology_uses_event_type_value(tmp_path):
    event = make_event(2)
    write_events_csv([event], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["split_topology"] == "normal_split"


def test_csv_bleach_risk_formatted_to_three_decimals(tmp_path):
    event = make_event(2, bleach_risk=0.123456)
    write_events_csv([event], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["bleach_risk"] == "0.123"


def test_csv_bleach_risk_empty_when_none(tmp_path):
    event = make_event(2, bleach_risk=None)
    write_events_csv([event], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["bleach_risk"] == ""


def test_csv_claude_notes_written_and_empty_when_none(tmp_path):
    e1 = make_event(2, claude_notes="asymmetric: clear split")
    e2 = make_event(3, claude_notes=None)
    write_events_csv([e1, e2], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["claude_notes"] == "asymmetric: clear split"
    assert rows[1]["claude_notes"] == ""


def test_csv_one_row_per_event(tmp_path):
    events = [make_event(i) for i in range(5)]
    write_events_csv(events, tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert len(rows) == 5
    assert [r["event_id"] for r in rows] == ["0", "1", "2", "3", "4"]


def test_csv_acd_columns_written(tmp_path):
    from src.classify import LineageEvent
    event = LineageEvent(
        track_id=2, parent_id=1, frame=10,
        event_type=EventType.NORMAL_SPLIT, classification_source="claude",
        confidence=0.9, centroid=(50.0, 75.0),
        acd_division_type="bipolar",
        misaligned_chromosomes=False,
        lagging_chromosome=True,
        anaphase_bridge=False,
        micronucleus=True,
    )
    write_events_csv([event], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["acd_division_type"] == "bipolar"
    assert rows[0]["lagging_chromosome"] == "1"
    assert rows[0]["micronucleus"] == "1"
    assert rows[0]["misaligned_chromosomes"] == "0"
    assert rows[0]["anaphase_bridge"] == "0"


def test_csv_acd_columns_empty_when_not_classified(tmp_path):
    event = make_event(2)  # no ACD fields set
    write_events_csv([event], tmp_path / "events.csv", source_video="v.avi")
    rows = _read_csv(tmp_path)
    assert rows[0]["acd_division_type"] == ""
    assert rows[0]["lagging_chromosome"] == ""
    assert rows[0]["micronucleus"] == ""
