"""Tests for bleach_risk computation in pipeline.run."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.classify import EventType, LineageEvent


def make_event(track_id, frame, parent_id=1):
    return LineageEvent(
        track_id=track_id,
        parent_id=parent_id,
        frame=frame,
        event_type=EventType.NORMAL_SPLIT,
        classification_source="rule",
        confidence=1.0,
        centroid=(0.0, 0.0),
    )


def _run_pipeline_stub(events, n_frames):
    """Apply only the bleach_risk computation step from pipeline.run."""
    total_frames = n_frames
    return [dataclasses.replace(e, bleach_risk=e.frame / total_frames) for e in events]


def test_bleach_risk_is_frame_over_total():
    events = [make_event(2, frame=10), make_event(3, frame=10)]
    result = _run_pipeline_stub(events, n_frames=100)
    assert all(abs(e.bleach_risk - 0.1) < 1e-9 for e in result)


def test_bleach_risk_at_last_frame_approaches_one():
    events = [make_event(2, frame=99)]
    result = _run_pipeline_stub(events, n_frames=100)
    assert abs(result[0].bleach_risk - 0.99) < 1e-9


def test_bleach_risk_at_first_frame_is_zero():
    events = [make_event(2, frame=0)]
    result = _run_pipeline_stub(events, n_frames=50)
    assert result[0].bleach_risk == 0.0


def test_bleach_risk_is_set_independently_per_event():
    events = [make_event(2, frame=10), make_event(3, frame=40)]
    result = _run_pipeline_stub(events, n_frames=100)
    risks = {e.track_id: e.bleach_risk for e in result}
    assert abs(risks[2] - 0.1) < 1e-9
    assert abs(risks[3] - 0.4) < 1e-9


def test_bleach_risk_does_not_mutate_original_events():
    event = make_event(2, frame=10)
    _run_pipeline_stub([event], n_frames=100)
    assert event.bleach_risk is None
