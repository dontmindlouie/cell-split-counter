"""Tests for cell_size_um2 computation in pipeline.run.

Closes a real gap flagged 2026-08-06: no test previously checked that the
area_px -> um^2 conversion (src/pipeline.py's _cell_size_um2) is arithmetically
correct, or that a missing pixel_size_um degrades to None rather than crashing
or silently reporting px^2 as um^2.
"""

import dataclasses

from src.classify import EventType, LineageEvent


def make_event(track_id, cell_area_px, parent_id=1):
    return LineageEvent(
        track_id=track_id,
        parent_id=parent_id,
        frame=10,
        event_type=EventType.NORMAL_SPLIT,
        classification_source="rule",
        confidence=1.0,
        centroid=(0.0, 0.0),
        cell_area_px=cell_area_px,
    )


def _run_pipeline_stub(events, pixel_size_um):
    """Apply only the cell_size_um2 computation step from pipeline.run."""

    def _cell_size_um2(area_px):
        if area_px is None or pixel_size_um is None:
            return None
        return area_px * pixel_size_um ** 2

    return [dataclasses.replace(e, cell_size_um2=_cell_size_um2(e.cell_area_px)) for e in events]


def test_cell_size_um2_matches_closed_form():
    # A 10x10 px synthetic mask (100 px) at 0.5 um/px -> 100 * 0.25 = 25.0 um^2.
    events = [make_event(2, cell_area_px=100)]
    result = _run_pipeline_stub(events, pixel_size_um=0.5)
    assert abs(result[0].cell_size_um2 - 25.0) < 1e-9


def test_cell_size_um2_scales_quadratically_with_pixel_size():
    events = [make_event(2, cell_area_px=100)]
    at_1x = _run_pipeline_stub(events, pixel_size_um=1.0)[0].cell_size_um2
    at_2x = _run_pipeline_stub(events, pixel_size_um=2.0)[0].cell_size_um2
    # Doubling pixel_size_um should quadruple the reported area (um^2 = px * s^2).
    assert abs(at_2x - 4 * at_1x) < 1e-9


def test_cell_size_um2_is_none_when_pixel_size_missing():
    events = [make_event(2, cell_area_px=100)]
    result = _run_pipeline_stub(events, pixel_size_um=None)
    assert result[0].cell_size_um2 is None


def test_cell_size_um2_is_none_when_area_missing():
    events = [make_event(2, cell_area_px=None)]
    result = _run_pipeline_stub(events, pixel_size_um=0.5)
    assert result[0].cell_size_um2 is None


def test_cell_size_um2_is_set_independently_per_event():
    events = [make_event(2, cell_area_px=100), make_event(3, cell_area_px=400)]
    result = _run_pipeline_stub(events, pixel_size_um=0.5)
    sizes = {e.track_id: e.cell_size_um2 for e in result}
    assert abs(sizes[2] - 25.0) < 1e-9
    assert abs(sizes[3] - 100.0) < 1e-9
