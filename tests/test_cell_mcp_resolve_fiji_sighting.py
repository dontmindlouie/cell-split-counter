"""Tests for resolve_fiji_sighting's stale-snap warning.

Added 2026-08-15 (researcher feedback): a click at (267,822) f348 snapped to
track 454 -- real, clean-looking, long-lived -- but get_lineage(454) showed it
was the already-settled daughter of a division ~120 frames earlier; it was
just idling nearby by the time of the click, and the real division at that
frame belonged to an unrelated track found only by widening the crop. A
snap being a real track is not evidence THAT track is doing anything at the
clicked frame -- this checks how long the snapped track has been running
before the click and flags it when that's suspiciously long.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    """Track 1: running since f0, still there at f200 -- a stale snap if the
    click lands at f200. Track 2: starts at f195, 5 min before the click -- a
    fresh snap that should NOT trigger the warning. Both sit at the same spot,
    1 um/px, 1 min/frame, so the click (100, 100) can be pointed at either by
    frame alone in tests that isolate one at a time.
    """
    rows = [{"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0, "area_um2": 50.0}
            for f in range(0, 201)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 250, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(250)],
    })
    return "fake"


def test_long_running_track_triggers_stale_snap_warning(fake, monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {1: {"parent": 0}})
    out = cell_mcp_server.resolve_fiji_sighting(fake, fiji_frame=201, x=100.0, y=100.0)
    assert "CHECK BEFORE TRUSTING THIS SNAP" in out
    assert "mother-link is that old too" in out


def test_freshly_started_track_does_not_trigger_the_warning(fake, monkeypatch):
    rows = [{"track_id": 2, "frame": f, "cx": 100.0, "cy": 100.0, "area_um2": 50.0}
            for f in range(195, 201)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {2: {"parent": 1}})
    out = cell_mcp_server.resolve_fiji_sighting(fake, fiji_frame=201, x=100.0, y=100.0)
    assert "CHECK BEFORE TRUSTING THIS SNAP" not in out


def test_no_lineage_csv_skips_the_check_entirely(fake, monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {})
    out = cell_mcp_server.resolve_fiji_sighting(fake, fiji_frame=201, x=100.0, y=100.0)
    assert "CHECK BEFORE TRUSTING THIS SNAP" not in out


def test_no_good_match_suggests_the_point_based_tools(fake, monkeypatch):
    """2026-08-16 feedback: a click with nothing nearby used to dead-end at
    'try list_nearby_tracks' -- with no track to anchor on, the actual next step
    is to look at the raw point directly via watch_location_over_time /
    show_cells_in_browser(kind='point'), which need no track at all."""
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {})
    out = cell_mcp_server.resolve_fiji_sighting(fake, fiji_frame=1, x=9000.0, y=9000.0)
    assert "NO GOOD MATCH" in out
    assert "watch_location_over_time(well='fake', x=9000, y=9000" in out
    assert "kind': 'point'" in out and "'x': 9000, 'y': 9000" in out


@pytest.fixture
def fake_um(monkeypatch):
    """2 um/px, so pixels and microns are numerically distinct -- the fixture the
    1 um/px `fake` fixture above cannot exercise, since there px == um and a
    unit-conversion bug would be invisible. Track 1 sits at pixel (50, 50); a
    click reported in MICRONS as (100, 100) converts to that same pixel spot."""
    rows = [{"track_id": 1, "frame": f, "cx": 50.0, "cy": 50.0, "area_um2": 50.0}
            for f in range(0, 201)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 2.0, "n_frames": 250, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(250)],
    })
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {1: {"parent": 0}})
    return "fake_um"


def test_stale_snap_message_uses_pixels_not_the_raw_fiji_unit(fake_um):
    """The suggested watch_location_over_time/show_cells_in_browser calls take
    full-frame PIXEL coordinates -- echoing back whatever unit Fiji happened to
    report (here, microns: 100, 100) would point them at the wrong place
    entirely (50 px off at 2 um/px)."""
    out = cell_mcp_server.resolve_fiji_sighting(fake_um, fiji_frame=201, x=100.0, y=100.0)
    assert "CHECK BEFORE TRUSTING THIS SNAP" in out
    assert "x=50, y=50" in out, out
    assert "x=100, y=100" not in out.split("CHECK BEFORE")[1]
