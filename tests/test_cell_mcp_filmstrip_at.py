"""Tests for watch_location_over_time -- the position-addressed filmstrip.

Its reason to exist is that every other tool addresses cells by track_id, so an object
the segmenter never caught cannot be asked about at all. The nearest-detection readout
is what makes that usable: it is how a caller tells "this is track 2036" from "nothing
tracked is anywhere near here".
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    """0.5 um/px, so a 10 px separation is 5 um."""
    tracks = pd.DataFrame([
        {"track_id": 1, "frame": 0, "cx": 100.0, "cy": 100.0},
        {"track_id": 2, "frame": 0, "cx": 110.0, "cy": 100.0},
        {"track_id": 3, "frame": 0, "cx": 300.0, "cy": 300.0},
        {"track_id": 1, "frame": 1, "cx": 100.0, "cy": 100.0},
    ])
    monkeypatch.setattr(cell_mcp, "_tracks", lambda well: tracks)
    monkeypatch.setattr(cell_mcp, "_manifest", lambda well: {
        "pixel_size_um": 0.5, "n_frames": 10, "width_px": 512, "height_px": 512,
    })
    return "fake"


def test_nearest_detection_reports_id_and_distance_in_um(fake):
    tid, dum = cell_mcp._nearest_detection(fake, 0, 100.0, 100.0)
    assert tid == 1 and dum == pytest.approx(0.0)
    tid, dum = cell_mcp._nearest_detection(fake, 0, 104.0, 100.0)
    assert tid == 1 and dum == pytest.approx(2.0), "4 px at 0.5 um/px"


def test_nearest_detection_excludes_the_anchor(fake):
    """In anchor mode the anchor is always nearest to itself at 0.0 um, which says
    nothing -- the useful answer is what ELSE is near the place being watched."""
    tid, _ = cell_mcp._nearest_detection(fake, 0, 100.0, 100.0)
    assert tid == 1
    tid, dum = cell_mcp._nearest_detection(fake, 0, 100.0, 100.0, exclude=1)
    assert tid == 2 and dum == pytest.approx(5.0)


def test_nearest_detection_is_none_when_the_frame_has_nothing(fake):
    assert cell_mcp._nearest_detection(fake, 7, 100.0, 100.0) is None
    # ...and when excluding the only cell present leaves nothing.
    assert cell_mcp._nearest_detection(fake, 1, 100.0, 100.0, exclude=1) is None


def test_requires_either_a_point_or_an_anchor(fake):
    with pytest.raises(ValueError, match="either x and y"):
        cell_mcp.watch_location_over_time(fake, start_frame=0, end_frame=2)
    with pytest.raises(ValueError, match="either x and y"):
        cell_mcp.watch_location_over_time(fake, start_frame=0, end_frame=2, x=1.0)


def test_rejects_an_unknown_anchor_track(fake):
    with pytest.raises(ValueError, match="anchor track 999 not found"):
        cell_mcp.watch_location_over_time(fake, start_frame=0, end_frame=2, anchor_track_id=999)


def test_rejects_an_inverted_range(fake):
    with pytest.raises(ValueError, match="empty range"):
        cell_mcp.watch_location_over_time(fake, start_frame=5, end_frame=1, x=1.0, y=1.0)
