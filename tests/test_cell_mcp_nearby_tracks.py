"""Tests for list_nearby_tracks -- the tool that hands the daughter question over.

Why it exists: Cellpose segments the daughters fine; the TRACKER declines to link
them, so on BeWo the objects you need are already in tracks.csv under ids nobody
connected. The first attempt at using that automatically -- take the two nearest new
starts to the mother -- chose tracks 3829 and 3879 on BeWo 969, which begin 2.3 um and
six frames apart and are ONE cell being re-acquired. The reviewer caught it on sight.

So the tool reports and does not decide, and the column that makes deciding possible
is `coexists_with`. Two daughters must be on screen at the same time; a run of tracks
at one spot with consecutive, non-overlapping spans is a single object losing its id,
no matter how close together the starts look. That is the same shape as the
co-existence contradiction that made the _bridge_track_gaps over-merge provable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def well(monkeypatch):
    """A mother ending at f10 at (100, 100), and three things near her afterwards:

    - 20 and 21: a genuine pair, both live f11-30, 10 px either side. They COEXIST.
    - 30, 31, 32: consecutive spans at one spot -- one object re-acquired twice.
    - 40: a neighbour that has been on screen since f0.
    """
    rows = []
    for f in range(11):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 100.0, "area_px": 400.0, "intensity_mean": 100.0})
    for f in range(11, 31):
        rows.append({"track_id": 20, "frame": f, "cx": 92.0, "cy": 100.0,
                     "area_um2": 50.0, "area_px": 200.0, "intensity_mean": 100.0})
        rows.append({"track_id": 21, "frame": f, "cx": 108.0, "cy": 100.0,
                     "area_um2": 50.0, "area_px": 200.0, "intensity_mean": 100.0})
    for tid, (a, b) in {30: (12, 14), 31: (15, 17), 32: (18, 25)}.items():
        for f in range(a, b + 1):
            rows.append({"track_id": tid, "frame": f, "cx": 100.0, "cy": 116.0,
                         "area_um2": 60.0, "area_px": 240.0, "intensity_mean": 100.0})
    for f in range(0, 31):
        rows.append({"track_id": 40, "frame": f, "cx": 112.0, "cy": 108.0,
                     "area_um2": 90.0, "area_px": 360.0, "intensity_mean": 100.0})

    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 300_000 for f in range(40)],
    })
    return "W"


def _row(out, tid):
    for line in out.splitlines():
        if line.split(" | ")[0] == str(tid):
            return line
    return ""


def test_a_real_pair_is_reported_as_coexisting(well):
    out = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=30)
    assert "21" in _row(out, 20).split("coexists_with")[-1] or "21" in _row(out, 20)
    assert "20" in _row(out, 21)


def test_a_re_acquired_object_coexists_with_none_of_its_own_chain(well):
    """30, 31 and 32 sit at one spot with consecutive spans. Distance cannot separate
    them from a sister pair; coexistence can, and that is the whole point."""
    out = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=30)
    for a, b in ((30, 31), (31, 32), (30, 32)):
        assert str(b) not in _row(out, a).split(" | ")[-1], \
            f"{a} and {b} have disjoint spans and must not read as coexisting"


def test_a_long_standing_neighbour_is_excluded_by_default(well):
    """A sister has to be NEW. Track 40 has been on screen since f0."""
    out = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=30)
    assert _row(out, 40) == ""
    out2 = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=30, new_only=False)
    assert _row(out2, 40) != "", "new_only=False must still be able to show it"


def test_it_decides_nothing(well):
    """The failure this replaces was a silent pick. The output must rank by distance
    and say outright that it is not claiming a division."""
    out = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=30)
    assert "nothing here is a claim that a division happened" in out
    assert "Two daughters must COEXIST" in out


def test_an_empty_neighbourhood_says_so_rather_than_returning_a_bare_table(well):
    out = cell_mcp_server.list_nearby_tracks(well, track_id=1, radius_um=2)
    assert "nothing segmented within" in out
    assert "new_only" in out, "must say the filter that may have caused the emptiness"


def test_it_can_anchor_on_a_point_instead_of_a_track(well):
    out = cell_mcp_server.list_nearby_tracks(well, x=100.0, y=100.0, frame=10, radius_um=30)
    assert "point (100, 100)" in out


def test_an_anchor_is_required(well):
    with pytest.raises(ValueError, match="either track_id"):
        cell_mcp_server.list_nearby_tracks(well)
