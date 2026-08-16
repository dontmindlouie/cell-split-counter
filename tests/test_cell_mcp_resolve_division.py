"""Tests for resolve_division -- resolving a REAL split into 2+ coexisting
daughters lineage.csv never linked, as opposed to resolve_lineage_chain's job
(one physical object losing and regaining its id).

Added 2026-08-15 (researcher feedback): most divisions found by hand were NOT
id hops -- they were genuine splits the tracker never linked, found only by
repeated manual list_nearby_tracks + coexistence-judging, often 3-4 hops deep.
This automates that same coexistence+distance+size test.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    """Mother (track 1, f0-4) splits into two coexisting daughters (10, 11,
    f5-15), one of which (10) itself wobble-hops once to 12 (f17-20) --
    a daughter that is not the same as a second sibling. Track 99 sits far
    away the whole time and must never be picked as a candidate.

    1 um/px, 1 min/frame.
    """
    rows = []
    for f in range(0, 5):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0})
    for f in range(5, 16):
        rows.append({"track_id": 10, "frame": f, "cx": 97.0, "cy": 100.0,
                     "area_um2": 25.0})
        rows.append({"track_id": 11, "frame": f, "cx": 104.0, "cy": 100.0,
                     "area_um2": 25.0})
    for f in range(17, 21):
        rows.append({"track_id": 12, "frame": f, "cx": 97.0, "cy": 103.0,
                     "area_um2": 24.0})
    for f in range(0, 30):
        rows.append({"track_id": 99, "frame": f, "cx": 500.0, "cy": 500.0,
                     "area_um2": 50.0})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(40)],
    })
    return "fake"


def test_resolves_a_real_split_into_two_daughters(fake):
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=20)
    assert "2 coexisting daughter(s)" in out
    assert "10:" in out.replace(" ", "") or "10:" in out
    assert "11:" in out
    assert "99" not in out


def test_daughter_hop_chain_is_collapsed_not_a_third_sibling(fake):
    """Track 12 is track 10's own wobble hop, not a third daughter -- it must be
    folded into 10's chain, not counted as a separate coexisting member."""
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=20)
    assert "2 coexisting daughter(s)" in out
    assert "12" in out  # present, but inside 10's chain note
    assert "3 coexisting" not in out


def test_stitched_set_includes_mother_and_full_chains(fake):
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=20)
    assert "Stitched set: [1, 10, 11, 12]" in out


def test_pure_hop_with_no_second_daughter_reports_no_split(fake, monkeypatch):
    """Only one candidate (10, which itself hops to 12) starts near the mother --
    nothing to coexist WITH, so this is a hop, not a split, and resolve_division
    must say so rather than report a one-member 'split'."""
    rows = cell_mcp_server._tracks(fake).to_dict("records")
    rows = [r for r in rows if r["track_id"] != 11]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=20)
    assert "none of them COEXIST" in out
    assert "resolve_lineage_chain" in out


def test_no_nearby_candidates_at_all(fake):
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=0)
    assert "No split resolves here" in out
    assert "resolve_lineage_chain" in out


def test_rejects_unknown_track(fake):
    with pytest.raises(ValueError, match="not found"):
        cell_mcp_server.resolve_division(fake, 999)


def test_distant_track_never_picked_as_a_candidate(fake):
    out = cell_mcp_server.resolve_division(fake, 1, before_min=0, after_min=20)
    assert "99" not in out
