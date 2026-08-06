"""Tests for _walk_chain / resolve_lineage_chain -- chasing a cell through id hops
caused by segmentation losing and regaining it (wobble), not a division.

The failure this exists to fix: a daughter's track ends at a wobble hop, the
strip's centring drops her from the mean because her span "ended", and the crop
snaps onto her sister alone -- read live as the cells having moved
(2026-08-05/06 researcher-test eval, well nTSC_ZO1_1-4_M1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    """Track 1 hops to 2, then to 3 -- one physical cell, two wobble re-acquisitions.

    Track 4 sits far away the whole time (never a candidate). Track 5 COEXISTS with
    1 (a real sibling, on screen at the same time) and must never be picked as a hop.
    1 um/px, 1 min/frame, so distances and gaps are easy to reason about in the raw
    numbers.
    """
    rows = []
    for f in range(0, 5):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0})
    for f in range(7, 10):
        rows.append({"track_id": 2, "frame": f, "cx": 103.0, "cy": 100.0,
                     "area_um2": 48.0})
    for f in range(12, 16):
        rows.append({"track_id": 3, "frame": f, "cx": 106.0, "cy": 101.0,
                     "area_um2": 47.0})
    for f in range(0, 16):
        rows.append({"track_id": 4, "frame": f, "cx": 400.0, "cy": 400.0,
                     "area_um2": 50.0})
    for f in range(0, 5):
        rows.append({"track_id": 5, "frame": f, "cx": 130.0, "cy": 100.0,
                     "area_um2": 50.0})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(40)],
    })
    return "fake"


def test_walks_through_two_wobble_hops(fake):
    chain = cell_mcp_server._walk_chain(fake, 1, direction="forward")
    assert [h["track_id"] for h in chain] == [2, 3]


def test_a_coexisting_track_is_never_picked_as_a_hop(fake, monkeypatch):
    """5 sits closer to 1 than 2 does but is ON SCREEN AT THE SAME TIME as 1 --
    that makes it a sibling, not a re-acquisition, and the coexistence test must
    reject it even though nothing else about it looks wrong."""
    rows = cell_mcp_server._tracks(fake).to_dict("records")
    rows = [r for r in rows if not (r["track_id"] == 5 and r["frame"] >= 3)]
    rows += [{"track_id": 5, "frame": f, "cx": 101.0, "cy": 100.0, "area_um2": 50.0}
             for f in range(0, 3)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    chain = cell_mcp_server._walk_chain(fake, 1, direction="forward")
    assert [h["track_id"] for h in chain] == [2, 3]
    assert all(h["track_id"] != 5 for h in chain)


def test_a_distant_track_is_never_picked(fake):
    """Track 4 starts inside the time window but 300 px away -- far outside any
    plausible cell radius, so it must never resolve as a hop."""
    chain = cell_mcp_server._walk_chain(fake, 1, direction="forward")
    assert all(h["track_id"] != 4 for h in chain)


def test_two_equally_close_candidates_stop_the_walk_rather_than_guess(fake, monkeypatch):
    rows = cell_mcp_server._tracks(fake).to_dict("records")
    rows += [{"track_id": 6, "frame": f, "cx": 103.0, "cy": 100.0, "area_um2": 48.0}
             for f in range(7, 10)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    chain = cell_mcp_server._walk_chain(fake, 1, direction="forward")
    assert chain == []


def test_no_candidate_in_range_returns_empty(fake):
    chain = cell_mcp_server._walk_chain(fake, 3, direction="forward")
    assert chain == []


def test_unknown_seed_track_returns_empty(fake):
    chain = cell_mcp_server._walk_chain(fake, 999, direction="forward")
    assert chain == []


def test_backward_direction_walks_the_mirror_image(fake):
    chain = cell_mcp_server._walk_chain(fake, 3, direction="backward")
    assert [h["track_id"] for h in chain] == [2, 1]


def test_resolve_lineage_chain_tool_reports_the_stitched_ids(fake):
    out = cell_mcp_server.resolve_lineage_chain(fake, 1)
    assert "2 forward hop(s) resolved" in out
    assert "Stitched chain: [1, 2, 3]" in out


def test_resolve_lineage_chain_says_why_it_found_nothing(fake):
    out = cell_mcp_server.resolve_lineage_chain(fake, 3)
    assert "no forward hop resolves" in out


def test_resolve_lineage_chain_rejects_unknown_track(fake):
    with pytest.raises(ValueError, match="not found"):
        cell_mcp_server.resolve_lineage_chain(fake, 999)


def test_resolve_family_auto_chains_a_recorded_daughter(fake, monkeypatch):
    """The fix for the actual complaint: a lone mother whose recorded daughter (2)
    later wobble-hops to 3 gets 3 folded into the member set automatically, so
    follow_cells_over_time's centring never sees her span "end" early."""
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {1: {"daughters": [2]}})
    members, added = cell_mcp_server._resolve_family(fake, [1], include_nearby=False)
    assert members == [1, 2, 3]
    assert added == [3]


def test_resolve_family_never_chains_the_anchor_itself(fake, monkeypatch):
    """ids[0] is the anchor (the mother, by this function's own convention) -- her
    track ending IS the division, not a wobble hop, so chain-walking from her would
    just rediscover a daughter the caller may have deliberately left off the list."""
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda well: {})
    members, added = cell_mcp_server._resolve_family(fake, [1], include_nearby=False)
    assert members == [1]
    assert added == []
