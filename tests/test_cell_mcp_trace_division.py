"""Tests for trace_division -- resolve_division's multi-generation extension.

Added 2026-08-16 (field feedback): resolve_division only resolves ONE hop's
worth of coexisting candidates. A real division on 20251016_ACTB_M1 needed 5-8
manual hops (get_lineage) to reach the real split, and even then a short-lived
side branch (one frame) was almost silently dropped. trace_division walks
forward through every generation automatically and never drops a terminal
branch, flagging short-lived ones instead.

Positions are spaced well outside _walk_chain's own (small, hardcoded) hop
radius so a generation's daughters are never mistaken for a wobble-hop of an
unrelated sibling -- that bit a first draft of this fixture (10 briefly
absorbed 20 as its own hop because they happened to be close together).
radius_um is passed explicitly and large enough to cover the wider spacing.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402

_R = dict(before_min=0, after_min=20, radius_um=70.0)


@pytest.fixture
def fake(monkeypatch):
    """Mother 1 (f0-4, at 100,100) splits into 10 (100,160 -- terminal, never
    splits again) and 11 (100,40), which itself splits a second generation
    later into 20 (85,20) and 21 (115,20) plus a short-lived one-frame
    fragment 22 (100,20). Track 99 sits far away and must never appear.
    1 um/px, 1 min/frame.
    """
    rows = []
    for f in range(0, 5):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0})
    for f in range(5, 16):
        rows.append({"track_id": 10, "frame": f, "cx": 100.0, "cy": 160.0,
                     "area_um2": 25.0})
        rows.append({"track_id": 11, "frame": f, "cx": 100.0, "cy": 40.0,
                     "area_um2": 25.0})
    for f in range(17, 26):
        rows.append({"track_id": 20, "frame": f, "cx": 85.0, "cy": 20.0,
                     "area_um2": 24.0})
        rows.append({"track_id": 21, "frame": f, "cx": 115.0, "cy": 20.0,
                     "area_um2": 24.0})
    rows.append({"track_id": 22, "frame": 17, "cx": 100.0, "cy": 20.0,
                 "area_um2": 20.0})
    for f in range(0, 30):
        rows.append({"track_id": 99, "frame": f, "cx": 500.0, "cy": 500.0,
                     "area_um2": 50.0})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(40)],
    })
    return "fake"


def test_walks_a_second_generation_split(fake):
    out = cell_mcp_server.trace_division(fake, 1, **_R)
    assert "2 generation" in out
    assert "10" in out
    assert "20" in out and "21" in out


def test_short_lived_terminal_is_flagged_not_dropped(fake):
    out = cell_mcp_server.trace_division(fake, 1, **_R)
    assert "22" in out
    assert "LIKELY FRAGMENT" in out


def test_terminal_branches_include_both_long_lived_daughters(fake):
    out = cell_mcp_server.trace_division(fake, 1, **_R)
    terminal_section = out.split("Terminal branches")[1]
    assert "10:" in terminal_section
    assert "20:" in terminal_section
    assert "21:" in terminal_section


def test_distant_track_never_appears(fake):
    out = cell_mcp_server.trace_division(fake, 1, **_R)
    assert "99" not in out


def test_max_generations_stops_the_walk(fake):
    """With max_generations=1, only the mother's own split (1 -> 10, 11) is
    walked -- 11's own further split into 20/21 is never explored, so 11 itself
    becomes a terminal branch rather than being walked into its daughters."""
    out = cell_mcp_server.trace_division(fake, 1, **{**_R, "max_generations": 1})
    assert "1 generation" in out
    terminal_section = out.split("Terminal branches")[1]
    assert "10:" in terminal_section
    assert "11:" in terminal_section
    assert "20" not in terminal_section


def test_rejects_unknown_track(fake):
    with pytest.raises(ValueError, match="not found"):
        cell_mcp_server.trace_division(fake, 999)


def test_all_members_traced_includes_every_generation(fake):
    out = cell_mcp_server.trace_division(fake, 1, **_R)
    tail = out.split("All members traced:")[1]
    for tid in ("1", "10", "11", "20", "21", "22"):
        assert tid in tail


@pytest.fixture
def wide_fake(monkeypatch):
    """Mother 1 (f0-4, at 100,100) splits into two long-lived daughters 30
    (100,220) and 31 (140,220) -- both 120-127um away, outside the default
    radius_um=70 (so the default search finds NOTHING, n_checked=0) but inside
    the 2.2x widened radius (154um). No other branch competes for them.

    Added 2026-08-17 (field feedback): the default search is not always wide
    enough -- track 12 on 20251016_ACTB_M1's real daughters (828/829) were only
    reachable a widened search away from an intermediate hop. This is the
    positive case; see the `fake` fixture above (10/11/20/21) for the negative
    guard -- a widened search must never steal a candidate that actually
    belongs to a sibling branch.
    """
    rows = []
    for f in range(0, 5):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0})
    for f in range(5, 25):
        rows.append({"track_id": 30, "frame": f, "cx": 100.0, "cy": 220.0,
                     "area_um2": 25.0})
        rows.append({"track_id": 31, "frame": f, "cx": 140.0, "cy": 220.0,
                     "area_um2": 25.0})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(40)],
    })
    return "fake"


def test_widen_retry_finds_daughters_outside_default_radius(wide_fake):
    out = cell_mcp_server.trace_division(wide_fake, 1, **_R)
    assert "30" in out and "31" in out
    assert "WIDENED SEARCH (level 1" in out
    terminal_section = out.split("Terminal branches")[1]
    assert "30:" in terminal_section
    assert "31:" in terminal_section


@pytest.fixture
def escalation_fake(monkeypatch):
    """Mother 1 (f0-4, at 100,100) splits into two long-lived daughters 40
    (100,350) and 41 (140,350) -- 250-253um away. Outside the default radius
    (70) AND the single-widen level (154), but inside the level-2 escalation
    cap (min(70*4.8, 300) = 300um) -- needs the ladder to actually reach a
    SECOND level, not just the one retry the previous single-widen version had.
    """
    rows = []
    for f in range(0, 5):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0})
    for f in range(5, 25):
        rows.append({"track_id": 40, "frame": f, "cx": 100.0, "cy": 350.0,
                     "area_um2": 25.0})
        rows.append({"track_id": 41, "frame": f, "cx": 140.0, "cy": 350.0,
                     "area_um2": 25.0})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 1.0, "n_frames": 40, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 60_000 for f in range(40)],
    })
    return "fake"


def test_escalation_reaches_a_second_level(escalation_fake):
    out = cell_mcp_server.trace_division(escalation_fake, 1, **_R)
    assert "40" in out and "41" in out
    assert "WIDENED SEARCH (level 2" in out
    terminal_section = out.split("Terminal branches")[1]
    assert "40:" in terminal_section
    assert "41:" in terminal_section


def test_escalation_stops_at_theft_rather_than_widening_further(fake):
    """Same 10/11/20/21 fixture as the sibling-theft guard above, but with a
    radius small enough that 10's OWN escalation doesn't reach 20/21 (140.8um
    away) until level 2 (30*4.8=144um) -- the ladder must still refuse them
    there (same theft rule at every level, not just level 1), leaving 10 an
    ordinary terminal. 11's own default search (30um) already reaches its real
    children directly (25um away) with no escalation needed, so 11 correctly
    resolves into 20/21 as a normal generation, not a terminal."""
    out = cell_mcp_server.trace_division(fake, 1, before_min=0, after_min=20, radius_um=30.0)
    assert "11 (via chain end 11) -> 20, 21" in out
    terminal_section = out.split("Terminal branches")[1]
    assert "10:" in terminal_section
    assert "11:" not in terminal_section
