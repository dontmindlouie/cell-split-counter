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
