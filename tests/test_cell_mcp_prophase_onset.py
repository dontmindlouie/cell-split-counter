"""Tests for _prophase_onset / find_prophase_onset -- condensation scored BACKWARD
past a mother's own track start, through any predecessor id-hop resolve_lineage_chain
resolves.

Why it exists: a track's first frame is where THAT ID started, not necessarily where
the biology did -- Cellpose can lose and regain a cell across a wobble before a
division just as it does after one (2026-08-05/06 researcher-test eval, well
nTSC_ZO1_1-4_M1). The forward half of this (cond/cond_f in find_candidates) already
exists; this is its mirror, reusing _walk_chain's backward hop-walk and the same
DNA-conservation gate _condensation itself uses.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


def _mother(first=20, last=40, area=100.0, mean=100.0):
    """Steady interphase nucleus at (100, 100) -- also the baseline window."""
    return [{"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
             "area_um2": area, "intensity_mean": mean,
             "intensity_integrated": area * 4 * mean}
            for f in range(first, last + 1)]


def _predecessor(frames=range(14, 20), area=50.0, mean_start=160.0, mean_step=16.0):
    """A condensed-looking object at the same spot, ending right before the mother's
    own track picks up -- the hop _walk_chain's backward walk should resolve.

    Brightness ramps across the frames so there is one unambiguous peak (the last
    frame, right at the handoff) rather than a tie a test would have to special-case.
    """
    out = []
    for i, f in enumerate(frames):
        mean = mean_start + mean_step * i
        out.append({"track_id": 0, "frame": f, "cx": 100.0, "cy": 100.0,
                    "area_um2": area, "intensity_mean": mean,
                    "intensity_integrated": area * 4 * mean})
    return out


def _crowd(n_frames):
    """Six ordinary cells far away that set the per-frame field median -- same
    convention as test_cell_mcp_condensation.py."""
    rows = []
    for f in range(n_frames):
        for k in range(6):
            rows.append({"track_id": 900 + k, "frame": f, "cx": 400.0 + 10 * k,
                         "cy": 400.0, "area_um2": 100.0,
                         "intensity_mean": 100.0, "intensity_integrated": 40000.0})
    return rows


def _well(monkeypatch, rows, n_frames=41):
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "pixel_size_um": 0.5, "n_frames": n_frames,
        "frame_timestamps_ms": [f * 300_000 for f in range(n_frames)],
    })
    return "fake"


def test_finds_the_peak_frame_in_a_resolved_predecessor(monkeypatch):
    fake = _well(monkeypatch, _mother() + _predecessor() + _crowd(41))
    result = cell_mcp_server._prophase_onset(fake, 1)
    assert result is not None
    assert result["prophase_frame"] == 19
    assert result["chain"] == [1, 0]
    assert result["dna"] == pytest.approx(1.2, abs=0.01)


def test_returns_none_when_no_predecessor_resolves(monkeypatch):
    fake = _well(monkeypatch, _mother() + _crowd(41))
    assert cell_mcp_server._prophase_onset(fake, 1) is None


def test_returns_none_when_predecessor_fails_the_dna_gate(monkeypatch):
    """A predecessor that resolves as a HOP (position/coexistence pass) but whose
    signal looks like a fragment (DNA far below baseline) must not be reported as
    prophase -- the gate is the same one _condensation uses forward."""
    fragment = [{"track_id": 0, "frame": f, "cx": 100.0, "cy": 100.0,
                "area_um2": 20.0, "intensity_mean": 100.0,
                "intensity_integrated": 20.0 * 4 * 100.0} for f in range(14, 20)]
    fake = _well(monkeypatch, _mother() + fragment + _crowd(41))
    assert cell_mcp_server._prophase_onset(fake, 1) is None


def test_tool_reports_the_resolved_chain_and_peak(monkeypatch):
    fake = _well(monkeypatch, _mother() + _predecessor() + _crowd(41))
    out = cell_mcp_server.find_prophase_onset(fake, 1)
    assert "prophase_frame 19" in out
    assert "chain [1, 0]" in out


def test_tool_names_the_no_predecessor_case_distinctly(monkeypatch):
    """Must not read like 'no prophase' -- it is a different, honest admission that
    checking earlier needs a human stepping frames by eye."""
    fake = _well(monkeypatch, _mother() + _crowd(41))
    out = cell_mcp_server.find_prophase_onset(fake, 1)
    assert "no predecessor track resolves" in out
    assert "watch_location_over_time" in out


def test_tool_names_the_failed_gate_case_distinctly(monkeypatch):
    fragment = [{"track_id": 0, "frame": f, "cx": 100.0, "cy": 100.0,
                "area_um2": 20.0, "intensity_mean": 100.0,
                "intensity_integrated": 20.0 * 4 * 100.0} for f in range(14, 20)]
    fake = _well(monkeypatch, _mother() + fragment + _crowd(41))
    out = cell_mcp_server.find_prophase_onset(fake, 1)
    assert "resolve" in out and "no frame in them passes" in out


def test_tool_rejects_unknown_track(monkeypatch):
    fake = _well(monkeypatch, _mother() + _crowd(41))
    with pytest.raises(ValueError, match="not found"):
        cell_mcp_server.find_prophase_onset(fake, 999)
