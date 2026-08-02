"""Tests for _collapse_sites -- folding recorded divisions that are one real event.

Why: a tracker that fails through a division does not fail once. On BeWo M2, tracks
1824 (f468-472), 1860 (f475-479) and 1883 (f479-496) are one cell at one spot, each
recorded as its own division with its own daughters, and they took three of the top
five rows of a cond-ranked sample of five. The reviewer asked for five candidates and
got three copies of one answer.

Two earlier rules failed in opposite directions and both are pinned here:

- Proximity alone chained f140 to f697 through an entire lineage and folded 81% of
  the pool, because a genuine daughter also begins where her mother ended.
- Gating on "do two new objects coexist at this site" folded 3 rows out of 1253: in a
  crowded BeWo field some pair always coexists, so the gate fires everywhere.

What separates a re-acquisition from a lineage is DURATION, which is what the span
cap encodes. Real fold rates under it: BeWo M2 1253 -> 797 (36%), RUES2 M12 1222 ->
1095 (10%) -- worse on the line that tracks worse, which is the expected direction.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


def _mk(monkeypatch, tracks_rows, n_frames=400):
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(tracks_rows))
    monkeypatch.setattr(cell_mcp, "_manifest", lambda w: {
        "pixel_size_um": 1.0, "n_frames": n_frames,
        # 5 min per frame: the 60 min span cap is 12 frames.
        "frame_timestamps_ms": [f * 300_000 for f in range(n_frames)],
    })
    return "W"


def _track(tid, f0, f1, x, y, area=100.0):
    return [{"track_id": tid, "frame": f, "cx": x, "cy": y,
             "area_um2": area, "area_px": area * 4, "intensity_mean": 100.0}
            for f in range(f0, f1 + 1)]


def _rows(*ids):
    return pd.DataFrame([{"track_id": t} for t in ids])


def test_a_re_acquired_mother_folds_into_the_first_recording(monkeypatch):
    """Three short mothers at one spot inside an hour: one cell, one site."""
    rows = _track(1, 0, 4, 100, 100) + _track(2, 5, 8, 101, 100) \
        + _track(3, 9, 11, 100, 101)
    w = _mk(monkeypatch, rows)
    sites = cell_mcp._collapse_sites(w, _rows(1, 2, 3), cell_mcp._tracks(w))
    assert len(sites) == 1
    rep, others = next(iter(sites.items()))
    assert sorted([rep, *others]) == [1, 2, 3]


def test_the_representative_is_the_longest_lived_recording(monkeypatch):
    """The recording of the event that saw the most of it."""
    rows = _track(1, 0, 1, 100, 100) + _track(2, 2, 9, 100, 100)
    w = _mk(monkeypatch, rows)
    sites = cell_mcp._collapse_sites(w, _rows(1, 2), cell_mcp._tracks(w))
    assert list(sites) == [2] and sites[2] == [1]


def test_nothing_is_discarded(monkeypatch):
    """A dropped row is a row nobody can audit -- the failure that made events.csv
    unusable. Folded ids come back so the caller can list them."""
    rows = _track(1, 0, 4, 100, 100) + _track(2, 5, 8, 100, 100)
    w = _mk(monkeypatch, rows)
    sites = cell_mcp._collapse_sites(w, _rows(1, 2), cell_mcp._tracks(w))
    assert sorted(sum([[k, *v] for k, v in sites.items()], [])) == [1, 2]


def test_a_distant_mother_is_a_separate_site(monkeypatch):
    rows = _track(1, 0, 4, 100, 100) + _track(2, 5, 8, 400, 400)
    w = _mk(monkeypatch, rows)
    sites = cell_mcp._collapse_sites(w, _rows(1, 2), cell_mcp._tracks(w))
    assert len(sites) == 2


def test_a_chain_does_not_walk_through_a_whole_lineage(monkeypatch):
    """THE failure of the first version. Each hop is close and soon after the last, so
    proximity alone folds them all -- but the group would then span hours, and no cell
    divides repeatedly in that time at one spot. The span cap breaks the chain."""
    rows = []
    for i, tid in enumerate((1, 2, 3, 4, 5, 6)):
        rows += _track(tid, i * 10, i * 10 + 8, 100, 100)
    w = _mk(monkeypatch, rows)
    sites = cell_mcp._collapse_sites(w, _rows(1, 2, 3, 4, 5, 6), cell_mcp._tracks(w))
    assert len(sites) > 1, "a 250-minute chain must not become one site"
    for rep, others in sites.items():
        g = [rep, *others]
        sub = cell_mcp._tracks(w)
        lo = int(sub[sub.track_id.isin(g)].frame.min())
        hi = int(sub[sub.track_id.isin(g)].frame.max())
        assert cell_mcp._minutes_between(w, lo, hi) <= cell_mcp._SITE_MAX_SPAN_MIN


def test_a_bundle_without_the_columns_gets_no_fold_rather_than_a_wrong_one(monkeypatch):
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0} for t in (1, 2)]))
    monkeypatch.setattr(cell_mcp, "_manifest", lambda w: {"pixel_size_um": 1.0})
    sites = cell_mcp._collapse_sites("W", _rows(1, 2), cell_mcp._tracks("W"))
    assert sites == {1: [], 2: []}
