"""Tests for find_candidates -- the whole-well triage tool.

Exists because every other tool answers a question about one track you already
picked, so "where do I even start" had no answer and two independent sessions
dropped out to Bash to compute it (13 and 15 calls). The invariants pinned here are
the honesty ones: it must not silently claim a sort it did not apply, and it must
not present topology as a verdict.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


@pytest.fixture
def well(tmp_path, monkeypatch):
    d = tmp_path / "W"
    (d).mkdir()
    pd.DataFrame([
        # mother, two daughters: one healthy split, one fragment-like
        {"track_id": 1, "parent_id": "", "first_frame": 0, "last_frame": 10,
         "n_daughters": 2, "daughter_ids": "2 3", "link_distance_px": "",
         "dna_ratio": "", "size_ratio": ""},
        {"track_id": 2, "parent_id": 1, "first_frame": 11, "last_frame": 40,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 5.0,
         "dna_ratio": 0.98, "size_ratio": 0.90},
        {"track_id": 3, "parent_id": 1, "first_frame": 11, "last_frame": 12,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 5.0,
         "dna_ratio": 0.98, "size_ratio": 0.90},
        {"track_id": 4, "parent_id": "", "first_frame": 0, "last_frame": 20,
         "n_daughters": 2, "daughter_ids": "5 6", "link_distance_px": "",
         "dna_ratio": "", "size_ratio": ""},
        {"track_id": 5, "parent_id": 4, "first_frame": 21, "last_frame": 60,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 3.0,
         "dna_ratio": 1.02, "size_ratio": 0.05},
        {"track_id": 6, "parent_id": 4, "first_frame": 21, "last_frame": 22,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 3.0,
         "dna_ratio": 1.02, "size_ratio": 0.05},
    ]).to_csv(d / "lineage.csv", index=False)

    monkeypatch.setattr(cell_mcp, "BUNDLE", tmp_path)
    monkeypatch.setattr(cell_mcp, "_manifest", lambda w: {
        "n_frames": 100, "pixel_size_um": 0.5, "width_px": 512, "height_px": 512,
        "lineage": {"source": "geometry"},
    })
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 256.0, "cy": 256.0} for t in range(1, 7)]))
    return "W"


def test_division_pool_ranks_the_fragment_first(well):
    out = cell_mcp.find_candidates(well, pool="division", sort_by="fragment_like")
    body = [l for l in out.splitlines() if l and l[0].isdigit()]
    assert body[0].startswith("4 |"), "the size_ratio 0.05 split must rank above the 0.90 one"


def test_track_end_pool_excludes_anything_with_daughters(well):
    out = cell_mcp.find_candidates(well, pool="track_end")
    ids = {int(l.split("|")[0]) for l in out.splitlines() if l and l[0].isdigit()}
    assert 1 not in ids and 4 not in ids, "mothers are not track-ends"
    assert {2, 3, 5, 6} & ids


def test_it_never_claims_a_sort_it_did_not_apply(well, monkeypatch):
    """A score sort on a pool with no scores used to return file order while the
    header said otherwise."""
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 256.0, "cy": 256.0} for t in range(1, 7)]))
    df = pd.read_csv(Path(cell_mcp.BUNDLE) / "W" / "lineage.csv")
    df["size_ratio"] = ""
    df["dna_ratio"] = ""
    df.to_csv(Path(cell_mcp.BUNDLE) / "W" / "lineage.csv", index=False)
    out = cell_mcp.find_candidates(well, pool="division", sort_by="fragment_like")
    assert "asked for fragment_like" in out and "sorted by frame" in out


def test_contested_pool_says_so_when_the_source_cannot_supply_it(well, monkeypatch):
    monkeypatch.setattr(cell_mcp, "_manifest", lambda w: {
        "n_frames": 100, "pixel_size_um": 0.5, "width_px": 512, "height_px": 512,
        "lineage": {"source": "ctc"},
    })
    out = cell_mcp.find_candidates(well, pool="contested")
    assert "ctc" in out and "no contested pool" in out


def test_near_edge_rows_are_counted_even_though_they_are_not_shown(well, monkeypatch):
    """Dropping is fine; dropping SILENTLY is not.

    The rows stay out of the listing -- a clipped nucleus has understated area and
    brightness, so it is a bad cell to study -- but they have to remain visible as a
    number, because an invisible count is what made events.csv dangerous.
    """
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 2.0, "cy": 2.0} for t in range(1, 7)]))
    out = cell_mcp.find_candidates(well, pool="division")
    assert "edge_clipped | 2 |" in out
    assert not [l for l in out.splitlines() if l and l[0].isdigit()]


def test_asking_for_the_edge_stratum_turns_the_edge_filter_off(well, monkeypatch):
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 2.0, "cy": 2.0} for t in range(1, 7)]))
    out = cell_mcp.find_candidates(well, pool="division", stratum="edge_clipped")
    assert [l for l in out.splitlines() if l and l[0].isdigit()]


def test_census_partitions_the_pool(well):
    out = cell_mcp.find_candidates(well, pool="division", limit=0)
    total = int(out.split("division pool, ")[1].split(" total")[0])
    counts = {l.split(" | ")[0]: int(l.split(" | ")[1])
              for l in out.splitlines() if " | " in l and l.split(" | ")[0] in
              {n for n, _ in cell_mcp._STRATA}}
    assert sum(counts.values()) == total
    assert "THE THRESHOLDS ARE UNVALIDATED" in out


def test_census_only_shows_no_rows(well):
    out = cell_mcp.find_candidates(well, pool="division", limit=0)
    assert "census only" in out
    assert not [l for l in out.splitlines() if l and l[0].isdigit()]


def test_unknown_pool_is_rejected(well):
    with pytest.raises(ValueError, match="unknown pool"):
        cell_mcp.find_candidates(well, pool="deaths")
