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

import cell_mcp_server  # noqa: E402


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

    monkeypatch.setattr(cell_mcp_server, "BUNDLE", tmp_path)
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "n_frames": 100, "pixel_size_um": 0.5, "width_px": 512, "height_px": 512,
        "lineage": {"source": "geometry"},
    })
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 256.0, "cy": 256.0} for t in range(1, 7)]))
    return "W"


def test_division_pool_ranks_the_fragment_first(well):
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    body = [l for l in out.splitlines() if l and l[0].isdigit()]
    assert body[0].startswith("4 |"), "the size_ratio 0.05 split must rank above the 0.90 one"


def test_track_end_pool_excludes_anything_with_daughters(well):
    out = cell_mcp_server.find_candidates(well, pool="track_end")
    ids = {int(l.split("|")[0]) for l in out.splitlines() if l and l[0].isdigit()}
    assert 1 not in ids and 4 not in ids, "mothers are not track-ends"
    assert {2, 3, 5, 6} & ids


def test_it_never_claims_a_sort_it_did_not_apply(well, monkeypatch):
    """A score sort on a pool with no scores used to return file order while the
    header said otherwise."""
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 256.0, "cy": 256.0} for t in range(1, 7)]))
    df = pd.read_csv(Path(cell_mcp_server.BUNDLE) / "W" / "lineage.csv")
    df["size_ratio"] = ""
    df["dna_ratio"] = ""
    df.to_csv(Path(cell_mcp_server.BUNDLE) / "W" / "lineage.csv", index=False)
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    assert "asked for fragment_like" in out and "sorted by frame" in out


def test_pool_and_sort_by_are_closed_enums_in_the_mcp_schema(well):
    """2026-08-16: pool/sort_by used to be bare str -- an unrecognized sort_by
    silently fell through to a default sort with a misleading "this pool carries
    no link scores" note, even when the real problem was a typo (e.g. "Duration"
    instead of "duration"). Literal types make the MCP layer reject that before
    find_candidates' body ever runs, with a message that names the typo."""
    import asyncio
    from cell_mcp_server.server import server

    async def _call(**kw):
        return await server.call_tool("find_candidates", {"well": well, **kw})

    with pytest.raises(Exception, match="sort_by"):
        asyncio.run(_call(sort_by="Duration"))
    with pytest.raises(Exception, match="pool"):
        asyncio.run(_call(pool="Division"))
    # A valid call still goes through the same path without raising.
    asyncio.run(_call(sort_by="duration", pool="division"))


def _rows(out: str) -> list[str]:
    return [l for l in out.splitlines() if l and l[0].isdigit() and " | " in l]


def test_the_stratum_is_printed_on_every_row_not_just_in_the_census(well):
    """It was computed and counted, but only shown in the summary block -- so a reader
    scanning the table ranked a known vanishing_daughter link as the cleanest row in
    the well (2026-08-01, track 115)."""
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    assert "| stratum |" in out
    for line in _rows(out):
        assert any(name in line for name, _ in cell_mcp_server._STRATA), line


def test_ratios_measured_across_a_stub_link_read_na_not_a_number(well):
    """Both mothers here have a 2-frame daughter. dna/size are measured ACROSS that
    link, so a printed 0.98 outranks the honest rows -- worse than a blank."""
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    for line in _rows(out):
        assert "n/a | n/a" in line, line
        assert "0.98" not in line and "1.02" not in line, line


def test_ratios_come_back_when_both_daughters_persist(well):
    df = pd.read_csv(Path(cell_mcp_server.BUNDLE) / "W" / "lineage.csv")
    df.loc[df.track_id == 3, "last_frame"] = 40  # no longer a stub
    df.to_csv(Path(cell_mcp_server.BUNDLE) / "W" / "lineage.csv", index=False)
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    row = next(l for l in _rows(out) if l.startswith("1 |"))
    assert "0.98" in row and "n/a" not in row, row


def test_dau_frames_can_never_disagree_with_the_vanishing_daughter_stratum(well):
    """One threshold, two consumers.

    Note the asymmetry, which is deliberate: strata are FIRST-match-wins, so a row
    with a stub daughter that also trips fragment_like is labelled fragment_like and
    never reaches the vanishing_daughter test. The census therefore UNDERSTATES how
    many links rest on a stub. Suppression must key off the span itself, not off the
    label, or those rows print ratios again through the side door.
    """
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    for line in _rows(out):
        spans = [int(x) for x in line.split("|")[3].strip().split("/") if x.strip().isdigit()]
        stub = min(spans) <= cell_mcp_server._STUB_DAUGHTER_FRAMES
        assert stub == ("n/a" in line), line
        if "vanishing_daughter" in line:
            assert stub, line
    assert "0" not in [l.split("|")[3].strip() for l in _rows(out)], \
        "a daughter that exists cannot last zero frames -- count inclusively"


def test_it_hands_back_a_call_centred_on_the_event_not_on_the_link(well):
    """Hand-building this call is where the documented trap gets sprung."""
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="fragment_like")
    assert "Ready to look" in out
    call = next(l for l in out.splitlines() if "follow_cells_over_time(" in l and "  1 " in l)
    assert "track_ids=[1, 2, 3]" in call, "the family, not the mother alone"
    assert "centre_frame=" in call


def test_contested_pool_says_so_when_the_source_cannot_supply_it(well, monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "n_frames": 100, "pixel_size_um": 0.5, "width_px": 512, "height_px": 512,
        "lineage": {"source": "ctc"},
    })
    out = cell_mcp_server.find_candidates(well, pool="contested")
    assert "ctc" in out and "no contested pool" in out


def test_near_edge_rows_are_counted_even_though_they_are_not_shown(well, monkeypatch):
    """Dropping is fine; dropping SILENTLY is not.

    The rows stay out of the listing -- a clipped nucleus has understated area and
    brightness, so it is a bad cell to study -- but they have to remain visible as a
    number, because an invisible count is what made events.csv dangerous.
    """
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 2.0, "cy": 2.0} for t in range(1, 7)]))
    out = cell_mcp_server.find_candidates(well, pool="division")
    assert "edge_clipped | 2 |" in out
    assert not [l for l in out.splitlines() if l and l[0].isdigit()]


def test_asking_for_the_edge_stratum_turns_the_edge_filter_off(well, monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 2.0, "cy": 2.0} for t in range(1, 7)]))
    out = cell_mcp_server.find_candidates(well, pool="division", stratum="edge_clipped")
    assert [l for l in out.splitlines() if l and l[0].isdigit()]


def test_census_partitions_the_pool(well):
    out = cell_mcp_server.find_candidates(well, pool="division", limit=0)
    total = int(out.split("division pool, ")[1].split(" total")[0])
    counts = {l.split(" | ")[0]: int(l.split(" | ")[1])
              for l in out.splitlines() if " | " in l and l.split(" | ")[0] in
              {n for n, _ in cell_mcp_server._STRATA}}
    assert sum(counts.values()) == total
    assert "THE THRESHOLDS ARE UNVALIDATED" in out


def test_census_only_shows_no_rows(well):
    out = cell_mcp_server.find_candidates(well, pool="division", limit=0)
    assert "census only" in out
    assert not [l for l in out.splitlines() if l and l[0].isdigit()]


def test_unknown_pool_is_rejected(well):
    with pytest.raises(ValueError, match="unknown pool"):
        cell_mcp_server.find_candidates(well, pool="deaths")


# --- sort_by="random": the only sort a number may be computed from ------------
#
# Added after a 2026-07-31 blind-scoring session drew its BeWo sample with
# sort_by="duration" and then computed per-stratum true-positive rates from it.
# duration is measured in FRAMES, so it is not neutral across wells with different
# cadence, and on a 5-case draw it plausibly oversampled long-lived non-dividers.


def test_random_sort_is_reproducible_under_a_seed(well):
    a = cell_mcp_server.find_candidates(well, pool="division", sort_by="random", seed=7)
    b = cell_mcp_server.find_candidates(well, pool="division", sort_by="random", seed=7)
    assert a == b, "a seeded draw someone cannot repeat cannot back a published number"


def test_random_sort_states_the_seed_and_warns_when_there_is_none(well):
    seeded = cell_mcp_server.find_candidates(well, pool="division", sort_by="random", seed=7)
    assert "seed=7" in seeded and "re-callable" in seeded
    loose = cell_mcp_server.find_candidates(well, pool="division", sort_by="random")
    assert "UNSEEDED" in loose


def test_random_sort_is_not_silently_downgraded(well):
    """The fallback path rewrites sort_by when a pool carries no scores; random
    carries none by definition and must survive that."""
    out = cell_mcp_server.find_candidates(well, pool="division", sort_by="random", seed=1)
    assert "sorted by random" in out
    assert "asked for random" not in out


def test_random_draw_covers_the_whole_pool_not_the_head_of_another_order(well):
    """Shuffle first, limit second. Drawing from the top of a ranked list is the
    exact bias this sort exists to remove."""
    seen = set()
    for s in range(40):
        out = cell_mcp_server.find_candidates(well, pool="division", sort_by="random",
                                       seed=s, limit=1)
        seen.update(l.split(" |")[0] for l in out.splitlines() if l and l[0].isdigit())
    assert seen == {"1", "4"}, f"only ever drew {seen}"


# --- sort_by="daughter_persistence" ---------------------------------------------
#
# Promoted 2026-08-06 from _daughter_spans's "printed but not scored" column
# (dau_min) to an actual sort, after a second independent session (nTSC,
# 2026-08-06) found the same split as the original 5-case read: real divisions'
# daughters ran 77-159 frames, artifacts' 1-10. `well`'s own fixture ties on
# dau_min (both mothers' worst daughter is a 2-frame stub), so this needs its own
# fixture with a genuine split.


@pytest.fixture
def persistence_well(tmp_path, monkeypatch):
    d = tmp_path / "P"
    d.mkdir()
    pd.DataFrame([
        # mother 1: both daughters persist -- a real division.
        {"track_id": 1, "parent_id": "", "first_frame": 0, "last_frame": 10,
         "n_daughters": 2, "daughter_ids": "2 3", "link_distance_px": "",
         "dna_ratio": "", "size_ratio": ""},
        {"track_id": 2, "parent_id": 1, "first_frame": 11, "last_frame": 110,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 5.0,
         "dna_ratio": 0.98, "size_ratio": 0.90},
        {"track_id": 3, "parent_id": 1, "first_frame": 11, "last_frame": 100,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 5.0,
         "dna_ratio": 0.98, "size_ratio": 0.90},
        # mother 4: one daughter is a short-lived stub -- the artifact shape.
        {"track_id": 4, "parent_id": "", "first_frame": 0, "last_frame": 20,
         "n_daughters": 2, "daughter_ids": "5 6", "link_distance_px": "",
         "dna_ratio": "", "size_ratio": ""},
        {"track_id": 5, "parent_id": 4, "first_frame": 21, "last_frame": 90,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 3.0,
         "dna_ratio": 1.02, "size_ratio": 0.05},
        {"track_id": 6, "parent_id": 4, "first_frame": 21, "last_frame": 23,
         "n_daughters": 0, "daughter_ids": "", "link_distance_px": 3.0,
         "dna_ratio": 1.02, "size_ratio": 0.05},
    ]).to_csv(d / "lineage.csv", index=False)
    monkeypatch.setattr(cell_mcp_server, "BUNDLE", tmp_path)
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "n_frames": 200, "pixel_size_um": 0.5, "width_px": 512, "height_px": 512,
        "lineage": {"source": "geometry"},
    })
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(
        [{"track_id": t, "frame": 0, "cx": 256.0, "cy": 256.0} for t in range(1, 7)]))
    return "P"


def test_daughter_persistence_ranks_the_surviving_pair_first(persistence_well):
    out = cell_mcp_server.find_candidates(persistence_well, pool="division",
                                          sort_by="daughter_persistence")
    body = [l for l in out.splitlines() if l and l[0].isdigit()]
    assert body[0].startswith("1 |"), \
        "mother 1's daughters (100, 90 frames) must outrank mother 4's (70, 3)"


def test_daughter_persistence_sorts_by_the_shorter_daughter_not_the_longer(persistence_well):
    """Ranking on the WORSE of the pair, not the better one, is the whole point --
    a row only earns a high rank if BOTH daughters held up. Sorting on the max
    would let mother 4's 70-frame daughter (5) hide its 3-frame stub sibling."""
    out = cell_mcp_server.find_candidates(persistence_well, pool="division",
                                          sort_by="daughter_persistence")
    body = [l for l in out.splitlines() if l and l[0].isdigit()]
    assert body[-1].startswith("4 |")


def test_daughter_persistence_is_not_silently_downgraded(persistence_well):
    out = cell_mcp_server.find_candidates(persistence_well, pool="division",
                                          sort_by="daughter_persistence")
    assert "sorted by daughter_persistence" in out
