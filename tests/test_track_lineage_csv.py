"""Cover src.track._write_lineage_csv on synthetic CTC tables.

Worth a test despite being a small function: it runs inside link_frames_trackastra,
partway through a multi-hour GPU run, so an exception here does not fail fast -- it
throws away the whole run. These cases are the ones that could raise (a bridged
group, a parent that got merged into its own child's canonical id) rather than
merely produce wrong numbers.
"""

import csv

import pandas as pd

from src.track import _write_lineage_csv


def _run(tmp_path, rows, canonical_of):
    df = pd.DataFrame(rows, columns=["label", "t1", "t2", "parent"])
    parent_of = {int(r["label"]): int(r["parent"]) for _, r in df.iterrows()}
    begin_of = {int(r["label"]): int(r["t1"]) for _, r in df.iterrows()}
    true_birth = {}
    for lbl, canon in canonical_of.items():
        if canon not in true_birth or begin_of[lbl] < begin_of[true_birth[canon]]:
            true_birth[canon] = lbl
    _write_lineage_csv(tmp_path, df, canonical_of, true_birth, parent_of, begin_of,
                       "t2", "label")
    return {int(r["track_id"]): r
            for r in csv.DictReader((tmp_path / "ctc_lineage.csv").open(encoding="utf-8"))}


def test_simple_division(tmp_path):
    # 1 spans 0-9 then divides into 2 and 3.
    out = _run(tmp_path, [(1, 0, 9, 0), (2, 10, 20, 1), (3, 10, 20, 1)],
               {1: 1, 2: 2, 3: 3})
    assert out[1]["parent_id"] == ""
    assert out[1]["n_daughters"] == "2"
    assert sorted(out[1]["daughter_ids"].split()) == ["2", "3"]
    assert out[2]["parent_id"] == "1"
    assert out[2]["first_frame"] == "10" and out[2]["last_frame"] == "20"
    assert out[3]["n_daughters"] == "0" and out[3]["daughter_ids"] == ""


def test_bridged_group_spans_both_segments_and_keeps_one_birth(tmp_path):
    # Label 5 is a resumed continuation of 4 after a one-frame detection gap, so
    # _bridge_track_gaps collapsed both onto canonical id 4. The group must report a
    # single span 0-20, and 5's own recorded parent must not become 4's parent.
    out = _run(tmp_path, [(4, 0, 9, 0), (5, 11, 20, 4)], {4: 4, 5: 4})
    assert set(out) == {4}
    assert out[4]["first_frame"] == "0" and out[4]["last_frame"] == "20"
    assert out[4]["parent_id"] == ""
    assert out[4]["n_daughters"] == "0"


def test_parent_merged_into_same_canonical_id_is_not_its_own_parent(tmp_path):
    # Pathological but reachable: bridging maps a label's parent onto the label's own
    # canonical id. Self-parenting would create a cycle for anything walking lineage.
    out = _run(tmp_path, [(7, 0, 9, 0), (8, 10, 20, 7)], {7: 7, 8: 7})
    assert out[7]["parent_id"] == ""
    assert "7" not in out[7]["daughter_ids"].split()
