"""Tests for src.track._bridge_track_gaps, especially the two guards that stop it
blending distinct cells onto one canonical track_id.

Before the guards, 3.25% of tracks on M12_RUES2 mixed two different cells, which is
what made a "track" untrustworthy as a unit of navigation -- the track and the
OFF-TRACK walk could sit on different cells in the same crop. Re-run offline against
M12's real masks, the guards take co-existing merged groups from 168 to 0 while
keeping 1,351 of 1,740 bridges.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track import _bridge_track_gaps  # noqa: E402

H = W = 40


def _video(placements: dict[int, list[tuple[int, int, int]]], n_frames: int) -> np.ndarray:
    """placements: {frame: [(label, cx, cy), ...]} -> a (T,H,W) label video."""
    vid = np.zeros((n_frames, H, W), dtype=np.uint16)
    for f, cells in placements.items():
        for lab, cx, cy in cells:
            vid[f, cy - 1:cy + 2, cx - 1:cx + 2] = lab
    return vid


def _ctc(spans: dict[int, tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"label": l, "begin": b, "end": e} for l, (b, e) in spans.items()])


def _run(spans, placements, n_frames):
    return _bridge_track_gaps(_ctc(spans), _video(placements, n_frames), "label", "begin", "end")


def test_a_simple_one_frame_gap_is_still_bridged():
    """The behaviour the function exists for: Cellpose drops a mask for one frame and
    Trackastra starts a new track. That must still merge."""
    spans = {1: (0, 2), 2: (4, 6)}
    places = {0: [(1, 10, 10)], 1: [(1, 10, 10)], 2: [(1, 10, 10)],
              4: [(2, 11, 10)], 5: [(2, 11, 10)], 6: [(2, 11, 10)]}
    out = _run(spans, places, 7)
    assert out[1] == out[2], "a lone nearby successor after a short gap should bridge"


def test_a_real_division_with_two_successors_is_not_bridged():
    spans = {1: (0, 2), 2: (3, 5), 3: (3, 5)}
    places = {0: [(1, 20, 20)], 1: [(1, 20, 20)], 2: [(1, 20, 20)],
              3: [(2, 18, 20), (3, 22, 20)], 4: [(2, 18, 20), (3, 22, 20)],
              5: [(2, 18, 20), (3, 22, 20)]}
    out = _run(spans, places, 6)
    assert out[1] != out[2] and out[1] != out[3], "daughters must stay separate"


def test_two_predecessors_claiming_one_successor_do_not_merge():
    """GUARD 2, and the mechanism behind the worst real damage: union-find is
    transitive, so without this both predecessors AND the successor land in one
    group -- and the two predecessors can then overlap heavily."""
    spans = {1: (0, 3), 2: (0, 3), 3: (5, 8)}
    places = {}
    for f in range(4):
        places[f] = [(1, 18, 20), (2, 22, 20)]
    for f in (5, 6, 7, 8):
        places[f] = [(3, 20, 20)]
    out = _run(spans, places, 9)
    assert out[1] != out[3], "an ambiguous successor must not be claimed"
    assert out[2] != out[3]
    assert out[1] != out[2], "and the two predecessors must certainly not merge"


def test_labels_that_coexist_never_share_a_canonical_id():
    """GUARD 1. One cell cannot be in two places at once, so overlapping spans are
    provably different cells whatever the geometry says."""
    spans = {1: (0, 10), 2: (5, 15)}
    places = {}
    for f in range(11):
        places.setdefault(f, []).append((1, 20, 20))
    for f in range(5, 16):
        places.setdefault(f, []).append((2, 21, 20))
    out = _run(spans, places, 16)
    assert out[1] != out[2], "co-existing labels must never be merged"


def test_the_overlap_guard_is_checked_against_the_whole_group_not_just_the_pair():
    """Chaining is the mechanism, so A~B then B~C must be rejected when A and C
    overlap even though neither pair overlaps its immediate partner."""
    # A: 0-3, C: 2-9 (overlaps A), B: 5-8 -- B is a plausible successor to A,
    # and C would chain in via B if only the pair were checked.
    spans = {1: (0, 3), 3: (2, 9), 2: (5, 8)}
    places = {}
    for f in range(4):
        places.setdefault(f, []).append((1, 20, 20))
    for f in range(2, 10):
        places.setdefault(f, []).append((3, 21, 21))
    for f in range(5, 9):
        places.setdefault(f, []).append((2, 20, 20))
    out = _run(spans, places, 10)
    assert out[1] != out[3], "A and C overlap and must never end up in one group"


def test_result_does_not_depend_on_dict_iteration_order():
    spans = {1: (0, 2), 2: (4, 6), 5: (0, 2), 6: (4, 6)}
    places = {}
    for f in (0, 1, 2):
        places[f] = [(1, 10, 10), (5, 30, 30)]
    for f in (4, 5, 6):
        places[f] = [(2, 11, 10), (6, 31, 30)]
    a = _run(spans, places, 7)
    b = _run(dict(reversed(list(spans.items()))), places, 7)
    assert a == b
