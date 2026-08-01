"""Tests for the condensation score -- the only morphology signal in find_candidates.

Why it exists: every other column describes who the tracker linked to whom. On BeWo
M2 that topology scored close to ANTI-correlated with a human reading metaphase
plates (2026-07-31 blind scoring: 0/4 `clean` cases real, 4/5 `vanishing_daughter`
real), because the tracker breaks THROUGH the division and the recorded daughters are
pre-mitotic debris. A signal that needs the link to be right fails wherever topology
already fails, and those are the same events.

The three cases pinned here are the ones the score has to tell apart, and they are
built synthetically because that is the only way to know the answer:

    condensation  area down, brightness UP,   total conserved
    fragment      area down, brightness flat, total DOWN
    death/bleach  area flat, brightness DOWN, total DOWN

Against the maintainer's 28 real labels the ranking scores AUC 0.68 overall and 0.75 on
BeWo -- informative, not decisive, and recorded here so nobody mistakes it for a
detector.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


def _well(monkeypatch, mother_rows, extra_rows=(), n_frames=60):
    rows = list(mother_rows) + list(extra_rows)
    # A crowd of ordinary cells far away: they set the per-frame field median that
    # brightness is measured against, and they must not fall inside the disc.
    for f in range(n_frames):
        for k in range(6):
            rows.append({"track_id": 900 + k, "frame": f, "cx": 400.0 + 10 * k,
                         "cy": 400.0, "area_um2": 100.0, "area_px": 400.0,
                         "intensity_mean": 100.0, "intensity_integrated": 40000.0})
    monkeypatch.setattr(cell_mcp, "_tracks", lambda w: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp, "_manifest", lambda w: {
        "pixel_size_um": 0.5, "n_frames": n_frames,
        "frame_timestamps_ms": [f * 300_000 for f in range(n_frames)],
    })
    return "fake"


def _mother(last=40, area=100.0, mean=100.0):
    """A steady interphase nucleus at (100, 100) up to `last`."""
    return [{"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
             "area_um2": area, "area_px": area * 4, "intensity_mean": mean,
             "intensity_integrated": area * 4 * mean}
            for f in range(last + 1)]


def _tail(kind, start=41, n=6):
    """What sits at the mother's position AFTER her track ends."""
    out = []
    for i in range(n):
        f = start + i
        if kind == "condensed":      # half the area, double the brightness
            a, m = 50.0, 200.0
        elif kind == "fragment":     # half the area, same brightness -> half the DNA
            a, m = 50.0, 100.0
        else:                        # dying: same area, half the brightness
            a, m = 100.0, 50.0
        out.append({"track_id": 2, "frame": f, "cx": 100.0, "cy": 100.0,
                    "area_um2": a, "area_px": a * 4, "intensity_mean": m,
                    "intensity_integrated": a * 4 * m})
    return out


def _lin():
    return pd.DataFrame([
        {"track_id": 1, "parent_id": "", "first_frame": 0, "last_frame": 40,
         "n_daughters": 2, "daughter_ids": "2 3"},
    ])


def _score(fake, well_rows=None):
    lin = _lin()
    rows = lin[lin.n_daughters >= 2].copy()
    s, f, d, a = cell_mcp._condensation(
        fake, rows, lin, cell_mcp._tracks(fake), 0.5)
    return s[0], f[0], d[0], a[0]


def test_condensation_scores_above_baseline(monkeypatch):
    fake = _well(monkeypatch, _mother(), _tail("condensed"))
    cond, frame, dna, area = _score(fake)
    assert cond > 1.5, f"halved area at doubled brightness must read high, got {cond}"
    assert 0.8 < dna < 1.2, "the DNA is all still there -- that is the point"
    assert area < 0.7, "and the object got smaller"


def test_a_fragment_does_not_score_as_condensation(monkeypatch):
    """The failure that shipped first: scoring (brightness up) x (area down) put M12's
    fragments on top at cond 93 with 65% of the DNA gone. Both shrink the area; only
    condensation raises the per-pixel signal."""
    fake = _well(monkeypatch, _mother(), _tail("fragment"))
    cond, frame, dna, area = _score(fake)
    assert cond != cond or cond < 1.2, f"flat brightness is not condensation ({cond})"


def test_a_dying_cell_does_not_score_as_condensation(monkeypatch):
    fake = _well(monkeypatch, _mother(), _tail("dying"))
    cond, frame, dna, area = _score(fake)
    # Exactly 1.0 is the right answer, not a near miss: the peak over the window is
    # the mother's own untroubled baseline, because nothing after it ever beat her.
    assert cond != cond or cond <= 1.01, f"brightness FELL ({cond})"


def test_the_peak_is_found_after_the_mothers_track_ends(monkeypatch):
    """The whole reason for the forward-heavy window. On BeWo the figure appears up to
    ~20 min past the link, which is where the tracker gave up, not where the cell
    divided."""
    fake = _well(monkeypatch, _mother(), _tail("condensed"))
    cond, frame, dna, area = _score(fake)
    assert frame > 40, f"peak at f{frame} -- must be able to look past last_frame"


def test_it_reads_objects_the_lineage_never_linked(monkeypatch):
    """Track 3 below is the condensed figure, and nothing links it to the mother --
    exactly BeWo 802, whose family had rows in only 10 of 28 window frames and scored
    NaN. Measuring a DISC rather than a member set is what fixes that, and it is what
    makes this signal independent of the tracking it is meant to check."""
    orphan = [{"track_id": 77, "frame": 41 + i, "cx": 100.0, "cy": 100.0,
               "area_um2": 50.0, "area_px": 200.0, "intensity_mean": 200.0,
               "intensity_integrated": 40000.0} for i in range(6)]
    fake = _well(monkeypatch, _mother(), orphan)
    cond, frame, dna, area = _score(fake)
    assert cond > 1.5, f"unlinked figure inside the disc must still count ({cond})"


def test_bleaching_does_not_read_as_lost_dna(monkeypatch):
    """BeWo 793: a whole-track baseline put every window frame at 0.60 of 'baseline'
    purely because the recording started brighter, the conservation gate rejected all
    28 of them, and a case a human read as prophase scored NaN. The baseline is the
    hour before the window for this reason."""
    rows = []
    for f in range(41):
        decay = 1.0 - 0.01 * f           # 40% dimmer by the end, steadily
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_um2": 100.0, "area_px": 400.0,
                     "intensity_mean": 100.0 * decay,
                     "intensity_integrated": 40000.0 * decay})
    tail = []
    for i in range(6):
        decay = 1.0 - 0.01 * (41 + i)
        tail.append({"track_id": 2, "frame": 41 + i, "cx": 100.0, "cy": 100.0,
                     "area_um2": 50.0, "area_px": 200.0,
                     "intensity_mean": 200.0 * decay,
                     "intensity_integrated": 40000.0 * decay})
    fake = _well(monkeypatch, rows, tail)
    cond, frame, dna, area = _score(fake)
    assert cond == cond, "a bleaching cell must not be scored NaN"
    assert cond > 1.4, f"condensation is still condensation under a bleach ({cond})"


def test_too_short_a_mother_scores_nan_rather_than_guessing(monkeypatch):
    """No history means no baseline. Inventing one manufactures the signal."""
    short = [{"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
              "area_um2": 100.0, "area_px": 400.0, "intensity_mean": 100.0,
              "intensity_integrated": 40000.0} for f in range(3)]
    lin = pd.DataFrame([{"track_id": 1, "parent_id": "", "first_frame": 0,
                         "last_frame": 2, "n_daughters": 2, "daughter_ids": "2 3"}])
    fake = _well(monkeypatch, short, _tail("condensed", start=3))
    s, f, d, a = cell_mcp._condensation(
        fake, lin[lin.n_daughters >= 2].copy(), lin, cell_mcp._tracks(fake), 0.5)
    assert np.isnan(s[0]) and f[0] == -1
