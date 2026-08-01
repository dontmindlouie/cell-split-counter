"""Tests for get_filmstrip_family -- the crop centred on a SET of tracks.

Its reason to exist: a division is the one event where the subject stops being one
object, which is exactly when a single-mask filmstrip goes OFF-TRACK and a fixed-point
crop starts depending on the cells not having migrated. Centring on whoever is present
makes "follow the mother, then the daughters' midpoint" fall out of membership rather
than a mode switch.

The invariants pinned here are the honesty ones, same as everywhere else in this
server: a held position must never be presentable as a measured one, and the crop must
not rescale between frames so that a rendering artifact can be read as biology.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    """A mother (f0-9) handing off to two daughters (f10-19) 20 px apart.

    0.5 um/px. The daughters MIGRATE -- +4 px per frame in x -- which is the case a
    fixed (x, y) crop cannot handle and the reason the member set exists.
    """
    rows = []
    for f in range(10):
        rows.append({"track_id": 1, "frame": f, "cx": 100.0, "cy": 100.0,
                     "area_px": 400.0, "n_masks_in_frame": 1, "intensity_mean": 100.0})
    for f in range(10, 20):
        drift = 4.0 * (f - 10)
        rows.append({"track_id": 2, "frame": f, "cx": 90.0 + drift, "cy": 100.0,
                     "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0})
        rows.append({"track_id": 3, "frame": f, "cx": 110.0 + drift, "cy": 100.0,
                     "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0})
    monkeypatch.setattr(cell_mcp, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp, "_manifest", lambda well: {
        "pixel_size_um": 0.5, "n_frames": 40, "width_px": 512, "height_px": 512,
    })
    monkeypatch.setattr(cell_mcp, "_frame_png",
                        lambda well, f: np.full((512, 512), 40, dtype=np.uint8))
    monkeypatch.setattr(cell_mcp, "_hours", lambda well, f: f * 0.1)
    return "fake"


def _strip(fake, ids, **kw):
    kw = {"start_frame": None, "end_frame": None, "max_images": 12, "crop_um": None,
          "color": False, "scale_bar": False, "marker": False,
          "before": 4, "after": 4, **kw}
    return cell_mcp._family_filmstrip_frames(
        fake, ids, kw["start_frame"], kw["end_frame"], kw["max_images"], kw["crop_um"],
        kw["color"], kw["scale_bar"], kw["marker"], kw["before"], kw["after"])


def test_window_is_chosen_around_the_membership_transition(fake):
    """Not the members' full span: that would open the mother's whole lifetime, which
    on a real well is hundreds of frames of nothing happening."""
    header, images = _strip(fake, [1, 2, 3])
    assert "transition at f10" in header
    assert "frames 6-14" in header


def test_centre_follows_the_mother_then_the_daughters_midpoint(fake):
    """The handoff needs no special case -- f9 has only the mother, f10 only the
    daughters, and the mean of whoever is present does the switching."""
    tracks = cell_mcp._tracks(fake)

    def centre(f):
        rows = tracks[tracks.frame == f]
        rows = rows[rows.track_id.isin([1, 2, 3])]
        return float(rows.cx.mean()), float(rows.cy.mean())

    assert centre(9) == (100.0, 100.0), "mother alone"
    assert centre(10) == (100.0, 100.0), "daughters' midpoint, symmetric about her"
    assert centre(14) == (116.0, 100.0), "midpoint has MIGRATED with the daughters"


def test_crop_is_auto_fitted_and_identical_for_every_frame(fake):
    """One size for the whole strip. Sizing per frame would rescale each image and the
    nuclei would appear to breathe, which reads as biology and is not."""
    header, images = _strip(fake, [1, 2, 3])
    assert "auto-fitted" in header
    assert len({img.shape for img in images}) == 1


def test_auto_fit_is_wide_enough_to_hold_both_daughters(fake):
    """Separation is 20 px = 10 um, plus each daughter's own radius. A hand-guessed
    crop_um is how a sibling ends up drifting out of frame halfway along the strip."""
    header, _ = _strip(fake, [1, 2, 3])
    width = float(header.split("Crop ")[1].split(" um")[0])
    assert width > 10.0


def test_a_frame_with_no_member_is_held_and_labelled_never_interpolated(fake):
    """An invented position rendered identically to a measured one is the failure this
    whole tool set exists to avoid."""
    header, _ = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39, max_images=12)
    assert "HELD" in header and "never interpolated" in header


def test_members_are_capped_and_chosen_once_by_size(fake, monkeypatch):
    """A cell shattering during necrosis can throw many ids. Re-picking the largest
    per frame would make the centre lurch as membership churned; picking once over the
    window keeps the strip on the same objects, jagged but stable."""
    rows = [{"track_id": t, "frame": f, "cx": 100.0 + t, "cy": 100.0,
             "area_px": 500.0 - 10 * t, "n_masks_in_frame": 1, "intensity_mean": 100.0}
            for t in range(1, 12) for f in range(5)]
    monkeypatch.setattr(cell_mcp, "_tracks", lambda well: pd.DataFrame(rows))
    header, _ = _strip(fake, list(range(1, 12)), start_frame=0, end_frame=4)
    assert "dropped to keep the centre stable" in header
    assert "largest by median area" in header


def test_unknown_ids_are_reported_rather_than_silently_ignored(fake):
    header, _ = _strip(fake, [1, 2, 3, 999])
    assert "NOT FOUND" in header and "999" in header


def test_all_members_are_missing_is_an_error_not_an_empty_strip(fake):
    with pytest.raises(ValueError, match="none of"):
        _strip(fake, [777, 888])


def test_no_ring_by_default_and_the_header_says_why(fake):
    """One ring among several members is ambiguous about what it is claiming."""
    header, _ = _strip(fake, [1, 2, 3])
    assert "Nothing is ringed" in header


def test_a_lone_mother_expands_to_her_recorded_daughters(fake, monkeypatch):
    monkeypatch.setattr(cell_mcp, "_lineage",
                        lambda well: {1: {"daughters": [2, 3]}})
    assert cell_mcp._resolve_family(fake, [1]) == [1, 2, 3]
    # Explicit sets are left exactly as given -- the caller may be disputing the link.
    assert cell_mcp._resolve_family(fake, [1, 2]) == [1, 2]


# --------------------------------------------------------------------- dropouts
#
# The case these pin is the one that actually went wrong on M12: daughter 2362 was not
# segmented at f353 while plainly visible in the pixels. The crop re-centred on her
# sister alone, the whole field shifted, and the shifted view was read first as the
# cells having moved and then as a lagging chromosome. Nothing in the image said the
# camera had panned -- which is the point: a reader cannot see a centring change.


@pytest.fixture
def dropout(monkeypatch, fake):
    """Same family as `fake`, but daughter 2 is not segmented at f14 -- a one-frame
    segmentation dropout in the middle of her span, with her sister still present."""
    rows = [r for r in cell_mcp._tracks(fake).to_dict("records")
            if not (r["track_id"] == 2 and r["frame"] == 14)]
    monkeypatch.setattr(cell_mcp, "_tracks", lambda well: pd.DataFrame(rows))
    return fake


def _centres(fake, ids, lo, hi):
    tracks = cell_mcp._tracks(fake)
    win = tracks[(tracks.track_id.isin(ids)) & (tracks.frame >= lo) & (tracks.frame <= hi)]
    pos = {}
    for r in win.itertuples():
        pos.setdefault(int(r.frame), []).append(r)
    return cell_mcp._resolve_family_centres(win, pos, list(range(lo, hi + 1)))


def test_a_dropout_does_not_swing_the_centre_onto_the_remaining_member(dropout):
    """The trade this makes, in numbers, on a family migrating 4 px/frame.

    Re-centring on the sister alone moves the crop 12 px in one frame -- a visible pan
    with nothing in the image to explain it. Holding the missing member's last measured
    position instead costs a 2 px lag, because a held position is one frame stale and
    there are two members sharing the mean. Six times smaller, and in the direction
    that keeps both cells in frame. The lag is real and bounded by the member's own
    drift; it is not a measured position and the label says so.
    """
    centres, _, _ = _centres(dropout, [1, 2, 3], 10, 19)
    x13, x14, x15 = centres[13][0], centres[14][0], centres[15][0]
    expected = (x13 + x15) / 2
    assert abs(x14 - expected) < 3.0, "held position lags by at most the member's drift"

    sister_only = float(
        cell_mcp._tracks(dropout).query("track_id == 3 and frame == 14").cx.iloc[0])
    assert abs(sister_only - expected) > 4 * abs(x14 - expected), "the swing it replaces"


def test_the_dropout_frame_names_the_missing_member(dropout):
    """'1 seen' alone reads as a cell having gone. The id is what tells a reader the
    difference between a segmentation failure and a biological event."""
    _, _, gapped = _centres(dropout, [1, 2, 3], 10, 19)
    assert gapped == {14: [2]}


def test_the_header_explains_that_a_gap_member_is_unringed(dropout):
    header, _ = _strip(dropout, [1, 2, 3], start_frame=10, end_frame=19, max_images=10)
    assert "segmentation dropout, not a departure" in header
    assert "NOT" in header and "ringed" in header
    assert "2" in header.split("a member (")[1].split(")")[0]


def test_a_member_is_never_held_outside_its_own_span(fake):
    """The mother must leave the mean after the handoff, or the crop never follows the
    daughters and the whole point of a member set is lost."""
    centres, _, gapped = _centres(fake, [1, 2, 3], 0, 19)
    assert gapped == {}, "no interior gaps in this fixture"
    assert centres[14][0] == pytest.approx(116.0), "daughters' midpoint, mother gone"
    assert centres[14][2] == 2 and centres[14][3] == 0


def test_a_frame_with_nothing_present_is_held_not_gap_filled(fake):
    """HELD and 'gap' are different claims: gap means a known member was missed here,
    HELD means the centre is simply the previous frame's, reused."""
    centres, held, gapped = _centres(fake, [1, 2, 3], 0, 25)
    assert set(range(20, 26)) <= held
    assert all(f not in gapped for f in range(20, 26))
    assert centres[25][:2] == centres[19][:2]
