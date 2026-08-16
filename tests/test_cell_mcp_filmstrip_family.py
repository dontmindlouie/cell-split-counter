"""Tests for follow_cells_over_time -- the crop centred on a SET of tracks.

Its reason to exist: a division is the one event where the subject stops being one
object, which is exactly when a single-mask filmstrip goes OFF-TRACK and a fixed-point
crop starts depending on the cells not having migrated. Centring on whoever is present
makes "follow the mother, then the daughters' midpoint" fall out of membership rather
than a mode switch.

The invariants pinned here are the honesty ones, same as everywhere else in this
server: a held position must never be presentable as a measured one, and the crop must
not rescale between frames so that a rendering artifact can be read as biology.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


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
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 0.5, "n_frames": 40, "width_px": 512, "height_px": 512,
        # 5 min per frame, so a window asked for in minutes has an exact frame answer.
        "frame_timestamps_ms": [f * 300_000 for f in range(40)],
    })
    monkeypatch.setattr(cell_mcp_server, "_frame_png",
                        lambda well, f: np.full((512, 512), 40, dtype=np.uint8))
    monkeypatch.setattr(cell_mcp_server, "_hours", lambda well, f: f * 0.1)
    return "fake"


def _window(header: str) -> tuple[int, int]:
    """The rendered frame range, found by SHAPE rather than by splitting on the first
    'frames ' in the header. The header carries prose that legitimately contains that
    word ahead of the range, so the naive split silently read a warning instead."""
    m = re.search(r"frames (\d+)-(\d+) \(", header)
    assert m, f"no frame range in header: {header[:200]}"
    return int(m.group(1)), int(m.group(2))


def _strip(fake, ids, **kw):
    # 20 min either side = 4 frames on this fake's 5 min cadence, which is what these
    # tests asserted back when the window was counted in frames.
    kw = {"start_frame": None, "end_frame": None, "max_images": 12, "crop_um": None,
          "color": False, "scale_bar": False, "marker": False,
          "before_min": 20.0, "after_min": 20.0, "stride_min": cell_mcp_server._STRIDE_MIN,
          "cap": cell_mcp_server.MAX_IMAGES, "added": None, **kw}
    start, end = kw.pop("start_frame"), kw.pop("end_frame")
    return cell_mcp_server._family_filmstrip_frames(fake, ids, start, end, **kw)


def test_window_is_chosen_around_the_membership_transition(fake):
    """Not the members' full span: that would open the mother's whole lifetime, which
    on a real well is hundreds of frames of nothing happening."""
    header, images = _strip(fake, [1, 2, 3])
    assert "transition at f10" in header
    assert "frames 6-14" in header


def test_centre_follows_the_mother_then_the_daughters_midpoint(fake):
    """The handoff needs no special case -- f9 has only the mother, f10 only the
    daughters, and the mean of whoever is present does the switching."""
    tracks = cell_mcp_server._tracks(fake)

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
    assert "auto-fit" in header
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
    assert "HELD" in header and "Neither is interpolated" in header


def test_a_mostly_held_strip_says_so_before_the_images_are_spent(fake):
    """A HELD frame costs what a real one costs and teaches nothing -- it is a frozen
    crop of a place. Learning that by looking at 8 of 12 rendered images, as happened
    on 2026-08-01, is the whole failure; the header has the counts before the render."""
    header, _ = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39, max_images=12)
    assert "WARNING" in header and "cover only" in header
    assert header.startswith("WARNING"), "a warning after the prose is one nobody reads"
    assert "re-centre" in header or "cut after_min" in header


def test_it_names_the_tracks_that_would_fill_a_held_strip(fake, monkeypatch):
    """The daughters the tracker never linked are the usual reason a strip goes HELD,
    and they are sitting right there in the crop. Naming them turns a dead render into
    the next call."""
    rows = cell_mcp_server._tracks(fake).to_dict("records")
    rows += [{"track_id": 4, "frame": f, "cx": 140.0, "cy": 100.0, "area_px": 200.0,
              "n_masks_in_frame": 1, "intensity_mean": 100.0} for f in range(20, 30)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    header, _ = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39, max_images=12)
    assert "WARNING" in header
    assert "NOT in this set" in header and "4 (" in header
    assert "track_ids=[1, 2, 3, 4]" in header, "hand back the call, not just the id"


def test_members_are_capped_and_chosen_once_by_lifetime(fake, monkeypatch):
    """A cell shattering during necrosis can throw many ids. Re-picking per frame would
    make the centre lurch as membership churned; picking once over the window keeps the
    strip on the same objects, jagged but stable.

    Ranked by LIFETIME, not size. Median area dropped BeWo 1893 -- f480-497, the
    surviving daughter -- for being the smallest object in the set, which is the one
    member the strip could not afford to lose."""
    rows = [{"track_id": t, "frame": f, "cx": 100.0 + t, "cy": 100.0,
             "area_px": 500.0 - 10 * t, "n_masks_in_frame": 1, "intensity_mean": 100.0}
            for t in range(1, 12) for f in range(t % 5 + 1)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    header, _ = _strip(fake, list(range(1, 12)), start_frame=0, end_frame=4)
    assert "dropped to keep the centre stable" in header
    assert "longest-lived" in header
    kept = header.split("tracks [")[1].split("]")[0]
    # 4, 9 (5 frames each) outlive 1, 6, 11 (2 frames) despite being smaller.
    assert "4" in kept and "9" in kept


def test_a_member_absent_from_the_window_is_not_called_dropped(fake):
    """"Dropped to keep the centre stable" claims a choice was made. A member whose
    track ended before the window is simply not there, and saying otherwise sent a
    reader looking for a decision that never happened."""
    header, _ = _strip(fake, [1, 2, 3], start_frame=15, end_frame=19)
    assert "not present anywhere in this window" in header
    assert "1" in header.split("not present anywhere")[0].split(".")[-1] or "1" in header


def test_unknown_ids_are_reported_rather_than_silently_ignored(fake):
    header, _ = _strip(fake, [1, 2, 3, 999])
    assert "NOT FOUND" in header and "999" in header


def test_a_kept_member_far_from_the_rest_is_flagged(fake, monkeypatch):
    """2026-08-15 feedback: dropping the SHORTEST members by lifetime can retain a
    long-persisting but spatially-unrelated track while dropping the short
    connecting fragments that actually told the story, producing a crop that
    visibly jumps between unrelated objects. Six long-lived members (1-6) get
    kept over two short ones (7, 8); member 6 sits thousands of microns from the
    rest and must be called out rather than discovered by watching the crop jump."""
    rows = [{"track_id": t, "frame": f, "cx": 100.0 + t, "cy": 100.0,
             "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0}
            for t in range(1, 6) for f in range(10)]
    rows += [{"track_id": 6, "frame": f, "cx": 5000.0, "cy": 100.0,
              "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0}
             for f in range(10)]
    rows += [{"track_id": t, "frame": f, "cx": 100.0 + t, "cy": 100.0,
              "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0}
             for t in (7, 8) for f in range(2)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    header, _ = _strip(fake, list(range(1, 9)), start_frame=0, end_frame=9)
    assert "dropped to keep the centre stable" in header
    assert "WARNING: 6" in header
    assert "farther from every other kept member than the crop is wide" in header


def test_kept_members_all_close_together_are_not_flagged(fake):
    """The far-member warning is conditioned on there being a drop -- a small,
    spatially coherent set (this fixture's normal case) must stay silent."""
    header, _ = _strip(fake, [1, 2, 3])
    assert "farther from every other kept member" not in header


def test_all_members_are_missing_is_an_error_not_an_empty_strip(fake):
    with pytest.raises(ValueError, match="none of"):
        _strip(fake, [777, 888])


def test_no_ring_by_default_and_the_header_says_why(fake):
    """One ring among several members is ambiguous about what it is claiming."""
    header, _ = _strip(fake, [1, 2, 3])
    assert "Nothing is ringed" in header


def test_a_lone_mother_expands_to_her_recorded_daughters(fake, monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_lineage",
                        lambda well: {1: {"daughters": [2, 3]}})
    assert cell_mcp_server._resolve_family(fake, [1], include_nearby=False) == ([1, 2, 3], [])
    # Explicit sets keep their recorded members -- the caller may be disputing the link.
    assert cell_mcp_server._resolve_family(fake, [1, 2], include_nearby=False) == ([1, 2], [])


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
    rows = [r for r in cell_mcp_server._tracks(fake).to_dict("records")
            if not (r["track_id"] == 2 and r["frame"] == 14)]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    return fake


def _centres(fake, ids, lo, hi):
    tracks = cell_mcp_server._tracks(fake)
    win = tracks[(tracks.track_id.isin(ids)) & (tracks.frame >= lo) & (tracks.frame <= hi)]
    pos = {}
    for r in win.itertuples():
        pos.setdefault(int(r.frame), []).append(r)
    return cell_mcp_server._resolve_family_centres(win, pos, list(range(lo, hi + 1)))


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
        cell_mcp_server._tracks(dropout).query("track_id == 3 and frame == 14").cx.iloc[0])
    assert abs(sister_only - expected) > 4 * abs(x14 - expected), "the swing it replaces"


def test_the_dropout_frame_names_the_missing_member(dropout):
    """'1 seen' alone reads as a cell having gone. The id is what tells a reader the
    difference between a segmentation failure and a biological event."""
    _, _, gapped = _centres(dropout, [1, 2, 3], 10, 19)
    assert gapped == {14: [2]}


def test_the_header_explains_that_a_gap_member_is_unringed(dropout):
    header, _ = _strip(dropout, [1, 2, 3], start_frame=10, end_frame=19, max_images=10)
    assert "segmentation dropout" in header
    assert "NOT" in header and "ringed" in header
    assert "2" in header.split("labelled 'gap' (member ")[1].split(")")[0]


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


def test_every_image_tool_discloses_that_brightness_is_not_comparable(fake):
    """src/ingest.py stretches each frame to its OWN 0.5/99.5 percentiles, so a real
    field-wide trend -- photobleaching over 40-70 h -- is flattened into looking
    constant. Nothing in an image can reveal that, and the numbers do not share the
    problem (build_bundle measures against the raw ND2), so the split has to be stated
    or a reader will compare brightness across frames and be wrong."""
    header, _ = _strip(fake, [1, 2, 3])
    assert "0.5/99.5" in header and "NOT comparable" in header
    assert "get_track_profile" in header, "must point at where brightness IS reliable"

    solo, _ = cell_mcp_server._filmstrip_frames(fake, 1, 0, 9, max_images=4, crop_um=40.0,
                                         color=False, scale_bar=False, marker=False)
    assert "0.5/99.5" in solo


def test_the_note_says_whether_this_bundle_can_undo_the_stretch(fake, monkeypatch):
    """A bundle that recorded the per-frame window can be put on a common scale; one
    that did not is stuck. Saying only "not comparable" leaves a reader unable to tell
    which case they are in, and the answer differs per bundle."""
    base = dict(cell_mcp_server._manifest(fake))
    assert "cannot be undone" in cell_mcp_server._display_note(fake), "no window recorded"

    monkeypatch.setattr(cell_mcp_server, "_manifest",
                        lambda w: {**base, "display_window": {"recorded": True}})
    note = cell_mcp_server._display_note(fake)
    assert "reversible" in note and "lo + png/255" in note
    assert "NOT comparable" in note, "still a caveat, just a recoverable one"


# --- the window is minutes, and it reaches forward ----------------------------
#
# All of the below exist because of one 2026-07-31 finding: on BeWo M2 the mitotic
# figure appears up to ~7 frames (~20 min) AFTER the frame where lineage.csv records
# the mother->daughter link (track 802: link ends f771, prometaphase f778, two
# objects by f782). The old window was +/- a fixed number of FRAMES centred near the
# link, so it rendered the lead-up and cut off before the outcome -- and a human
# scoring off it called real mitoses artifacts, inverting the census strata.


def test_window_is_measured_in_minutes_not_frames(fake):
    """20 min on this fake's 5 min cadence is 4 frames. The same 20 min on a well
    shot every 3 min must be ~7 frames -- that is the whole point, and it is why a
    frame count gave BeWo the shortest look at the line that needed the longest."""
    header, _ = _strip(fake, [1, 2, 3], before_min=20.0, after_min=20.0)
    assert "frames 6-14" in header
    header2, _ = _strip(fake, [1, 2, 3], before_min=10.0, after_min=40.0)
    assert "frames 8-18" in header2


def test_window_reaches_much_further_forward_by_default(fake):
    """The transition is where the TRACKER stopped linking, not where the cell
    divided. Symmetric defaults are what hid the outcome.

    The exact ratio isn't load-bearing (widened 2026-08-13 on TSC_batch2_M13_WGD
    evidence that BEFORE also needed real headroom -- a real prophase landed 195 min
    before the transition, 6.5x the old 30-min default), just that AFTER stays
    bigger than BEFORE -- the visible outcome reliably lands soon after the
    transition on every line checked so far, which the old *2 factor overstated."""
    assert cell_mcp_server._WINDOW_AFTER_MIN > cell_mcp_server._WINDOW_BEFORE_MIN
    header, _ = _strip(fake, [1, 2, 3],
                       before_min=cell_mcp_server._WINDOW_BEFORE_MIN,
                       after_min=cell_mcp_server._WINDOW_AFTER_MIN)
    lo, hi = _window(header)
    assert hi - 10 > 10 - lo, "must look further past the transition than before it"


def test_a_minute_window_is_never_quietly_shorter_than_asked(fake):
    """Rounding inward would make a 90 min window mean 85 on some wells, silently."""
    header, _ = _strip(fake, [1, 2, 3], before_min=12.0, after_min=12.0)
    lo, hi = _window(header)
    assert cell_mcp_server._minutes_between(fake, lo, 10) >= 12.0
    assert cell_mcp_server._minutes_between(fake, 10, hi) >= 12.0


def test_the_header_states_the_window_in_minutes(fake):
    header, _ = _strip(fake, [1, 2, 3])
    assert re.search(r"frames \d+-\d+ \(\d+ min\)", header), "frame numbers alone hide cadence"


# --- sampling: gap-free when it fits, time-spaced when it does not ------------


def test_an_explicit_short_range_is_rendered_gap_free(fake):
    """A researcher who names a frame range wants that range. Their cost is eyes on
    images, and the skipped frame is where anaphase was -- it is 1-2 frames long."""
    header, images = _strip(fake, [1, 2, 3], start_frame=8, end_frame=15,
                            max_images=None)
    assert "no sampling gaps" in header
    assert len(images) == 8


def test_a_pinned_max_images_is_still_honoured(fake):
    header, images = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39,
                            max_images=5)
    assert len(images) == 5
    assert "max_images=5" in header


def test_a_long_window_is_thinned_by_time_and_says_the_spacing(fake):
    """Not by a frame count: a fixed count means different time resolution per well."""
    header, images = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39,
                            max_images=None, stride_min=10.0)
    assert len(images) <= cell_mcp_server.MAX_IMAGES
    assert "min apart" in header


def test_a_page_may_show_more_frames_than_the_context_cap(fake):
    """Images on an HTML page cost no context -- they go to disk and to a human's
    browser -- so the model's token budget has no business shrinking them."""
    assert cell_mcp_server.MAX_IMAGES_PAGE > cell_mcp_server.MAX_IMAGES
    _, images = _strip(fake, [1, 2, 3], start_frame=0, end_frame=39,
                       max_images=None, cap=cell_mcp_server.MAX_IMAGES_PAGE)
    assert len(images) == 40, "the whole range fits under the page cap"


# --- daughters the lineage never recorded ------------------------------------
#
# "Only tracking 1 daughter, would be nice to have midpoint" -- maintainer note, scoring
# case 12 (track 969), where the mitosis is plainly visible (pro 768, meta 777, ana
# 792) but only one daughter is linked, so the crop follows half the event and the
# real sister drifts out of frame.


def test_an_unlinked_sister_is_found_by_position(monkeypatch, fake):
    """A daughter the tracker never connected still has to APPEAR as a new object next
    to her mother. That is what makes her findable from geometry alone."""
    rows = list(cell_mcp_server._tracks(fake).to_dict("records"))
    # Track 1 ends at f9 at (100, 100); 2 and 3 are her recorded daughters. Track 50
    # begins at f10 right beside her and is linked to nobody.
    for f in range(10, 20):
        rows.append({"track_id": 50, "frame": f, "cx": 104.0, "cy": 100.0,
                     "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0,
                     "area_um2": 50.0})
    for r in rows:
        r.setdefault("area_um2", r["area_px"] * 0.25)
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda w: {1: {"daughters": [2, 3]}})

    members, added = cell_mcp_server._resolve_family(fake, [1], include_nearby=True)
    assert 50 in members and added == [50]


def test_a_long_standing_neighbour_is_not_mistaken_for_a_daughter(monkeypatch, fake):
    """The discriminator is that a sister is NEW. A cell that has been on screen the
    whole time and merely happens to be close is a neighbour, and sweeping it in turns
    'the sister' into 'the neighbourhood'."""
    rows = list(cell_mcp_server._tracks(fake).to_dict("records"))
    for f in range(0, 20):
        rows.append({"track_id": 60, "frame": f, "cx": 104.0, "cy": 100.0,
                     "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0,
                     "area_um2": 50.0})
    for r in rows:
        r.setdefault("area_um2", r["area_px"] * 0.25)
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda w: {1: {"daughters": [2, 3]}})

    members, added = cell_mcp_server._resolve_family(fake, [1], include_nearby=True)
    assert 60 not in members and added == []


def test_the_header_says_a_member_is_there_by_position_not_lineage(monkeypatch, fake):
    """Showing an unrecorded object silently would be claiming the lineage vouches for
    it. It does not, and that is a different claim from a recorded daughter."""
    rows = list(cell_mcp_server._tracks(fake).to_dict("records"))
    for f in range(10, 20):
        rows.append({"track_id": 50, "frame": f, "cx": 104.0, "cy": 100.0,
                     "area_px": 200.0, "n_masks_in_frame": 1, "intensity_mean": 100.0,
                     "area_um2": 50.0})
    for r in rows:
        r.setdefault("area_um2", r["area_px"] * 0.25)
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: pd.DataFrame(rows))
    header, _ = _strip(fake, [1, 2, 3, 50], added=[50])
    assert "by POSITION" in header and "50" in header


# --- the merged tool's dispatch -------------------------------------------------
# get_filmstrip and get_filmstrip_family used to be two tools. They were merged
# because the single-track one was never being reached for: a names-only tool list
# made "the filmstrip cluster" look interchangeable, and the family form with a
# one-element list had quietly become the tool everyone actually called. The two
# BACKENDS still differ, though -- one mask vs a member set -- so the dispatch is
# now the thing that has to stay honest.

def test_track_id_and_track_ids_are_not_interchangeable(monkeypatch, fake):
    """track_ids=[N] adds N's recorded daughters; track_id=N does not.

    Merging the two tools would be a silent behaviour change if a scalar id started
    pulling in daughters -- a reviewer asking to see ONE mask would get a crop that
    re-centres on objects the tracker chose for them.
    """
    monkeypatch.setattr(cell_mcp_server, "_lineage", lambda w: {1: {"daughters": [2, 3]}})
    members, _ = cell_mcp_server._resolve_family(fake, [1])
    assert members == [1, 2, 3]

    seen = {}

    def spy_single(well, track_id, *a, **k):
        seen["single"] = track_id
        return "hdr", []

    def spy_family(well, members, *a, **k):
        seen["family"] = list(members)
        return "hdr", []

    monkeypatch.setattr(cell_mcp_server, "_filmstrip_frames", spy_single)
    monkeypatch.setattr(cell_mcp_server, "_family_filmstrip_frames", spy_family)

    cell_mcp_server.follow_cells_over_time(fake, track_id=1)
    assert seen == {"single": 1}

    seen.clear()
    cell_mcp_server.follow_cells_over_time(fake, track_ids=[1])
    assert seen == {"family": [1, 2, 3]}


def test_neither_or_both_is_refused_rather_than_guessed(fake):
    """Picking one for the caller would pick the wrong backend half the time."""
    with pytest.raises(ValueError, match="exactly one"):
        cell_mcp_server.follow_cells_over_time(fake)
    with pytest.raises(ValueError, match="exactly one"):
        cell_mcp_server.follow_cells_over_time(fake, track_id=1, track_ids=[1])


def test_a_single_track_still_defaults_to_the_60um_crop(monkeypatch, fake):
    """crop_um=None means auto-fit for a member set but 60 for one mask -- the old
    get_filmstrip default. Passing None straight through would hand _filmstrip_frames
    a None where it wants a float."""
    seen = {}

    def spy(well, tid, sf, ef, *, crop_um, **k):
        seen["crop"] = crop_um
        return "hdr", []

    monkeypatch.setattr(cell_mcp_server, "_filmstrip_frames", spy)
    cell_mcp_server.follow_cells_over_time(fake, track_id=1)
    assert seen["crop"] == 60.0
