import numpy as np

from src.segment import CellMask
from src.track import link_frames


def make_cell(frame: int, mask_id: int, y0: int, x0: int, h: int, w: int) -> CellMask:
    """A solid rectangular cell -- makes overlap fractions easy to predict by hand."""
    return CellMask(
        frame=frame,
        mask_id=mask_id,
        bbox=(y0, y0 + h, x0, x0 + w),
        local_mask=np.ones((h, w), dtype=bool),
        centroid=(x0 + w / 2, y0 + h / 2),
        area=float(h * w),
    )


def test_single_cell_continues_same_track_across_frames():
    masks_by_frame = {
        0: [make_cell(0, 1, y0=0, x0=0, h=10, w=10)],
        1: [make_cell(1, 1, y0=0, x0=0, h=10, w=10)],  # identical position -> same cell
    }
    tracks = link_frames(masks_by_frame)

    track_ids = {n.track_id for n in tracks}
    assert track_ids == {0}  # one continuous track, no split
    frames = sorted(n.frame for n in tracks if n.track_id == 0)
    assert frames == [0, 1]


def test_one_cell_splitting_into_two_is_detected():
    # frame 0: one tall cell. frame 1: it has split into a top half and bottom half.
    masks_by_frame = {
        0: [make_cell(0, 1, y0=0, x0=0, h=20, w=10)],
        1: [
            make_cell(1, 1, y0=0, x0=0, h=10, w=10),
            make_cell(1, 2, y0=10, x0=0, h=10, w=10),
        ],
    }
    tracks = link_frames(masks_by_frame)

    parent = next(n for n in tracks if n.frame == 0)
    assert len(parent.children) == 2

    child_nodes = [n for n in tracks if n.frame == 1]
    assert len(child_nodes) == 2
    assert {n.track_id for n in child_nodes} == set(parent.children)
    assert all(n.parent_id == parent.track_id for n in child_nodes)


def test_unrelated_cell_entering_frame_gets_a_new_track():
    masks_by_frame = {
        0: [make_cell(0, 1, y0=0, x0=0, h=10, w=10)],
        1: [
            make_cell(1, 1, y0=0, x0=0, h=10, w=10),  # same cell, continues
            make_cell(1, 2, y0=500, x0=500, h=10, w=10),  # far away, unrelated, new track
        ],
    }
    tracks = link_frames(masks_by_frame)

    frame1_nodes = [n for n in tracks if n.frame == 1]
    assert len(frame1_nodes) == 2
    new_track_ids = {n.track_id for n in frame1_nodes} - {0}
    assert len(new_track_ids) == 1  # one genuinely new track_id for the unrelated cell


def test_cell_disappearing_does_not_error():
    masks_by_frame = {
        0: [make_cell(0, 1, y0=0, x0=0, h=10, w=10)],
        1: [],  # cell died or exited frame
    }
    tracks = link_frames(masks_by_frame)
    assert len(tracks) == 1  # only the frame-0 node; nothing carries into frame 1


def test_empty_input_returns_empty_list():
    assert link_frames({}) == []
