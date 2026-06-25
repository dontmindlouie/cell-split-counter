from src.classify import EventType, classify_events
from src.segment import CellMask
from src.track import TrackNode


def fake_mask(frame: int) -> CellMask:
    return CellMask(frame=frame, mask_id=0, bbox=(0, 1, 0, 1), local_mask=None, centroid=(0, 0), area=1.0)


def node(track_id, parent_id, frame, children=None):
    return TrackNode(track_id=track_id, parent_id=parent_id, frame=frame, mask=fake_mask(frame), children=children or [])


def test_no_children_produces_no_events():
    tracks = [node(track_id=1, parent_id=None, frame=0)]
    assert classify_events(tracks, None) == []


def test_normal_split_emits_one_event_per_daughter():
    tracks = [node(track_id=1, parent_id=None, frame=10, children=[2, 3])]
    events = classify_events(tracks, None)

    assert len(events) == 2
    assert {e.track_id for e in events} == {2, 3}
    assert all(e.parent_id == 1 for e in events)
    assert all(e.frame == 11 for e in events)
    assert all(e.event_type == EventType.NORMAL_SPLIT for e in events)


def test_three_way_split_is_classified_as_multi_way():
    tracks = [node(track_id=1, parent_id=None, frame=10, children=[2, 3, 4])]
    events = classify_events(tracks, None)

    assert len(events) == 3
    assert all(e.event_type == EventType.MULTI_WAY_SPLIT for e in events)


def test_cascade_within_window_is_suppressed_but_descendant_origin_still_tracked():
    # track 1 splits into 2,3 at frame 11 (real event).
    # track 2 'splits' again at frame 12 into 4,5 -- mask-flicker noise, same underlying event.
    # track 4 splits again at frame 41 -- far enough later (>20 frames from the ORIGINAL frame 11)
    # to be a real, distinct second division.
    tracks = [
        node(track_id=1, parent_id=None, frame=10, children=[2, 3]),
        node(track_id=2, parent_id=1, frame=11, children=[4, 5]),
        node(track_id=4, parent_id=2, frame=40, children=[6, 7]),
    ]
    events = classify_events(tracks, None, cascade_window=20)

    assert [e.track_id for e in events] == [2, 3, 6, 7]
    assert [e.frame for e in events] == [11, 11, 41, 41]
    assert events[2].parent_id == 4 and events[3].parent_id == 4


def test_cascade_just_outside_window_is_not_suppressed():
    # same as above, but the second split is only 15 frames after the first -- still within
    # the 20-frame cascade window -- versus one that's 25 frames after, which is outside it.
    tracks = [
        node(track_id=1, parent_id=None, frame=10, children=[2, 3]),  # split at frame 11
        node(track_id=2, parent_id=1, frame=34, children=[4, 5]),  # split at frame 35: 35-11=24 > 20
    ]
    events = classify_events(tracks, None, cascade_window=20)

    assert len(events) == 4  # both splits kept, since the second is outside the cascade window
    assert {e.track_id for e in events} == {2, 3, 4, 5}
