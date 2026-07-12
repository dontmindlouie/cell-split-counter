import math

from src.classify import EventType, _nearest_neighbor_info, classify_events, classify_track_ends
from src.segment import CellMask
from src.track import TrackNode


def fake_mask(cx: float = 0.0, cy: float = 0.0, area: float = 1.0) -> CellMask:
    return CellMask(frame=0, mask_id=0, bbox=(0, 1, 0, 1), local_mask=None, centroid=(cx, cy), area=area)


def node(track_id, parent_id, frame, children=None, cx=0.0, cy=0.0, area=1.0):
    return TrackNode(
        track_id=track_id,
        parent_id=parent_id,
        frame=frame,
        mask=fake_mask(cx, cy, area=area),
        children=children or [],
    )


def split_nodes(parent_tid, child_tids, parent_frame, persist=5, cx_offsets=None):
    """Build a parent node + daughter nodes persisting for `persist` frames.

    cx_offsets: list of x-centroid offsets per daughter (default: spread daughters apart).
    Used to control daughter separation for persistence tests.
    """
    if cx_offsets is None:
        cx_offsets = [i * 50.0 for i in range(len(child_tids))]
    result = [node(parent_tid, None, parent_frame, children=list(child_tids))]
    split_frame = parent_frame + 1
    for f in range(split_frame, split_frame + persist):
        for i, c in enumerate(child_tids):
            result.append(node(c, parent_tid if f == split_frame else None, f, cx=cx_offsets[i]))
    return result


def test_no_children_produces_no_events():
    tracks = [node(track_id=1, parent_id=None, frame=0)]
    assert classify_events(tracks, None) == []


def test_normal_split_emits_one_event_per_daughter():
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=5)
    events = classify_events(tracks, None)

    assert len(events) == 2
    assert {e.track_id for e in events} == {2, 3}
    assert all(e.parent_id == 1 for e in events)
    assert all(e.frame == 11 for e in events)
    assert all(e.event_type == EventType.NORMAL_SPLIT for e in events)


def test_three_way_split_is_classified_as_multi_way():
    tracks = split_nodes(1, [2, 3, 4], parent_frame=10, persist=5)
    events = classify_events(tracks, None)

    assert len(events) == 3
    assert all(e.event_type == EventType.MULTI_WAY_SPLIT for e in events)


def test_single_child_node_emits_no_event():
    # A node with exactly 1 child is a track-ID continuation artifact, not a real
    # division -- previously mislabeled MULTI_WAY_SPLIT by the `else` branch
    # catching len(children) == 1 (found 2026-07-06).
    tracks = split_nodes(1, [2], parent_frame=10, persist=5)
    events = classify_events(tracks, None)

    assert events == []


def test_single_child_continuation_still_propagates_origin_for_cascade_suppression():
    # track 1 splits into 2,3 at frame 11 (real event). track 2 then has a
    # single-child "continuation" to track 4 at frame 12 (not a real split --
    # should emit nothing). track 4 then re-splits at frame 13, well within
    # cascade_window of the ORIGINAL frame-11 split -- should still be suppressed
    # as cascade noise via the propagated origin, not treated as a fresh event.
    tracks = (
        split_nodes(1, [2, 3], parent_frame=10, persist=1)
        + [node(2, None, 12, children=[4])]
        + split_nodes(4, [5, 6], parent_frame=12, persist=5)
        + [node(3, None, f) for f in range(12, 17)]
    )
    events = classify_events(tracks, None, cascade_window=20)

    assert {e.track_id for e in events} == {2, 3}
    assert {e.frame for e in events} == {11}


def test_cascade_within_window_is_suppressed_but_descendant_origin_still_tracked():
    # track 1 splits into 2,3 at frame 11 (real event).
    # track 2 'splits' again at frame 12 into 4,5 -- mask-flicker noise, same underlying event.
    # track 4 splits again at frame 41 -- far enough later (>20 frames from the ORIGINAL frame 11)
    # to be a real, distinct second division.
    # min_daughter_persistence=1 isolates cascade behaviour from persistence checking.
    tracks = (
        split_nodes(1, [2, 3], parent_frame=10, persist=1)  # daughters at frame 11 only
        + split_nodes(2, [4, 5], parent_frame=11, persist=1)
        + split_nodes(4, [6, 7], parent_frame=40, persist=5)
        + [node(3, None, f) for f in range(12, 16)]   # track 3 continues
    )
    events = classify_events(tracks, None, cascade_window=20)

    assert {e.track_id for e in events} == {2, 3, 6, 7}
    assert {e.frame for e in events} == {11, 41}
    split_1_events = [e for e in events if e.frame == 11]
    assert all(e.parent_id == 1 for e in split_1_events)
    split_2_events = [e for e in events if e.frame == 41]
    assert all(e.parent_id == 4 for e in split_2_events)


def test_cascade_just_outside_window_is_not_suppressed():
    # Second split is 24 frames after the first (> cascade_window=20) -- should NOT be suppressed.
    tracks = (
        split_nodes(1, [2, 3], parent_frame=10, persist=5)
        + split_nodes(2, [4, 5], parent_frame=34, persist=5)  # split_frame=35, 35-11=24 > 20
        + [node(3, None, f) for f in range(12, 41)]
    )
    events = classify_events(tracks, None, cascade_window=20)

    assert len(events) == 4
    assert {e.track_id for e in events} == {2, 3, 4, 5}


def test_short_lived_daughters_still_emit_with_low_confidence():
    # Daughters survive only 1 frame -- below confidence_max_frames=10 -- kept but low confidence.
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=1)
    events = classify_events(tracks, None, confidence_max_frames=10)

    assert len(events) == 2
    assert all(e.confidence == 0.1 for e in events)


def test_persistent_daughters_pass_and_get_confidence_score():
    # Daughters survive confidence_max_frames=5 -- confidence should be 1.0.
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=5)
    events = classify_events(tracks, None, confidence_max_frames=5)

    assert len(events) == 2
    assert all(e.confidence == 1.0 for e in events)


def test_partial_persistence_gives_fractional_confidence():
    # Daughters survive 3 frames out of confidence_max_frames=6 -- confidence = 0.5.
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=3)
    events = classify_events(tracks, None, confidence_max_frames=6)

    assert len(events) == 2
    assert all(abs(e.confidence - 0.5) < 1e-6 for e in events)


def test_track_stopping_before_video_end_emits_death():
    tracks = [node(1, None, 0), node(1, None, 1), node(1, None, 2)]
    events = classify_track_ends(tracks, last_frame=10, min_track_frames=1)

    assert len(events) == 1
    e = events[0]
    assert e.track_id == 1
    assert e.frame == 2
    assert e.event_type == EventType.DEATH
    assert e.classification_source == "rule"


def test_track_alive_at_last_frame_emits_no_event():
    # The track's last node IS the video's last frame -- the video ended, not the cell.
    tracks = [node(1, None, 0), node(1, None, 1), node(1, None, 2)]
    events = classify_track_ends(tracks, last_frame=2, min_track_frames=1)

    assert events == []


def test_split_track_end_emits_no_death_event():
    # The node right before a split has children set -- classify_events already covers
    # it, classify_track_ends must not double-emit a DEATH for the same stop.
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=5)
    events = classify_track_ends(tracks, last_frame=100, min_track_frames=1)

    assert all(e.track_id != 1 for e in events)


def test_death_parent_id_traced_from_birth_node():
    # parent_id is only set on a track's birth-frame node -- classify_track_ends must
    # look it up from there, not from the (parent_id=None) final node.
    tracks = [node(2, parent_id=1, frame=5), node(2, None, 6), node(2, None, 7)]
    events = classify_track_ends(tracks, last_frame=100, min_track_frames=1)

    assert len(events) == 1
    assert events[0].parent_id == 1
    assert events[0].frame == 7


def test_multiple_tracks_evaluated_independently():
    tracks = [
        node(1, None, 0), node(1, None, 1), node(1, None, 2),  # dies at frame 2
        node(2, None, 0), node(2, None, 1), node(2, None, 5),  # still alive at video end
    ]
    events = classify_track_ends(tracks, last_frame=5, min_track_frames=1)

    assert len(events) == 1
    assert events[0].track_id == 1


def test_short_lived_track_end_is_dropped_entirely():
    # 3 frames total (0,1,2) is below the default min_track_frames=5 -- a segmentation
    # blip, not a plausible death candidate, so it should not appear at all.
    tracks = [node(1, None, 0), node(1, None, 1), node(1, None, 2)]
    events = classify_track_ends(tracks, last_frame=100)

    assert events == []


def test_track_end_confidence_scales_with_duration():
    # Track spans frames 0-9 (duration 10) out of confidence_max_frames=20 -> 0.5.
    tracks = [node(1, None, f) for f in range(10)]
    events = classify_track_ends(tracks, last_frame=100, min_track_frames=1, confidence_max_frames=20)

    assert len(events) == 1
    assert abs(events[0].confidence - 0.5) < 1e-6


def test_track_end_confidence_capped_at_one():
    # Track spans frames 0-24 (duration 25), above confidence_max_frames=20 -> capped at 1.0.
    tracks = [node(1, None, f) for f in range(25)]
    events = classify_track_ends(tracks, last_frame=100, min_track_frames=1, confidence_max_frames=20)

    assert len(events) == 1
    assert events[0].confidence == 1.0


def test_nearest_neighbor_info_returns_none_with_no_other_cells():
    own = node(1, None, 0, cx=0.0, cy=0.0)
    assert _nearest_neighbor_info([own], own) == (None, None)


def test_nearest_neighbor_info_picks_closest_centroid_when_sizes_equal():
    own = node(1, None, 0, cx=0.0, cy=0.0, area=100.0)
    near = node(2, None, 0, cx=10.0, cy=0.0, area=100.0)
    far = node(3, None, 0, cx=50.0, cy=0.0, area=100.0)
    dist, area = _nearest_neighbor_info([own, near, far], own)
    assert dist == 10.0
    assert area == 100.0


def test_nearest_neighbor_info_prefers_smaller_edge_gap_over_smaller_centroid_distance():
    # A farther but much larger neighbor can have less actual clearance (edge-to-edge)
    # than a closer but small one -- _nearest_neighbor_info must pick the neighbor with
    # the smaller true gap, not the smaller raw centroid distance. Mirrors the real
    # 2026-07-11 misattribution root cause where a "safe"-looking raw distance still
    # belonged to a neighbor large enough to nearly reach the candidate's centroid.
    own = node(1, None, 0, cx=0.0, cy=0.0)
    close_small = node(2, None, 0, cx=15.0, cy=0.0, area=1.0)  # gap ~= 15 - ~0.56 = ~14.4
    far_large = node(3, None, 0, cx=30.0, cy=0.0, area=math.pi * 25 ** 2)  # gap = 30 - 25 = 5
    dist, area = _nearest_neighbor_info([own, close_small, far_large], own)
    assert dist == 30.0
    assert area == far_large.mask.area


def test_classify_events_populates_neighbor_area_px():
    tracks = split_nodes(1, [2, 3], parent_frame=10, persist=5)
    tracks.append(node(99, None, frame=10, cx=20.0, cy=0.0, area=42.0))  # neighbor in the split frame
    events = classify_events(tracks, None)
    assert all(e.neighbor_area_px == 42.0 for e in events)


def test_classify_track_ends_populates_neighbor_area_px():
    tracks = [node(1, None, f) for f in range(10)]
    # Neighbor track 99 is present at frame 9 (the death frame being measured) but
    # stays alive past last_frame so it doesn't itself emit a death event.
    tracks.append(node(99, None, frame=9, cx=5.0, cy=0.0, area=77.0))
    tracks.append(node(99, None, frame=100, cx=5.0, cy=0.0, area=77.0))
    events = classify_track_ends(tracks, last_frame=100, min_track_frames=1)
    assert len(events) == 1
    assert events[0].neighbor_area_px == 77.0
