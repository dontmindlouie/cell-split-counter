"""Lineage graph rules: classify each split/end event as normal, anomalous, or ambiguous."""

from dataclasses import dataclass
from enum import Enum

from src.track import TrackNode


class EventType(str, Enum):
    NORMAL_SPLIT = "normal_split"
    FAILED_SPLIT = "failed_split"
    MULTI_WAY_SPLIT = "multi_way_split"
    ROI_EXIT = "roi_exit"
    DEATH = "death"
    AMBIGUOUS = "ambiguous"


@dataclass
class LineageEvent:
    track_id: int
    parent_id: int | None
    frame: int
    event_type: EventType
    classification_source: str  # "rule" or "claude"
    confidence: float


def classify_events(
    tracks: list[TrackNode], roi: tuple[int, int, int, int] | None, cascade_window: int = 20
) -> list[LineageEvent]:
    """Walk the lineage graph and emit one LineageEvent per split point.

    v1 scope: only split events (NORMAL_SPLIT / MULTI_WAY_SPLIT) are classified.
    Failed splits, ROI exits, death, and abnormality review are deferred -- see
    project scope notes -- so `roi` is accepted but unused for now.

    A real division's two daughters sit right next to each other afterward, and their
    Cellpose masks keep flickering (touching/separating slightly) for many frames --
    each flicker looks like "another split" to the tracker, producing a long cascade
    of spurious events for one real division. If a track that was itself born from a
    split splits again within cascade_window frames, treat it as noise from the same
    event rather than a new one: don't emit it, but keep propagating its origin frame
    so deeper cascades are suppressed too. tracks must be in non-decreasing frame order
    (true as built by link_frames) for the propagation below to see parents before
    their children.
    """
    origin_frame: dict[int, int] = {}  # track_id -> frame of the split that "really" produced it
    events = []
    for node in tracks:
        if not node.children:
            continue
        split_frame = node.frame + 1  # daughters first appear in the next frame
        parent_origin = origin_frame.get(node.track_id)
        is_cascade_noise = parent_origin is not None and (split_frame - parent_origin) <= cascade_window

        if is_cascade_noise:
            for child_track_id in node.children:
                origin_frame[child_track_id] = parent_origin
            continue

        event_type = EventType.NORMAL_SPLIT if len(node.children) == 2 else EventType.MULTI_WAY_SPLIT
        for child_track_id in node.children:
            origin_frame[child_track_id] = split_frame
            events.append(
                LineageEvent(
                    track_id=child_track_id,
                    parent_id=node.track_id,
                    frame=split_frame,
                    event_type=event_type,
                    classification_source="rule",
                    confidence=1.0,
                )
            )
    return events
