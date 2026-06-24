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


def classify_events(tracks: list[TrackNode], roi: tuple[int, int, int, int] | None) -> list[LineageEvent]:
    """Walk the lineage graph and emit one LineageEvent per split point.

    v1 scope: only split events (NORMAL_SPLIT / MULTI_WAY_SPLIT) are classified.
    Failed splits, ROI exits, death, and abnormality review are deferred -- see
    project scope notes -- so `roi` is accepted but unused for now.
    """
    events = []
    for node in tracks:
        if not node.children:
            continue
        event_type = EventType.NORMAL_SPLIT if len(node.children) == 2 else EventType.MULTI_WAY_SPLIT
        for child_track_id in node.children:
            events.append(
                LineageEvent(
                    track_id=child_track_id,
                    parent_id=node.track_id,
                    frame=node.frame + 1,  # daughters first appear in the next frame
                    event_type=event_type,
                    classification_source="rule",
                    confidence=1.0,
                )
            )
    return events
