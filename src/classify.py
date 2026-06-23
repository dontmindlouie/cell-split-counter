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


def classify_events(tracks: list[TrackNode], roi: tuple[int, int, int, int]) -> list[LineageEvent]:
    """Walk the lineage graph and emit one LineageEvent per split/end point.

    Anomaly candidates that the rules can't confidently resolve are marked AMBIGUOUS
    and left for src.review.review_ambiguous to adjudicate with Claude vision.
    """
    raise NotImplementedError
