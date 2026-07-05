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
    confidence: float           # current best estimate (tracker initially; Claude overwrites on review).
                                 # written to CSV as "claude_confidence" -- see docs/output_schema.md
    tracker_confidence: float | None = None  # original trackastra persistence score, never overwritten.
                                              # written to CSV as "tracker_persistence_score"
    centroid: tuple[float, float] | None = None  # (cx, cy) of parent cell at split frame
    claude_notes: str | None = None  # "split_type: reason" from Claude vision review
    bleach_risk: float | None = None  # frame / total_frames; proxy for photobleaching accumulation
    # ACD division classification (populated by review_ambiguous's combined verify+classify call)
    acd_division_type: str | None = None  # bipolar / tripolar / multipolar / unknown
    misaligned_chromosomes: bool | None = None
    lagging_chromosome: bool | None = None
    anaphase_bridge: bool | None = None
    micronucleus: bool | None = None
    anomaly_notes: str | None = None  # interesting anomaly flagged for case study


def _daughter_persistence(
    node_by_tid_frame: dict[tuple[int, int], TrackNode],
    child_tids: list[int],
    split_frame: int,
    max_lookahead: int,
) -> int:
    """Count how many consecutive frames ALL daughters remain as separate tracks.

    Returns an int in [0, max_lookahead]. 0 means at least one daughter vanishes
    at split_frame itself (shouldn't happen normally). A shape-change false positive
    typically returns 1-2; a real division typically returns max_lookahead or close.
    """
    for lookahead in range(max_lookahead):
        f = split_frame + lookahead
        if not all((tid, f) in node_by_tid_frame for tid in child_tids):
            return lookahead
    return max_lookahead


def classify_events(
    tracks: list[TrackNode],
    roi: tuple[int, int, int, int] | None,
    cascade_window: int = 20,
    confidence_max_frames: int = 10,
) -> list[LineageEvent]:
    """Walk the lineage graph and emit one LineageEvent per split point.

    v1 scope: only split events (NORMAL_SPLIT / MULTI_WAY_SPLIT) are classified.
    Failed splits, ROI exits, death, and abnormality review are deferred -- see
    project scope notes -- so `roi` is accepted but unused for now.

    Two mechanisms:

    1. Cascade noise (binary suppress): a real division's daughters sit adjacent and
       their Cellpose masks flicker (touching/separating slightly) across many frames.
       If a track born from a split re-splits within cascade_window frames, treat as
       noise. Origin frame is propagated to daughters so deeper cascades are suppressed.

    2. Daughter persistence (confidence score only, no suppression): a shape-change or
       z-plane false positive briefly produces 2 masks then reverts to 1 the next frame;
       real daughters stay separate for many frames. All splits are kept so recall stays
       at 100%, but scored:
         confidence = min(1.0, persistence_frames / confidence_max_frames)
       Low-confidence events (confidence < 1.0) are routed to Claude vision review.

    tracks must be in non-decreasing frame order (true as built by link_frames).
    """
    node_by_tid_frame: dict[tuple[int, int], TrackNode] = {
        (n.track_id, n.frame): n for n in tracks
    }
    origin_frame: dict[int, int] = {}
    events = []

    for node in tracks:
        if not node.children:
            continue
        split_frame = node.frame + 1
        parent_origin = origin_frame.get(node.track_id)
        is_cascade_noise = parent_origin is not None and (split_frame - parent_origin) <= cascade_window

        if is_cascade_noise:
            for child_track_id in node.children:
                origin_frame[child_track_id] = parent_origin
            continue

        persistence = _daughter_persistence(
            node_by_tid_frame, node.children, split_frame, confidence_max_frames
        )
        confidence = min(1.0, persistence / confidence_max_frames)
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
                    confidence=confidence,
                    centroid=node.mask.centroid,
                )
            )
    return events
