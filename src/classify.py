"""Lineage graph rules: classify each split/end event as normal, anomalous, or ambiguous."""

import math
from dataclasses import dataclass
from enum import Enum

from src.track import TrackNode


class EventType(str, Enum):
    NORMAL_SPLIT = "normal_split"
    FAILED_SPLIT = "failed_split"
    MULTI_WAY_SPLIT = "multi_way_split"
    DEATH = "death"
    AMBIGUOUS = "ambiguous"


# Centroid distance (px) from any frame boundary below which an event is flagged near_edge.
# Partial visibility at the image boundary produces messier/more uncertain classifications
# (see 2026-07-04 finding: a near-edge event got the most complex label stack in the dataset).
# Decision: flag, don't exclude -- keep near-edge splits in total confirmed-split counts (the
# division is still real) but exclude them from anomaly-subtype-rate analysis downstream.
NEAR_EDGE_MARGIN_PX = 100


@dataclass
class LineageEvent:
    track_id: int
    parent_id: int | None
    frame: int
    event_type: EventType
    classification_source: str  # "rule", or the resolved vision model name (e.g. "claude-haiku-4-5", a GPT deployment name)
    confidence: float           # current best estimate (tracker initially; AI review overwrites on review).
                                 # written to CSV as "ai_confidence" -- see docs/output_schema.md
    tracker_confidence: float | None = None  # original trackastra persistence score, never overwritten.
                                              # written to CSV as "tracker_persistence_score"
    centroid: tuple[float, float] | None = None  # (cx, cy) of parent cell at split frame
    ai_notes: str | None = None              # "split_type: reason" from AI vision review
    raw_ai_confidence: float | None = None  # model's self-reported confidence before any post-hoc floor is applied
    review_error: bool = False              # True when the vision API call failed and the event was passed through unchanged
    bleach_risk: float | None = None  # frame / total_frames; proxy for photobleaching accumulation
    # ACD division classification (populated by review_ambiguous's combined verify+classify call)
    acd_division_type: str | None = None  # bipolar / tripolar / multipolar / unknown
    misaligned_chromosomes: bool | None = None
    lagging_chromosome: bool | None = None
    anaphase_bridge: bool | None = None
    micronucleus: bool | None = None
    binucleation: bool | None = None  # one cell body, two nuclei that don't progressively separate
    anomaly_notes: str | None = None  # interesting anomaly flagged for case study
    likely_division_dropout: bool | None = None  # vision review of a DEATH event suspects the
        # track end is a tracking/segmentation failure during mitotic entry (prophase chromatin
        # condensation obscuring the mask), not genuine cell death. None if never reviewed
        # (see review_deaths in review.py). Backlog item 23/27, 2026-07-10.
    near_edge: bool | None = None  # centroid within NEAR_EDGE_MARGIN_PX of any frame boundary
    cell_area_px: float | None = None  # parent cell's Cellpose mask area at the split frame
    cell_size_um2: float | None = None  # cell_area_px converted via per-acquisition pixel size, if known
    neighbor_distance_px: float | None = None  # centroid distance to the neighbor cell mask
        # picked by _nearest_neighbor_info (see docstring there -- smallest edge-to-edge gap,
        # not necessarily the closest centroid) in the same frame this event's
        # centroid/cell_area_px were measured from (not event.frame, which for splits is one
        # frame later -- see classify_events). None if no other cell mask exists in that frame.
        # Used by review.py to size the vision-review marker so it can't enclose a
        # simultaneously-dividing neighbor (2026-07-08 marker spike; edge-gap selection added
        # 2026-07-11 after that spike's radius formula still misattributed a large nearby
        # neighbor at a nominally "safe" raw distance -- see neighbor_area_px below).
    neighbor_area_px: float | None = None  # that same neighbor's own Cellpose mask area (pixel
        # count). A big neighbor can encroach on the marker box even at a raw centroid distance
        # that looks safe for a same-sized neighbor -- adaptive_radius (src/review.py) uses this
        # to size the box off the actual gap between the candidate's centroid and the neighbor's
        # OWN edge, not just centroid-to-centroid distance. None if no other cell mask exists.
    eccentricity: float | None = None  # regionprops shape descriptor at the same frame as
        # centroid/cell_area_px, 0 (circle) to ~1 (elongated) -- spike, no vision review,
        # not yet validated as a real/noise or anomaly signal (2026-07-09).
    solidity: float | None = None  # area / convex_hull_area, same frame -- 1.0 fully convex,
        # lower means concave/irregular outline (e.g. mid-division pinching, blebbing).
    split_type: str | None = None  # the vision model's own characterization of a confirmed real
        # split: "symmetric" | "asymmetric" | "multi_way" | "failed". Independent of
        # `event_type`/`split_topology`, which come from Trackastra's lineage-graph topology --
        # the two can disagree (e.g. tracker only found 2 children but the model visually saw
        # 3 daughters and reports "multi_way"). A confirmed "failed" split_type gets promoted to
        # `EventType.FAILED_SPLIT` in review.py (see 2026-07-09 un-shelving of that event type).


def _build_frame_index(tracks: list[TrackNode]) -> dict[int, list[TrackNode]]:
    index: dict[int, list[TrackNode]] = {}
    for node in tracks:
        index.setdefault(node.frame, []).append(node)
    return index


def _nearest_neighbor_info(frame_nodes: list[TrackNode], own: TrackNode) -> tuple[float | None, float | None]:
    """(centroid distance, area) of the neighbor with the smallest edge-to-edge gap to `own`.

    Not simply the nearest centroid -- a farther-but-larger neighbor can encroach on the
    candidate's marker box more than a small, nearby one. Approximates each neighbor as a
    circle of its own Cellpose area (radius = sqrt(area / pi)) and picks whichever neighbor
    minimizes (centroid_distance - neighbor_radius), i.e. the neighbor whose own body comes
    closest to `own`'s centroid. Root-caused 2026-07-11: a real misattribution case had a raw
    centroid distance of 24.5px that looked "safe" under the old pure-distance formula, but the
    neighbor's own mask was large enough that its edge nearly reached `own`'s centroid.
    """
    ox, oy = own.mask.centroid
    best_gap: float | None = None
    best: tuple[float, float] | None = None
    for node in frame_nodes:
        if node is own:
            continue
        nx, ny = node.mask.centroid
        d = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
        n_radius = math.sqrt(node.mask.area / math.pi) if node.mask.area else 0.0
        gap = d - n_radius
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = (d, node.mask.area)
    return best if best is not None else (None, None)


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


def classify_track_ends(
    tracks: list[TrackNode],
    last_frame: int,
    min_track_frames: int = 5,
    confidence_max_frames: int = 20,
) -> list[LineageEvent]:
    """Emit a DEATH event for tracks that stop -- without splitting -- before the video ends.

    A track's last node has no children in two distinct cases: it split (already
    covered by classify_events) or it just stops -- the cell died, drifted out of
    the focal plane, or Cellpose/Trackastra simply lost it. Tracking topology alone
    can't tell those apart, and this dataset's dominant failure mode (documented
    elsewhere in this project) is the tracker losing a healthy cell for a few frames,
    not real death -- so a raw "did it stop" signal isn't trustworthy on its own,
    same lesson classify_events already learned for splits via daughter persistence:

    1. Tracks shorter than min_track_frames are dropped entirely, not emitted. A
       track that only ever existed for 1-2 frames before vanishing is a segmentation
       blip, not a plausible death candidate -- no evidence value at all.
    2. Surviving tracks get a persistence-style confidence: min(1.0, track_duration /
       confidence_max_frames). A track that barely clears the cutoff is a much
       weaker death candidate than one alive for confidence_max_frames+ -- both are
       kept (unlike case 1), but a downstream reader can still filter on confidence.

    ROI_EXIT (a stop near the frame boundary) is deliberately not classified here at
    all -- pipeline.py drops those before they reach this function's output, since a
    cell walking out of the field of view carries no biological information worth
    reporting, unlike an unexplained stop away from the edge.

    Tracks still alive at last_frame are excluded -- the video simply ended first,
    that's not a death.

    Not validated against ground truth (unlike splits, this project has no death/
    track-end ground truth to score against) -- treat min_track_frames/
    confidence_max_frames as an untuned first pass, not a validated threshold.
    """
    last_node_by_track: dict[int, TrackNode] = {}
    first_frame_by_track: dict[int, int] = {}
    parent_of_track: dict[int, int] = {}
    for node in tracks:
        prev = last_node_by_track.get(node.track_id)
        if prev is None or node.frame > prev.frame:
            last_node_by_track[node.track_id] = node
        first = first_frame_by_track.get(node.track_id)
        if first is None or node.frame < first:
            first_frame_by_track[node.track_id] = node.frame
        if node.parent_id is not None:
            parent_of_track[node.track_id] = node.parent_id

    frame_index = _build_frame_index(tracks)
    events = []
    for track_id, node in last_node_by_track.items():
        if node.children or node.frame >= last_frame:
            continue
        duration = node.frame - first_frame_by_track[track_id] + 1
        if duration < min_track_frames:
            continue
        neighbor_distance_px, neighbor_area_px = _nearest_neighbor_info(frame_index.get(node.frame, []), node)
        events.append(
            LineageEvent(
                track_id=track_id,
                parent_id=parent_of_track.get(track_id),
                frame=node.frame,
                event_type=EventType.DEATH,
                classification_source="rule",
                confidence=min(1.0, duration / confidence_max_frames),
                centroid=node.mask.centroid,
                cell_area_px=node.mask.area,
                neighbor_distance_px=neighbor_distance_px,
                neighbor_area_px=neighbor_area_px,
                eccentricity=node.mask.eccentricity,
                solidity=node.mask.solidity,
            )
        )
    return events


def classify_events(
    tracks: list[TrackNode],
    roi: tuple[int, int, int, int] | None,
    cascade_window: int = 20,
    confidence_max_frames: int = 10,
) -> list[LineageEvent]:
    """Walk the lineage graph and emit one LineageEvent per split point.

    v1 scope: only split events (NORMAL_SPLIT / MULTI_WAY_SPLIT) are classified here.
    Track ends (DEATH) are handled separately by classify_track_ends, since they need
    to know where the video ends, not just the lineage graph. Failed splits and
    ambiguous-abnormality-only tracks are still deferred -- so `roi` is accepted but
    unused for now.

    A node with exactly 1 child is not a split at all (a track-ID continuation
    artifact -- see the 1-child branch below) and emits no event.

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
    frame_index = _build_frame_index(tracks)
    origin_frame: dict[int, int] = {}
    events = []

    for node in tracks:
        if not node.children:
            continue
        if len(node.children) == 1:
            # Not a real division -- a single-child node is a track-ID continuation
            # artifact (e.g. from the gap-bridging fix occasionally merging one real
            # daughter into an unrelated track), not a split. Previously fell through
            # to the `else` branch below and got mislabeled MULTI_WAY_SPLIT (found
            # 2026-07-06: every multi_way_split row in one run was a singleton).
            # Propagate any existing origin so cascade-noise detection still works
            # correctly for a later real split on this continued lineage.
            child_track_id = node.children[0]
            if node.track_id in origin_frame:
                origin_frame[child_track_id] = origin_frame[node.track_id]
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
        # neighbor distance is measured at node.frame, the same frame centroid/cell_area_px
        # come from -- NOT split_frame (node.frame + 1), which is one frame later.
        neighbor_distance_px, neighbor_area_px = _nearest_neighbor_info(frame_index.get(node.frame, []), node)
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
                    cell_area_px=node.mask.area,
                    neighbor_distance_px=neighbor_distance_px,
                    neighbor_area_px=neighbor_area_px,
                    eccentricity=node.mask.eccentricity,
                    solidity=node.mask.solidity,
                )
            )
    return events
