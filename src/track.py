"""Deterministic frame-to-frame linking of cell masks into lineage tracks."""

from dataclasses import dataclass, field

from src.segment import CellMask


@dataclass
class TrackNode:
    track_id: int
    parent_id: int | None
    frame: int
    mask: CellMask
    children: list[int] = field(default_factory=list)  # track_ids spawned from this node


def link_frames(masks_by_frame: dict[int, list[CellMask]], iou_threshold: float = 0.3) -> list[TrackNode]:
    """Match cell masks across consecutive frames by IoU/centroid distance.

    One mask matching one mask next frame extends the same track_id.
    One mask matching multiple masks next frame is a split event (new track_ids for each child,
    parent_id set to the originating track).
    """
    raise NotImplementedError
