"""Deterministic frame-to-frame linking of cell masks into lineage tracks."""

from dataclasses import dataclass, field

import numpy as np

from src.segment import CellMask


@dataclass
class TrackNode:
    track_id: int
    parent_id: int | None
    frame: int
    mask: CellMask
    children: list[int] = field(default_factory=list)  # track_ids spawned from this node


def _overlap_fraction(a: CellMask, b: CellMask) -> float:
    """Intersection over the smaller mask's area.

    Plain IoU under-matches a dividing cell: each daughter is much smaller than the
    parent, so intersection / union stays low even when the daughter clearly sits
    where the parent used to be. Normalizing by the smaller area keeps that match
    strong enough to detect.

    Masks are stored cropped to their own bounding box (see CellMask), so the
    intersection has to be computed over the overlapping region of the two boxes,
    not by directly comparing the arrays.
    """
    ay0, ay1, ax0, ax1 = a.bbox
    by0, by1, bx0, bx1 = b.bbox
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy0 >= iy1 or ix0 >= ix1:
        return 0.0

    sub_a = a.local_mask[iy0 - ay0 : iy1 - ay0, ix0 - ax0 : ix1 - ax0]
    sub_b = b.local_mask[iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0]
    intersection = np.logical_and(sub_a, sub_b).sum()
    smaller_area = min(a.area, b.area)
    return 0.0 if smaller_area == 0 else intersection / smaller_area


def link_frames(masks_by_frame: dict[int, list[CellMask]], overlap_threshold: float = 0.3) -> list[TrackNode]:
    """Match cell masks across consecutive frames by overlap fraction.

    Each cur-frame cell is assigned to whichever prev-frame cell it overlaps most.
    A prev cell claimed by exactly one cur cell continues its track_id. A prev cell
    claimed by 2+ cur cells is a split: each cur cell gets a new track_id, with
    parent_id set to the prev cell's track. A prev cell claimed by no cur cell
    simply stops (death or exited frame). A cur cell unmatched to any prev cell is
    a new track (entered frame).
    """
    nodes: list[TrackNode] = []
    next_track_id = 0

    frame_indices = sorted(masks_by_frame.keys())
    if not frame_indices:
        return nodes

    # track_id -> (TrackNode, CellMask) for that track's current frame
    live: dict[int, TrackNode] = {}
    for cell in masks_by_frame[frame_indices[0]]:
        node = TrackNode(track_id=next_track_id, parent_id=None, frame=cell.frame, mask=cell)
        nodes.append(node)
        live[next_track_id] = node
        next_track_id += 1

    for prev_idx, cur_idx in zip(frame_indices, frame_indices[1:]):
        prev_cells = masks_by_frame[prev_idx]
        cur_cells = masks_by_frame[cur_idx]
        prev_tids = [tid for tid, n in live.items() if n.frame == prev_idx]
        prev_cell_by_tid = {tid: live[tid].mask for tid in prev_tids}

        # best prev match (by track_id) for each cur cell, or None if no good overlap
        best_parent: list[int | None] = []
        for c in cur_cells:
            scores = [(_overlap_fraction(prev_cell_by_tid[tid], c), tid) for tid in prev_tids]
            best = max(scores, default=(0.0, None))
            best_parent.append(best[1] if best[0] >= overlap_threshold else None)

        children_by_parent: dict[int, list[int]] = {}
        for cur_i, tid in enumerate(best_parent):
            if tid is not None:
                children_by_parent.setdefault(tid, []).append(cur_i)

        new_live: dict[int, TrackNode] = {}
        for parent_tid, cur_indices in children_by_parent.items():
            parent_node = live[parent_tid]
            if len(cur_indices) == 1:
                cell = cur_cells[cur_indices[0]]
                node = TrackNode(track_id=parent_tid, parent_id=None, frame=cur_idx, mask=cell)
                nodes.append(node)
                new_live[parent_tid] = node
            else:
                for cur_i in cur_indices:
                    cell = cur_cells[cur_i]
                    node = TrackNode(track_id=next_track_id, parent_id=parent_tid, frame=cur_idx, mask=cell)
                    nodes.append(node)
                    new_live[next_track_id] = node
                    parent_node.children.append(next_track_id)
                    next_track_id += 1

        for cur_i, c in enumerate(cur_cells):
            if best_parent[cur_i] is None:
                node = TrackNode(track_id=next_track_id, parent_id=None, frame=cur_idx, mask=c)
                nodes.append(node)
                new_live[next_track_id] = node
                next_track_id += 1

        live = new_live

    return nodes
