"""Deterministic frame-to-frame linking of cell masks into lineage tracks."""

from dataclasses import dataclass, field
from pathlib import Path

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


def _label_to_cellmask(label_map: np.ndarray, label_id: int, frame: int) -> CellMask:
    mask = label_map == label_id
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return CellMask(
        frame=frame,
        mask_id=label_id,
        bbox=(y0, y1, x0, x1),
        local_mask=mask[y0:y1, x0:x1].copy(),
        centroid=(float(xs.mean()), float(ys.mean())),
        area=float(mask.sum()),
    )


def _normalize_on_memmap(float_frames: np.ndarray, subsample: int = 4) -> np.ndarray:
    """Percentile-normalize a float32 memmap in-place.

    Replicates trackastra.utils.utils.normalize but operates on an already-float32
    disk-backed memmap, avoiding the 9GB RAM spike from astype(float32) on the full
    video at once. In-place ops page through the memmap without loading it all to RAM.
    """
    if subsample is not None and all(s > 64 * subsample for s in float_frames.shape[-2:]):
        y = float_frames[..., ::subsample, ::subsample]
    else:
        y = float_frames
    mi, ma = np.percentile(y, (1, 99.8)).astype(np.float32)
    float_frames -= mi
    float_frames /= ma - mi + 1e-8
    return float_frames


def _make_float32_memmap(frames: np.ndarray) -> np.ndarray:
    """Write uint8 frames to a float32 memmap one frame at a time (16MB per frame)."""
    T, H, W = frames.shape
    memmap_dir = Path(getattr(frames, "filename", "data/frames/_memmap")).parent
    float_path = memmap_dir / "frames_float32.dat"
    float_frames = np.memmap(float_path, dtype=np.float32, mode="w+", shape=(T, H, W))
    print("  writing float32 memmap to avoid Trackastra RAM spike...", flush=True)
    for i in range(T):
        float_frames[i] = frames[i].astype(np.float32)
    float_frames.flush()
    print("  done", flush=True)
    return float_frames


def link_frames_trackastra(frames: np.ndarray, labels: np.ndarray) -> list[TrackNode]:
    """Track cells using Trackastra and return a TrackNode list with lineage.

    frames: (T, H, W) uint8 grayscale
    labels: (T, H, W) uint16 Cellpose label maps

    Trackastra assigns stable Cell_IDs across frames and detects divisions.
    The returned TrackNodes have parent_id and children set for split events,
    so classify_events() scores daughter persistence correctly.
    """
    import trackastra.model.model_api as _model_api
    import trackastra.tracking.utils as _ttu
    import torch
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Trackastra.from_pretrained("general_2d", device=device)

    # Trackastra's normalize() casts the full (T,H,W) array to float32 in RAM (~9GB for
    # 575 frames at 2048x2048). Pre-write float32 to disk memmap instead, then replace
    # normalize so it operates on the on-disk array without the RAM spike.
    #
    # IMPORTANT: model_api.py uses `from ..utils import normalize` — a LOCAL binding.
    # Patching trackastra.utils.utils.normalize has no effect on that local binding.
    # We must patch trackastra.model.model_api.normalize directly.
    float_frames = _make_float32_memmap(frames)
    _orig_normalize = _model_api.normalize
    _model_api.normalize = lambda x, **kw: _normalize_on_memmap(float_frames)

    try:
        # model.track() returns (nx.DiGraph, tracked_masks) in trackastra 0.5.3+
        track_graph, tracked_video = model.track(frames, labels, mode="greedy")
    finally:
        _model_api.normalize = _orig_normalize

    # graph_to_ctc allocates (T,H,W) uint16 tracked masks via np.stack in RAM (~4.5GB).
    # Patch np.stack in that module to redirect large uint16 outputs to a disk memmap.
    memmap_dir = Path(getattr(frames, "filename", "data/frames/_memmap")).parent
    tracked_masks_path = memmap_dir / "tracked_masks.dat"
    T, H, W = labels.shape
    _orig_stack = _ttu.np.stack

    def _stack_to_memmap(arrays, *args, **kwargs):
        # Intercept call 1: np.stack([zeros_like(m) for m in labels]) → (T,H,W) uint16
        # Create a zero memmap directly instead of allocating 4.5GB in RAM.
        if (isinstance(arrays, (list, tuple)) and len(arrays) == T
                and np.asarray(arrays[0]).shape == (H, W)
                and np.asarray(arrays[0]).dtype == labels.dtype):
            return np.memmap(tracked_masks_path, dtype=labels.dtype, mode="w+", shape=(T, H, W))
        # Intercept call 2: np.stack(masks) where masks is already our (T,H,W) memmap.
        # graph_to_ctc calls np.stack a second time on the already-filled array — a no-op copy
        # that would OOM at 4.5GB. Return the memmap unchanged.
        if (isinstance(arrays, np.ndarray) and arrays.shape == (T, H, W)
                and arrays.dtype == labels.dtype):
            return arrays
        return _orig_stack(arrays, *args, **kwargs)

    _ttu.np.stack = _stack_to_memmap
    try:
        # graph_to_ctc recomputes relabeled masks; columns: track_id, t_start, t_end, parent_id
        # check=False skips _check_ctc_df which iterates all frames (slow, not needed here)
        df_ctc, tracked_video = graph_to_ctc(track_graph, labels, check=False)
    finally:
        _ttu.np.stack = _orig_stack

    # Column names vary by trackastra version (label/t1/t2/parent or track_id/t_start/t_end/parent_id)
    l_col, b_col, e_col, p_col = df_ctc.columns[:4]

    # Build parent lookup: label -> parent_label (0 if no parent)
    parent_of: dict[int, int] = {
        int(row[l_col]): int(row[p_col]) for _, row in df_ctc.iterrows()
    }
    begin_of: dict[int, int] = {
        int(row[l_col]): int(row[b_col]) for _, row in df_ctc.iterrows()
    }

    # Build nodes — one TrackNode per (label, frame) occurrence in tracked_video
    nodes: list[TrackNode] = []
    # Track the last node for each label so we can set children on it
    last_node: dict[int, TrackNode] = {}

    for t, label_map in enumerate(tracked_video):
        for label_id in np.unique(label_map):
            if label_id == 0:
                continue
            label_id = int(label_id)
            parent_label = parent_of.get(label_id, 0)
            is_birth_frame = (begin_of.get(label_id, 0) == t)

            # parent_id on the TrackNode is the label of the mother track,
            # but only set it on the birth frame node (first appearance after division)
            parent_id = parent_label if (parent_label > 0 and is_birth_frame) else None

            node = TrackNode(
                track_id=label_id,
                parent_id=parent_id,
                frame=t,
                mask=_label_to_cellmask(label_map, label_id, t),
            )
            nodes.append(node)

            # Wire children onto the parent's last-seen node
            if parent_id is not None and parent_label in last_node:
                parent_node = last_node[parent_label]
                if label_id not in parent_node.children:
                    parent_node.children.append(label_id)

            last_node[label_id] = node

    return nodes
