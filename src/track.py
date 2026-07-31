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


def _bridge_track_gaps(
    df_ctc,
    tracked_video,
    l_col: str,
    b_col: str,
    e_col: str,
    max_gap_frames: int = 3,
    max_gap_dist: float = 40.0,
) -> dict[int, int]:
    """Bridge single-successor tracking gaps into one continuous lineage.

    Confirmed root cause (2026-07-06): Cellpose's mask for a real, continuously-
    dividing cell can briefly vanish for 1+ frames (small/dim object, borderline
    detection), and Trackastra creates a brand-new, disconnected track when it
    reappears rather than continuing the original one -- even with Trackastra's
    own delta_t parameter raised (tested up to 4, and in ilp mode; the edge is
    either never generated across the gap or the model's learned feature
    similarity doesn't recognize the object as continuous). A project-wide scan
    found this pattern implicated in the majority of this project's confirmed
    real missed divisions (Tom20 GT events near frames 60/62, 185, 212, and a
    previously-uncounted division near frame 59) -- worth fixing here rather
    than relying on Trackastra to solve it internally.

    A REAL split produces two nearby successors (the daughters) -- that's normal
    and must NOT be bridged. Only tracks with EXACTLY ONE nearby successor within
    the gap window are merged; two or more successors are left alone.

    Two guards stop that merging from running away (added 2026-07-31 after the
    original version blended distinct cells onto one id in 3.25% of M12_RUES2's
    tracks, which is what made a "track" untrustworthy as a unit of navigation):

    1. NO TEMPORAL OVERLAP. One cell cannot be in two places at once, so two labels
       whose frame spans overlap are provably different cells and must never share a
       canonical id. The one-successor rule alone does not prevent this, because
       union-find is transitive: if two DIFFERENT predecessors each bridge to the
       same successor, all three land in one group and the two predecessors can
       overlap heavily. Measured on M12: 226 of 1,060 merged groups contained
       co-existing labels, the worst overlapping by 118 frames, and chaining built
       groups of up to 11 labels -- a single cell needing ten bridges is not
       credible. The check is against the whole group, not the pair, because
       chaining is the mechanism.
    2. ONE CLAIMANT PER SUCCESSOR. The original asked "does this track have exactly
       one successor?" but never "does this successor have exactly one predecessor?"
       A successor two tracks both want is ambiguous, and guessing picks wrong half
       the time, so it is left unbridged.

    Returns a mapping from every original Trackastra label to its canonical
    (post-merge) label -- labels merged into the same lineage share one value.
    """
    from collections import defaultdict

    begin_of = {int(row[l_col]): int(row[b_col]) for _, row in df_ctc.iterrows()}
    end_of = {int(row[l_col]): int(row[e_col]) for _, row in df_ctc.iterrows()}
    video_end = len(tracked_video) - 1

    from skimage.measure import regionprops

    centroid_cache: dict[int, dict[int, tuple[float, float]]] = {}

    def _centroid(label: int, frame: int) -> tuple[float, float] | None:
        if frame not in centroid_cache:
            frame_arr = np.asarray(tracked_video[frame])
            centroid_cache[frame] = {
                int(p.label): (float(p.centroid[1]), float(p.centroid[0]))
                for p in regionprops(frame_arr)
            }
        return centroid_cache[frame].get(label)

    starts_by_frame: dict[int, list[int]] = defaultdict(list)
    for lbl, f0 in begin_of.items():
        starts_by_frame[f0].append(lbl)

    parent_uf: dict[int, int] = {lbl: lbl for lbl in begin_of}

    def find(x: int) -> int:
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent_uf[rb] = ra

    # Every label's own frame span, for the overlap guard below.
    spans: dict[int, tuple[int, int]] = {lbl: (begin_of[lbl], end_of[lbl]) for lbl in begin_of}
    members: dict[int, list[int]] = {lbl: [lbl] for lbl in begin_of}

    def _would_overlap(a: int, b: int) -> bool:
        """True if merging a's group with b's would put two co-existing labels together."""
        for x in members[find(a)]:
            bx0, bx1 = spans[x]
            for y in members[find(b)]:
                by0, by1 = spans[y]
                if bx0 <= by1 and by0 <= bx1:
                    return True
        return False

    # Pass 1: collect every proposed bridge, so a successor wanted by two different
    # predecessors can be spotted before anything is merged.
    proposals: list[tuple[int, int]] = []
    for lbl, f_end in end_of.items():
        if f_end >= video_end:
            continue  # track runs to the end of the video, not a gap
        end_c = _centroid(lbl, f_end)
        if end_c is None:
            continue
        ex, ey = end_c
        successors = []
        for gap in range(1, max_gap_frames + 1):
            f_start = f_end + gap
            if f_start > video_end:
                break
            for other in starts_by_frame.get(f_start, []):
                if other == lbl:
                    continue
                start_c = _centroid(other, f_start)
                if start_c is None:
                    continue
                if ((start_c[0] - ex) ** 2 + (start_c[1] - ey) ** 2) ** 0.5 <= max_gap_dist:
                    successors.append(other)
        if len(successors) == 1:
            proposals.append((lbl, successors[0]))

    # Guard 2: a successor claimed by more than one predecessor is ambiguous.
    claimants: dict[int, int] = defaultdict(int)
    for _lbl, succ in proposals:
        claimants[succ] += 1

    # Pass 2: merge, skipping ambiguous claims and anything that would co-exist.
    # Sorted so the result does not depend on dict iteration order.
    for lbl, succ in sorted(proposals):
        if claimants[succ] > 1:
            continue
        if _would_overlap(lbl, succ):
            continue
        ra, rb = find(lbl), find(succ)
        if ra != rb:
            merged_members = members[ra] + members[rb]
            union(lbl, succ)
            members[find(lbl)] = merged_members

    return {lbl: find(lbl) for lbl in begin_of}


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


def _write_lineage_csv(
    memmap_dir,
    df_ctc,
    canonical_of: dict[int, int],
    true_birth_label: dict[int, int],
    parent_of: dict[int, int],
    begin_of: dict[int, int],
    e_col: str,
    l_col: str,
) -> None:
    """Persist the mother/daughter graph for EVERY track, canonical ids, no verdicts.

    Trackastra's CTC table is the only complete lineage this pipeline ever has, and
    until 2026-07-29 it was discarded here: the graph survived only as far as
    classify_events, which copied a parent_id onto the events it happened to emit.
    That left events.csv as the sole record of lineage -- partial by construction
    (938 of 5163 tracks in TSC_batch2_M12_RUES2) and welded to the AI's verdict
    columns, so anything wanting pure topology had to read a file full of answers.

    Written next to the memmaps rather than into the run root because it is a
    tracking artifact keyed to canonical_labels.json's remap; scripts/build_bundle.py
    lifts it into the bundle as lineage.csv.

    Topology only. A daughter here means "Trackastra linked these across a division",
    which is not a claim that the division was real -- judging that is the reviewer's
    job, and keeping the two apart is what makes this safe to ship in an eval bundle.
    """
    import csv as _csv

    end_of = {int(row[l_col]): int(row[e_col]) for _, row in df_ctc.iterrows()}

    # Collapse raw labels into canonical tracks: a bridged group spans the earliest
    # begin to the latest end across all of its member labels.
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    for lbl, canon in canonical_of.items():
        b, e = begin_of[lbl], end_of[lbl]
        first_frame[canon] = min(b, first_frame.get(canon, b))
        last_frame[canon] = max(e, last_frame.get(canon, e))

    # Only the true birth label of a group may carry a parent -- a resumed label is a
    # continuation of the same cell, not a new birth (same rule the node builder uses).
    parent: dict[int, int | None] = {}
    for canon, birth_lbl in true_birth_label.items():
        raw = parent_of.get(birth_lbl, 0)
        canon_parent = canonical_of.get(raw, raw)
        parent[canon] = canon_parent if canon_parent > 0 and canon_parent != canon else None

    daughters: dict[int, list[int]] = {}
    for canon, p in parent.items():
        if p is not None:
            daughters.setdefault(p, []).append(canon)

    path = memmap_dir / "ctc_lineage.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["track_id", "parent_id", "first_frame", "last_frame",
                    "n_daughters", "daughter_ids"])
        for canon in sorted(first_frame):
            kids = sorted(daughters.get(canon, []))
            w.writerow([canon, parent.get(canon) if parent.get(canon) is not None else "",
                        first_frame[canon], last_frame[canon],
                        len(kids), " ".join(str(k) for k in kids)])
    print(f"  wrote lineage for {len(first_frame)} canonical tracks -> {path}", flush=True)


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
    print(f"  writing float32 memmap to avoid Trackastra RAM spike... (0/{T})", flush=True)
    for i in range(T):
        float_frames[i] = frames[i].astype(np.float32)
        if (i + 1) % 100 == 0 or i == T - 1:
            print(f"    {i + 1}/{T}", flush=True)
    float_frames.flush()
    print("  done", flush=True)
    return float_frames


def link_frames_trackastra(
    frames: np.ndarray, labels: np.ndarray, mode: str = "greedy", delta_t: int = 1
) -> list[TrackNode]:
    """Track cells using Trackastra and return a TrackNode list with lineage.

    frames: (T, H, W) uint8 grayscale
    labels: (T, H, W) uint16 Cellpose label maps
    mode: "greedy" (default, fast) or "ilp" (global optimum via motile/SCIP --
        catches divisions where two daughters are compact/adjacent enough that
        greedy's local frame-to-frame matching collapses them into one track;
        see docs/investigation_notes.md 2026-07-03 entry)
    delta_t: how many frames apart the candidate graph will consider linking
        (Trackastra's own parameter, library default 1 = adjacent frames only).
        Raising this bridges brief gaps where a cell's mask is missing for 1+
        frames -- validated 2026-07-06 via direct track-ID tracing: a real
        division's daughter (track 410) vanished for exactly one frame and
        reappeared as a disconnected new track (678) under delta_t=1, breaking
        the lineage link back to the split. delta_t=2 lets the graph consider
        edges across that gap directly.

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

    # Pre-compute paths used by multiple patches below.
    T, H, W = labels.shape
    memmap_dir = Path(getattr(frames, "filename", "data/frames/_memmap")).parent
    tracked_masks_path = memmap_dir / "tracked_masks.dat"

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

    # model.track() internally calls apply_solution_graph_to_masks which does
    # np.zeros_like(masks_original) → 4.49 GB RAM allocation. We don't use that
    # return value (graph_to_ctc recomputes tracked masks on disk below), so replace
    # it with a stub that returns a tiny dummy array.
    #
    # IMPORTANT: model_api.py uses `from ..tracking import apply_solution_graph_to_masks`
    # (local binding) — must patch _model_api directly.
    _orig_asm = _model_api.apply_solution_graph_to_masks
    _model_api.apply_solution_graph_to_masks = lambda g, m, **kw: np.zeros((1,), dtype=m.dtype)

    print(f"  running Trackastra model.track() (mode={mode}, device={device}, {T} frames)...", flush=True)
    try:
        # model.track() returns (nx.DiGraph, tracked_masks) in trackastra 0.5.3+
        track_graph, tracked_video = model.track(frames, labels, mode=mode, delta_t=delta_t)
    finally:
        _model_api.normalize = _orig_normalize
        _model_api.apply_solution_graph_to_masks = _orig_asm
    print("  model.track() done", flush=True)

    # graph_to_ctc allocates (T,H,W) uint16 tracked masks via np.stack in RAM (~4.5GB).
    # Patch np.stack in that module to redirect large uint16 outputs to a disk memmap.
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

    # Bridge single-successor tracking gaps (see _bridge_track_gaps docstring) before
    # building nodes, so downstream classify_events sees one continuous lineage instead
    # of an artificially orphaned track when a real division's daughter briefly vanishes.
    canonical_of = _bridge_track_gaps(df_ctc, tracked_video, l_col, b_col, e_col)

    # For each merged group, only the earliest (true) birth label may carry a real
    # parent -- later "resumed" labels in the same group are continuations, not births.
    true_birth_label: dict[int, int] = {}  # canonical_id -> raw label with min begin_of
    for lbl, canon in canonical_of.items():
        if canon not in true_birth_label or begin_of[lbl] < begin_of[true_birth_label[canon]]:
            true_birth_label[canon] = lbl

    _write_lineage_csv(memmap_dir, df_ctc, canonical_of, true_birth_label,
                       parent_of, begin_of, e_col, l_col)

    # Build nodes — one TrackNode per (label, frame) occurrence in tracked_video.
    # Use regionprops for a single pass per frame: instead of N separate boolean array
    # comparisons (label_map == label_id for each label), compute all cell properties in
    # one C-level scan. Also skip storing local_mask — downstream classify/review only
    # needs centroid, which regionprops provides directly.
    from skimage.measure import regionprops

    nodes: list[TrackNode] = []
    last_node: dict[int, TrackNode] = {}

    for t, label_map in enumerate(tracked_video):
        label_arr = np.asarray(label_map)  # load frame from memmap once
        for prop in regionprops(label_arr):
            label_id = int(prop.label)
            track_id = canonical_of.get(label_id, label_id)
            birth_label = true_birth_label.get(track_id, label_id)
            parent_label_raw = parent_of.get(birth_label, 0)
            parent_label = canonical_of.get(parent_label_raw, parent_label_raw)
            is_birth_frame = (label_id == birth_label and begin_of.get(birth_label, 0) == t)
            parent_id = parent_label if (parent_label > 0 and is_birth_frame) else None

            # regionprops bbox: (min_row, min_col, max_row, max_col) exclusive
            r0, c0, r1, c1 = prop.bbox
            node = TrackNode(
                track_id=track_id,
                parent_id=parent_id,
                frame=t,
                mask=CellMask(
                    frame=t,
                    mask_id=label_id,
                    bbox=(r0, r1, c0, c1),  # (y0, y1, x0, x1)
                    local_mask=None,         # not needed; saves ~1 GB of bool copies
                    centroid=(float(prop.centroid[1]), float(prop.centroid[0])),  # (cx, cy)
                    area=float(prop.area),
                    eccentricity=float(prop.eccentricity),
                    solidity=float(prop.solidity),
                ),
            )
            nodes.append(node)

            # Wire children onto the parent's last-seen node
            if parent_id is not None and parent_label in last_node:
                parent_node = last_node[parent_label]
                if track_id not in parent_node.children:
                    parent_node.children.append(track_id)

            last_node[track_id] = node

    return nodes
