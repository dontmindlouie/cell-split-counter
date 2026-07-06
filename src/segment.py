"""Per-frame cell segmentation via Cellpose."""

import cProfile
import os
import pstats
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cellpose import models

_model: models.CellposeModel | None = None
_PROFILE = os.environ.get("PROFILE_SEGMENTATION") == "1"


def _log_memory(label: str) -> None:
    import resource  # Linux-only; only imported when actually profiling (cloud)
    import torch
    ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB->MB on Linux
    gpu_alloc_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else 0
    print(f"  [mem] {label}: RAM peak={ram_mb:.0f}MB  GPU alloc peak={gpu_alloc_mb:.0f}MB  GPU reserved peak={gpu_reserved_mb:.0f}MB", flush=True)


def _get_model() -> models.CellposeModel:
    global _model
    if _model is None:
        _model = models.CellposeModel(gpu=True, model_type="cyto3")
    return _model


@dataclass
class CellMask:
    frame: int
    mask_id: int
    bbox: tuple[int, int, int, int]  # y0, y1, x0, x1 (exclusive)
    local_mask: np.ndarray  # boolean array cropped to bbox, NOT full frame size
    centroid: tuple[float, float]
    area: float


def segment_frame(frame_path: Path, frame_index: int) -> list[CellMask]:
    """Run Cellpose on a single frame and return one CellMask per detected cell.

    Masks are stored cropped to each cell's bounding box rather than at full frame
    size -- a 2048x2048 bool array is 4MB regardless of cell size, and holding one per
    cell per frame for a whole video exhausts memory. Cropped, each cell costs tens of
    KB instead.
    """
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    label_map, _, _ = _get_model().eval(img, diameter=None, channels=[0, 0])

    cell_masks = []
    for mask_id in np.unique(label_map):
        if mask_id == 0:
            continue
        mask = label_map == mask_id
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        centroid = (float(xs.mean()), float(ys.mean()))
        cell_masks.append(
            CellMask(
                frame=frame_index,
                mask_id=int(mask_id),
                bbox=(y0, y1, x0, x1),
                local_mask=mask[y0:y1, x0:x1].copy(),
                centroid=centroid,
                area=float(mask.sum()),
            )
        )
    return cell_masks


def segment_all(frame_paths: list[Path]) -> dict[int, list[CellMask]]:
    """Segment every frame, keyed by frame index."""
    return {i: segment_frame(p, i) for i, p in enumerate(frame_paths)}


_EVAL_BATCH = 64  # frames per Cellpose eval() call — amortizes per-call overhead


def segment_video_arrays(
    frame_paths: list[Path],
    memmap_dir: Path | None = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment all frames and return raw arrays for Trackastra input.

    Returns (frames, labels) both shaped (T, H, W):
      frames: uint8 grayscale pixel values
      labels: uint16 integer label maps (0 = background, N = cell N)

    Arrays are written to memory-mapped files under memmap_dir (defaults to a
    temp dir alongside the first frame). This keeps RAM usage to one batch of
    frames at a time during segmentation — Trackastra then pages from disk on
    demand rather than holding the full video in RAM.

    Frames are sent to Cellpose in batches (model.eval() accepts a list of
    images) rather than one call per frame — each call carries its own fixed
    dispatch/normalization overhead, which one-at-a-time calls pay per frame.

    cellprob_threshold/flow_threshold default to Cellpose's own library defaults
    (0.0, 0.4). Lowering cellprob_threshold and/or raising flow_threshold makes
    detection more permissive/sensitive -- validated 2026-07-05 via
    scripts/test_sensitive_thresholds.py against known small/dim objects that
    default settings miss entirely (GT events 9/10 on Tom20): cellprob=-4.0,
    flow=0.8 recovered mask presence at a known-real, previously-undetected
    location on 6/7 tested frames vs. 3/7 at default, with modest mask-count
    increase (no runaway over-segmentation observed in that spot-check).
    """
    model = _get_model()
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    T, H, W = len(frame_paths), first.shape[0], first.shape[1]

    if memmap_dir is None:
        memmap_dir = frame_paths[0].parent / "_memmap"
    memmap_dir.mkdir(parents=True, exist_ok=True)

    frames_path = memmap_dir / "frames.dat"
    labels_path = memmap_dir / "labels.dat"
    raw_frames = np.memmap(frames_path, dtype=np.uint8,  mode="w+", shape=(T, H, W))
    label_maps = np.memmap(labels_path, dtype=np.uint16, mode="w+", shape=(T, H, W))

    profiler = cProfile.Profile() if _PROFILE else None
    if profiler:
        profiler.enable()

    for batch_num, start in enumerate(range(0, T, _EVAL_BATCH)):
        end = min(start + _EVAL_BATCH, T)
        print(f"  segmenting frames {start+1}-{end}/{T}", end="\r", flush=True)
        imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frame_paths[start:end]]
        masks, _, _ = model.eval(
            imgs, diameter=None, channels=[0, 0],
            cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
        )
        for offset, (img, mask) in enumerate(zip(imgs, masks)):
            raw_frames[start + offset] = img
            label_maps[start + offset] = mask.astype(np.uint16)
        if _PROFILE and batch_num % 5 == 0:
            _log_memory(f"after batch {batch_num} (frame {end}/{T})")
    print()

    if profiler:
        profiler.disable()
        _log_memory("segmentation complete")
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        print("  [profile] top 30 by cumulative time:")
        stats.print_stats(30)

    raw_frames.flush()
    label_maps.flush()
    return raw_frames, label_maps


def load_video_arrays(
    frame_paths: list[Path],
    memmap_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load existing segmentation memmaps without re-running Cellpose.

    Raises FileNotFoundError if the memmaps don't exist yet — run without
    --reuse-masks first to produce them.
    """
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    T, H, W = len(frame_paths), first.shape[0], first.shape[1]

    if memmap_dir is None:
        memmap_dir = frame_paths[0].parent / "_memmap"

    frames_path = memmap_dir / "frames.dat"
    labels_path = memmap_dir / "labels.dat"

    if not frames_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Segmentation memmaps not found in {memmap_dir} — run without --reuse-masks first"
        )

    expected_frames = T * H * W
    expected_labels = T * H * W * 2
    actual_frames = frames_path.stat().st_size
    actual_labels = labels_path.stat().st_size
    if actual_frames != expected_frames or actual_labels != expected_labels:
        raise ValueError(
            f"Memmap size mismatch — expected {expected_frames/1e9:.2f} GB / {expected_labels/1e9:.2f} GB "
            f"but found {actual_frames/1e9:.2f} GB / {actual_labels/1e9:.2f} GB. "
            f"Memmaps are stale (different frame count or resolution). Run without --reuse-masks to regenerate."
        )

    raw_frames = np.memmap(frames_path, dtype=np.uint8,  mode="r", shape=(T, H, W))
    label_maps = np.memmap(labels_path, dtype=np.uint16, mode="r", shape=(T, H, W))
    print(f"  loaded existing memmaps from {memmap_dir} ({T} frames, {H}x{W})")
    return raw_frames, label_maps
