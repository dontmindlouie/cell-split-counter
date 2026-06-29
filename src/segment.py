"""Per-frame cell segmentation via Cellpose."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cellpose import models

_model: models.CellposeModel | None = None


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


def segment_video_arrays(
    frame_paths: list[Path],
    memmap_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment all frames and return raw arrays for Trackastra input.

    Returns (frames, labels) both shaped (T, H, W):
      frames: uint8 grayscale pixel values
      labels: uint16 integer label maps (0 = background, N = cell N)

    Arrays are written to memory-mapped files under memmap_dir (defaults to a
    temp dir alongside the first frame). This keeps RAM usage to a single frame
    at a time during segmentation — Trackastra then pages from disk on demand
    rather than holding the full video in RAM.
    """
    import tempfile
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

    for i, path in enumerate(frame_paths):
        print(f"  segmenting frame {i+1}/{T}", end="\r", flush=True)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        label_map, _, _ = model.eval(img, diameter=None, channels=[0, 0])
        raw_frames[i] = img
        label_maps[i] = label_map.astype(np.uint16)
    print()
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

    raw_frames = np.memmap(frames_path, dtype=np.uint8,  mode="r", shape=(T, H, W))
    label_maps = np.memmap(labels_path, dtype=np.uint16, mode="r", shape=(T, H, W))
    print(f"  loaded existing memmaps from {memmap_dir} ({T} frames, {H}x{W})")
    return raw_frames, label_maps
