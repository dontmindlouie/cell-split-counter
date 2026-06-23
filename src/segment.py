"""Per-frame cell segmentation via Cellpose."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CellMask:
    frame: int
    mask_id: int
    mask: np.ndarray  # boolean array, same shape as frame
    centroid: tuple[float, float]
    area: float


def segment_frame(frame_path: Path, frame_index: int) -> list[CellMask]:
    """Run Cellpose on a single frame and return one CellMask per detected cell."""
    raise NotImplementedError


def segment_all(frame_paths: list[Path]) -> dict[int, list[CellMask]]:
    """Segment every frame, keyed by frame index."""
    return {i: segment_frame(p, i) for i, p in enumerate(frame_paths)}
