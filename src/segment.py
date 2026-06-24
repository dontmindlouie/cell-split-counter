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
        _model = models.CellposeModel(gpu=False, model_type="cyto3")
    return _model


@dataclass
class CellMask:
    frame: int
    mask_id: int
    mask: np.ndarray  # boolean array, same shape as frame
    centroid: tuple[float, float]
    area: float


def segment_frame(frame_path: Path, frame_index: int) -> list[CellMask]:
    """Run Cellpose on a single frame and return one CellMask per detected cell."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    label_map, _, _ = _get_model().eval(img, diameter=None, channels=[0, 0])

    cell_masks = []
    for mask_id in np.unique(label_map):
        if mask_id == 0:
            continue
        mask = label_map == mask_id
        ys, xs = np.nonzero(mask)
        centroid = (float(xs.mean()), float(ys.mean()))
        cell_masks.append(
            CellMask(frame=frame_index, mask_id=int(mask_id), mask=mask, centroid=centroid, area=float(mask.sum()))
        )
    return cell_masks


def segment_all(frame_paths: list[Path]) -> dict[int, list[CellMask]]:
    """Segment every frame, keyed by frame index."""
    return {i: segment_frame(p, i) for i, p in enumerate(frame_paths)}
