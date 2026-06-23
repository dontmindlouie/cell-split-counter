"""Frame extraction and ROI cropping from input video."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestConfig:
    video_path: Path
    frame_interval_sec: float
    roi: tuple[int, int, int, int]  # x, y, w, h


def extract_frames(config: IngestConfig, out_dir: Path) -> list[Path]:
    """Extract frames from the video at config.frame_interval_sec, cropped to config.roi.

    Returns paths to the written frame images, in chronological order.
    """
    raise NotImplementedError
