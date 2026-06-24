"""Frame extraction and ROI cropping from input video."""

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class IngestConfig:
    video_path: Path
    frame_step: int  # take every Nth raw frame; acquisition rate is fixed by the microscope, not video fps
    roi: tuple[int, int, int, int] | None  # x, y, w, h; None = full frame


def extract_frames(config: IngestConfig, out_dir: Path) -> list[Path]:
    """Extract every config.frame_step'th frame from the video, cropped to config.roi.

    Returns paths to the written frame images, in chronological order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {config.video_path}")

    paths = []
    raw_index = 0
    kept_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_index % config.frame_step == 0:
            if config.roi is not None:
                x, y, w, h = config.roi
                frame = frame[y : y + h, x : x + w]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_path = out_dir / f"frame_{kept_index:05d}_raw{raw_index:05d}.png"
            cv2.imwrite(str(out_path), frame_gray)
            paths.append(out_path)
            kept_index += 1
        raw_index += 1

    cap.release()
    return paths
