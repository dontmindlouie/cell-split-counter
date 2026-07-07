"""Frame extraction and ROI cropping from input video."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class IngestConfig:
    video_path: Path
    frame_step: int  # take every Nth raw frame; acquisition rate is fixed by the microscope, not video fps
    roi: tuple[int, int, int, int] | None  # x, y, w, h; None = full frame


def get_pixel_size_um(video_path: Path) -> float | None:
    """Return the acquisition's µm/pixel, if the source format carries it.

    ND2 files (Nikon NIS-Elements) embed real per-acquisition voxel size --
    varies by objective/zoom, so this is NOT a fixed constant even within one
    imaging project (confirmed 2026-07-03: Bewo's ND2s are 0.57 µm/px, Tom20's
    M2 ND2 is 0.432 µm/px). AVI exports carry no reliable equivalent metadata --
    returns None, and callers should fall back to an explicit override or leave
    size-in-µm fields blank rather than guess.
    """
    if video_path.suffix.lower() != ".nd2":
        return None
    import nd2

    with nd2.ND2File(video_path) as f:
        return f.voxel_size().x


def extract_frames(config: IngestConfig, out_dir: Path) -> list[Path]:
    """Extract every config.frame_step'th frame from the video, cropped to config.roi.

    Returns paths to the written frame images, in chronological order. ND2 files
    (Nikon NIS-Elements native format) are read directly via the nd2 package;
    everything else goes through cv2.VideoCapture as before.
    """
    if config.video_path.suffix.lower() == ".nd2":
        return _extract_frames_nd2(config, out_dir)
    return _extract_frames_cv2(config, out_dir)


def _extract_frames_cv2(config: IngestConfig, out_dir: Path) -> list[Path]:
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


def _rescale_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Percentile-rescale a uint16 frame to uint8.

    Clips to the 0.5-99.5 percentile range rather than true min/max so a few hot
    pixels don't crush the rest of the frame's contrast to near-black.
    """
    lo, hi = np.percentile(frame, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
    return scaled.astype(np.uint8)


def _extract_frames_nd2(config: IngestConfig, out_dir: Path) -> list[Path]:
    import nd2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    kept_index = 0
    with nd2.ND2File(config.video_path) as f:
        total = f.sizes.get("T", f.shape[0])
        for raw_index in range(total):
            if raw_index % config.frame_step != 0:
                continue
            frame = f.read_frame(raw_index)
            if config.roi is not None:
                x, y, w, h = config.roi
                frame = frame[y : y + h, x : x + w]
            frame_gray = _rescale_to_uint8(frame)
            out_path = out_dir / f"frame_{kept_index:05d}_raw{raw_index:05d}.png"
            cv2.imwrite(str(out_path), frame_gray)
            paths.append(out_path)
            kept_index += 1
    return paths
