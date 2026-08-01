"""Frame extraction and ROI cropping from input video."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_DISPLAY_COLOR_FILENAME = "_display_color.json"
_DISPLAY_WINDOW_FILENAME = "_display_windows.json"


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


def get_display_color_rgb(video_path: Path) -> list[int] | None:
    """Return the ND2's own display LUT as [r, g, b] (0-255), or None.

    NIS writes this from the acquisition preset -- e.g. H2B-mCherry channels carry
    (255, 0, 0). Reproducing it is what makes a rendered frame match what a
    researcher sees in Fiji, instead of us inventing a colormap. Non-ND2 sources
    (AVI, etc.) carry no equivalent metadata and always return None.
    """
    if video_path.suffix.lower() != ".nd2":
        return None
    import nd2

    with nd2.ND2File(video_path) as f:
        color = getattr(f.metadata.channels[0].channel, "color", None)
        return [color.r, color.g, color.b] if color else None


def colorize(grey: np.ndarray, display_color_rgb: list[int] | None) -> np.ndarray:
    """Tint a grayscale frame with the acquisition's own display LUT.

    Returns the frame unchanged (single channel) when no LUT is known, rather than
    a fake all-white recolor -- keeps sources with no real color metadata (AVI,
    older runs) from tripling their crop storage for no visual gain. Where a real
    LUT exists (single-channel fluorescence, e.g. mCherry's (255,0,0)) at least two
    of the three output channels are constant/zero, so PNG compresses the result
    close to grayscale size despite being 3-channel.
    """
    if not display_color_rgb:
        return grey
    r, g, b = (v / 255.0 for v in display_color_rgb)
    return cv2.merge([(grey * b).astype(np.uint8),
                      (grey * g).astype(np.uint8),
                      (grey * r).astype(np.uint8)])


@lru_cache(maxsize=32)
def read_display_color(frame_dir: Path) -> list[int] | None:
    """Look up the display LUT saved alongside a frame directory by extract_frames."""
    p = frame_dir / _DISPLAY_COLOR_FILENAME
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("display_color_rgb")


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


def _rescale_to_uint8(frame: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Percentile-rescale a uint16 frame to uint8, returning the window used.

    Clips to the 0.5-99.5 percentile range rather than true min/max so a few hot
    pixels don't crush the rest of the frame's contrast to near-black.

    The window is computed per frame, which is what Cellpose wants -- it sees a
    consistently-contrasted image even as the field bleaches over 40-70 h. The cost is
    that the 8-bit result is no longer comparable frame to frame: a field-wide decay
    rescales itself back to full range and renders as no decay at all. So the window is
    returned and recorded, which makes the stretch REVERSIBLE
    (raw ~= lo + png/255 * (hi - lo)) without re-exporting anything or changing a single
    pixel Cellpose reads.
    """
    lo, hi = (float(v) for v in np.percentile(frame, [0.5, 99.5]))
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.uint8), (lo, hi)
    scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255
    return scaled.astype(np.uint8), (lo, hi)


def _extract_frames_nd2(config: IngestConfig, out_dir: Path) -> list[Path]:
    import nd2

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    windows: list[tuple[float, float]] = []
    kept_index = 0
    with nd2.ND2File(config.video_path) as f:
        color = getattr(f.metadata.channels[0].channel, "color", None)
        display_color_rgb = [color.r, color.g, color.b] if color else None
        (out_dir / _DISPLAY_COLOR_FILENAME).write_text(
            json.dumps({"display_color_rgb": display_color_rgb})
        )
        read_display_color.cache_clear()  # out_dir may reuse a path from an earlier run

        total = f.sizes.get("T", f.shape[0])
        for raw_index in range(total):
            if raw_index % config.frame_step != 0:
                continue
            frame = f.read_frame(raw_index)
            if config.roi is not None:
                x, y, w, h = config.roi
                frame = frame[y : y + h, x : x + w]
            frame_gray, window = _rescale_to_uint8(frame)
            windows.append(window)
            out_path = out_dir / f"frame_{kept_index:05d}_raw{raw_index:05d}.png"
            cv2.imwrite(str(out_path), frame_gray)
            paths.append(out_path)
            kept_index += 1
    (out_dir / _DISPLAY_WINDOW_FILENAME).write_text(json.dumps(
        {"note": "per-frame 0.5/99.5 percentile window used to make the 8-bit PNGs; "
                 "raw ~= lo + png/255 * (hi - lo). Without it, apparent brightness is "
                 "not comparable between frames.",
         "windows": [[round(lo, 2), round(hi, 2)] for lo, hi in windows]}))
    return paths
