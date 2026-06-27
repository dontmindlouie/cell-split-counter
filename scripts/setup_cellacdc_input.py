"""Convert a raw AVI into Cell-ACDC's expected TIFF stack format.

Cell-ACDC expects: <experiment>/<position>/Images/<basename>_<channel>.tif

Usage:
  python setup_cellacdc_input.py                          # smoke_test.avi, 20 frames
  python setup_cellacdc_input.py --src path/to/video.avi  # full video, all frames
  python setup_cellacdc_input.py --frames 100             # first 100 frames
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

DEFAULT_SRC = Path("G:/Projects/cell-split-counter/data/smoke_test.avi")
OUT_TIFF = Path("G:/Projects/cell-split-counter-spike-cellacdc/data/cellacdc_input/Position_1/Images/video_Tom20.tif")
RESIZE_TO = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--frames", type=int, default=None, help="Max frames to read (default: all)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.src))
    if not cap.isOpened():
        print(f"Cannot open {args.src}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(args.frames, total) if args.frames else total
    print(f"Reading {n}/{total} frames from {args.src.name}...")

    frames = []
    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.resize(gray, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
        frames.append(gray)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    cap.release()

    stack = np.stack(frames, axis=0)
    print(f"Stack: {stack.shape} {stack.dtype}")
    OUT_TIFF.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(OUT_TIFF), stack, photometric="minisblack")
    print(f"Written to {OUT_TIFF}")

    # Remove stale Cell-ACDC outputs so the next run re-segments from scratch
    for stale in ["video_segm.npz", "video_acdc_output.csv"]:
        p = OUT_TIFF.parent / stale
        if p.exists():
            p.unlink()
            print(f"Removed stale {stale}")


if __name__ == "__main__":
    main()
