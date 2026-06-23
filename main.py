"""CLI entrypoint: python main.py <video_path> <roi_x> <roi_y> <roi_w> <roi_h>."""

import sys
from pathlib import Path

from src.ingest import IngestConfig
from src.pipeline import run

if __name__ == "__main__":
    video_path = Path(sys.argv[1])
    roi = tuple(int(x) for x in sys.argv[2:6])

    config = IngestConfig(video_path=video_path, frame_interval_sec=5.0, roi=roi)
    run(config, frame_dir=Path("data/frames"), output_dir=Path("data/output"))
