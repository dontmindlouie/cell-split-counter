"""CLI entrypoint: python main.py <video_path> [frame_step]."""

import sys
from pathlib import Path

from src.ingest import IngestConfig
from src.pipeline import run

if __name__ == "__main__":
    video_path = Path(sys.argv[1])
    frame_step = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    config = IngestConfig(video_path=video_path, frame_step=frame_step, roi=None)
    run(config, frame_dir=Path("data/frames"), output_dir=Path("data/output"))
