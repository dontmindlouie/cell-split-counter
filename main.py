"""CLI entrypoint: python main.py <video_path> [frame_step]."""

import sys
from pathlib import Path

from src.ingest import IngestConfig
from src.pipeline import run

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tracker", choices=["iou", "trackastra"], default="iou")
    args = parser.parse_args()

    config = IngestConfig(video_path=args.video_path, frame_step=args.frame_step, roi=None)
    run(config, frame_dir=Path("data/frames"), output_dir=Path("data/output"), tracker=args.tracker)
