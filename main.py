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
    parser.add_argument("--tracker", choices=["iou", "trackastra"], default="trackastra")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame index to process (0-indexed)")
    parser.add_argument("--end-frame", type=int, default=None, help="last frame index (exclusive); default = all")
    parser.add_argument("--debug-crops", action="store_true", help="save Claude review crops to data/debug/crops/")
    args = parser.parse_args()

    config = IngestConfig(video_path=args.video_path, frame_step=args.frame_step, roi=None)
    run(
        config,
        frame_dir=Path("data/frames"),
        output_dir=Path("data/output"),
        tracker=args.tracker,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_debug_crops=args.debug_crops,
    )
