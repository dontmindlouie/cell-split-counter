"""CLI entrypoint: python main.py <video_path> [frame_step]."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
    parser.add_argument("--debug-crops", action="store_true", help="save Claude review crops to data/review_crops/")
    parser.add_argument("--classify-divisions", action="store_true", help="run ACD division type classifier on high-confidence events")
    parser.add_argument("--reuse-masks", action="store_true", help="skip Cellpose and load existing memmaps from data/frames/_memmap/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"), help="directory for events.csv and summary.json (default: data/output)")
    parser.add_argument("--frame-dir", type=Path, default=Path("data/frames"), help="directory for extracted frames (default: data/frames)")
    args = parser.parse_args()

    config = IngestConfig(video_path=args.video_path, frame_step=args.frame_step, roi=None)
    run(
        config,
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        tracker=args.tracker,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_debug_crops=args.debug_crops,
        classify_divisions=args.classify_divisions,
        reuse_masks=args.reuse_masks,
    )
