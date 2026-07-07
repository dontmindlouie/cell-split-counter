"""CLI entrypoint: python main.py <video_path> [frame_step]."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Claude's returned descriptions can contain non-ASCII characters (e.g. "->" as an
# actual arrow glyph). Windows defaults redirected stdout to the system codepage
# (cp1252), which can't encode them -- crashing mid-review with all prior work lost
# since events.csv is only written after review_ambiguous returns. Force UTF-8 with
# replacement instead of erroring.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.ingest import IngestConfig
from src.pipeline import run

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tracker", choices=["iou", "trackastra"], default="trackastra")
    parser.add_argument("--tracker-mode", choices=["greedy", "ilp"], default="greedy", help="Trackastra linking mode (default: greedy; ilp catches compact/adjacent divisions greedy collapses, see docs/investigation_notes.md)")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame index to process (0-indexed)")
    parser.add_argument("--end-frame", type=int, default=None, help="last frame index (exclusive); default = all")
    parser.add_argument("--debug-crops", action="store_true", help="save Claude review crops to data/review_crops/")
    parser.add_argument("--reuse-masks", action="store_true", help="skip Cellpose and load existing memmaps from data/frames/_memmap/")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0, help="Cellpose sensitivity knob; lower = more permissive (default 0.0, library default)")
    parser.add_argument("--flow-threshold", type=float, default=0.4, help="Cellpose sensitivity knob; higher = more permissive (default 0.4, library default)")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for events.csv and summary.json (default: data/output/<video filename stem>)")
    parser.add_argument("--frame-dir", type=Path, default=Path("data/frames"), help="directory for extracted frames (default: data/frames)")
    parser.add_argument("--vision-backend", choices=["claude", "gpt"], default="claude", help="vision review model: claude (default, higher precision) or gpt (Azure OpenAI, lower precision but draws down Azure credit instead of Anthropic API spend -- see src/review_gpt.py)")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("data/output") / args.video_path.stem

    config = IngestConfig(video_path=args.video_path, frame_step=args.frame_step, roi=None)
    run(
        config,
        frame_dir=args.frame_dir,
        output_dir=output_dir,
        tracker=args.tracker,
        tracker_mode=args.tracker_mode,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_debug_crops=args.debug_crops,
        reuse_masks=args.reuse_masks,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        vision_backend=args.vision_backend,
    )
