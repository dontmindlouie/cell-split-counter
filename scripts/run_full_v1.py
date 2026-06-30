"""Run the full v1 pipeline on a video at full resolution.

Expect ~4 hours on CPU (cellpose cyto3, ~26s/frame x 575 frames). Run overnight.

NOTE: superseded by main.py which uses pipeline.run() (Trackastra + Claude review).
Kept for reference only — prefer: python main.py data/raw/<video>.avi
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import classify_events
from src.ingest import IngestConfig, extract_frames
from src.output import write_events_csv, write_summary_json
from src.segment import segment_all
from src.track import link_frames

VIDEO = Path("data/raw/your_video.avi")  # configure for your dataset
FRAME_DIR = Path("data/frames")
OUT_DIR = Path("data/output")


def main() -> None:
    t0 = time.time()
    config = IngestConfig(video_path=VIDEO, frame_step=1, roi=None)

    print("extracting frames...", flush=True)
    frame_paths = extract_frames(config, FRAME_DIR)
    print(f"extracted {len(frame_paths)} frames in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    masks_by_frame = segment_all(frame_paths)
    print(f"segmented all frames in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    tracks = link_frames(masks_by_frame)
    events = classify_events(tracks, config.roi)
    print(f"tracked + classified in {time.time() - t2:.1f}s -- {len(events)} events", flush=True)

    write_events_csv(events, OUT_DIR / "events.csv", source_video=VIDEO.name)
    write_summary_json(events, {"video_path": str(VIDEO)}, OUT_DIR / "summary.json")
    print(f"done. total runtime {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
