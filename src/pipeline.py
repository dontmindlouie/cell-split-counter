"""End-to-end orchestration: video in, events out."""

from pathlib import Path

from src.classify import classify_events
from src.ingest import IngestConfig, extract_frames
from src.output import write_events_csv, write_summary_json
from src.segment import segment_all
from src.track import link_frames

# v1 scope: detect 1->2 (and 1->N) splits only. Abnormality classification and the
# Claude-vision review step (src.review) are deferred until split detection itself
# is validated against ground truth.


def run(config: IngestConfig, frame_dir: Path, output_dir: Path) -> None:
    frame_paths = extract_frames(config, frame_dir)
    masks_by_frame = segment_all(frame_paths)
    tracks = link_frames(masks_by_frame)
    events = classify_events(tracks, config.roi)

    write_events_csv(events, output_dir / "events.csv", source_video=config.video_path.name)
    write_summary_json(events, {"video_path": str(config.video_path)}, output_dir / "summary.json")
