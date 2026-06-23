"""End-to-end orchestration: video in, events out."""

from pathlib import Path

from src.classify import classify_events
from src.ingest import IngestConfig, extract_frames
from src.output import write_events_csv, write_summary_json
from src.review import review_ambiguous
from src.segment import segment_all
from src.track import link_frames


def run(config: IngestConfig, frame_dir: Path, output_dir: Path) -> None:
    frame_paths = extract_frames(config, frame_dir)
    masks_by_frame = segment_all(frame_paths)
    tracks = link_frames(masks_by_frame)
    events = classify_events(tracks, config.roi)
    events = review_ambiguous(events, frame_dir)

    write_events_csv(events, output_dir / "events.csv")
    write_summary_json(events, {"video_path": str(config.video_path)}, output_dir / "summary.json")
