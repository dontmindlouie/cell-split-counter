"""End-to-end orchestration: video in, events out."""

import dataclasses
from pathlib import Path

from src.classify import classify_events
from src.ingest import IngestConfig, extract_frames
from src.output import write_events_csv, write_summary_json
from src.review import review_ambiguous
from src.segment import segment_all, segment_video_arrays
from src.track import link_frames, link_frames_trackastra


def run(config: IngestConfig, frame_dir: Path, output_dir: Path, tracker: str = "trackastra") -> None:
    frame_paths = extract_frames(config, frame_dir)

    if tracker == "trackastra":
        frames_arr, labels_arr = segment_video_arrays(frame_paths)
        tracks = link_frames_trackastra(frames_arr, labels_arr)
    else:
        masks_by_frame = segment_all(frame_paths)
        tracks = link_frames(masks_by_frame)

    events = classify_events(tracks, config.roi)

    total_frames = len(frame_paths)
    events = [dataclasses.replace(e, bleach_risk=e.frame / total_frames) for e in events]

    events = review_ambiguous(events, frame_dir)

    write_events_csv(events, output_dir / "events.csv", source_video=config.video_path.name)
    write_summary_json(events, {"video_path": str(config.video_path)}, output_dir / "summary.json")
