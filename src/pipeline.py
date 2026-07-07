"""End-to-end orchestration: video in, events out."""

import dataclasses
from pathlib import Path

import cv2

from src.classify import NEAR_EDGE_MARGIN_PX, classify_events
from src.ingest import IngestConfig, extract_frames, get_pixel_size_um
from src.output import write_events_csv, write_summary_json
from src.review import review_ambiguous
from src.segment import load_video_arrays, segment_all, segment_video_arrays
from src.track import link_frames, link_frames_trackastra


def run(
    config: IngestConfig,
    frame_dir: Path,
    output_dir: Path,
    tracker: str = "trackastra",
    tracker_mode: str = "greedy",
    start_frame: int = 0,
    end_frame: int | None = None,
    save_debug_crops: bool = False,
    reuse_masks: bool = False,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    segmentation_model: str = "cyto3",
    pixel_size_um: float | None = None,
    vision_backend: str = "claude",
    gpt_reasoning_effort: str = "medium",
    min_gpt_confidence: float = 0.85,
) -> None:
    if pixel_size_um is None:
        pixel_size_um = get_pixel_size_um(config.video_path)

    frame_paths = extract_frames(config, frame_dir)

    if start_frame != 0 or end_frame is not None:
        frame_paths = frame_paths[start_frame:end_frame]
        print(f"  frame range: {start_frame}–{end_frame or len(frame_paths) + start_frame} ({len(frame_paths)} frames)")

    if tracker == "trackastra":
        if reuse_masks:
            frames_arr, labels_arr = load_video_arrays(frame_paths)
        else:
            frames_arr, labels_arr = segment_video_arrays(
                frame_paths, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
                model_type=segmentation_model,
            )
        tracks = link_frames_trackastra(frames_arr, labels_arr, mode=tracker_mode)
    else:
        masks_by_frame = segment_all(frame_paths, model_type=segmentation_model)
        tracks = link_frames(masks_by_frame)

    events = classify_events(tracks, config.roi)

    total_frames = len(frame_paths)
    frame_h, frame_w = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE).shape

    def _is_near_edge(centroid: tuple[float, float] | None) -> bool | None:
        if centroid is None:
            return None
        cx, cy = centroid
        m = NEAR_EDGE_MARGIN_PX
        return cx < m or cx > frame_w - m or cy < m or cy > frame_h - m

    def _cell_size_um2(area_px: float | None) -> float | None:
        if area_px is None or pixel_size_um is None:
            return None
        return area_px * pixel_size_um ** 2

    events = [
        dataclasses.replace(
            e, bleach_risk=e.frame / total_frames, near_edge=_is_near_edge(e.centroid),
            cell_size_um2=_cell_size_um2(e.cell_area_px),
        )
        for e in events
    ]

    # upper_threshold=inf: every non-suppressed event gets a Claude verdict, notes, AND
    # division-type/abnormality classification in one combined call, instead of
    # persistence-confirmed (confidence>=1.0) events skipping review.
    # max_reviews raised so busy videos don't silently fall back to rule-only
    # classification once the default 50-split-point cap is hit.
    vision_usage: dict = {}
    events = review_ambiguous(
        events, frame_dir, upper_threshold=float("inf"), max_reviews=10_000,
        backend=vision_backend, save_debug_crops=save_debug_crops, usage_out=vision_usage,
        gpt_reasoning_effort=gpt_reasoning_effort, min_gpt_confidence=min_gpt_confidence,
    )

    write_events_csv(events, output_dir / "events.csv", source_video=config.video_path.name)
    write_summary_json(
        events, {"video_path": str(config.video_path), "pixel_size_um": pixel_size_um},
        output_dir / "summary.json", claude_usage=vision_usage
    )
