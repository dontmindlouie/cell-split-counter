"""GPT-vision review of ambiguous lineage events — alternate backend to review.py's
Claude-based _review_and_classify.

Mirrors _review_and_classify exactly (same frame cropping/sampling, same system
prompt, same JSON schema) but calls Azure OpenAI's chat completions API instead of
the Anthropic Messages API, so the two backends are interchangeable behind
review_ambiguous(backend=...).

Spike results (2026-07-06, Tom20 video, 180 shared candidate splits, gpt-5-mini):
recall parity with Claude Haiku 4.5 (96.7%) but precision 20.0% vs Claude's 34.1% --
GPT-5-mini is markedly more permissive calling ambiguous crops "real" with this same
prompt. Kept as an option for spending down Azure credit, not as the default.
"""

import json
from pathlib import Path

import cv2
import base64
from openai import AzureOpenAI

from src.classify import LineageEvent
from src.review import (
    _DEATH_SYSTEM,
    _EMPTY_DEATH_VERDICT,
    _EMPTY_VERDICT,
    _FRAME_STRIDE,
    _FRAMES_AFTER,
    _FRAMES_BEFORE,
    _MARKER_PROMPT_LINE,
    _SYSTEM,
    _build_dense_debug_window,
    _build_frame_window,
    _build_review_content,
    _crop_image,
    _crop_image_with_offset,
    _death_relation_labels,
    _death_trailing_caption,
    _draw_corner_ticks,
    _estimate_cost_usd,
    _save_debug_crops,
    _write_verdict_txt,
    adaptive_radius,
)

# Re-export for callers that imported estimate_cost_usd directly from this module.
estimate_cost_usd = _estimate_cost_usd


def _load_image_block(
    path: Path,
    centroid: tuple[float, float] | None = None,
    marker_radius: float | None = None,
) -> dict:
    if marker_radius is not None and centroid is not None:
        crop, x0, y0 = _crop_image_with_offset(path, centroid)
        crop = _draw_corner_ticks(crop, centroid[0] - x0, centroid[1] - y0, radius=marker_radius)
    else:
        crop = _crop_image(path, centroid)
    ok, buf = cv2.imencode(".png", crop)
    raw = buf.tobytes() if ok else path.read_bytes()
    data = base64.standard_b64encode(raw).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def review_and_classify_gpt(
    client: AzureOpenAI,
    event: LineageEvent,
    frame_dir: Path,
    deployment: str,
    debug_dir: Path | None = None,
    usage_log: list[dict] | None = None,
    reasoning_effort: str = "low",
) -> dict:
    """GPT equivalent of review._review_and_classify. Identical crop window and prompt."""
    indexed_paths = _build_frame_window(event, frame_dir)
    if not indexed_paths:
        return _EMPTY_VERDICT.copy()

    event_debug_dir: Path | None = None
    if debug_dir is not None:
        dense_paths = _build_dense_debug_window(event, frame_dir)
        event_debug_dir = _save_debug_crops(dense_paths, event, debug_dir)

    content = _build_review_content(indexed_paths, event, _load_image_block)

    system = _SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER, stride=_FRAME_STRIDE) + _MARKER_PROMPT_LINE
    response = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=2000,
        reasoning_effort=reasoning_effort,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )

    if usage_log is not None:
        usage_log.append({
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        })

    text = response.choices[0].message.content.strip()
    parsed = json.loads(text)

    if event_debug_dir is not None:
        _write_verdict_txt(event_debug_dir, parsed)

    return parsed


def review_death_gpt(
    client: AzureOpenAI,
    event: LineageEvent,
    frame_dir: Path,
    deployment: str,
    debug_dir: Path | None = None,
    usage_log: list[dict] | None = None,
    reasoning_effort: str = "low",
) -> dict:
    """GPT equivalent of review._review_death_event. Identical frame window and prompt."""
    indexed_paths = _build_frame_window(event, frame_dir)
    if not indexed_paths:
        return _EMPTY_DEATH_VERDICT.copy()

    event_debug_dir: Path | None = None
    if debug_dir is not None:
        dense_paths = _build_dense_debug_window(event, frame_dir)
        event_debug_dir = _save_debug_crops(dense_paths, event, debug_dir)

    relation_target, at_target_label = _death_relation_labels()
    content = _build_review_content(
        indexed_paths, event, _load_image_block,
        relation_target=relation_target, at_target_label=at_target_label,
        trailing_caption_fn=_death_trailing_caption,
    )

    system = _DEATH_SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER, stride=_FRAME_STRIDE)
    response = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=1500,
        reasoning_effort=reasoning_effort,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    )

    if usage_log is not None:
        usage_log.append({
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        })

    text = response.choices[0].message.content.strip()
    parsed = json.loads(text)

    if event_debug_dir is not None:
        _write_verdict_txt(event_debug_dir, parsed)

    return parsed
