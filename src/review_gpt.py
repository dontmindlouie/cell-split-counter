"""GPT-vision review of ambiguous lineage events -- alternate backend to review.py's
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

import base64
import json
from pathlib import Path

import cv2
from openai import AzureOpenAI

from src.classify import LineageEvent
from src.review import (
    _FRAME_STRIDE,
    _FRAMES_AFTER,
    _FRAMES_BEFORE,
    _SYSTEM,
    _crop_image,
    _find_frame,
)

# USD per million tokens (Azure OpenAI GlobalStandard rate). Only models actually
# used are listed; an unrecognized model skips the cost estimate rather than guessing.
_MODEL_PRICING_PER_MTOK = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_price, out_price) in _MODEL_PRICING_PER_MTOK.items():
        if model.startswith(prefix):
            return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return None


def _load_image_block(path: Path, centroid: tuple[float, float] | None = None) -> dict:
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
    before_indices = [event.frame - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
    after_indices = [event.frame + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
    indices = [i for i in before_indices if i >= 0] + [event.frame] + after_indices
    indexed_paths = [(i, p) for i in indices if (p := _find_frame(frame_dir, i)) is not None]

    if not indexed_paths:
        return {
            "verdict": "false_positive", "confidence": 0.0, "split_type": None,
            "description": "no frames found", "acd_division_type": None,
            "misaligned_chromosomes": None, "lagging_chromosome": None,
            "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None,
        }

    before_count = sum(1 for i, _ in indexed_paths if i < event.frame)
    after_count = len(indexed_paths) - before_count

    event_debug_dir: Path | None = None
    if debug_dir is not None:
        event_debug_dir = debug_dir / f"frame_{event.frame:05d}_parent_{event.parent_id}"
        event_debug_dir.mkdir(parents=True, exist_ok=True)
        for pos, (idx, path) in enumerate(indexed_paths):
            label = "before" if idx < event.frame else ("split" if idx == event.frame else "after")
            crop = _crop_image(path, event.centroid)
            cv2.imwrite(str(event_debug_dir / f"{pos:02d}_{label}_{idx:05d}.png"), crop)

    content = [_load_image_block(p, event.centroid) for _, p in indexed_paths]
    content.append({
        "type": "text",
        "text": (
            f"Candidate split at frame {event.frame}: track {event.parent_id} -> daughter tracks. "
            f"{before_count} frames before and {after_count} frames after the split are shown, "
            f"each {_FRAME_STRIDE} frames apart."
        ),
    })

    response = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=2000,
        reasoning_effort=reasoning_effort,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER, stride=_FRAME_STRIDE)},
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
        (event_debug_dir / "verdict.txt").write_text(
            f"verdict:    {parsed.get('verdict', '')}\n"
            f"confidence: {float(parsed.get('confidence', 0.0)):.2f}\n"
            f"notes:      {parsed.get('description', '')}\n",
            encoding="utf-8",
        )

    return parsed
