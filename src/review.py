"""Claude-vision review of ambiguous lineage events flagged by classify.py."""

import base64
import dataclasses
import json
import os
from pathlib import Path

import anthropic

from src.classify import LineageEvent

_FRAMES_BEFORE = 2
_FRAMES_AFTER = 3

# System prompt — uses .format() so JSON braces must be doubled.
_SYSTEM = """\
You are reviewing microscopy timelapse images to verify a candidate cell division event.
You will receive frames in chronological order: {before} frames before the split and {after} \
frames after it. The split is expected to occur between the last before-frame and the first \
after-frame.

Real divisions: one cell rounds up, elongates along a cleavage plane, and becomes two \
distinct, persistent daughter cells.
False positives arise from: (1) shape-change flickering where a cell briefly appears as two \
touching masks, (2) z-plane focus drift that momentarily blurs one cell into two blobs, \
(3) a tracking ID swap with a nearby unrelated cell.

Respond with a JSON object — no other text:
{{"verdict": "real" | "false_positive", "confidence": <float 0.0–1.0>, "reason": "<one sentence>"}}"""


def _find_frame(frame_dir: Path, index: int) -> Path | None:
    matches = list(frame_dir.glob(f"frame_{index:05d}_*.png"))
    return matches[0] if matches else None


def _load_image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _review_split(
    client: anthropic.Anthropic,
    event: LineageEvent,
    frame_dir: Path,
    model: str,
) -> tuple[str, float]:
    """Ask Claude whether the split at event.frame is real. Returns (verdict, confidence)."""
    indices = list(range(max(0, event.frame - _FRAMES_BEFORE), event.frame + _FRAMES_AFTER + 1))
    indexed_paths = [(i, p) for i in indices if (p := _find_frame(frame_dir, i)) is not None]

    if not indexed_paths:
        return "false_positive", 0.0

    before_count = sum(1 for i, _ in indexed_paths if i < event.frame)
    after_count = len(indexed_paths) - before_count

    content = [_load_image_block(p) for _, p in indexed_paths]
    content.append({
        "type": "text",
        "text": (
            f"Candidate split at frame {event.frame}: track {event.parent_id} → daughter tracks. "
            f"{before_count} frames before and {after_count} frames after the split are shown."
        ),
    })

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER),
        messages=[{"role": "user", "content": content}],
    )

    text = (
        response.content[0].text.strip()
        .removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )
    parsed = json.loads(text)
    return str(parsed["verdict"]), float(parsed["confidence"])


def review_ambiguous(
    events: list[LineageEvent],
    frame_dir: Path,
    *,
    confidence_threshold: float = 1.0,
    model: str = "claude-haiku-4-5",
    max_reviews: int = 50,
) -> list[LineageEvent]:
    """Re-classify low-confidence events using Claude vision.

    Events at or above confidence_threshold pass through unchanged.
    For events below the threshold, Claude reviews the frame window around the split
    and returns a verdict; classification_source is updated to "claude" and confidence
    reflects Claude's assessment (0.0 for false positives).

    max_reviews caps the number of unique split points sent to Claude — daughter events
    from the same split share one API call, so the actual call count is at most
    min(unique_split_points, max_reviews). Events beyond the cap pass through unchanged.
    Raise max_reviews only after you have a sense of typical event counts per video.

    Daughter events from the same split share one Claude call — frames are identical
    so there is no point querying twice.
    """
    to_review = [e for e in events if e.confidence < confidence_threshold]
    passing = [e for e in events if e.confidence >= confidence_threshold]

    if not to_review:
        return events

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — cannot run Claude vision review")

    client = anthropic.Anthropic(api_key=api_key)

    # Cache verdicts per split point (parent_id, frame) so daughters share one API call.
    split_verdict: dict[tuple[int | None, int], tuple[str, float]] = {}
    reviewed = []

    for event in to_review:
        key = (event.parent_id, event.frame)

        if key not in split_verdict:
            if len(split_verdict) >= max_reviews:
                # Cap reached — pass remaining events through unchanged.
                reviewed.append(event)
                continue

            try:
                verdict, confidence = _review_split(client, event, frame_dir, model)
            except Exception:
                # API failure: preserve original rule-based classification.
                split_verdict[key] = ("real", event.confidence)
                reviewed.append(event)
                continue
            split_verdict[key] = (verdict, confidence)

        verdict, confidence = split_verdict[key]
        reviewed.append(dataclasses.replace(
            event,
            classification_source="claude",
            confidence=confidence if verdict == "real" else 0.0,
        ))

    return passing + reviewed
