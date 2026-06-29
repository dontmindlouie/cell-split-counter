"""Claude-vision review of ambiguous lineage events flagged by classify.py."""

import base64
import dataclasses
import json
import os
import shutil
from pathlib import Path

import anthropic
import cv2
import numpy as np

from src.classify import LineageEvent

_FRAMES_BEFORE = 2
_FRAMES_AFTER = 3
_CROP_RADIUS = 192  # px around centroid; 384px box comfortably fits parent + both daughters

# System prompt — uses .format() so JSON braces must be doubled.
_SYSTEM = """\
You are reviewing microscopy timelapse images to verify and characterize a candidate cell division event.
You will receive frames in chronological order: {before} frames before the split and {after} \
frames after it. The split is expected to occur between the last before-frame and the first \
after-frame.

Real divisions: one cell rounds up, elongates along a cleavage plane, and becomes two \
distinct, persistent daughter cells.
False positives arise from: (1) shape-change flickering where a cell briefly appears as two \
touching masks, (2) z-plane focus drift that momentarily blurs one cell into two blobs, \
(3) a tracking ID swap with a nearby unrelated cell.

For real divisions, also classify the split type:
- "symmetric": daughters are approximately equal in size (typical mitosis)
- "asymmetric": one daughter is clearly larger than the other (stem cell-like or budding)
- "multi_way": three or more daughters visible

Respond with a JSON object — no other text:
{{"verdict": "real" | "false_positive", "confidence": <float 0.0–1.0>, "split_type": "symmetric" | "asymmetric" | "multi_way" | null, "description": "<one or two sentences describing what is observed>"}}"""


def _find_frame(frame_dir: Path, index: int) -> Path | None:
    matches = list(frame_dir.glob(f"frame_{index:05d}_*.png"))
    return matches[0] if matches else None


def _crop_image(path: Path, centroid: tuple[float, float] | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if centroid is None:
        return img
    cx, cy = int(centroid[0]), int(centroid[1])
    h, w = img.shape
    y0, y1 = max(0, cy - _CROP_RADIUS), min(h, cy + _CROP_RADIUS)
    x0, x1 = max(0, cx - _CROP_RADIUS), min(w, cx + _CROP_RADIUS)
    return img[y0:y1, x0:x1]


def _load_image_block(path: Path, centroid: tuple[float, float] | None = None) -> dict:
    crop = _crop_image(path, centroid)
    ok, buf = cv2.imencode(".png", crop)
    raw = buf.tobytes() if ok else path.read_bytes()
    data = base64.standard_b64encode(raw).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _review_split(
    client: anthropic.Anthropic,
    event: LineageEvent,
    frame_dir: Path,
    model: str,
    debug_dir: Path | None = None,
) -> tuple[str, float, str]:
    """Ask Claude whether the split at event.frame is real. Returns (verdict, confidence, notes)."""
    indices = list(range(max(0, event.frame - _FRAMES_BEFORE), event.frame + _FRAMES_AFTER + 1))
    indexed_paths = [(i, p) for i in indices if (p := _find_frame(frame_dir, i)) is not None]

    if not indexed_paths:
        return "false_positive", 0.0

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
    split_type = parsed.get("split_type") or ""
    description = parsed.get("description", "")
    notes = f"{split_type}: {description}".strip(": ") if split_type else description
    verdict = str(parsed["verdict"])
    confidence = float(parsed["confidence"])

    if event_debug_dir is not None:
        (event_debug_dir / "verdict.txt").write_text(
            f"verdict:    {verdict}\nconfidence: {confidence:.2f}\nnotes:      {notes}\n"
        )

    return verdict, confidence, notes


def review_ambiguous(
    events: list[LineageEvent],
    frame_dir: Path,
    *,
    lower_threshold: float = 0.05,
    upper_threshold: float = 1.0,
    model: str = "claude-haiku-4-5",
    max_reviews: int = 50,
    save_debug_crops: bool = False,
) -> list[LineageEvent]:
    """Three-tier confidence routing for split events.

    Tier 1 — suppress (confidence < lower_threshold): daughters vanished in 0 frames,
      definitely noise. Dropped from output entirely.
    Tier 2 — Claude review (lower_threshold <= confidence < upper_threshold): ambiguous;
      Claude inspects the frame window and returns a verdict.
    Tier 3 — auto-confirm (confidence >= upper_threshold): daughters persisted long enough
      to be confident; pass through unchanged.

    max_reviews caps unique split points sent to Claude (daughters share one call).
    Events beyond the cap pass through unchanged in Tier 2.
    """
    suppressed  = [e for e in events if e.confidence < lower_threshold]
    to_review   = [e for e in events if lower_threshold <= e.confidence < upper_threshold]
    passing     = [e for e in events if e.confidence >= upper_threshold]

    if suppressed:
        print(f"  suppressed {len(suppressed)} events (confidence < {lower_threshold})")

    if not to_review:
        return passing

    debug_dir: Path | None = None
    if save_debug_crops:
        debug_dir = frame_dir.parent / "debug" / "crops"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True)
        print(f"  saving crops to {debug_dir}")

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
                verdict, confidence, notes = _review_split(client, event, frame_dir, model, debug_dir=debug_dir)
            except Exception:
                split_verdict[key] = ("real", event.confidence, "")
                reviewed.append(event)
                continue
            risk_str = f" bleach={event.bleach_risk:.2f}" if event.bleach_risk is not None else ""
            print(f"  frame={event.frame:3d} parent={event.parent_id} [{verdict}]{risk_str} {notes}")
            split_verdict[key] = (verdict, confidence, notes)

        verdict, confidence, notes = split_verdict[key]
        reviewed.append(dataclasses.replace(
            event,
            classification_source="claude",
            confidence=confidence if verdict == "real" else 0.0,
            tracker_confidence=event.tracker_confidence if event.tracker_confidence is not None else event.confidence,
            claude_notes=notes if verdict == "real" else None,
        ))

    return passing + reviewed
