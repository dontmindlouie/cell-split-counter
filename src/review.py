"""Claude-vision review of ambiguous lineage events flagged by classify.py."""

import base64
import dataclasses
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import cv2

from src.classify import LineageEvent
from src.config import CLAUDE_MODEL

_FRAMES_BEFORE = 8  # see metaphase plate alignment; widened 2026-07-03 sweep, see docs/investigation_notes.md
_FRAMES_AFTER = 8   # see cytokinesis + any micronuclei forming; widened 2026-07-03 sweep, see docs/investigation_notes.md
_FRAME_STRIDE = 3   # sample every Nth frame instead of consecutive frames, see docs/investigation_notes.md 2026-07-03
_CROP_RADIUS = 192  # px around centroid; 384px box comfortably fits parent + both daughters

# USD per million tokens. Only models we've actually used are listed; an unrecognized
# model reports token counts but skips the cost estimate rather than guessing pricing.
_MODEL_PRICING_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (in_price, out_price) in _MODEL_PRICING_PER_MTOK.items():
        if model.startswith(prefix):
            return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return None

# System prompt — uses .format() so JSON braces must be doubled.
# Single call does both false-positive verification and division-type/abnormality
# classification, so every reviewed event comes back fully annotated instead of
# requiring a second opt-in pass (review_division_type used to be separate and
# defaulted off, which silently left events with notes but no classification).
_SYSTEM = """\
You are reviewing fluorescence microscopy timelapse images of cells to verify and characterize \
a candidate cell split event. You will receive {before} frames before and {after} frames after \
the suspected split point, sampled every {stride} frames (not consecutive) to cover a longer \
time span.

STEP 1 — IS THIS A REAL SPLIT?
Real splits: one cell rounds up, elongates along a cleavage plane, and becomes two distinct \
persistent daughter cells. Splits can be slow — a nucleus may still look like a single \
waisted/hourglass shape partway through the sequence and only clearly resolve into two \
separated lobes by the last frame or two. Judge the whole trend across the sequence, not just \
the frame nearest the split: progressive elongation and constriction that is still resolving by \
the final frame is real division evidence, even without full separation in every frame shown.
False positives arise from: (1) shape-change flickering where a cell briefly appears as two \
touching masks then reverts, with no net progression toward separation, (2) z-plane focus drift \
that momentarily blurs one cell into two blobs, (3) a tracking ID swap with a nearby unrelated cell.

STEP 2 — IF REAL, CHARACTERIZE THE EVENT:
Split type:
- "symmetric": daughters approximately equal in size (typical mitosis)
- "asymmetric": one daughter is clearly larger (stem cell-like or budding)
- "multi_way": three or more daughters visible
- "failed": cytokinesis began but daughters re-fused

ACD classification (spindle geometry from chromosome staining):
- "bipolar": chromosomes separate into exactly 2 groups (normal)
- "tripolar": 3 groups (extra spindle pole)
- "multipolar": 4 or more groups
- "unknown": image quality too poor to determine

Chromosomal abnormalities — examine each frame carefully:
- misaligned_chromosomes: one or more chromosomes offset from the metaphase plate pre-split
- lagging_chromosome: a single chromosome or fragment trailing between the two separating masses
- anaphase_bridge: a thin continuous chromatin thread connecting the two separating chromosome masses
- micronucleus: a small distinct bright spot separate from the main daughter nucleus in post-split frames
- binucleation: a single cell body containing two separate, similarly-sized nuclei that do NOT \
progressively separate (karyokinesis without cytokinesis) -- distinct from a normal division still \
in progress, and distinct from split_type "failed" (which re-fuses back into one nucleus)

Also note any other interesting anomaly worth case study (unusual morphology, unexpected behavior, etc.)

Respond with a JSON object — no other text:
{{"verdict": "real" | "false_positive", \
"confidence": <float 0.0-1.0>, \
"split_type": "symmetric" | "asymmetric" | "multi_way" | "failed" | null, \
"description": "<one or two sentences>", \
"acd_division_type": "bipolar" | "tripolar" | "multipolar" | "unknown" | null, \
"misaligned_chromosomes": true | false | null, \
"lagging_chromosome": true | false | null, \
"anaphase_bridge": true | false | null, \
"micronucleus": true | false | null, \
"binucleation": true | false | null, \
"anomaly_notes": "<describe any interesting anomaly worth case study, or null>"}}

Set split_type, acd_division_type, all boolean fields, and anomaly_notes to null for false positives."""


def _find_frame(frame_dir: Path, index: int) -> Path | None:
    matches = list(frame_dir.glob(f"frame_{index:05d}_*.png"))
    return matches[0] if matches else None


def _crop_image(path: Path, centroid: tuple[float, float] | None = None):
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


def _review_and_classify(
    client: anthropic.Anthropic,
    event: LineageEvent,
    frame_dir: Path,
    model: str,
    debug_dir: Path | None = None,
    usage_log: list[dict] | None = None,
) -> dict:
    """Single API call: verify the split is real AND classify type/abnormalities.

    Returns the full parsed JSON dict (verdict, confidence, split_type, description,
    acd_division_type, misaligned_chromosomes, lagging_chromosome, anaphase_bridge,
    micronucleus, binucleation, anomaly_notes).
    """
    before_indices = [event.frame - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
    after_indices = [event.frame + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
    indices = [i for i in before_indices if i >= 0] + [event.frame] + after_indices
    indexed_paths = [(i, p) for i in indices if (p := _find_frame(frame_dir, i)) is not None]

    if not indexed_paths:
        return {
            "verdict": "false_positive", "confidence": 0.0, "split_type": None,
            "description": "no frames found", "acd_division_type": None,
            "misaligned_chromosomes": None, "lagging_chromosome": None,
            "anaphase_bridge": None, "micronucleus": None, "binucleation": None,
            "anomaly_notes": None,
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
            f"Candidate split at frame {event.frame}: track {event.parent_id} → daughter tracks. "
            f"{before_count} frames before and {after_count} frames after the split are shown, "
            f"each {_FRAME_STRIDE} frames apart."
        ),
    })

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER, stride=_FRAME_STRIDE),
        messages=[{"role": "user", "content": content}],
    )

    if usage_log is not None:
        usage_log.append({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

    text = (
        response.content[0].text.strip()
        .removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    )
    parsed = json.loads(text)

    if event_debug_dir is not None:
        (event_debug_dir / "verdict.txt").write_text(
            f"verdict:    {parsed.get('verdict', '')}\n"
            f"confidence: {float(parsed.get('confidence', 0.0)):.2f}\n"
            f"notes:      {parsed.get('description', '')}\n",
            encoding="utf-8",
        )

    return parsed


def review_ambiguous(
    events: list[LineageEvent],
    frame_dir: Path,
    *,
    lower_threshold: float = 0.05,
    upper_threshold: float = 1.0,
    backend: str = "claude",
    model: str | None = None,
    max_reviews: int = 50,
    save_debug_crops: bool = False,
    max_workers: int = 10,
    usage_out: dict | None = None,
    gpt_reasoning_effort: str = "medium",
    min_gpt_confidence: float = 0.85,
) -> list[LineageEvent]:
    """Three-tier confidence routing for split events.

    Tier 1 — suppress (confidence < lower_threshold): daughters vanished in 0 frames,
      definitely noise. Dropped from output entirely.
    Tier 2 — vision review (lower_threshold <= confidence < upper_threshold): ambiguous;
      the model inspects the frame window and returns a verdict.
    Tier 3 — auto-confirm (confidence >= upper_threshold): daughters persisted long enough
      to be confident; pass through unchanged.

    backend selects the vision model: "claude" (default, Claude Haiku 4.5 via the
    Anthropic API) or "gpt" (GPT via Azure OpenAI, config.GPT_DEPLOYMENT). model overrides
    the default deployment/model name for the chosen backend.

    GPT-backend tuning, per the 2026-07-06/07 spike on the same 180-candidate baseline:
    - gpt_reasoning_effort: "low" (20.0% precision), "medium" (22.1% raw, best option --
      "high" burns most of its 2000-token budget on invisible reasoning tokens and fails
      outright on ~68% of calls with an empty response, not currently usable).
    - min_gpt_confidence: real verdicts below this (using GPT's own self-reported
      confidence, not the tracker's) are downgraded to false_positive. At 0.85 with
      medium effort this brings precision to 36.5% (recall 90.0%, F1 0.519) -- slightly
      beating Claude's F1 (0.504) on this same dataset. Set to 0.0 to disable filtering.
      Only applied to the "gpt" backend; no effect on "claude".

    max_reviews caps unique split points sent for review (daughters share one call).
    Events beyond the cap pass through unchanged in Tier 2. Unique split points are
    reviewed concurrently (max_workers at a time) -- these are network-bound API calls,
    not local compute, so parallelizing them is safe and doesn't compete with Cellpose/GPU.

    If usage_out is given, it's populated (in place) with token/cost totals for this
    review pass: {"api_calls", "input_tokens", "output_tokens", "estimated_cost_usd"}.
    estimated_cost_usd is None for models without a known price in the backend's pricing table.
    """
    if backend not in ("claude", "gpt"):
        raise ValueError(f"backend must be 'claude' or 'gpt', got {backend!r}")
    suppressed  = [e for e in events if e.confidence < lower_threshold]
    to_review   = [e for e in events if lower_threshold <= e.confidence < upper_threshold]
    passing     = [e for e in events if e.confidence >= upper_threshold]

    if suppressed:
        print(f"  suppressed {len(suppressed)} events (confidence < {lower_threshold})")

    if not to_review:
        return passing

    debug_dir: Path | None = None
    if save_debug_crops:
        debug_dir = frame_dir.parent / "review_crops"
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True)
        print(f"  saving crops to {debug_dir}")

    if backend == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set — cannot run Claude vision review")
        client = anthropic.Anthropic(api_key=api_key)
        resolved_model = model or CLAUDE_MODEL
        review_fn = _review_and_classify
    else:
        from openai import AzureOpenAI

        from src.config import GPT_DEPLOYMENT
        from src.review_gpt import review_and_classify_gpt

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set — cannot run GPT vision review")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview")
        resolved_model = model or GPT_DEPLOYMENT
        review_fn = review_and_classify_gpt

    # One representative event per unique split point, in order, capped at max_reviews.
    to_call: dict[tuple[int | None, int], LineageEvent] = {}
    for event in to_review:
        key = (event.parent_id, event.frame)
        if key not in to_call and len(to_call) < max_reviews:
            to_call[key] = event

    usage_log: list[dict] = []

    def _call(key: tuple[int | None, int], event: LineageEvent):
        try:
            if backend == "gpt":
                return key, review_fn(client, event, frame_dir, resolved_model, debug_dir=debug_dir, usage_log=usage_log, reasoning_effort=gpt_reasoning_effort)
            return key, review_fn(client, event, frame_dir, resolved_model, debug_dir=debug_dir, usage_log=usage_log)
        except Exception as exc:
            # Fail open (treat as real, at the tracker's own confidence) rather than silently
            # dropping a possibly-genuine division -- but log why, since this used to swallow
            # the error entirely and looked identical to a real Claude confirmation downstream.
            print(f"  [review error] frame={event.frame} parent={event.parent_id}: {type(exc).__name__}: {exc}")
            return key, {"verdict": "real", "confidence": event.confidence, "split_type": None,
                         "description": "", "acd_division_type": None, "misaligned_chromosomes": None,
                         "lagging_chromosome": None, "anaphase_bridge": None, "micronucleus": None,
                         "binucleation": None, "anomaly_notes": None}

    split_result: dict[tuple[int | None, int], dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call, key, event) for key, event in to_call.items()]
        for future in as_completed(futures):
            key, result = future.result()
            split_result[key] = result
            event = to_call[key]
            verdict = result.get("verdict", "real")
            risk_str = f" bleach={event.bleach_risk:.2f}" if event.bleach_risk is not None else ""
            print(f"  frame={event.frame:3d} parent={event.parent_id} [{verdict}]{risk_str} {result.get('description', '')}")

    total_input = sum(u["input_tokens"] for u in usage_log)
    total_output = sum(u["output_tokens"] for u in usage_log)
    if backend == "claude":
        cost = _estimate_cost_usd(resolved_model, total_input, total_output)
    else:
        from src.review_gpt import estimate_cost_usd as _estimate_cost_usd_gpt
        cost = _estimate_cost_usd_gpt(resolved_model, total_input, total_output)
    cost_str = f", est. ${cost:.4f}" if cost is not None else ""
    print(f"  {backend} usage: {len(usage_log)} calls, {total_input:,} input tokens, {total_output:,} output tokens{cost_str}")
    if usage_out is not None:
        usage_out.update({
            "api_calls": len(usage_log),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": cost,
        })

    reviewed = []
    for event in to_review:
        key = (event.parent_id, event.frame)
        if key not in split_result:
            # Beyond the max_reviews cap -- pass through unchanged.
            reviewed.append(event)
            continue
        r = split_result[key]
        verdict = r.get("verdict", "real")
        confidence = float(r.get("confidence", event.confidence))
        if backend == "gpt" and verdict == "real" and confidence < min_gpt_confidence:
            verdict = "false_positive"
        is_real = verdict == "real"
        split_type = r.get("split_type") or ""
        description = r.get("description", "")
        notes = f"{split_type}: {description}".strip(": ") if split_type else description
        reviewed.append(dataclasses.replace(
            event,
            classification_source=resolved_model,
            confidence=confidence if is_real else 0.0,
            tracker_confidence=event.tracker_confidence if event.tracker_confidence is not None else event.confidence,
            claude_notes=notes,
            acd_division_type=r.get("acd_division_type") if is_real else None,
            misaligned_chromosomes=r.get("misaligned_chromosomes") if is_real else None,
            lagging_chromosome=r.get("lagging_chromosome") if is_real else None,
            anaphase_bridge=r.get("anaphase_bridge") if is_real else None,
            micronucleus=r.get("micronucleus") if is_real else None,
            binucleation=r.get("binucleation") if is_real else None,
            anomaly_notes=r.get("anomaly_notes") if is_real else None,
        ))

    return passing + reviewed

