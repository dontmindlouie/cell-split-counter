"""Vision review of ambiguous lineage events flagged by classify.py.

Supports two backends (selectable via review_ambiguous(backend=...)):
  - "claude"  Anthropic Claude Haiku (ANTHROPIC_API_KEY)  — higher precision
  - "gpt"     Azure OpenAI GPT       (AZURE_OPENAI_*)     — lower cost

Shared helpers (_build_frame_window, _save_debug_crops, _write_verdict_txt,
_EMPTY_VERDICT, _estimate_cost_usd) are imported by review_gpt.py so both
backends stay in sync without duplicating crop / verdict logic.
"""

import base64
import dataclasses
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import cv2

from src.classify import EventType, LineageEvent
from src.config import CLAUDE_MODEL

_FRAMES_BEFORE = 8  # see metaphase plate alignment; widened 2026-07-03 sweep, see docs/investigation_notes.md
_FRAMES_AFTER = 8   # see cytokinesis + any micronuclei forming; widened 2026-07-03 sweep, see docs/investigation_notes.md
_FRAME_STRIDE = 3   # sample every Nth frame instead of consecutive frames, see docs/investigation_notes.md 2026-07-03
_CROP_RADIUS = 192  # px around centroid; 384px box comfortably fits parent + both daughters

# Corner-bracket marker drawn on every crop to disambiguate the candidate cell from a
# simultaneously-dividing neighbor (2026-07-08 spike, spike/crop-marker-v2). Radius is
# adaptive per event (see adaptive_radius) so the box can't enclose a nearby neighbor
# or clip the candidate's own division -- see adaptive_radius's docstring for the two
# regressions (neighbor misattribution, GPT fragmentation misread) this fixes.
_TICK_RADIUS = 55       # px from centroid, used when no neighbor is nearby
_TICK_RADIUS_MIN = 15   # px floor -- below this the brackets start overlapping the cell body
_TICK_LEN = 14          # px, length of each corner-bracket arm
_TICK_COLOR = (60, 170, 230)  # BGR, dim/muted orange-amber -- not saturated cyan
_TICK_THICKNESS = 2
_EDGE_MARGIN = 6        # px -- keep ticks from being clipped exactly at the canvas edge

_MARKER_PROMPT_LINE = (
    "\n\nThe candidate cell is indicated by four corner brackets (thin, dim orange) "
    "in each image, positioned clear of the cell itself, not touching it. Evaluate "
    "only the cell centered within the brackets -- ignore division or anomaly "
    "activity on any other cell visible in the frame."
)


def adaptive_radius(
    neighbor_distance_px: float | None,
    margin: float = 5.0,
    fraction: float = 0.5,
    radius_min: int = _TICK_RADIUS_MIN,
    cell_area_px: float | None = None,
    size_k: float = 1.3,
) -> int:
    """Shrink the bracket radius so the marked box can't enclose a nearby neighbor,
    while not shrinking it below what the candidate cell's own size needs.

    Caps the radius at `fraction` of the distance to the nearest neighbor (minus a
    small margin) so the box structurally cannot contain both cells. Composed with a
    size floor (`size_k * candidate's own Cellpose-derived radius`, from
    `cell_area_px` via area=pi*r^2) so a genuinely large candidate cell doesn't get a
    box tight enough to clip its own division -- see the 2026-07-08 marker spike
    (spike/crop-marker-v2) for the n=8 validation on both Claude Haiku and GPT-5-mini
    that landed on fraction=0.5, margin=5.0, size_k=1.3 as one shared formula for
    both backends. Returns the fixed default radius if no neighbor distance is known.
    """
    if neighbor_distance_px is None:
        return _TICK_RADIUS
    floor = radius_min
    if cell_area_px is not None:
        cell_own_radius = math.sqrt(cell_area_px / math.pi)
        floor = max(floor, size_k * cell_own_radius)
    computed = neighbor_distance_px * fraction - margin
    return int(max(floor, min(_TICK_RADIUS, computed)))


def _draw_corner_ticks(crop, local_cx: float, local_cy: float, radius: float | None = None):
    """4 short L-shaped brackets at `radius` (default _TICK_RADIUS) from the point,
    pointing inward. Not a continuous ring, not touching the cell itself.

    Corner positions are clamped to the crop's actual bounds -- an edge-clamped crop
    (candidate near the frame boundary) can be narrower than the nominal 384px box.
    """
    out = crop.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    r = radius if radius is not None else _TICK_RADIUS
    l = _TICK_LEN
    m = _EDGE_MARGIN
    corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # dx, dy sign per corner
    for dx, dy in corners:
        cx = max(m, min(w - m, local_cx + dx * r))
        cy = max(m, min(h - m, local_cy + dy * r))
        hx = max(m, min(w - m, cx - dx * l))  # horizontal arm, far end clamped too
        cv2.line(out, (int(cx), int(cy)), (int(hx), int(cy)), _TICK_COLOR, _TICK_THICKNESS, cv2.LINE_AA)
        vy = max(m, min(h - m, cy - dy * l))  # vertical arm
        cv2.line(out, (int(cx), int(cy)), (int(cx), int(vy)), _TICK_COLOR, _TICK_THICKNESS, cv2.LINE_AA)
    return out

# USD per million tokens for every backend we've actually deployed.
# An unrecognised model reports token counts but skips the cost estimate.
_MODEL_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),   # Anthropic
    "gpt-5-mini":       (0.25, 2.00),   # Azure OpenAI
    "gpt-5.4-mini":     (0.75, 4.50),   # Azure OpenAI
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return estimated USD cost or None if the model isn't in the pricing table."""
    for prefix, (in_price, out_price) in _MODEL_PRICING_PER_MTOK.items():
        if model.startswith(prefix):
            return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price
    return None


# Verdict dict returned when no frame files are found for a candidate event.
_EMPTY_VERDICT: dict = {
    "verdict": "false_positive", "confidence": 0.0, "split_type": None,
    "description": "no frames found", "acd_division_type": None,
    "misaligned_chromosomes": None, "lagging_chromosome": None,
    "anaphase_bridge": None, "micronucleus": None, "binucleation": None,
    "anomaly_notes": None,
}


def _build_frame_window(
    event: "LineageEvent", frame_dir: Path
) -> list[tuple[int, Path]]:
    """Return (frame_index, path) pairs for the review window around event.frame."""
    before_indices = [event.frame - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
    after_indices  = [event.frame + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
    indices = [i for i in before_indices if i >= 0] + [event.frame] + after_indices
    return [(i, p) for i in indices if (p := _find_frame(frame_dir, i)) is not None]


def _save_debug_crops(
    indexed_paths: list[tuple[int, Path]],
    event: "LineageEvent",
    debug_dir: Path,
) -> Path:
    """Save per-event PNG crops under debug_dir; return the event subfolder path.

    Includes the same marker sent to the vision model, at the same radius, so debug
    crops show exactly what the model saw rather than a cleaner unmarked version.
    """
    event_debug_dir = debug_dir / f"frame_{event.frame:05d}_parent_{event.parent_id}"
    event_debug_dir.mkdir(parents=True, exist_ok=True)
    radius = adaptive_radius(event.neighbor_distance_px, cell_area_px=event.cell_area_px)
    for pos, (idx, path) in enumerate(indexed_paths):
        label = "before" if idx < event.frame else ("split" if idx == event.frame else "after")
        if event.centroid is not None:
            crop, x0, y0 = _crop_image_with_offset(path, event.centroid)
            crop = _draw_corner_ticks(crop, event.centroid[0] - x0, event.centroid[1] - y0, radius=radius)
        else:
            crop = _crop_image(path, None)
        cv2.imwrite(str(event_debug_dir / f"{pos:02d}_{label}_{idx:05d}.png"), crop)
    return event_debug_dir


def _write_verdict_txt(event_debug_dir: Path, parsed: dict) -> None:
    """Write a human-readable verdict summary alongside the debug crops."""
    (event_debug_dir / "verdict.txt").write_text(
        f"verdict:    {parsed.get('verdict', '')}\n"
        f"confidence: {float(parsed.get('confidence', 0.0)):.2f}\n"
        f"notes:      {parsed.get('description', '')}\n",
        encoding="utf-8",
    )

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
False positives arise from: (1) z-plane focus drift that momentarily blurs one cell into two \
blobs with no real division machinery ever visible (no rounding, no elongation, no cleavage \
furrow), (2) a tracking ID swap with a nearby unrelated cell. Do NOT call a real division attempt \
a false positive just because it doesn't finish: if the cell visibly rounds up, elongates along a \
cleavage plane, and a waist/furrow forms between two masses before they re-fuse back into one, \
that is a REAL event -- report verdict "real" with split_type "failed" (see Step 2), not \
false_positive.

STEP 2 — IF REAL, CHARACTERIZE THE EVENT:
Split type:
- "symmetric": daughters approximately equal in size (typical mitosis)
- "asymmetric": one daughter is clearly larger (stem cell-like or budding)
- "multi_way": three or more daughters visible
- "failed": cytokinesis began (rounding, elongation, cleavage furrow) but the daughters re-fused \
into one cell before separating -- this is a real, biologically meaningful event, not a false positive

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


def _crop_image_with_offset(path: Path, centroid: tuple[float, float]):
    """Like _crop_image, but also returns the crop's (x0, y0) offset in the source
    image -- needed to place the marker at the correct local position."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    cx, cy = int(centroid[0]), int(centroid[1])
    h, w = img.shape
    y0, y1 = max(0, cy - _CROP_RADIUS), min(h, cy + _CROP_RADIUS)
    x0, x1 = max(0, cx - _CROP_RADIUS), min(w, cx + _CROP_RADIUS)
    return img[y0:y1, x0:x1], x0, y0


def _crop_image(path: Path, centroid: tuple[float, float] | None = None):
    if centroid is None:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return _crop_image_with_offset(path, centroid)[0]


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
    indexed_paths = _build_frame_window(event, frame_dir)
    if not indexed_paths:
        return _EMPTY_VERDICT.copy()

    before_count = sum(1 for i, _ in indexed_paths if i < event.frame)
    after_count = len(indexed_paths) - before_count

    event_debug_dir: Path | None = None
    if debug_dir is not None:
        event_debug_dir = _save_debug_crops(indexed_paths, event, debug_dir)

    radius = adaptive_radius(event.neighbor_distance_px, cell_area_px=event.cell_area_px)
    content = [_load_image_block(p, event.centroid, marker_radius=radius) for _, p in indexed_paths]
    content.append({
        "type": "text",
        "text": (
            f"Candidate split at frame {event.frame}: track {event.parent_id} → daughter tracks. "
            f"{before_count} frames before and {after_count} frames after the split are shown, "
            f"each {_FRAME_STRIDE} frames apart."
        ),
    })

    system = _SYSTEM.format(before=_FRAMES_BEFORE, after=_FRAMES_AFTER, stride=_FRAME_STRIDE) + _MARKER_PROMPT_LINE
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
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
        _write_verdict_txt(event_debug_dir, parsed)

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
                         "binucleation": None, "anomaly_notes": None, "review_error": True}

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
    cost = _estimate_cost_usd(resolved_model, total_input, total_output)
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
        raw_confidence = float(r.get("confidence", event.confidence))
        confidence = raw_confidence
        split_type = r.get("split_type") or ""
        # min_gpt_confidence was tuned for "is this really a COMPLETED division" -- applying
        # it uniformly to "failed" splits suppresses the very category this fix exists to
        # surface. Real-API validation (2026-07-09, smoke run + repeated-sampling on M4) found
        # the model's own confidence for a genuine failed division clusters at 0.72-0.82,
        # consistently BELOW the 0.85 floor, so every single one would have been silently
        # downgraded back to false_positive before ever reaching the FAILED_SPLIT
        # reclassification below. Skip the floor for split_type=="failed" -- it's already
        # excluded from confirmed-split counts regardless of confidence, and raw_ai_confidence
        # still preserves the number for anyone who wants their own cutoff.
        if backend == "gpt" and verdict == "real" and confidence < min_gpt_confidence and split_type != "failed":
            verdict = "false_positive"
        is_real = verdict == "real"
        description = r.get("description", "")
        notes = f"{split_type}: {description}".strip(": ") if split_type else description

        # A confirmed "failed" split (cytokinesis began but daughters re-fused) is a real,
        # biologically meaningful event but NOT a completed division -- reclassify it out of
        # normal_split/multi_way_split so aggregate confirmed-split counts don't include it.
        # See EventType.FAILED_SPLIT (un-shelved 2026-07-09 -- Trackastra now gives the
        # daughter-fate tracking that originally blocked this).
        new_event_type = event.event_type
        if is_real and split_type == "failed" and event.event_type in (
            EventType.NORMAL_SPLIT, EventType.MULTI_WAY_SPLIT
        ):
            new_event_type = EventType.FAILED_SPLIT

        # The model can visually see 3+ daughters even when Trackastra's lineage graph only
        # found 2 children (e.g. two daughters touching closely enough that Cellpose/tracking
        # merged them) -- flag this mismatch rather than silently undercounting multi_way splits.
        if is_real and split_type == "multi_way" and event.event_type == EventType.NORMAL_SPLIT:
            print(f"  [split_type mismatch] frame={event.frame} parent={event.parent_id}: "
                  f"tracker topology says normal_split (2 children) but model reports multi_way")

        reviewed.append(dataclasses.replace(
            event,
            event_type=new_event_type,
            classification_source=resolved_model,
            confidence=confidence if is_real else 0.0,
            raw_ai_confidence=raw_confidence,
            review_error=bool(r.get("review_error", False)),
            tracker_confidence=event.tracker_confidence if event.tracker_confidence is not None else event.confidence,
            ai_notes=notes,
            split_type=split_type if (is_real and split_type) else None,
            acd_division_type=r.get("acd_division_type") if is_real else None,
            misaligned_chromosomes=r.get("misaligned_chromosomes") if is_real else None,
            lagging_chromosome=r.get("lagging_chromosome") if is_real else None,
            anaphase_bridge=r.get("anaphase_bridge") if is_real else None,
            micronucleus=r.get("micronucleus") if is_real else None,
            binucleation=r.get("binucleation") if is_real else None,
            anomaly_notes=r.get("anomaly_notes") if is_real else None,
        ))

    return passing + reviewed

