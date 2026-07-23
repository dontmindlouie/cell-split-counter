"""Dense-window death re-review experiment.

Tests whether feeding the vision model EVERY consecutive frame (stride 1) around a
death/track-end event -- instead of the production stride-3 sparse sample -- lets it
correctly discriminate genuine necrosis from tracking/segmentation dropout during
mitosis. Motivated by the 2026-07-22 raw-mask trace (see docs/investigation_notes.md):
23/24 "divided halfway through" events showed a TRUE Cellpose detection gap that
resolved within 30 frames (median 3, max 28), which is well within Azure's 50-image
cap at stride 1 -- so density, not width, is the lever being tested here.

IMPORTANT -- do not reuse the likely_division_dropout flag already sitting in M4's
events.csv as a baseline: that file was overwritten by the 2026-07-21/22 M2-M6 batch
rerun, so it reflects a DIFFERENT (later, non-deterministic) set of API calls than the
ones Batch B was originally stratified/labeled against. Comparing today's dense result
against that stale flag would not be apples-to-apples. Instead this script makes BOTH
a fresh sparse call (same window/prompt as production review_death_gpt) and a fresh
dense call for every golden-set event, in the same run, so the two are directly
comparable to each other even if neither matches the historically-reported 61-96%
figures exactly (expected, since GPT-5-mini review is known non-deterministic --
see scripts/eval_harness/README.md).

Usage:
    python scripts/eval_harness/rereview_deaths_dense.py --dry-run
    python scripts/eval_harness/rereview_deaths_dense.py
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv; load_dotenv()

from openai import AzureOpenAI

from scripts.eval_harness.death_golden_set import load_death_golden_set
from scripts.eval_harness.score_death_events import print_report, score
from src.classify import EventType, LineageEvent
from src.config import GPT_DEPLOYMENT
from src.review import (
    _DEATH_SYSTEM,
    _FRAME_STRIDE,
    _build_dense_debug_window,
    _build_frame_window,
    _build_review_content,
    _death_relation_labels,
)
from src.review_gpt import _load_image_block

REPO = Path(__file__).resolve().parents[2]
M4_EVENTS_CSV = REPO / "data/output/202660629_Bewop920x_M4/events.csv"
FRAME_DIR = REPO / "data/output/202660629_Bewop920x_M4/frames"
OUT_CSV = REPO / "data/output/dense_death_rereview_results.csv"

DENSE_BEFORE_FRAMES = 5   # real frames, not stride units -- matches the 2026-07-22 trace
DENSE_AFTER_FRAMES = 35   # covers the max observed gap (28) plus margin, ~41 images total

REASONING_EFFORT = "medium"  # matches main.py's production default


def load_events(track_ids: set[int]) -> dict[int, LineageEvent]:
    rows = list(csv.DictReader(M4_EVENTS_CSV.open(encoding="utf-8")))
    events = {}
    for r in rows:
        tid = int(r["track_id"])
        if tid not in track_ids or r["split_topology"] != "death":
            continue
        events[tid] = LineageEvent(
            track_id=tid,
            parent_id=int(float(r["parent_id"])) if r.get("parent_id") else None,
            frame=int(r["peak_frame"]),
            event_type=EventType.DEATH,
            classification_source="rereview_deaths_dense",
            confidence=0.0,
            centroid=(float(r["centroid_x"]), float(r["centroid_y"])),
            cell_area_px=float(r["cell_area_px"]) if r.get("cell_area_px") else None,
            neighbor_distance_px=float(r["neighbor_distance_px"]) if r.get("neighbor_distance_px") else None,
            neighbor_area_px=float(r["neighbor_area_px"]) if r.get("neighbor_area_px") else None,
        )
    return events


def _dense_trailing_caption(event, split_position, total, before_count, after_count) -> str:
    frame_ref = f" (image {split_position} of {total} above)" if split_position is not None else ""
    return (
        f"Track end at frame {event.frame}{frame_ref}: track {event.parent_id} is lost here, "
        f"no further detection. {before_count} consecutive frames before and {after_count} "
        f"consecutive frames after the track-end point are shown -- EVERY frame in this range, "
        f"not a sample -- in strict chronological order from earliest to latest."
    )


_DENSE_SYSTEM = _DEATH_SYSTEM.replace(
    "You will receive {before} frames before and {after} frames after the track-end point, "
    "sampled every {stride} frames (not consecutive), in strict chronological order earliest "
    "to latest.",
    "You will receive {before} frames before and {after} frames after the track-end point -- "
    "EVERY consecutive frame in this range, not a sample -- in strict chronological order "
    "earliest to latest.",
)


def review_sparse(client: AzureOpenAI, event: LineageEvent, usage_log: list) -> dict:
    """Fresh call, identical to production review_death_gpt (stride-3, +/-8)."""
    indexed_paths = _build_frame_window(event, FRAME_DIR)
    relation_target, at_target_label = _death_relation_labels()
    content = _build_review_content(
        indexed_paths, event, _load_image_block,
        relation_target=relation_target, at_target_label=at_target_label,
    )
    system = _DEATH_SYSTEM.format(before=8, after=8, stride=_FRAME_STRIDE)
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT, max_completion_tokens=1500, reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
    )
    usage_log.append({"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens})
    return json.loads(response.choices[0].message.content.strip())


def review_dense(client: AzureOpenAI, event: LineageEvent, usage_log: list) -> dict:
    """Dense (stride-1) call over a tighter, empirically-sized window."""
    indexed_paths = _build_dense_debug_window(
        event, FRAME_DIR, frames_before=DENSE_BEFORE_FRAMES, frames_after=DENSE_AFTER_FRAMES // _FRAME_STRIDE + 1
    )
    # _build_dense_debug_window's before/after args are in _FRAME_STRIDE units (it multiplies
    # internally) -- rebuild directly by real-frame count instead so DENSE_BEFORE/AFTER_FRAMES
    # mean what they say.
    lo = max(0, event.frame - DENSE_BEFORE_FRAMES)
    hi = event.frame + DENSE_AFTER_FRAMES
    from src.review import _find_frame
    indexed_paths = [(i, p) for i in range(lo, hi + 1) if (p := _find_frame(FRAME_DIR, i)) is not None]
    if len(indexed_paths) > 50:
        raise ValueError(f"track {event.track_id}: dense window has {len(indexed_paths)} frames, exceeds Azure's 50-image cap")

    relation_target, at_target_label = _death_relation_labels()
    content = _build_review_content(
        indexed_paths, event, _load_image_block,
        relation_target=relation_target, at_target_label=at_target_label,
        trailing_caption_fn=_dense_trailing_caption,
    )
    before_count = sum(1 for i, _ in indexed_paths if i < event.frame)
    after_count = len(indexed_paths) - before_count - (1 if event.frame in [i for i, _ in indexed_paths] else 0)
    system = _DENSE_SYSTEM.format(before=before_count, after=after_count, stride=1)
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT, max_completion_tokens=1500, reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
    )
    usage_log.append({"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens})
    return json.loads(response.choices[0].message.content.strip()), len(indexed_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N golden-set events (for a cheap smoke test)")
    args = parser.parse_args()

    golden = load_death_golden_set()
    events = load_events(set(golden.labels))
    missing = set(golden.labels) - set(events)
    if missing:
        print(f"WARNING: {len(missing)} golden-set tracks not found in M4 events.csv: {sorted(missing)[:10]}...")

    items = list(events.items())
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} events to review (sparse + dense each = {2 * len(items)} API calls)")
    for tid, ev in items[:3]:
        lo, hi = max(0, ev.frame - DENSE_BEFORE_FRAMES), ev.frame + DENSE_AFTER_FRAMES
        print(f"  track {tid}: frame={ev.frame}  dense window [{lo},{hi}] = {hi - lo + 1} frames")

    if args.dry_run:
        return

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview")

    usage_log: list = []
    sparse_preds: dict[int, bool | None] = {}
    dense_preds: dict[int, bool | None] = {}
    rows_out = []

    for i, (tid, ev) in enumerate(items, 1):
        try:
            sparse_result = review_sparse(client, ev, usage_log)
            sparse_preds[tid] = sparse_result.get("likely_division_dropout")
        except Exception as exc:
            print(f"  track {tid}: sparse call FAILED: {exc}")
            sparse_result = {"error": str(exc)}
            sparse_preds[tid] = None

        try:
            dense_result, n_frames = review_dense(client, ev, usage_log)
            dense_preds[tid] = dense_result.get("likely_division_dropout")
        except Exception as exc:
            print(f"  track {tid}: dense call FAILED: {exc}")
            dense_result = {"error": str(exc)}
            n_frames = 0
            dense_preds[tid] = None

        expected = golden.labels[tid]
        s_ok = "OK" if sparse_preds[tid] == expected else "MISS"
        d_ok = "OK" if dense_preds[tid] == expected else "MISS"
        print(f"[{i}/{len(items)}] track {tid:>6} expected={expected!s:<5} sparse={sparse_preds[tid]!s:<5}[{s_ok}] dense={dense_preds[tid]!s:<5}[{d_ok}] (dense n_frames={n_frames})")

        rows_out.append({
            "track_id": tid, "expected_dropout": expected,
            "sparse_dropout": sparse_preds[tid], "sparse_confidence": sparse_result.get("confidence"),
            "sparse_description": sparse_result.get("description"),
            "dense_dropout": dense_preds[tid], "dense_confidence": dense_result.get("confidence"),
            "dense_description": dense_result.get("description"), "dense_n_frames": n_frames,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {OUT_CSV}")

    from src.review import _estimate_cost_usd
    cost = sum(_estimate_cost_usd(GPT_DEPLOYMENT, u["input_tokens"], u["output_tokens"]) or 0 for u in usage_log)
    print(f"total cost: ${cost:.4f} ({len(usage_log)} calls)")

    sparse_result = score(sparse_preds, golden)
    dense_result = score(dense_preds, golden)
    print_report("Fresh sparse (production window, today's calls)", sparse_result)
    print_report("Dense (stride-1, today's calls)", dense_result)


if __name__ == "__main__":
    main()
