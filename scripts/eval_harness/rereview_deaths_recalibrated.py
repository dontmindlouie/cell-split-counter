"""Recalibrated-prior death re-review experiment.

Frame-sampling changes (wide window: 45%, dense stride-1: 55.1%) both underperformed
the plain production default (63.3%, sparse/narrow, fresh calls today -- see
rereview_deaths_dense.py). In all three conditions errors are one-directional: the
model calls "real death" when truth is "still alive" (0 false positives on the necrosis
side in every condition). That's a calibration problem, not a visual-evidence problem --
more/wider/denser frames didn't fix it because the frames were never the bottleneck.

This tests a prompt that explicitly states the empirical base rate (~96% of tracker
"death" events are actually tracking dropout, per Batch B ground truth) and inverts the
decision default: assume dropout unless there is clear positive evidence of genuine
degenerative death, rather than treating the two options symmetrically. Same narrow/
sparse window as production (the best-performing window so far) -- only the prompt
changes, to isolate the effect.

Usage:
    python scripts/eval_harness/rereview_deaths_recalibrated.py --dry-run
    python scripts/eval_harness/rereview_deaths_recalibrated.py
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
from scripts.eval_harness.rereview_deaths_dense import load_events, REASONING_EFFORT
from scripts.eval_harness.score_death_events import print_report, score
from src.config import GPT_DEPLOYMENT
from src.review import _FRAME_STRIDE, _build_frame_window, _build_review_content, _death_relation_labels
from src.review_gpt import _load_image_block

REPO = Path(__file__).resolve().parents[2]
FRAME_DIR = REPO / "data/output/202660629_Bewop920x_M4/frames"
OUT_CSV = REPO / "data/output/recalibrated_death_rereview_results.csv"

_RECALIBRATED_SYSTEM = """\
You are reviewing fluorescence microscopy timelapse images of cells. A cell-tracking \
algorithm lost track of a cell at the point shown (the track ends here) and, by default, \
classified this as cell death purely from how long the track persisted -- it has no way \
to distinguish genuine death from a tracking/segmentation failure. You will look at the \
actual images to make that call.

IMPORTANT CALIBRATION NOTE, from prior ground-truth review of this exact pipeline: of \
tracker-classified "death" events checked against real outcomes, roughly 95% turned out \
to be tracking/segmentation dropout during a real division, not genuine death. Genuine \
death is the rare case, not the common one. Treat TRACKING DROPOUT as your default \
assumption for an ambiguous or unclear case -- only call GENUINE DEATH when you see \
clear, specific positive evidence of degeneration, not merely the absence of a fully \
confirmed division.

You will receive {before} frames before and {after} frames after the track-end point, \
sampled every {stride} frames (not consecutive), in strict chronological order earliest \
to latest. Each image is preceded by a text label stating its position in the sequence and \
its frame offset relative to the track-end point -- use these labels, not visual guesswork, \
to determine temporal order and direction.

There are two possibilities:

1. GENUINE DEATH (rare -- requires clear positive evidence): the cell shows unambiguous \
degenerative changes -- fragmenting into several small irregular pieces (blebs/apoptotic \
bodies), progressively shrinking or dimming without any rounding or division machinery \
ever appearing, or simply disintegrating -- AND no new distinct cell mass appears in its \
place afterward. If you are not confident you can see this specific pattern, this is NOT \
the right call.

2. TRACKING DROPOUT (common -- the default assumption): the segmentation algorithm lost a \
real division in progress. Positive signs include the cell rounding up, chromatin \
condensing into a single bright compact mass, or the cell's outline/boundary becoming \
indistinct or temporarily disappearing -- but even in the ABSENCE of clear positive signs \
either way, tracking dropout is still more likely than genuine death given the base rate \
above. Only override this default toward genuine death if the fragmentation/degeneration \
evidence in possibility 1 is clearly present.

Respond with a JSON object — no other text:
{{"likely_division_dropout": true | false, \
"confidence": <float 0.0-1.0, your confidence in this call>, \
"description": "<one or two sentences>", \
"anomaly_notes": "<any other interesting observation, or null>"}}"""


def review_recalibrated(client: AzureOpenAI, event, usage_log: list) -> dict:
    indexed_paths = _build_frame_window(event, FRAME_DIR)
    relation_target, at_target_label = _death_relation_labels()
    content = _build_review_content(
        indexed_paths, event, _load_image_block,
        relation_target=relation_target, at_target_label=at_target_label,
    )
    system = _RECALIBRATED_SYSTEM.format(before=8, after=8, stride=_FRAME_STRIDE)
    response = client.chat.completions.create(
        model=GPT_DEPLOYMENT, max_completion_tokens=1500, reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
    )
    usage_log.append({"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens})
    return json.loads(response.choices[0].message.content.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    golden = load_death_golden_set()
    events = load_events(set(golden.labels))
    items = list(events.items())
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} events to review (recalibrated prompt, sparse window, 1 call each)")
    if args.dry_run:
        print(_RECALIBRATED_SYSTEM.format(before=8, after=8, stride=_FRAME_STRIDE))
        return

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview")

    usage_log: list = []
    preds: dict[int, bool | None] = {}
    rows_out = []

    for i, (tid, ev) in enumerate(items, 1):
        try:
            result = review_recalibrated(client, ev, usage_log)
            preds[tid] = result.get("likely_division_dropout")
        except Exception as exc:
            print(f"  track {tid}: call FAILED: {exc}")
            result = {"error": str(exc)}
            preds[tid] = None

        expected = golden.labels[tid]
        ok = "OK" if preds[tid] == expected else "MISS"
        print(f"[{i}/{len(items)}] track {tid:>6} expected={expected!s:<5} recalibrated={preds[tid]!s:<5}[{ok}]")

        rows_out.append({
            "track_id": tid, "expected_dropout": expected, "recalibrated_dropout": preds[tid],
            "confidence": result.get("confidence"), "description": result.get("description"),
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

    result = score(preds, golden)
    print_report("Recalibrated prompt (sparse window, today's calls)", result)


if __name__ == "__main__":
    main()
