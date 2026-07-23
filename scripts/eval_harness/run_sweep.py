"""Tier A hands-off sweep runner: for each config, re-run review (via main.py
--reuse-masks against the cached M4 segmentation) with different vision-review
knobs, score against the frozen golden set, and append to the shared results log.

No re-segmentation, no new human review -- only the vision-review step differs
per config, which is exactly what makes marker/prompt/confidence/backend variables
"Tier A" (see golden_set.py's docstring). Frame-sampling changes are Tier B and do
NOT belong in this runner -- they invalidate the (parent_id, peak_frame) key space
this scorer depends on.

Each config run makes REAL vision-backend API calls (Azure OpenAI or Claude) against
every ambiguous/anomaly-flagged split candidate in the frame range -- this costs real
money/quota per config, it is not free just because segmentation is cached. Use
--dry-run first to see exactly what would run.

Death-event review is OFF by default (--no-review-deaths) even though main.py reviews
deaths by default -- scorer.py never scores death rows, so paying for ~1457 death
review calls per config here would be pure waste. Opt back in per-config with
{"review_deaths": true} if a future scorer extension needs it.

REPEATS: GPT-5-mini vision review is NOT deterministic run-to-run on this pipeline
(found 2026-07-16 -- a config re-run with identical settings swung recall from 0.583 to
0.250, as large as the swing credited to an actual config change). A single run per
config is not enough to trust a score difference as a real config effect. Add
{"repeats": N} to a config dict to run it N times (as "<config_id>_rep1",
"<config_id>_rep2", ...); a min/max/mean variance summary prints after the last repeat
and every repeat still logs its own row to results_log.py for later inspection.

Usage:
    python scripts/eval_harness/run_sweep.py --dry-run
    python scripts/eval_harness/run_sweep.py --sweep-file scripts/eval_harness/sweeps/example.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_harness.golden_set import DEFAULT_RUN_LABELS, load_golden_set
from scripts.eval_harness.results_log import append_result
from scripts.eval_harness.scorer import score_events_csv

REPO_ROOT = Path(__file__).resolve().parents[2]

# The video every Tier A sweep reuses masks/tracking from -- same source .nd2 either
# way, --reuse-masks still opens it for pixel-size metadata (src/pipeline.py's
# get_pixel_size_um), it just skips re-running Cellpose/trackastra.
# Configure for your own dataset -- this is a local, machine-specific path, not
# something a public checkout has by default.
DEFAULT_VIDEO = REPO_ROOT.parent / "202660629_Bewop920x_M4.nd2"
SWEEP_OUTPUT_ROOT = REPO_ROOT / "data/output/eval_harness_sweeps"

# Two selectable frame-dir fixtures. "full" is the fixture the golden set's
# (parent_id, peak_frame) keys are anchored to -- see scripts/eval_harness/README.md.
# "200frame" is a separate, cheaper segmentation (own Cellpose run, does NOT share the
# full cache's memmaps) covering only frames 0-199 of the same video -- built and
# validated 2026-07-16: scores 24/56 golden events, and every one of the 32
# "unmatched" keys was individually confirmed to have peak_frame >= 200 (reduced
# *coverage*, not a mismatch) -- ~$0.80/run vs. ~$6.84/run for "full", useful for
# repeats where the absolute event count matters less than confirming a config
# difference is real vs. non-determinism noise.
#
# --end-frame IS required for "200frame" (found the hard way, real run failure): its
# frame_dir still has all 848 extracted PNGs sitting alongside the 200-frame memmap
# (only the segmentation/tracking step was capped at build time, not extraction), so
# src/pipeline.py's extract_frames() returns all 848 paths and load_video_arrays()
# then expects an 848-frame memmap unless --end-frame slices frame_paths back down to
# 200 first (src/pipeline.py: `frame_paths = frame_paths[start_frame:end_frame]` runs
# BEFORE load_video_arrays). Without it: "Memmap size mismatch -- expected 0.89 GB /
# 1.78 GB but found 0.21 GB / 0.42 GB" (0.21/0.89 = 200/848 exactly).
FRAME_DIR_FIXTURES = {
    "full": REPO_ROOT / "data/output/202660629_Bewop920x_M4/frames",
    "200frame": REPO_ROOT / "data/output/202660629_Bewop920x_M4_200frame_fixture/frames",
}
FIXTURE_END_FRAME = {"full": None, "200frame": 200}
FIXTURE_EXPECTED_GOLDEN = {"full": 56, "200frame": 24}


def build_main_args(config: dict, output_dir: Path) -> list[str]:
    fixture = config.get("fixture", "full")
    frame_dir = FRAME_DIR_FIXTURES[fixture]
    end_frame = FIXTURE_END_FRAME[fixture]
    args = [
        str(DEFAULT_VIDEO),
        "--reuse-masks",
        "--frame-dir", str(frame_dir),
        "--output-dir", str(output_dir),
        "--no-debug-crops",
    ]
    if end_frame is not None:
        args += ["--end-frame", str(end_frame)]
    if "vision_backend" in config:
        args += ["--vision-backend", config["vision_backend"]]
    if "gpt_reasoning_effort" in config:
        args += ["--gpt-reasoning-effort", config["gpt_reasoning_effort"]]
    if "min_gpt_confidence" in config:
        args += ["--min-gpt-confidence", str(config["min_gpt_confidence"])]
    # Default OFF: scorer.py never scores death rows (golden_set.py's keys are all
    # split events), so reviewing deaths on every sweep config would just re-pay for
    # ~1457 API calls per config with zero effect on the score. Opt back in with
    # {"review_deaths": true} in a config dict if a future scorer extension needs it.
    if not config.get("review_deaths"):
        args.append("--no-review-deaths")
    return args


def run_one(config: dict, run_id: str, dry_run: bool) -> dict | None:
    output_dir = SWEEP_OUTPUT_ROOT / run_id
    main_args = build_main_args(config, output_dir)

    print(f"\n=== {run_id} ===")
    print(f"python main.py {' '.join(main_args)}")
    if dry_run:
        return None

    subprocess.run([sys.executable, "main.py", *main_args], cwd=REPO_ROOT, check=True)

    events_csv = output_dir / "events.csv"
    golden = load_golden_set()
    score = score_events_csv(events_csv, golden)

    p = f"{score['precision']:.3f}" if score["precision"] is not None else "n/a"
    r = f"{score['recall']:.3f}" if score["recall"] is not None else "n/a"
    f1 = f"{score['f1']:.3f}" if score["f1"] is not None else "n/a"
    print(f"  scored {score['n_scored']}/{score['n_golden_total']}  precision={p} recall={r} f1={f1}")
    fixture = config.get("fixture", "full")
    expected_scored = FIXTURE_EXPECTED_GOLDEN[fixture]
    if score["n_unmatched_golden_keys"] and score["n_scored"] != expected_scored:
        # Reduced coverage from the 200frame fixture is expected (24/56, see
        # FRAME_DIR_FIXTURES' docstring) -- only warn when the scored count doesn't
        # even match that known-good number, which would mean something else is wrong.
        print(f"  WARNING: {score['n_unmatched_golden_keys']} golden keys unmatched, "
              f"expected {expected_scored}/{score['n_golden_total']} scored for fixture={fixture!r} "
              f"but got {score['n_scored']} -- check frame-dir/reuse-masks setup")

    append_result(
        config_id=run_id,
        score=score,
        run_labels_scored=DEFAULT_RUN_LABELS,
        vision_backend=config.get("vision_backend", ""),
        gpt_reasoning_effort=config.get("gpt_reasoning_effort", ""),
        min_gpt_confidence=str(config.get("min_gpt_confidence", "")),
        extra_flags=json.dumps({k: v for k, v in config.items() if k != "config_id"}),
        notes=config.get("notes", ""),
    )
    return score


def run_one_config(config: dict, dry_run: bool) -> None:
    config_id = config["config_id"]
    repeats = config.get("repeats", 1)

    scores = []
    for rep in range(1, repeats + 1):
        run_id = config_id if repeats == 1 else f"{config_id}_rep{rep}"
        score = run_one(config, run_id, dry_run)
        if score is not None:
            scores.append(score)

    if repeats > 1 and scores:
        precisions = [s["precision"] for s in scores if s["precision"] is not None]
        recalls = [s["recall"] for s in scores if s["recall"] is not None]
        f1s = [s["f1"] for s in scores if s["f1"] is not None]
        print(f"\n  --- {config_id}: {len(scores)} repeats ---")
        if precisions:
            print(f"  precision: min={min(precisions):.3f} max={max(precisions):.3f} mean={mean(precisions):.3f} range={max(precisions)-min(precisions):.3f}")
        if recalls:
            print(f"  recall:    min={min(recalls):.3f} max={max(recalls):.3f} mean={mean(recalls):.3f} range={max(recalls)-min(recalls):.3f}")
        if f1s:
            print(f"  f1:        min={min(f1s):.3f} max={max(f1s):.3f} mean={mean(f1s):.3f} range={max(f1s)-min(f1s):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-file", type=Path, help="JSON file: list of config dicts, each needs a unique config_id")
    parser.add_argument("--dry-run", action="store_true", help="print the main.py invocations without running them or spending API credit")
    args = parser.parse_args()

    if args.sweep_file:
        configs = json.loads(args.sweep_file.read_text(encoding="utf-8"))
    else:
        configs = [{"config_id": "baseline_gpt_medium", "vision_backend": "gpt", "gpt_reasoning_effort": "medium"}]

    for config in configs:
        run_one_config(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
