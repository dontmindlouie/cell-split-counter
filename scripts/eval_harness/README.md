# Eval harness (Tier A)

Built 2026-07-15, continuing the design from session `a9a4ff4e` (2026-07-13/14),
itself the follow-up to the human-review-ground-truth backlog. Read this cold if
you're an AI agent landing here with no prior context.

## What this is

A frozen golden dataset + a deterministic scorer + parameterized run configs + an
append-only results log -- the standard pattern used by promptfoo/OpenAI Evals/
braintrust for LLM-config sweeps. Lets you sweep vision-review config variables and
get precision/recall/F1 against real human-labeled ground truth, with **zero new
human review per sweep** and **zero re-segmentation cost**.

## Why "Tier A" specifically

Config variables split into two categories, and this harness only covers one of them:

- **Tier A** (marker geometry, marker-motion-tracking, prompt wording, confidence
  threshold, vision backend): only changes *how the model judges a fixed picture* --
  the same candidate split events (same `parent_id`/`peak_frame`) exist no matter what
  you sweep. Scoreable today, this harness.
- **Tier B** (frame count, frame spacing/density): changes segmentation and tracking
  topology itself -- different candidate events entirely. The golden set's
  `(parent_id, peak_frame)` keys **do not carry over** across Tier B configs. Needs
  new spatial (x/y + frame-range) ground truth first, not built here.

Don't add frame-sampling variables to `run_sweep.py`'s config schema -- they'll silently
produce a mismatched/meaningless score (or, if you're unlucky, false-credit collisions
of the exact kind `scripts/score_against_ground_truth.py`'s docstring warns about).

## Pieces

- `golden_set.py` -- loads the 56 usable binary labels (real/false_positive) for M4
  full + M4 neighborfix from `cell-split-counter-shared-data/human_review/human_review_compiled_2026-07-13.csv`.
  Drops `unsure`/freeform-note rows and one genuinely conflicting duplicate review
  (2 independent reviewers disagreed on `parent_id=2493, peak_frame=315`).
- `scorer.py` -- joins a run's `events.csv` against the golden set by key, derives
  each row's effective verdict the same way `scripts/reports/spot_check_review.py`
  does (`real` if `ai_confidence > 0` or `review_error == "1"` fail-open, else
  `false_positive`), computes precision/recall/F1.
- `results_log.py` -- appends one row per scored config to
  `cell-split-counter-shared-data/eval_harness/results_log.csv`. This *is* the
  "reliability over time" report the backlog asked for.
- `run_sweep.py` -- for each config in a JSON list, invokes `main.py --reuse-masks`
  against the cached golden-set video's segmentation (see `DEFAULT_VIDEO`/
  `FRAME_DIR_FIXTURES`, configure for your own dataset), scores the result, logs it.
  `--dry-run` prints invocations without spending API
  credit -- always dry-run a new sweep file first. Death review is skipped by default
  (`--no-review-deaths`) since `scorer.py` doesn't score death rows -- opt back in
  per-config with `{"review_deaths": true}` if that ever changes.

## Important caveat on absolute numbers

The golden set was built by `spot_check_review.py`'s *stratified per-bucket sampling*
(even sampling across `near_edge`/`confirmed_high`/`gpt_floor_downgrade`/`false_positive`/
etc. buckets), not random sampling. Class balance here (12 real / 44 false_positive)
reflects the sampling strategy, not true corpus-wide prevalence. **Treat precision/recall
from this harness as valid for relative comparison across Tier A configs (same golden
set, same stratification bias, held constant) -- not as an absolute pipeline
performance number.**

## Cost warning

Every config in a sweep makes real vision-backend API calls (Azure OpenAI `gpt-5-mini`
or Claude) against every ambiguous/anomaly-flagged split candidate in the M4 range --
`--reuse-masks` only skips the Cellpose segmentation step, not the review step being
tested. This costs real money and draws down the same Azure quota noted as
already-maxed in `project_azure_aoai_diagnostic_backlog` memory. Always `--dry-run`
first, and don't fire off a big sweep file without checking scope/cost.

## Not done yet

- No CLI knob for marker geometry (radius, thickness) or marker-motion-tracking --
  those live as constants/unimplemented feature in `src/review.py`. Adding a flag
  there is what plugs them into this harness; the harness itself doesn't block on it.
- No results-log visualization/diffing tool -- currently just an appendable CSV, read
  it directly or with pandas.
- Not wired into CI or any other automation -- run manually.
