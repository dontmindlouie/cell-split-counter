# Cell Split Counter

Detects, tracks, and classifies cell division events in microscopy time-lapse video
(ND2/AVI) for a real research-scientist end user, who previously did this by eye in
Fiji. Hybrid architecture: classical CV (Cellpose segmentation + Trackastra
deep-learning tracking) handles per-frame detection and lineage-linking
deterministically and cheaply; a vision-language model is used only as a targeted
second opinion on ambiguous candidates (borderline confidence, anomaly-flagged
geometry) rather than as a per-frame classifier — keeps cost and consistency in
check on multi-thousand-frame videos with hundreds of dividing cells. Config/prompt
changes to the review step are validated against a frozen, human-labeled golden
dataset via a dedicated eval harness rather than by feel — see
[Testing & Evaluation](#testing--evaluation).

## Pipeline
1. **Ingest** ✅ — extract frames from video, optional ROI crop.
2. **Segment** ✅ — Cellpose 3.x (`cyto3` model) finds cell masks per frame; frames/labels written to disk via memmap to avoid OOM.
3. **Track** ✅ — Trackastra deep-learning tracker links masks into per-cell lineages across frames.
4. **Classify** ✅ — persistence-filtered split + death detection; confidence-tiered routing (suppress / vision review / auto-confirm).
5. **Review** ✅ — `src/review.py` / `src/review_gpt.py` send ambiguous events to a vision model (`--vision-backend gpt`, Azure OpenAI, **default** — or `claude`, Anthropic) for real/FP verdict + split topology. Every reviewed crop is marked with a size-aware corner-bracket indicator (`adaptive_radius()`) so the model can't misattribute the verdict to a neighboring cell.
6. **Classify divisions** ✅ — `--classify-divisions` flag runs ACD classifier on confirmed events (bipolar/tripolar/multipolar + 4 abnormality flags).
7. **Package** ✅ — `scripts/package_events.py` outputs per-event folders with before/after crops + info.txt for papers/presentations.
8. **Output** ✅ — `events.csv` (see `docs/output_schema.md` for the full, current column list) + `summary.json`.
9. **Document** ✅ — every pipeline run auto-generates a self-contained `README.md` + `index.csv` in its output directory (`scripts/generate_package_readme.py`, wired into `src/pipeline.py`), so a run is self-describing when handed to someone (or their AI assistant) without repo access.
10. **QA tooling** ✅ — `scripts/reports/spot_check_review.py` generates a blind, self-contained HTML page for manually auditing a sample of review verdicts against your own judgment (stratified across risk buckets, live agreement scoring). `scripts/reports/researcher_browser.py` generates a filterable/sortable gallery of a run's **potentially interesting events** (splits, anomaly-flagged divisions, deaths) with AI verdicts/notes visible from the start — recall over precision, sized to shrink a researcher's haystack without dropping real events. `scripts/reports/death_shape_browser.py` is a shape-outlier-ranked gallery specifically for death events (no vision review runs on deaths, so this is the primary QA surface for that event type).

## Testing & Evaluation

Three layers, each answering a different question:

- **Unit tests** (`tests/`, run with `pytest`) — 118 tests across 9 files covering
  segmentation/tracking memmap handling, classification rules, output formatting,
  and the vision-review request/response parsing (mocked, no live API key or GPU
  needed to run the suite).
- **Eval harness** (`scripts/eval_harness/`) — sweeps vision-review config
  (prompt wording, confidence floor, backend choice) against a frozen, human-labeled
  golden set of real events and scores precision/recall/F1, with results logged
  append-only for tracking changes over time. Answers "did this config change
  actually help, measured against real human judgment" instead of eyeballing a
  handful of examples. See `scripts/eval_harness/README.md` for the full design,
  including an explicit Tier A (review-config changes, scoreable today) vs. Tier B
  (segmentation/tracking changes, would need new ground truth) split, and honest
  caveats on what the golden set's stratified sampling does and doesn't tell you.
- **Human QA tooling** (`spot_check_review.py`, `researcher_browser.py`'s
  annotation/export flow) — for validating a specific real run against a human
  reviewer's own judgment, not a fixed golden set.

The `Status` section below is intentionally candid about where the pipeline's own
confidence score is and isn't trustworthy — that finding came from this same
QA tooling, not from an assumption.

## Status
Full pipeline implemented; ground-truth-scored recall/precision (63.6%/88.0%, F1 0.739) comes from one 575-frame validation video (Tom20, 33 GT events) run under an earlier review architecture — **not re-validated since**, and not necessarily representative of newer runs/videos. Repeated blind spot-checks (`spot_check_review.py`) across three independent samples on production Bewo runs found the *auto-confirmed, no-further-review* `>=0.85` confidence tier agrees with independent human judgment only ~12-25% of the time, well below the auto-rejected false-positive tier's ~60-87% — i.e. the model's own confidence score is not a reliable proxy for correctness at the high end. Treat any confidence-based accuracy claim as unverified until you've spot-checked your own run.
See `docs/output_schema.md` for column reference.

## Setup

### Prerequisites
- Python 3.10 or 3.11
- An Azure OpenAI resource for the default `gpt` vision-review backend, **or** an [Anthropic API key](https://console.anthropic.com/) if using `--vision-backend claude`

### Install

**If your machine has an NVIDIA GPU with CUDA:**
```bash
# Install GPU torch first so cellpose and trackastra see it
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**CPU-only (no NVIDIA GPU, or unsure):**
```bash
pip install -r requirements.txt
```
On CPU, segmentation runs ~26s/frame at 2048×2048 — expect ~4 hours for a 575-frame video.
Confirm GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### API key
Create a `.env` file in the project root. For the default GPT backend:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
```
For the Claude backend (`--vision-backend claude`) instead/as well:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Run
```bash
# Full pipeline on a video file (GPT vision review by default)
python main.py --video data/raw/your_video.avi

# Use Claude instead of the default GPT backend
python main.py --video data/raw/your_video.avi --vision-backend claude

# Skip re-segmentation if masks already exist on disk
python main.py --reuse-masks

# Include ACD division type classification on confirmed events
python main.py --reuse-masks --classify-divisions

# Package confirmed events into per-event folders for papers/presentations
python scripts/package_events.py

# Generate a self-contained README + index for a delivered output package
python scripts/generate_package_readme.py data/output/your_run_folder

# Blind QA spot-check of review verdicts against your own judgment
python scripts/reports/spot_check_review.py data/output/your_run_folder

# Filterable gallery of a run's potentially interesting events (splits, anomalies, deaths)
python scripts/reports/researcher_browser.py data/output/your_run_folder
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

118 tests, synthetic fixtures throughout — no live API key or GPU required. See
[Testing & Evaluation](#testing--evaluation) above for how this fits together with
the eval harness and human QA tooling.
