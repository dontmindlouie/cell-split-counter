# Cell Split Counter

Detects and classifies cell division events in microscopy video (Fiji/AVI etc).

## Pipeline
1. **Ingest** ✅ — extract frames from video, optional ROI crop.
2. **Segment** ✅ — Cellpose 3.x (`cyto3` model) finds cell masks per frame; frames/labels written to disk via memmap to avoid OOM.
3. **Track** ✅ — Trackastra deep-learning tracker links masks into per-cell lineages across frames.
4. **Classify** ✅ — persistence-filtered split + death detection; confidence-tiered routing (suppress / vision review / auto-confirm).
5. **Review** ✅ — `src/review.py` / `src/review_gpt.py` send ambiguous events to a vision model (`--vision-backend gpt`, Azure OpenAI, **default** — or `claude`, Anthropic) for real/FP verdict + split topology. Every reviewed crop is marked with a size-aware corner-bracket indicator (`adaptive_radius()`) so the model can't misattribute the verdict to a neighboring cell.
6. **Classify divisions** ✅ — `--classify-divisions` flag runs ACD classifier on confirmed events (bipolar/tripolar/multipolar + 4 abnormality flags).
7. **Package** ✅ — `scripts/package_events.py` outputs per-event folders with before/after crops + info.txt for papers/presentations.
8. **Output** ✅ — `events.csv` (see `docs/output_schema.md` for the full, current column list) + `summary.json`.
9. **Document** ✅ — `scripts/generate_package_readme.py` writes a self-contained README.md + index.csv into an output package directory, so it's self-describing when handed to someone (or their AI assistant) without repo access. (Currently a manual step, not run automatically at the end of a pipeline run — see backlog.)
10. **QA tooling** ✅ — `scripts/reports/spot_check_review.py` generates a blind, self-contained HTML page for manually auditing a sample of review verdicts against your own judgment (stratified across risk buckets, live agreement scoring). `scripts/reports/researcher_browser.py` generates a filterable/sortable gallery of confirmed events with AI verdicts visible, for exploring what happened in a run.

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

# Filterable gallery of confirmed events for exploring a run
python scripts/reports/researcher_browser.py data/output/your_run_folder
```

For development/tests only:
```bash
pip install -r requirements-dev.txt
```
