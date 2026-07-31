# Cell Split Counter

Lets a research scientist explore multi-thousand-frame microscopy time-lapse video
(ND2/AVI) of dividing cells from inside Claude Code, instead of scrubbing through it
by eye in Fiji. **The product is the tooling, not a detector.** A month spent chasing
classifier accuracy on hand-crafted geometry features hit the field's known ceiling —
cross-cell-line collapse — before the project reframed: a domain expert reviewing
real pixels is already a better classifier than anything trainable on a few hundred
labels, so the leverage is in giving her random access to 8GB of video and somewhere
to put her measurements, not in trying to automate her out of the loop. See
[`docs/bundle_README.md`](docs/bundle_README.md) for the schema and
[`cell_mcp.py`](cell_mcp.py) for the tool surface.

**Two stages, run on different machines:**

1. **Preprocessing** (this repo's classical pipeline below, GPU machine) — Cellpose
   segmentation + Trackastra tracking turn a raw ND2 into per-frame masks and
   lineages, and a lightweight vision-review pass filters that down to *candidate*
   events. That detector output ships as `data/candidates/<well>/candidates.csv`,
   deliberately **outside** the bundle: it is a starting point to browse, never a
   verdict to trust on its own, and keeping it beside the frames had it read as an
   answer key. See [`docs/bundle_README.md`](docs/bundle_README.md)'s caveats.
2. **Bundle export + MCP server** (`scripts/build_bundle.py`, `cell_mcp.py`,
   read-only, pure-python, no GPU/torch/Cellpose) — packages the ~5% of a run that's
   actually needed to navigate it later (indexed frames, track table, lineage,
   calibration) into a ~0.2GB-per-well bundle, then serves it to a researcher's own
   Claude Code session over a local stdio MCP: `list_wells`, `list_tracks`,
   `get_track_profile`, `get_frame`, `get_filmstrip`, `get_lineage`, `measure`,
   `get_neighbourhood_stats`, `get_filmstrip_at`, `show_cells`. She clones the repo, points
   `CELL_BUNDLE_DIR` at a bundle, and asks Claude what a given cell is doing — no
   ND2 reader, no GPU, nothing to keep running.

The preprocessing pipeline below produces the raw material for stage 2. Its own
config/prompt changes are validated against a frozen, human-labeled golden dataset
via a dedicated eval harness rather than by feel — see
[Testing & Evaluation](#testing--evaluation).

## Preprocessing pipeline (stage 1)
1. **Ingest** ✅ — extract frames from video, optional ROI crop.
2. **Segment** ✅ — Cellpose 3.x (`cyto3` model) finds cell masks per frame; frames/labels written to disk via memmap to avoid OOM.
3. **Track** ✅ — Trackastra deep-learning tracker links masks into per-cell lineages across frames.
4. **Classify** ✅ — persistence-filtered split + death detection; confidence-tiered routing (suppress / vision review / auto-confirm).
5. **Review** ✅ — `src/review.py` / `src/review_gpt.py` send ambiguous events to a vision model (`--vision-backend gpt`, Azure OpenAI, **default** — or `claude`, Anthropic) for real/FP verdict + split topology. Every reviewed crop is marked with a size-aware corner-bracket indicator (`adaptive_radius()`) so the model can't misattribute the verdict to a neighboring cell.
6. **Classify divisions** ✅ — `--classify-divisions` flag runs ACD classifier on confirmed events (bipolar/tripolar/multipolar + 4 abnormality flags).
7. **Package** ✅ — `scripts/package_events.py` outputs per-event folders with before/after crops + info.txt for papers/presentations.
8. **Output** ✅ — `events.csv` (see `docs/output_schema.md` for the full, current column list) + `summary.json`.
9. **Document** ✅ — every pipeline run auto-generates a self-contained `README.md` + `index.csv` in its output directory (`scripts/generate_package_readme.py`, wired into `src/pipeline.py`), so a run is self-describing when handed to someone (or their AI assistant) without repo access.
10. **QA tooling** ✅ — `scripts/reports/spot_check_review.py` generates a blind, self-contained HTML page for manually auditing a sample of review verdicts against your own judgment (stratified across risk buckets, live agreement scoring). `scripts/reports/researcher_browser.py` generates a filterable/sortable gallery of a run's **potentially interesting events** (splits, anomaly-flagged divisions, deaths) with AI verdicts/notes visible from the start — recall over precision, sized to shrink a researcher's haystack without dropping real events. `scripts/reports/death_shape_browser.py` is a shape-outlier-ranked gallery specifically for death events (no vision review runs on deaths, so this is the primary QA surface for that event type). These three predate the bundle/MCP reframe and operate on a raw pipeline `run_dir` (`events.csv` + `review_crops/`), not a bundle — still useful on the analysis machine for auditing a run in progress, but not what ships to a researcher.

## MCP tooling (stage 2, the product)

Everything below runs on the researcher's own machine, with no GPU and no repo
dependencies beyond `mcp`/`numpy`/`pandas`/`opencv-python-headless` (see
`pyproject.toml`) — the pipeline's heavy stack (torch, Cellpose, Trackastra, `nd2`)
never installs on her side.

**Building a bundle** (analysis machine, after a pipeline run):
```bash
python scripts/build_bundle.py data/output/your_run_folder \
    --nd2 "path/to/source.nd2" --out data/bundle --cell-line RUES2
```
Writes `data/bundle/your_run_folder/` — indexed frame PNGs, 16-bit label maps,
`tracks.csv`, `lineage.csv` (with per-link DNA/size scores), and a `manifest.json`
carrying calibration read from the ND2 (pixel size, per-frame timestamps, the
acquisition's own display color). Machine candidates go to `data/candidates/` instead,
outside the bundle. ~0.2GB
per well vs. ~8GB for the source run directory. `docs/bundle_README.md` is copied
into the bundle root so it travels without this repo.

**Serving it** (researcher's machine): the committed `.mcp.json` points a `uv`-managed
subprocess at `cell_mcp.py`. She clones the repo, sets `CELL_BUNDLE_DIR` to wherever
she copied a bundle, opens Claude Code in the repo root, and approves the MCP server
once. `UV_PROJECT_ENVIRONMENT=.venv-mcp` is pinned in `.mcp.json` so `uv run` never
touches a full pipeline `.venv` if one happens to exist on the same machine.

**Tools** (`cell_mcp.py`, 11 total — docstrings are written as the model-facing schema,
see the file itself for full argument docs):
- `list_wells()` / `list_tracks(well, filters...)` — orientation, the `ls`.
- `get_track_profile(well, track_id)` — free (no images) sparkline of a track's own
  area/brightness/solidity over time; decides where a `get_filmstrip` call is worth
  spending.
- `get_frame(well, frame)` — one whole field of view.
- `get_filmstrip(well, track_id, start_frame, end_frame, ...)` — the main tool: a
  cell followed over time as re-centred close-up crops, with the ND2's own display
  color and an optional tracking-ring marker. May request frames outside a track's
  own lifetime (a division usually happens in the gap between a track ending and its
  daughters' tracks starting).
- `get_filmstrip_at(well, start_frame, end_frame, x, y | anchor_track_id, ...)` —
  the same crops addressed by **position** rather than by mask. Answers "what
  happened here" instead of "what happened to track N", which matters because every
  other tool addresses cells by `track_id`: an object the segmenter never caught
  otherwise cannot be asked about at all. Also the tool for when a mask-following
  crop keeps losing the cell — segmentation fails hardest exactly during mitosis and
  death. Reports the nearest tracked cell per frame instead of ringing anything.
- `get_lineage(well, track_id)` — mother/daughter links.
- `measure(well, track_id, frame)` — real units (µm², hours), not pixels or frames.
- `get_neighbourhood_stats(well, track_id, frame)` — compares one cell against its
  nearest neighbours and the whole field at a single frame, as a z-score — separates
  "this cell is dim" from "the whole field is dim right now."
- `show_cells(well, events)` — writes a self-contained HTML page (images
  embedded as base64) of specific cells/frame-ranges, to hand off rather than
  describe in chat.
- `annotate(well, track_id, outcome_class, ...)` — appends stage marks
  (condensation/metaphase/anaphase/exit) and outcome class to a separate,
  append-only `annotations.csv`, which is what turns her usage of these tools
  into labeled data as a byproduct.

## Testing & Evaluation

Three layers, each answering a different question:

- **Unit tests** (`tests/`, run with `pytest`) — 134 tests across 12 files covering
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

134 tests, synthetic fixtures throughout — no live API key or GPU required. See
[Testing & Evaluation](#testing--evaluation) above for how this fits together with
the eval harness and human QA tooling.
