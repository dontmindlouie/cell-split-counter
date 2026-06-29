# Cell Split Counter

Detects and classifies cell division events in microscopy video (Fiji/AVI etc).

## Pipeline
1. **Ingest** ✅ — extract frames from video, optional ROI crop.
2. **Segment** ✅ — Cellpose 3.x (`cyto3` model) finds cell masks per frame; frames/labels written to disk via memmap to avoid OOM.
3. **Track** ✅ — Trackastra deep-learning tracker links masks into per-cell lineages across frames.
4. **Classify** ✅ — persistence-filtered split detection; three-tier confidence routing (suppress / Claude review / auto-confirm).
5. **Review** ✅ — `src/review.py` sends ambiguous events to Claude Haiku for real/FP verdict + split topology.
6. **Classify divisions** ✅ — `--classify-divisions` flag runs ACD classifier on confirmed events (bipolar/tripolar/multipolar + 4 abnormality flags).
7. **Package** ✅ — `scripts/package_events.py` outputs per-event folders with before/after crops + info.txt for papers/presentations.
8. **Output** ✅ — `events.csv` (19 columns) + `summary.json`.

## Status
Full pipeline implemented and validated against the ACTB_Tom20 video (575 frames, 33 GT events).
Results at confidence > 0: 63.6% recall, 88.0% precision, F1 0.739.
See `docs/output_schema.md` for column reference.

## Setup

### Prerequisites
- Python 3.10 or 3.11
- An [Anthropic API key](https://console.anthropic.com/) for the Claude vision review step

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
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Run
```bash
# Full pipeline on a video file
python main.py --video data/raw/your_video.avi

# Skip re-segmentation if masks already exist on disk
python main.py --reuse-masks

# Include ACD division type classification on confirmed events
python main.py --reuse-masks --classify-divisions

# Package confirmed events into per-event folders for papers/presentations
python scripts/package_events.py
```

For development/tests only:
```bash
pip install -r requirements-dev.txt
```
