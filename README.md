# Cell Split Counter

Detects and classifies cell division events in microscopy video (Fiji/AVI etc).

## Pipeline
1. **Ingest** ✅ — extract every Nth frame from video, optional ROI crop.
2. **Segment** ✅ — Cellpose (CNN) finds cell masks per frame.
3. **Track** ✅ — deterministic mask-overlap matching links masks into per-cell lineages across frames.
4. **Classify** ✅ (partial) — graph rules over the lineage detect normal split / multi-way split only, with cascade-noise suppression. Failed split, exit-from-ROI, death, and AMBIGUOUS flagging are not implemented yet.
5. **Review** ❌ not implemented — `src/review.py` is a stub; would send AMBIGUOUS-flagged frames to Claude vision for a second opinion, but nothing currently produces an AMBIGUOUS event for it to act on.
6. **Output** ✅ — CSV of lineage events + JSON summary metadata.

## Status
v1 implemented and validated against real video: ingest → segment → track → classify →
output all work end-to-end. Detects normal/multi-way splits only -- no abnormality
classification or Claude-vision review yet (`src/review.py` is a stub). See
`docs/output_schema.md` for the output format and known limitations.

## Setup
```
pip install --index-url https://download.pytorch.org/whl/cu124 torch  # GPU build; install BEFORE requirements.txt
pip install -r requirements.txt                                       # cellpose etc. will see torch already satisfies its dependency
```
Cellpose segmentation is ~4x faster on GPU (~6s/frame vs ~26s/frame at 2048x2048 on
this project's hardware). If you skip the first line, `pip install -r requirements.txt`
pulls a CPU-only torch via cellpose and `src/segment.py`'s `gpu=True` will silently do
nothing useful. Run `python -c "import torch; print(torch.cuda.is_available())"` to confirm.

For development/tests only:
```
pip install -r requirements-dev.txt
```
