# Cell Split Counter

Detects and classifies cell division events in microscopy video (Fiji/AVI etc).

## Pipeline
1. **Ingest** — extract frames from video at fixed interval, crop to ROI square.
2. **Segment** — Cellpose (CNN) finds cell masks per frame.
3. **Track** — deterministic IoU/centroid matching links masks into per-cell lineages across frames.
4. **Classify** — graph rules over the lineage detect: normal split, failed split, multi-way split, exit-from-ROI, death. Ambiguous cases get flagged.
5. **Review** — flagged frames get sent to Claude vision for a second opinion.
6. **Output** — CSV of lineage events + JSON summary metadata (total splits, anomaly counts, video metadata).

## Status
Scaffolding only — no pipeline code yet.

## Setup
```
pip install -r requirements.txt
```
