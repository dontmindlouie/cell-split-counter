# Investigation notes

Running log of spot-check findings that don't belong in code comments but shouldn't
be lost between sessions. Newest entries on top.

## 2026-07-03 — Missed split at frame 2, pixel (432, 454)

**Source video:** an `.nd2` timelapse processed via the Azure cloud GPU path
(`data/output/nd2_m2_cloud/<video>/events.csv`). 1230 total events.

**Spot check:** visually confirmed in Fiji a split at frame index 2 (0-indexed; Fiji
shows this as slice "3/848" since ImageJ slice numbers are 1-indexed), pixel (432, 454).
Note: Fiji's status bar shows calibrated microns first and pixel coords in parens —
`x=245.16 (432), y=257.64 (454)` — easy to misread the micron value as the pixel value.

**Finding: not recorded.** Zero rows in events.csv with `peak_frame == 2`, anywhere in
the frame. Nearest candidate is a false-positive event at frame 4 (parent_id 247,
33px away), which Claude correctly/incorrectly rejected as a "tracking artifact" —
review_crops/frame_00004_parent_247/verdict.txt: confidence 0.95 false_positive.

**Root cause: segmentation, not tracking.** Cropped the raw frames directly (not the
review crops) around (432, 454) for frames 0, 1, 2, 3, 4, 6, 8, 10, 12:
- Frames 0–4: a bright "figure-8" doublet (two adjacent bright puncta) is visible,
  essentially unchanged in position/shape.
- Frame 8: signal fading, much less distinct.
- Frame 12: feature is gone — blended into surrounding nucleus texture, no two
  clearly-separated round daughter nuclei diverge outward in this window.

This pattern (bright doublet that fades rather than resolving into two full-size
round nuclei) points at Cellpose never segmenting this into two separate mask
instances. `link_frames_trackastra` (src/track.py:163) only *links* masks that
Cellpose already produced — it has no mechanism to invent a split when the upstream
label map never contained two instances at that location. So **this is not a
"move off trackastra" problem** — swapping trackers doesn't fix a segmentation gap.
Trackastra would only help if masks *are* correctly split per-frame but get
mis-linked across frames (ID swaps, motion ambiguity) — a different failure mode.

**Segmentation call site:** `src/segment.py:52` and `src/segment.py:123`, both:
```python
model.eval(img, diameter=None, channels=[0, 0])
```
with `model_type="cyto3"` (src/segment.py:29).

**Untested hypotheses, roughly in order of likely impact:**
1. `cellprob_threshold` — unset (Cellpose default 0.0). The doublet is fainter/smaller
   than typical whole-nucleus objects; may be falling below detection sensitivity or
   merging into a neighboring mask. Try lowering (e.g. -1).
2. `diameter` — currently `None` (auto-estimated per batch), which tunes to the
   dominant population size (full nuclei). A compact dividing-chromosome cluster is
   smaller than that population average. Try pinning an explicit smaller diameter.
3. `model_type="cyto3"` — general cytoplasm/whole-cell model, not trained on
   condensed-chromatin/DNA-stain morphology specifically. Cellpose's `nuclei`
   pretrained model, or a small fine-tune, might resolve doublets like this better.

**Next step (not yet done):** re-segment just this frame window (frames 0–10) with
1–2 parameter variants on the desktop devbox (Tailscale+SSH) and check whether the
doublet actually resolves into two label instances. Desktop devbox SSH details not
yet recorded in this repo — pending.
