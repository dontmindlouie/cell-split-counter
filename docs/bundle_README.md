# Cell microscopy bundle — what's in here

Copied into each bundle directory by `scripts/build_bundle.py`. Written for
someone (or some AI agent) arriving with no other context.

Each subdirectory is **one well** — one field of view, filmed continuously for
1–3 days. Wells are NOT replicates of each other: they are different cell lines.
Check `manifest.json` → `cell_line` before comparing anything across wells.

```
<well>/
  frames/frame_00042.png   raw image, one per timepoint, 0-based
  labels/frame_00042.png   16-bit PNG; pixel value = track_id of the cell there,
                           0 = background. Lossless.
  tracks.csv               one row per cell shape per frame
  events.csv               candidate division/death events found by the pipeline
  manifest.json            calibration + acquisition metadata
  summary.json             the original pipeline run's own summary
```

## Two things that will produce wrong numbers if ignored

**1. The time between frames is not constant.** Do not multiply frame counts by a
fixed interval. On the TSC batch2 wells the microscope requested 180 s but
actually delivered a 294 s median, ranging from 175 s to 871 s, because it was
visiting 16 positions per cycle and fell behind. Assuming the nominal value
understates elapsed time by ~39%. Use `manifest.frame_timestamps_ms`, which is
exact per frame, or the `time_ms` column in `tracks.csv`.

**2. `(track_id, frame)` is not a unique key.** The tracker sometimes merges two
genuinely distinct cells under one `track_id`. `n_masks_in_frame` says how many
shapes shared that id in that frame; `manifest.track_multiplicity.suspect_tracks`
lists ids where this happens in more than half the track's frames. Those ids
should be excluded from any measurement — averaging two cells 18 µm apart
produces a position where no cell is. Corpus-wide this affects ~1.4% of tracks.
`events.csv` uses the same ids, so it inherits the same caveat.

## tracks.csv columns

| column | meaning |
|---|---|
| `track_id` | the cell's identity over time |
| `frame` | 0-based timepoint |
| `time_ms` | elapsed ms since frame 0 — real, not assumed |
| `cx`, `cy` | centre position in pixels |
| `area_px`, `area_um2` | size; the µm² version uses this acquisition's own calibration |
| `bbox_*` | bounding box, y0/x0/y1/x1 |
| `intensity_mean`, `intensity_integrated` | brightness, measured on the original 16-bit data |
| `raw_label` | pre-merge tracker label; real key is (track_id, frame, raw_label) |
| `n_masks_in_frame` | >1 means this id covers several shapes here — see above |

## What the images actually show

The marker is **H2B-mCherry**: a red fluorescent protein fused to a histone, so
it labels **chromatin only**. There is no membrane or cytoplasm signal. Practical
consequences:

- The detected shapes are **nuclei, not whole cells**. A cell with two nuclei
  appears as two separate shapes and cannot be distinguished from two adjacent
  cells by image alone.
- Because histones are stoichiometric with DNA, `intensity_integrated` tracks
  **DNA content**. It should stay roughly flat through mitosis while chromatin
  condenses, and should be about double in the whole-genome-duplicated (WGD) line.
- Mitotic stages are clearly visible: chromatin condenses, aligns into a plate,
  then separates into two masses.

**Resolution limit.** Pixels are ~0.57 µm, which is about 3× coarser than the
optics resolve. Nuclei (~12 µm, ~21 px) and metaphase plates (~8 µm, ~14 px) are
well sampled. Sub-nuclear features are not: a micronucleus is ~2.6 px, a lagging
chromosome ~1.7 px, an anaphase bridge ~0.5 px. Do not claim to see structures at
that scale — the pixels to support such a claim were never recorded.

**Brightness is not comparable across frames** in `frames/*.png`. Each frame was
independently rescaled to 8-bit for display. For real intensity, use the
`intensity_*` columns, which were measured on the original 16-bit data.
`manifest.intensity_curve` holds both the per-frame mean (which rises as cells
proliferate) and the per-cell median (the density-controlled signal that shows
actual photobleaching).
