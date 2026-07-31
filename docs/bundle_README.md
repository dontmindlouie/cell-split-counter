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
  lineage.csv              mother/daughter links between track_ids
  manifest.json            calibration + acquisition metadata
  summary.json             the original pipeline run's own summary
  annotations.csv          human-verified verdicts -- NOT written by build_bundle.py,
                           only exists once someone calls the MCP server's annotate()
                           tool. Append-only; a row is never edited or removed.
  browsers/*.html          self-contained HTML galleries written by render_browser()
```

## annotations.csv columns

Written by `cell_mcp.py`'s `annotate()` tool, one row per call, never overwritten.
Deliberately a SEPARATE file from `events.csv` (machine-generated, gets replaced on
every pipeline re-run, and only has rows for events the detector found in the first
place — so a human verdict recorded onto it would be capped at the detector's own
recall). `event_id` links back to `events.csv` when the pipeline also flagged the
same event; empty means the human found it and the pipeline didn't.

| column | meaning |
|---|---|
| `timestamp`, `annotator` | when and who |
| `well`, `cell_line`, `condition` | so a cohort rollup across wells is a query, not a rewrite |
| `track_id` | the cell |
| `event_id` | optional link to a row in `events.csv`; empty if human-found only |
| `outcome_class` | free text, e.g. "divides" / "dies" / "neither" |
| `condensation_frame`, `metaphase_frame`, `anaphase_frame`, `exit_frame` | the four stage marks; any may be empty |
| `parent_id`, `daughter_ids` | may override `lineage.csv` if that link was determined to be wrong |
| `notes` | free text |

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

**3. A track usually ends at the moment its cell divides.** The two daughters get
new `track_id`s, so the division is not contained in any single track — it falls in
the gap between the mother's last frame and the daughters' first. Every labelled
event in the M12_RUES2 review set (130/130) sits exactly on a track boundary for
this reason. Use `lineage.csv` to cross that gap, and read frames on both sides of
it rather than only within one track.

**4. A track can just as easily *begin* mid-division, not just end there.** The
same rounding that breaks the mother's side of a division also breaks the
daughter's: as chromatin condenses the nucleus goes round and dim, Cellpose loses
the mask for a few frames, and the daughter's track only starts once it's
detectable again — sometimes already mid-anaphase or later. So a track's own
`first_frame` is not necessarily where the cell's story starts. If a track looks
like it's already mid-event in its very first frame, check frames *before*
`first_frame` (via the mother, from `lineage.csv`, or by widening a filmstrip
request past the track's own start) before concluding the lead-up can't be seen.

## lineage.csv columns

| column | meaning |
| --- | --- |
| `track_id` | the cell |
| `parent_id` | the track it was born from; empty = none recorded |
| `first_frame`, `last_frame` | its span (complete-coverage bundles only) |
| `n_daughters`, `daughter_ids` | space-separated track_ids born from this one |

Topology only, no verdicts: a daughter link means the tracker connected two ids
across a division, not that the division was real or normal. Check
`manifest.lineage.coverage` first — `complete` covers every track, while `partial`
was recovered from `events.csv` and therefore only covers tracks the pipeline
flagged, so a missing parent there means **unknown**, not orphan.

**Links are unreliable in both directions, not just by omission.** Absence being
"unknown" is the familiar half; the other half is that **presence is not evidence
either**. Verified cases in `TSC_batch2_M12_RUES2` alone: three textbook anaphases
(2036, 4714, 5286) with *no* daughters recorded; a recorded daughter of 3908 (4170)
that is a ~6 µm² micronucleus budding off rather than a second nucleus; and 6425's
recorded mother 4866 being a healthy neighbouring cell a few microns away rather
than its parent. Two cheap checks before trusting a link: a real division roughly
**halves the mother's area and it stays halved**, and the daughters' areas should
**sum to about the mother's** — a "daughter" far smaller than half is a fragment or
micronucleus, not a cell. Compare the frame spans too: a "daughter" starting long
after the mother ends, or overlapping it heavily, is an id-linking artifact.

**Bundles built before 2026-07-30 have no `solidity` column** — it was added to
`scripts/build_bundle.py` on that date. Absence means "not yet rebuilt," not
"solidity is zero here."

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
| `solidity` | area / convex-hull area, 0–1. Drops when a mask rounds up or briefly fragments — the strongest single geometric signal found so far for locating mitosis (stronger than area or intensity alone); a dip that recovers over a few frames is more informative than the raw value at any one frame |
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
