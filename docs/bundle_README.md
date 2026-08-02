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
  lineage.csv              mother/daughter links between track_ids
  manifest.json            calibration + acquisition metadata, plus a `provenance`
                           block: when the bundle was built, from which run dir, at
                           which git commit, and -- separately -- when the upstream
                           segmentation/tracking it was built FROM was written
  annotations.csv          human-verified verdicts -- NOT written by build_bundle.py,
                           only exists once someone calls the MCP server's annotate()
                           tool. Append-only; a row is never edited or removed.
  browsers/*.html          self-contained HTML pages written by show_cells_in_browser()
```

## Where the detector's guesses went

There is deliberately **no `events.csv` in the bundle**. The pipeline's candidate
division/death events now live outside it, at `data/candidates/<well>/candidates.csv`.

The move fixed a category error rather than a data problem: a list of machine
*candidates* sat beside the frames named and counted as if it were a list of
*findings*. Everything that went wrong downstream followed from that — `summary.json`
counted its rows as events, `lineage.csv` was derived from it, and the MCP evaluation
had to hide the file to stay valid, which is the clearest possible tell that it read
as an answer key. The bundle now holds **data plus human verdicts**; machine guesses
live somewhere else, under a name nobody mistakes for a verdict.

`summary.json` is **gone from the bundle too** (2026-07-31), for the same reason and
one more: it was the detector's tally, so it inherited the row-counting bug below, and
it went stale the instant tracking was re-run while still reading as the bundle's
authoritative answer. There is deliberately no replacement inside the bundle. Counts
belong to whoever computed them, stamped with when — the tools compute them on demand.

`candidates.csv` is one row per **event** (not per daughter — see below),
`ai_confidence` and `split_type` are dropped, and it is the file `annotations.csv`'s
`event_id` refers to.

> ⚠️ **If you have older numbers, they are wrong by 2× on divisions.** Splits used to
> be written one row per daughter while deaths were one row per event, and anything
> counting rows — including the retired `summary.json`, whose counts were row counts —
> doubled the division side only, so the error did not cancel. Across the 21 bundled
> wells this inflated the corpus from a true 22,585 events to 33,478 rows. **Twelve of
> the 21 wells flip qualitatively**: M14_WGD read 1.21 divisions per death when the
> truth is 0.60, so "more divisions than deaths" becomes its opposite. Relative
> ordering between wells survives (the factor is uniform), but no absolute claim does.

## annotations.csv columns

Written by `cell_mcp.py`'s `annotate()` tool, one row per call, never overwritten.
Deliberately a SEPARATE file from `candidates.csv` (machine-generated, gets replaced on
every pipeline re-run, and only has rows for events the detector found in the first
place — so a human verdict recorded onto it would be capped at the detector's own
recall). `event_id` links back to `candidates.csv` when the pipeline also flagged the
same event; empty means the human found it and the pipeline didn't.

| column | meaning |
|---|---|
| `timestamp`, `annotator` | when and who |
| `well`, `cell_line`, `condition` | so a cohort rollup across wells is a query, not a rewrite |
| `track_id` | the cell |
| `event_id` | optional link to a row in `candidates.csv`; empty if human-found only |
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

**2. `(track_id, frame)` used to not be a unique key. As of the 2026-07-31 rebuild
it is.** The tracker used to merge two genuinely distinct cells under one
`track_id`, so `n_masks_in_frame` could exceed 1 and
`manifest.track_multiplicity.suspect_tracks` listed the worst offenders, which had
to be excluded from any measurement — averaging two cells 18 µm apart produces a
position where no cell is.

That was a bug in gap bridging, not a property of the imaging, and it is fixed.
Across all 21 rebuilt wells and all four cell lines, `suspect_tracks` is empty and
`n_masks_in_frame` is 1 in every row — including the WGD wells, where lobed
genome-doubled nuclei made a genuine multi-mask case most plausible. Un-merging
recovered **1,742 tracks, +3.1%** corpus-wide.

Both fields are kept so the check stays possible rather than assumed. If either
ever reads non-1 again, treat it as a regression in bridging and not as data.
**Any bundle whose `manifest.provenance` is missing predates this fix**, and its
merged ids are still in there.

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
| `first_frame`, `last_frame` | its span |
| `n_daughters`, `daughter_ids` | space-separated track_ids born from this one |
| `link_distance_px` | mother's last centroid to this daughter's first (geometry source only) |
| `dna_ratio` | both daughters' `intensity_integrated` at f+1 over the mother's at f. ~1.0 when DNA is conserved, i.e. a real division (geometry source only) |
| `size_ratio` | smaller daughter's area over larger's. Near 1 for a symmetric division; a low value is the micronucleus/fragment signature (geometry source only) |
| `alt_parents` | other mothers that were also in range, as `id:px`. Non-empty means `parent_id` was a nearest-wins tie-break, not an unambiguous fact (geometry source only) |

`manifest.lineage.source` says where the graph came from. **`ctc`** is Trackastra's
own table, written at tracking time — complete, unscored. **`geometry`** is rebuilt
from `tracks.csv` adjacency (a mother ending at f linked to two births at f+1 within
40 px, each daughter assigned to its nearest eligible mother) — also complete, and
carrying the three score columns above. The old **`events`** source is retired: it
was partial by construction (2,364 of 5,163 tracks on M12_RUES2) and welded topology
to a file of AI verdicts.

**The scores are not filters.** Weak links are still written, so the graph stays pure
topology; the columns exist so you can discount one without opening an image. The
case they were built from: track 6425's recorded mother 4866 has `dna_ratio` 0.94 —
which looks healthy — but `size_ratio` 0.10, and it is a micronucleus beside a
nucleus rather than a daughter (confirmed by eye 2026-07-30). A low size ratio with a
healthy DNA ratio is the fragment signature: the large object carries the DNA and the
small one is debris. Geometry cannot tell those apart, and neither could the events
graph — the difference is that this one shows you the numbers.

**No absolute cutoff is applied, deliberately.** `get_lineage` reports each score's
percentile against *this well's own* links rather than a fixed threshold, because a
cutoff tuned on one cell line does not transfer — WGD nuclei are large and lobed
where RUES2 are compact. Which tail is suspicious differs by measure: a low
`size_ratio` means a fragment, while a high one is just a symmetric division, so only
the low tail is called out. `dna_ratio` is flagged at **both** ends — far below 1
means signal went missing, far above means the pair carries more DNA than the mother
had, so something else was included in the mask (track 189 on M12_RUES2 reads 2.16).

`alt_parents` exists for the same reason. Where two mothers end in the same frame
near each other, both reach the same births and the nearest one wins — 224 of 2,110
links on M12_RUES2 are contested that way. The winner is a tie-break, so the
runners-up ship next to it rather than being silently dropped.

Topology only, no verdicts: a daughter link means two ids were connected across a
division, not that the division was real or normal.

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
