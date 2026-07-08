# `data/output/events.csv` column reference

One row per detected event: either a **daughter cell** produced by a split (a
normal 1->2 split produces 2 rows; a multi-way split produces N rows, all
sharing the same frame and parent_id), or a **track end** (`death`) -- a track
that stops without splitting, one row per track. Track ends near the frame
boundary (more likely a cell walking out of the field of view than dying) and
tracks that only lasted a few frames (more likely a segmentation blip than a
real death candidate) are dropped silently, not reported at all -- see the
`split_topology` entry below.

Column order groups by what you'd use together: identity, then location/geometry
(frame + coordinates + size, since finding an event and checking its size both
start from "where is it"), then lineage, then review/confidence, then
classification. (Reordered 2026-07-07 -- `centroid_x`/`centroid_y` used to sit
near the end, far from `peak_frame`, despite usually being needed together.)

| Column | Meaning |
|---|---|
| `event_id` | Row counter, no other meaning. |
| `source_video` | Filename of the input video this row came from. One run = one video; this column makes the CSV self-describing if you're comparing output from multiple runs side by side. |
| `frame_range` | **Approximation only.** `[peak_frame - 10, peak_frame]` -- a fixed lookback window, not a real measurement of the metaphase-anaphase window like the ground-truth sheet's `Frame` column. Detecting the true window needs per-frame morphology classification, which is out of scope for v1. |
| `peak_frame` | The actual detected frame: the first frame the daughter cells appear as 2+ separate masks instead of 1. **0-indexed** (frame 0 = the first frame of the video) -- the ground-truth sheet is 1-indexed, so add 1 before comparing (`scripts/score_against_ground_truth.py` already does this). |
| `centroid_x` / `centroid_y` | Pixel coordinates of the dividing cell at `peak_frame`, raw frame space (no transform). Matches `review_crops` crop centers. Placed next to `peak_frame` since locating an event needs both together. |
| `near_edge` | `1`/`0` — centroid within `NEAR_EDGE_MARGIN_PX` (100px) of any frame boundary. Partial visibility at the image boundary produces messier/more uncertain classifications. **Flag, don't exclude**: near-edge splits are still real and belong in total confirmed-split counts, but exclude them (`near_edge != 1`) when computing anomaly-subtype rates (micronucleus %, anaphase_bridge %, etc.), since those need the whole cell visible. |
| `cell_area_px` | The parent cell's Cellpose mask area (pixel count) at the split frame. Always populated regardless of pixel-size availability. |
| `cell_size_um2` | `cell_area_px` converted to real units via the acquisition's µm/pixel, when known. **Blank if pixel size is unknown** -- auto-detected from ND2 metadata (varies per acquisition, e.g. Bewo's ND2s are 0.57 µm/px, Tom20's M2 ND2 is 0.432 µm/px; never a fixed constant), or set explicitly via `--pixel-size-um` for AVI sources. Check `summary.json`'s `pixel_size_um` field to see what value (if any) was used for a given run. |
| `split_topology` | Despite the column name (predates non-split events), this is the general event-type column: `normal_split` (1->2 daughters), `multi_way_split` (1->3+), or `death` (track stopped, without splitting, before the video ended, away from the frame boundary). Two categories of track-end are dropped entirely rather than reported: stops within `NEAR_EDGE_MARGIN_PX` of the frame boundary (more likely a cell walking out of the field of view than dying -- no biological content either way) and tracks shorter than `classify_track_ends`'s `min_track_frames` (default 5 -- a track that only ever existed for a couple of frames is a segmentation blip, not a plausible death candidate). `death` rows are rule-only (`classification_source="rule"`) — v1 can't distinguish a real death from a tracking dropout (segmentation losing the cell), so `claude_confidence` here is a track-duration persistence score (`min(1.0, track_duration_frames / confidence_max_frames)`, default `confidence_max_frames=20`), not model confidence, and not yet validated against ground truth (this project has no death/track-end ground truth to score against, unlike splits). Treat any `death` row as "track ended here, survived at least a few frames first," not a confirmed biological death. Distinct from ACD division type (bipolar/tripolar/multipolar), which describes spindle geometry and will be a separate column when the division classifier is wired in. **Bug fixed 2026-07-07:** a single-child node (a track-ID continuation artifact, not a real division) used to be mislabeled `multi_way_split` -- `classify_events` now skips these entirely, emitting no event. Output from runs before this fix may still have singleton `multi_way_split` rows; check sibling count before trusting `multi_way_split` counts in older packages. |
| `track_id` | The track ID assigned to *this* daughter cell going forward. |
| `parent_id` | The track ID of the cell that split to produce this daughter -- i.e. look up other rows with this same `track_id` value to trace lineage back another generation. |
| `classification_source` | `"rule"` for auto-confirmed events; `"claude"` once vision review has run on the event (also used for the `gpt` backend -- the column name predates that option). |
| `claude_confidence` | Rule stage: `persistence_frames / confidence_max_frames` capped at 1.0 (same number as `tracker_persistence_score`, before the vision model looks at it). After review: the model's own 0.0–1.0 confidence if verdict was real, forced to `0.0` if false positive. **Filter on `claude_confidence > 0` to get confirmed events — there is no validated stricter cutoff (e.g. >=0.5, >=0.7) for the current pipeline version; see gotcha below and `tracker_persistence_score`'s entry for why not to sweep a threshold.** |
| `tracker_persistence_score` | The persistence-based score computed *before* the vision model ever reviews the candidate (daughter masks surviving N frames / max frames) — kept after review so you can compare tracker behavior against the model's verdict. **Not a reliable real-vs-noise discriminator on its own**: in crowded fields, real divisions and false positives have scored similarly on this metric (0.1–0.2 for both). Don't filter on this column — filter on `claude_confidence`. |
| `claude_notes` | The vision model's free-text description from the review call. Populated regardless of verdict (real or false_positive). |
| `bleach_risk` | `peak_frame / total_frames` (0.0–1.0). Proxy for photobleaching accumulation — higher values mean the event occurred later in the timelapse, where SiR-DNA signal may be degraded. Treat division-type classifications with higher skepticism as this value approaches 1.0. |
| `acd_division_type` | `bipolar` / `tripolar` / `multipolar` — spindle geometry from the ACD classifier. Only populated for confirmed real events (`claude_confidence > 0`). |
| `misaligned_chromosomes` / `lagging_chromosome` / `anaphase_bridge` / `micronucleus` / `binucleation` | `1`/`0` abnormality flags from the review model's read, only populated for confirmed real events. `anaphase_bridge` has a documented history of over-calling subtle nuclear indentations as bridges — treat it with more skepticism than the other four. `binucleation` (added 2026-07-07) is one cell body containing two nuclei that don't progressively separate -- distinct from `split_topology=failed` (which re-fuses back into one nucleus). |
| `anomaly_notes` | The vision model's free-text notes on anything unusual beyond the four flagged categories. |

**Important gotcha:** `claude_confidence` is forced to `0.0` whenever Claude's verdict was
`false_positive`, regardless of how confident Claude actually was in that rejection.
`review_crops/<event>/verdict.txt` (if present) preserves Claude's *raw* confidence for
the verdict it actually gave (real or false_positive) — the two numbers measure different
things and will disagree for rejected candidates.

**Filtering guidance:** use `claude_confidence > 0`, full stop — don't sweep a numeric
threshold on top of it. Earlier notes referencing a `>=0.70` cutoff came from a small
20-frame smoke test on an older review-window architecture and don't generalize; the
current validated numbers (8+8 frames @ stride-3 review window, `docs/investigation_notes.md`)
are reported at the confidence>0 / confidence==0 split, not a swept threshold. Claude's raw
confidence value also has several points of run-to-run variance (no fixed temperature/seed),
so treating it as a precise dial isn't warranted yet.

## Known limitations (v1)

- **Cascade noise:** real divisions can still trigger a few spurious re-detections
  if daughter cells flicker between touching/separating across many consecutive
  frames; `classify_events`'s `cascade_window` parameter suppresses re-splits of a
  just-split track within N frames, but isn't perfect.
- **`frame_range` is not the real metaphase-anaphase window**, see above.
- **Slow/gradual divisions are systematically under-detected** — a division that takes
  many frames to visibly separate can be missed entirely or mis-timed (root-caused on
  Tom20 GT event 6, reproduced independently on Bewo M3).
- **Events near the frame edge** tend to produce noisier `acd_division_type`/abnormality
  calls, since only part of the cell is visible.
- Validated with ground truth only against a single 575-frame test video (Tom20). Other
  videos/cell lines have been run but not scored against ground truth.

## `data/output/summary.json`

Aggregate counts only: `video_path`, `pixel_size_um` (µm/pixel actually used for
this run's `cell_size_um2` column, `null` if unknown), `total_events`,
`event_counts` (by `division_type`), and `vision_usage` (api_calls/tokens/cost
for whichever `--vision-backend` was used, on runs generated after 2026-07-06;
named `claude_usage` before 2026-07-08).
