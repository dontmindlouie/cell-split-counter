# `data/output/events.csv` column reference

One row per **daughter cell** produced by a detected split (a normal 1->2 split
produces 2 rows; a multi-way split produces N rows, all sharing the same frame
and parent_id).

| Column | Meaning |
|---|---|
| `event_id` | Row counter, no other meaning. |
| `source_video` | Filename of the input video this row came from (e.g. `20251016_ACTB_Tom20 - Denoised_Tom20.avi (blue).avi`). One run = one video; this column makes the CSV self-describing if you're comparing output from multiple runs side by side. |
| `frame_range` | **Approximation only.** `[peak_frame - 10, peak_frame]` -- a fixed lookback window, not a real measurement of the metaphase-anaphase window like the ground-truth sheet's `Frame` column. Detecting the true window needs per-frame morphology classification, which is out of scope for v1. |
| `peak_frame` | The actual detected frame: the first frame the daughter cells appear as 2+ separate masks instead of 1. **0-indexed** (frame 0 = the first frame of the video) -- the ground-truth sheet is 1-indexed, so add 1 before comparing (`scripts/score_against_ground_truth.py` already does this). |
| `split_topology` | `normal_split` (1->2 daughters) or `multi_way_split` (1->3+). Tracks the number of daughters from the lineage graph. Distinct from ACD division type (bipolar/tripolar/multipolar), which describes spindle geometry and will be a separate column when the division classifier is wired in. |
| `track_id` | The track ID assigned to *this* daughter cell going forward. |
| `parent_id` | The track ID of the cell that split to produce this daughter -- i.e. look up other rows with this same `track_id` value to trace lineage back another generation. |
| `confidence` | Rule-based: `persistence_frames / confidence_max_frames` capped at 1.0. Claude-reviewed: Claude's own 0.0–1.0 confidence if real, 0.0 if false positive. |
| `classification_source` | `"rule"` for auto-confirmed events; `"claude"` once vision review has run on the event. |
| `bleach_risk` | `peak_frame / total_frames` (0.0–1.0). Proxy for photobleaching accumulation — higher values mean the event occurred later in the timelapse, where SiR-DNA signal may be degraded. Treat Claude division-type classifications with higher skepticism as this value approaches 1.0. |

## Known limitations (v1)

- **Cascade noise:** real divisions can still trigger a few spurious re-detections
  if daughter cells flicker between touching/separating across many consecutive
  frames; `classify_events`'s `cascade_window` parameter suppresses re-splits of a
  just-split track within N frames, but isn't perfect.
- **No abnormality detection.** Misaligned chromosomes, lagging chromosomes,
  micronuclei, and anaphase bridges -- all present in the ground-truth sheet -- are
  not detected.
- **`frame_range` is not the real metaphase-anaphase window**, see above.
- Only validated against `20251016_ACTB_Tom20 - Denoised_Tom20.avi (blue).avi`
  (575 frames). The other raw sample (`iPSC_nTSC_ZO1_1_RED.avi`, 9,431 frames) has
  not been run through the pipeline.

## `data/output/summary.json`

Aggregate counts only: `video_path`, `total_events`, `event_counts` (by
`division_type`).
