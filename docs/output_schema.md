# `data/output/events.csv` column reference

One row per **daughter cell** produced by a detected split (a normal 1->2 split
produces 2 rows; a multi-way split produces N rows, all sharing the same frame
and parent_id).

| Column | Meaning |
|---|---|
| `event_id` | Row counter, no other meaning. |
| `source_video` | Filename of the input video this row came from. One run = one video; this column makes the CSV self-describing if you're comparing output from multiple runs side by side. |
| `frame_range` | **Approximation only.** `[peak_frame - 10, peak_frame]` -- a fixed lookback window, not a real measurement of the metaphase-anaphase window like the ground-truth sheet's `Frame` column. Detecting the true window needs per-frame morphology classification, which is out of scope for v1. |
| `peak_frame` | The actual detected frame: the first frame the daughter cells appear as 2+ separate masks instead of 1. **0-indexed** (frame 0 = the first frame of the video) -- the ground-truth sheet is 1-indexed, so add 1 before comparing (`scripts/score_against_ground_truth.py` already does this). |
| `split_topology` | `normal_split` (1->2 daughters) or `multi_way_split` (1->3+). Tracks the number of daughters from the lineage graph. Distinct from ACD division type (bipolar/tripolar/multipolar), which describes spindle geometry and will be a separate column when the division classifier is wired in. |
| `track_id` | The track ID assigned to *this* daughter cell going forward. |
| `parent_id` | The track ID of the cell that split to produce this daughter -- i.e. look up other rows with this same `track_id` value to trace lineage back another generation. |
| `claude_confidence` | Rule stage: `persistence_frames / confidence_max_frames` capped at 1.0 (same number as `tracker_persistence_score`, before Claude looks at it). After Claude review: Claude's own 0.0–1.0 confidence if verdict was real, forced to `0.0` if false positive. **Filter on `claude_confidence > 0` to get confirmed events — there is no validated stricter cutoff (e.g. >=0.5, >=0.7) for the current pipeline version; see gotcha below and `tracker_persistence_score`'s entry for why not to sweep a threshold.** |
| `classification_source` | `"rule"` for auto-confirmed events; `"claude"` once vision review has run on the event. |
| `bleach_risk` | `peak_frame / total_frames` (0.0–1.0). Proxy for photobleaching accumulation — higher values mean the event occurred later in the timelapse, where SiR-DNA signal may be degraded. Treat Claude division-type classifications with higher skepticism as this value approaches 1.0. |
| `tracker_persistence_score` | The persistence-based score computed *before* Claude ever reviews the candidate (daughter masks surviving N frames / max frames) — kept after review so you can compare tracker behavior against Claude's verdict. **Not a reliable real-vs-noise discriminator on its own**: in crowded fields, real divisions and false positives have scored similarly on this metric (0.1–0.2 for both). Don't filter on this column — filter on `claude_confidence`. |
| `centroid_x` / `centroid_y` | Pixel coordinates of the dividing cell at `peak_frame`, raw frame space (no transform). Matches `review_crops` crop centers. |
| `claude_notes` | Claude's free-text description from the review call. Populated regardless of verdict (real or false_positive). |
| `acd_division_type` | `bipolar` / `tripolar` / `multipolar` — spindle geometry from the ACD classifier. Only populated for confirmed real events (`claude_confidence > 0`). |
| `misaligned_chromosomes` / `lagging_chromosome` / `anaphase_bridge` / `micronucleus` | `1`/`0` abnormality flags from Claude's read, only populated for confirmed real events. `anaphase_bridge` has a documented history of over-calling subtle nuclear indentations as bridges — treat it with more skepticism than the other three. |
| `anomaly_notes` | Claude's free-text notes on anything unusual beyond the four flagged categories. |
| `near_edge` | `1`/`0` — centroid within `NEAR_EDGE_MARGIN_PX` (100px) of any frame boundary. Partial visibility at the image boundary produces messier/more uncertain classifications. **Flag, don't exclude**: near-edge splits are still real and belong in total confirmed-split counts, but exclude them (`near_edge != 1`) when computing anomaly-subtype rates (micronucleus %, anaphase_bridge %, etc.), since those need the whole cell visible. |

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

Aggregate counts only: `video_path`, `total_events`, `event_counts` (by
`division_type`).
