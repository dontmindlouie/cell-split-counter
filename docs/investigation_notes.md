# Investigation notes

Running log of spot-check findings that don't belong in code comments but shouldn't
be lost between sessions. Newest entries on top.

## 2026-07-03 (follow-up 4): more frames beats a bigger gap — settled on 8+8 @ stride-3

After merging the combined verify+classify architecture (5+5 frames @ stride-3) with the
validated stride-3 fix (originally tested at 2+3 @ stride-3), the merged parameter
combination hadn't itself been validated. Each video frame is 3 min of real acquisition
time, so swept 5 configs against the same 180 previously-Claude-reviewed greedy-mode
candidates to decide between widening the *count* of frames shown vs. widening the
*gap* (stride) between them:

| config              | before/after | stride | detected | recall        | precision | F1    | missed GT |
|----------------------|:---:|:---:|---|---------------|-----------|-------|---|
| baseline (2+3 @ s3)  | 2/3 | 3 | 59 | 30/33 = 90.9% | 79.7%     | 0.849 | 177, 185, 212 |
| merged (5+5 @ s3)    | 5/5 | 3 | 68 | 31/33 = 93.9% | 75.0%     | 0.834 | 177, 185 |
| denser (5+5 @ s2)    | 5/5 | 2 | 59 | 30/33 = 90.9% | 76.3%     | 0.829 | 177, 185, 212 |
| sparser (5+5 @ s5)   | 5/5 | 5 | 91 | 31/33 = 93.9% | 65.9%     | 0.775 | 185, 212 |
| **more frames (8+8 @ s3)** | 8/8 | 3 | 85 | **32/33 = 97.0%** | 64.7% | 0.776 | **177 only** |

**Tightening the gap ("every other frame," stride=2) gave no benefit** — identical
recall to baseline, slightly worse precision. **Just widening the gap (stride=5) without
more frames** matches the merged config's recall but costs more precision for no gain.
**Showing more frames at the same gap (8+8 @ stride-3) won clearly** — 97.0% recall,
missing only frame 177, which is the one GT event every single config in this sweep
still misses (consistent with it being a genuine upstream segmentation/tracking gap,
not a review-window/prompt-fixable case, per the earlier follow-ups). **Decision: set
`_FRAMES_BEFORE=8, _FRAMES_AFTER=8, _FRAME_STRIDE=3`** (was 5+5 @ stride-3).

**Caveat — run-to-run variance is real:** re-running the identical baseline (2+3 @ s3)
config gave 90.9% recall here vs. 84.8% in the earlier validation pass — a ~6pp swing
from Claude's inherent non-determinism (no fixed seed/temperature=0), same candidate
set both times. Treat exact percentage-point gaps between configs as directional, not
precise; the "more frames helps, denser gap doesn't" pattern held consistently enough
across the sweep to trust, but a couple of points of difference between any two configs
here shouldn't be over-read.

## 2026-07-03 (follow-up 3): stride-3 review window fix validated — recall 51.5%→84.8% on greedy alone, ilp not needed

Acted on the "remaining open thread" from follow-up 2: widened `review.py`'s crop
sampling from consecutive frames to stride-3 (same frame *count*, wider time span) and
added an explicit prompt hint that gradual constriction still resolving by the last
frame counts as a real division, not just a fully-separated pair. Re-reviewed every
previously-Claude-reviewed split from both the `greedy` and `ilp` full validation-video
runs through the new prompt/window (418 unique splits total, Claude Haiku), scored
against the same 33-event ground truth:

| mode              | detected | recall        | precision   | F1    | missed GT frames |
|-------------------|----------|---------------|-------------|-------|---|
| greedy, pre-fix   | 14       | 17/33 = 51.5% | 13/14=92.9% | 0.663 | 56,60,62,65,65,99,177,185,212,266,363,394,427,440,514,520 |
| **greedy, fixed** | 45       | 28/33 = 84.8% | 37/45=82.2% | 0.835 | 60,62,177,185,212 |
| ilp, pre-fix      | 16       | 15/33 = 45.5% | 11/16=68.8% | 0.547 | (16 missed, incl. 545,567) |
| ilp, fixed        | 48       | 28/33 = 84.8% | 38/48=79.2% | 0.819 | 60,62,177,212,427 |

**The review-window/prompt fix was the real lever, not the tracker linking mode.**
Applying it to plain `greedy` alone recovers recall from 51.5% to 84.8% — including
frame 56 (the exact case from follow-up 2 that motivated trying `ilp` in the first
place) plus most of the other previously-missed frames. Precision drops modestly
(92.9%→82.2%), which is an acceptable trade here: the pipeline's output is a curated
candidate set for human review, not an automated count, so a missed real event is
costlier than one extra reviewed-and-dismissed false positive (+24 true positives for
+7 false positives on greedy).

**This also settles the tracker-mode question:** with the fix applied, `greedy` and
`ilp` land at identical recall (84.8%), but `greedy` still has better precision (82.2%
vs. 79.2%) and is cheaper to run (no global-optimization solve). **`ilp` provides no
remaining benefit once the review fix is in place — greedy stays the default, exactly
as follow-up 1 concluded, but now for a stronger reason: it isn't just "not a clean
win," it's fully redundant.**

**Remaining open thread, narrowed:** only 5 GT events are still genuinely missed
(60, 62, 177, 185/212 depending on mode) — down from 16. These look like real
segmentation/tracking gaps rather than review-prompt issues (the review fix had no
way to help with events the tracker never candidated at all), separate from
everything fixed here. Not investigated yet.

## 2026-07-03: Trackastra greedy vs ilp linking mode

**Trigger:** the frame-2 split spot-check (cell at (432, 454)) was visible in Fiji and confirmed
present in Cellpose's segmentation output, but never reached `events.csv` under the
default `greedy` linking mode. Root cause traced to `link_frames_trackastra` — greedy's
local frame-to-frame matching collapsed the two daughter masks into one track, while
Trackastra's `ilp` mode (exact global optimization, `model.track(..., mode="ilp")`)
correctly split them into two tracks and `classify_events` registered a `NORMAL_SPLIT`.

Runtime check before committing to a full-video `ilp` run: 100-frame/26k-node graph
solved `OPTIMAL` in 32.5s vs. greedy's 19.5s (~1.7x slower, not exponential) —
extrapolated to ~3 min for the second, denser validation video's 575 frames, so `ilp` was tractable.

**Full validation-video result — ilp did not generalize as a fix:**

| mode   | detected | recall       | precision   | F1    |
|--------|----------|--------------|-------------|-------|
| greedy | 14       | 17/33 = 51.5%| 13/14=92.9% | 0.663 |
| ilp    | 16       | 15/33 = 45.5%| 11/16=68.8% | 0.547 |

(Scored with the corrected filter — `classification_source == "claude" and confidence == 0.0`
— see fix below. An earlier pass showing ILP at ~100% recall was a scoring-script bug.)

`ilp` fixed the one single-cell case it was built to catch, but on the denser validation video it produced *more*
false positives (5 vs. greedy's 1: frames 156, 220, 298, 298, 488) and did not recover
any of greedy's 16 missed GT events — it missed everything greedy missed, plus 2 more
(frames 545, 567). **Conclusion: `ilp` is not a straightforward upgrade over `greedy`
for this dataset/density; keep `greedy` as default.** The original spot-check case that motivated
this may be a low-cell-density-specific win (few adjacent daughters, cheap global
solve) that doesn't hold in the denser validation video's ~250-cell/frame crowded field. `--tracker-mode`
flag is kept on `main.py`/`pipeline.py`/`track.py` for future experimentation.

## 2026-07-03 (follow-up): why greedy AND ilp both miss the same 16 GT events

Dug into the frame 56/60/62/65/65 cluster (GT events 6–10, rows 24–28), since
misses shared by both linking modes point to a cause upstream (segmentation) or
downstream (Claude review) of the linking algorithm, not the algorithm itself.

**Scoring bug found:** two of these "missed" GT rows are not real 1→2 divisions at
all — row 25 (peak 65) is annotated `"2 to 2 non-divison"` and row 26 (also peak 65)
is annotated `"2 to2 or 3"` (the researcher's own uncertainty note). `parse_ground_truth_peaks` in
`scripts/score_against_ground_truth.py` reads column C (peak frame) unconditionally
and never checks column D (Division Type), so both rows count toward the 33-event
denominator and inflate the miss count. This explains why frame 65 appears twice in
every missed-GT list. Not yet fixed in the scoring script — division-type filtering
should be added before trusting recall numbers precisely, though the effect here is
small (2/33 events).

**Visual check on the real misses (frames 56, 60, 62):** both greedy and ilp *do*
produce low-confidence candidate split events near these frames (tracker_confidence
0.1–0.2) — e.g. greedy's `parent_430` at frame 59 (targeting GT event 9, peak 60).
Dumped crops (`scripts/dump_crops.py`) and inspected directly: the candidate is a
small, dim, two-lobed structure that looks visually static and near-identical across
the entire frames 57–62 window — no dynamic separation, no rounding-then-splitting
kinetics. Claude's rejection ("no clear division... static population") looks correct
for *this candidate*. Combined with the earlier finding on `parent_411` (frame 55,
centroid ~240px from the GT-implied location, reviewing the wrong cell entirely —
inferred via manual Fiji cross-check, no GT coordinates exist in the source data),
the pattern holds: in this crowded region, the candidates that *do* surface near GT
peak frames are dim/noisy objects unrelated to the true dividing cell, while the real
dividing cell is never candidated by the tracker at all. **This looks like a genuine
segmentation/tracking miss on the actual dividing cell, not a Claude review failure**
— Claude is correctly rejecting the wrong candidates it's being shown.

**Caveat:** the ground-truth sheet has no spatial (x/y) coordinates, only frame
ranges and freeform text — so "wrong cell" can only be inferred by visual cross-
reference (Fiji or crop inspection), never confirmed against a stored ground-truth
location. Any future claim about *which* cell in a crowded frame corresponds to a
GT event should be treated as an informed guess, not a verified match.

## 2026-07-03 (follow-up 2): traced GT event 6 (frame 56) end-to-end — greedy misses it, ILP finds it, Claude wrongly rejects it

Wrote a standalone scan (`_label_to_cellmask` + `_overlap_fraction` from `src/track.py`,
applied directly to the raw `labels.dat` memmap — bypassing Trackastra entirely) over
frames 50-70 to find every geometric 1→2 mask split in the region, independent of
what either tracker candidated. Found ~20 splits in that window. Most are the same
2-3 locations recurring every couple of frames (e.g. centroid ~(1442, 344) recurs at
almost every frame pair from 52 through 67) — these are the chronic dim/small cells
that flicker between 1 and 2 Cellpose labels without any real biological change (same
objects already identified as false positives via `parent_411`/`parent_430` above).

One split stood out as a **true one-off, not a recurring flicker**: frame 56→57,
prev_label=64 at centroid (1006.6, 487.2), splitting into two daughters — landing
exactly on GT event 6's peak frame (56).

- **`greedy` mode: complete miss.** No event anywhere near this centroid/frame in
  the greedy-mode run's `events.csv` — greedy's local matching absorbed this split
  without ever generating a candidate node for Claude or the scorer to see.
- **`ilp` mode: found it exactly.** The ilp-mode run's `events.csv` has an event at frame 57,
  parent_id 57, centroid (1006.557, 487.217) — an exact match to the raw scan.
  Confidence 0.0 (Claude verdict: false_positive, `tracker_confidence` only 0.1).
- **Visual check via `dump_crops.py` (frames 55-60):** the center object is a
  single, subtly waisted/hourglass-shaped nucleus at frames 55-58 (looks like one
  cell to the eye), and only becomes a clearly separated two-lobed/peanut shape by
  frame 60 — classic **slow-division kinetics** (progressive constriction, not an
  instant 1→2 jump). Claude's crop window does include frame 60 and still called it
  "no clear morphological changes" — given how visually apparent the two-lobed shape
  is by frame 60, **this looks like a genuine Claude review miss on a real, slow
  division**, not a correct rejection of noise (unlike the `parent_411`/`parent_430`
  cases, which really were static/noisy on inspection).

**Conclusion for this event:** the miss is a compound failure across the whole
pipeline, not one single stage — `greedy` never candidates it at all; `ilp` does,
but at low tracker_confidence (0.1) because the daughters are still merging visually
frame-to-frame; and Claude's vision review, working from a fixed-size crop window,
called a real slow division a false positive. Fixing recall on events like this
would need either (a) switching to `ilp` (already shown to regress precision
elsewhere — see comparison table above, not a clean win) and/or (b) improving Claude
review's handling of slow/gradual divisions — e.g. a wider frame window, or an
explicit prompt hint that gradual elongation-then-constriction counts as a division
in progress even without full separation by the shown window's end.

**Remaining open thread:** the other two GT events in this cluster (60, 62) weren't
traced to the same level of certainty — the raw scan's recurring-flicker locations
near those frames were ruled out as noise, but no clean one-off match (like the
frame-56 case) was found for them within 50-70. They may be in a part of the frame
range not covered by this scan window, or may be genuine full segmentation misses
(Cellpose never producing 2 masks for the real dividing cell at all). Not pursued
further this session.

**Note:** the original, shorter write-up of this frame-2/pixel-(432,454) miss
(segmentation-vs-tracking root cause, untested Cellpose hypotheses) was superseded by
the fuller investigation above and dropped here to avoid duplication — see the
"Trackastra greedy vs ilp linking mode" section below for the full trace.
