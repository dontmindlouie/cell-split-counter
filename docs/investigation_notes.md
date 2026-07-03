# Investigation notes

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
