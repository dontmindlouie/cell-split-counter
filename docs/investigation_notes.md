# Investigation notes

Running log of spot-check findings that don't belong in code comments but shouldn't
be lost between sessions. Newest entries on top.

## 2026-07-21: Batch A/B human ground truth — failed-split precision ~0%, death/dropout confirmed at 96%, mitotic rounding as candidate root cause

Full results for the two targeted review batches (`batch_review_viewer.py`, hand-labeled
by the maintainer): `verdicts_Batch_A_-_Failed_Split_Review.csv` (30/30 complete) and
`verdicts_Batch_B_-_Death_vs_Dropout_Review.csv` (50/50 complete). Batch A stratifies
`gpt_medium_floor85`'s `failed_split` events by confidence tercile; Batch B stratifies
the base M4 run's death events by the model's own `likely_division_dropout` flag
(25 flagged / 25 not).

### Batch A: failed-split characterization

| verdict | count | % |
|---|---:|---:|
| not a real division attempt | 23 | 77% |
| unsure | 7 | 23% |
| genuine failed division | 0 | 0% |

Zero of 30 were confirmed as a genuine failed division (furrow forms, then re-fuses) —
the exact question this batch was built to answer. Of the 7 "unsure" rows, 2 carry an
explicit override note ("successful mitosis, ignore unsure") — a real division did
happen, it just *succeeded*, contradicting the model's "failed" label; "unsure" was
only picked because "successful" wasn't one of the three offered verdict options. 2 more
are a genuine partial case ("parent successful, daughter could fail") the binary
failed/not-failed frame doesn't capture. 3 are unresolved with no clarifying note.

Net: at minimum 25/30 (83%), arguably all 30/30, are not a real failed division under
this confidence-floor config. Combined with the prior n=5 spot-check (also 0/5
agreement, see memory), this is now a decisive, much larger sample landing on the same
conclusion: `split_type=failed` has ~0% precision for what a human calls a real failed
division, at least under `gpt_medium_floor85`.

### Batch B: death vs. dropout

| model's `likely_division_dropout` | still alive (tracking lost it) | real death | unsure |
|---|---:|---:|---:|
| 1 (flagged dropout) | 25/25 (100%) | 0 | 0 |
| 0 ("plain"/confident death) | 23/25 (92%) | 1 | 1 |

48/50 (96%) overall: the cell was actually still alive, tracking just lost it. The
model's own dropout flag gives **no discriminating signal** here — the pool it was
*not* flagging as suspicious is nearly as likely (92% vs. 100%) to turn out alive as the
pool it flagged. This is real ground truth, sharper than the earlier cross-dataset
estimate (69-84%, see memory) — and it says the current flag isn't doing useful
discriminating work at all.

24/50 (48%) of notes explicitly say "divided halfway through" — not a generic
disappearance, an active division tracking dropped mid-way. This skewed toward the
*non*-flagged pool (19 of 24) — the opposite of what the flag would ideally catch.

Two individual cases worth a closer look later: track 1122 ("cytokinesis failure,
interesting, but technically not death" — fits none of the three verdict options), and
track 9975 (the one real death — "the researcher keeps saying necrosis not death," a taxonomy
granularity not currently distinguished).

### Candidate root cause: mitotic rounding

the maintainer's own pattern from reviewing the crops: cell elongates, then appears to vanish
for a few frames, then reappears already split — around what looks like the
metaphase-to-anaphase transition. Checked against the literature (not verified against
this pipeline's own raw masks yet — see caveat below), this maps cleanly onto
well-documented mitotic mechanics, not something specific to this codebase:

- **Rounding (prophase → metaphase):** mitotic cells detach from the substrate and round
  into a near-sphere, driven by actomyosin cortex contraction plus a hydrostatic
  pressure buildup — cortical stiffness increases ~2-4x (Stewart et al. 2011, *Nature*;
  Frontiers 2020 review). In phase-contrast imaging this produces a sharp
  brightness/contrast change right before division (more diffracted light from the
  spherical shape) plus documented "shade-off"/"halo" optical artifacts that
  "complicate standard image processing" for segmentation. A drastically different,
  feature-poor, high-contrast blob is a plausible failure mode for both Cellpose
  segmentation and Trackastra's frame-to-frame linking, independent of any bug here.
- **Elongation (anaphase B):** a distinct, well-timed phase — spindle poles separate
  fast initially, then a second, slightly slower phase specifically coincides with
  visible cell elongation, a few minutes after anaphase onset, before the cleavage
  furrow becomes visible (~6 min after anaphase onset).
- **Split (telophase/cytokinesis):** furrow ingression completes the physical
  separation into two daughters.

This is a genuine, literature-grounded explanation for why a brightfield/phase-contrast
tracker would specifically struggle right around division — not proof this is exactly
what Trackastra/Cellpose are doing here, just a strong, testable hypothesis: if true,
the failure window should cluster tightly around the few-frame rounding/elongation
transition rather than being randomly distributed.

**Not yet done:** confirm against this pipeline's own raw Cellpose masks for a few of
the 24 "divided halfway through" events, same method as the 2026-07-04 raw-mask trace
below, rather than resting on general literature alone. If confirmed, the fix is
probably segmentation-side (e.g. loosening `cellprob_threshold` specifically for
hyper-round/bright objects) or tracker-side (a rounding-aware gap-bridging heuristic),
not a review-prompt fix.

Sources: [Mitotic cell rounding (Wikipedia)](https://en.wikipedia.org/wiki/Mitotic_cell_rounding),
[Hydrostatic pressure and the actomyosin cortex drive mitotic cell rounding (Nature)](https://www.nature.com/articles/nature09642),
[The Mechanics of Mitotic Cell Rounding (Frontiers)](https://www.frontiersin.org/journals/cell-and-developmental-biology/articles/10.3389/fcell.2020.00687/full),
[Mathematical imaging methods for mitosis analysis in live-cell phase contrast microscopy (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1046202317300518),
[Anaphase B (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5192431/).

## 2026-07-04 (follow-up): raw Cellpose mask trace for GT events 9/10 — refines the condensation hypothesis: it's intermittent low-confidence detection, not fragmentation

Wrote `scripts/debug_raw_masks.py` to dump raw per-frame Cellpose mask count/area/
centroid for a spatial ROI directly from `labels.dat`, bypassing every tracker.
Ran it on the events-9/10 region (`--roi 200 320 440 580 --frames 44 70`) to check
the manual-Fiji hypothesis below (pre-division chromatin condensation confusing
segmentation into fragments). The raw data tells a more specific story than that
guess:

- **Frame 52:** the object first appears in Cellpose's output at all — area 684
  (~43% of a normal nucleus's ~1580px in this frame), centroid (260, 526).
- **Frames 53-54: gone.** No mask anywhere near this location.
- **Frames 55-58:** reappears, area 419-920, drifting position frame to frame.
- **Frame 59: gone again.**
- **Frame 60:** reappears at area 286 (~18% of normal size).
- **Frames 61-62:** splits into two spatially distinct small pieces — (263, 485)
  and (265, 519), areas ~230-300 each. This is `ilp`'s candidate (frame 62,
  centroid 264.7/520.1, event_id 70/71 in `tom20_ilp_stride3/events.csv` —
  matches the (265, 519) piece almost exactly).
- **Frames 63-66:** both pieces persist, drifting apart.
- **Frame 67: the (265, 520) piece — the one daughter `ilp` actually
  candidated — disappears from segmentation entirely.** Matches the "bottom
  sister disappears" observation from the manual Fiji check below, but now
  confirmed at the raw-mask level, not just visually.
- **Frames 68-70:** only the other daughter piece remains.

(Two unrelated normal-sized cells also pass through this ROI across these frames —
a ~1550-1800px nucleus drifting steadily through the whole window, and a second
one entering around frame 61 — inflating the raw `n_masks` count per frame; neither
is part of this event.)

**Revised conclusion:** this isn't fragmentation from condensed/punctate chromatin
confusing Cellpose into multiple pieces — it's an object that's small/dim enough
that Cellpose's detection is unreliable on it, full stop. It flickers in and out of
the segmentation output *before* the split (frames 53-54, 59) and the identical
thing happens to one daughter *after* the split (frame 67). This explains `greedy`'s
complete miss structurally: no frame-to-frame overlap linker, however good, can
bridge a gap where the object has zero mask on the missing frame — it's not a
linking-algorithm shortcoming, there's nothing there to link to.

**Cheaper thing to try before building the "interesting event" detector:**
Cellpose's `cellprob_threshold`/`flow_threshold` params (currently left at
defaults) directly control sensitivity to dim/small objects like this one. Worth
testing a lower `cellprob_threshold` on just this ROI/frame range first — if that
stops the flickering, it's a one-line config change instead of new architecture.
Not yet tried.

**Also noticed, not chased down:** `ilp`'s two daughter rows for this event
(track_id 570/571) both store the *identical* centroid (264.7, 520.1) in
`events.csv`, despite the raw masks showing two clearly distinct daughter
locations at frame 62 ((263, 485) and (265, 519)). Something in how per-daughter
centroids get written looks off for this event specifically — worth a quick look
whenever someone's back in `src/output.py`/`src/classify.py`, but not investigated
this session.

## 2026-07-04: manual Fiji cross-reference on GT events 9/10 (frames 60/62) — greedy never candidates the real cell at all; ilp does, but Claude still rejects it even with the stride-3 fix

Followed up the "remaining open thread" from follow-up 3 (60, 62 still listed as
genuinely missed) by scrubbing the raw video by hand in Fiji around the frame 55-70
region instead of relying on crops. Found the real dividing cell by tracking a
faint punctate signal backward and forward from frame ~60:

- Frame 46: clean, nothing unusual at this location.
- Frames ~49-51: a bright, fragmented/punctate chromatin cluster appears — reads as
  a cell entering mitosis (condensation), well before any mask-visible split.
- Frames ~59-60: still a messy fragmented signal, not yet two clean lobes — this is
  the object visible at 75% zoom that looked "obviously dividing but small."
- Frame ~66: resolves into two distinct small puncta (the daughters).
- Frame ~68: one of the two ("bottom sister") has disappeared.

Raw pixel location: approximately (250-270, 490-540). Cross-referenced this against
`events.csv` from all 6 archived Tom20 run variants
(`H:\Archives\cell-split-counter\output-2026-07-04\`):

| run variant | candidate near (260, 520) / frame 55-70? |
|---|---|
| `tom20_greedy_fresh` | none |
| `tom20_greedy_fresh_stride3` | none |
| `tom20_greedy_rereviewed` | none |
| `tom20_ilp` | **event_id 70/71**, frame_range 52-62, peak_frame 62, centroid (264.7, 520.1), tracker_confidence 0.50, **confidence 0.0 (Claude: false_positive)** |
| `tom20_ilp_rereviewed` | same event, same verdict |
| `tom20_ilp_stride3` | same event, same verdict — **the 8+8-frame review-window fix that rescued frame 56 does not rescue this one** |

**`greedy` never candidates this real division in any config, including the
"recovered" 8+8@stride-3 config that follow-up 4's sweep table credits with only
missing frame 177.** That table's recall number for GT events 9/10 is almost
certainly a **false credit** — the scorer matches purely by frame proximity (the
ground-truth spreadsheet has no x/y coordinates at all, per the standing caveat in
follow-up/follow-up-2 above), so `greedy` picking up *some* nearby candidate within
tolerance at roughly the right frame gets scored as a "hit" even though it isn't the
cell manually confirmed here. **97% recall (8+8@stride-3, greedy) should not be
trusted at face value for GT events 9/10 without spatial cross-referencing** —
same "wrong cell" failure mode already documented for `parent_411`/frame 55, now
confirmed for this event specifically.

**`ilp` does find the real cell** (centroid match within ~15px of the manual
Fiji location, tracker_confidence 0.50 — correctly in the "route to Claude" middle
band) **but Claude rejects it as a false positive in all three ilp variants
tested**, unaffected by the stride-3 window widening. `claude_notes` is empty for
this event in the stored CSV despite `classification_source=claude`, so the actual
rejection reasoning wasn't preserved — possibly a re-occurrence of the "notes
discarded on false_positive verdict" bug supposedly fixed 2026-07-02, or this
archived run predates that fix; not yet determined which.

**Why Claude likely rejects it, unlike frame 56:** frame 56's fix worked because the
prompt now explicitly credits gradual elongation-then-constriction of *one coherent
nucleus shape* as a division in progress. This event looks visually different — a
fragmented, multi-punctate cluster (condensed/condensing chromatin) for ~10 frames
*before* resolving into two objects, not a single shape progressively pinching in
two. The existing prompt hint may not cover "starts as scattered fragments, not one
shape" as a valid division pattern. Additionally, even if accepted, the disappearing
"bottom sister" by frame 68 (~2 frames after resolving) would likely then trip the
`min_daughter_persistence>=3` filter as a shape-change artifact, even though this
looks like a genuine division whose daughter drops out of focus/detection shortly
after splitting.

**Net conclusion:** this GT event is a compound miss like frame 56, but a *harder*
one — `greedy` never tracks it (segmentation/tracking-stage failure, unresolved),
and `ilp` tracks it correctly but Claude's review still rejects it even after the
fix that worked for frame 56 (review-stage failure, distinct visual pattern from
frame 56's). Fixing this class of event would need either a segmentation-side fix
(so `greedy` doesn't lose a fragmenting/condensing nucleus) or a prompt addition
specifically for "fragmented punctate cluster resolving into two objects" as a
distinct division signature from simple elongation-constriction.

**Not yet checked:** GT event 10 (frame 62) may be the same physical event as
event 9 (frame 60) given the tight frame proximity and shared centroid match above
(peak_frame 62 in the ilp candidate) — worth confirming with the researcher whether her sheet's
rows 27/28 are actually two separate divisions or one event logged with an uncertain
peak frame, similar to the frame-65 double-row issue already found.

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
