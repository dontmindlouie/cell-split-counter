the maintainer wants the cell-split-counter pipeline ([[project_cell_split_counter]]) to eventually
capture "interesting events" beyond just cell divisions. Investigated scope 2026-07-07 and
shipped the first, smallest piece; a second, bigger piece is scoped but not started.

**Architecture finding:** `track.py` builds a complete per-frame lineage graph (every cell's
mask/area/centroid/parent/children each frame) — this part was already event-agnostic.
`classify.py`'s `classify_events()` only ever looked at nodes with 2+ children (splits);
childless track-ends were defined in the `EventType` enum but never emitted — the old
docstring literally said "deferred." Vision review (`review.py`) was also split-only:
abnormality flags (binucleation, micronucleus, etc.) only ever got populated on split
candidates, so a cell showing an anomaly *without* ever dividing was structurally invisible
to the whole pipeline.

**Shipped 2026-07-07, final form after two passes:** added `classify_track_ends()` in
`src/classify.py` — emits a `DEATH` event for any track whose last node has no children and
ends before the video's last frame (tracks still alive at video-end are excluded, that's not
a death). First pass also emitted `ROI_EXIT` for track-ends near the frame boundary, but
the maintainer called this out as having zero biological content (a cell that walked out of the
field of view didn't *do* anything) — so `ROI_EXIT` was removed from the `EventType` enum
entirely, and near-edge track-ends are now silently dropped, not reported. Separately, raw
"did it stop" was recognized as an untrustworthy signal on its own, since this dataset's
dominant failure mode is the tracker losing a healthy cell for a few frames, not real death —
so `classify_track_ends` now also (a) drops tracks shorter than `min_track_frames` (default 5)
outright as segmentation blips, and (b) scores survivors with a duration-based persistence
confidence (`min(1.0, track_duration_frames / confidence_max_frames)`, default
`confidence_max_frames=20`) instead of a flat 1.0 — same lesson `classify_events` already
learned for splits via daughter persistence. Verified on two real smoke runs (a low-density
sample well, 60 frames): first pass gave 79 death + 33 roi_exit; after this fix, 26 death (confidence
0.25–1.0), 0 roi_exit, splits unaffected (56 rows, same cost). DEATH events remain **rule-only,
never sent to Claude vision review** — v1 still can't distinguish a real death from a tracking
dropout from topology alone, that's a known, documented limitation, not a bug to chase.

**Non-obvious follow-up work this required:** four existing downstream scripts
(`score_against_ground_truth.py`, `csv_to_xlsx.py`, `dump_crops.py`, `package_events.py`) all
dedupe rows by `(parent_id, peak_frame)` and filter on `claude_confidence > 0` to mean "confirmed
split" — since DEATH rows also have a nonzero confidence, they would have silently leaked into
split-only tooling (inflating GT recall scores via false frame-proximity credit, populating
"confirmed_splits"/xlsx exports, getting packaged as divisions with nonsensical crop windows).
All four now explicitly filter to `split_topology in ("normal_split", "multi_way_split")` before
their existing dedup/confidence logic. **Why this matters if this class of pivot happens again:**
any new non-split `EventType` added to the CSV needs the same audit of every script that reads
`events.csv` and assumes "row = split candidate."

**Review mechanism, clarified 2026-07-07 (relevant to the next step below):** the vision-review
path has two separable layers, both in `src/review.py`. (1) Candidate selection
(`review_ambiguous()`) decides *which* events get sent at all — currently only ever fed split
candidates. (2) Crop building + the API call (`_review_and_classify()` / `review_gpt.py`'s
equivalent) is actually generic: frame index + centroid + radius → cropped image sequence, sent
with a prompt. The crop-building code doesn't know or care that it's looking at a split — only
the candidate-selection criterion (topological: "node with 2+ children") and the prompt text are
split-specific. So extending to other "interesting events" doesn't need new crop/API plumbing,
just a different candidate generator feeding the same crop code with a different prompt.

**Next step, scoped but NOT started (the maintainer is unsure yet how involved this should get):**
catching morphology anomalies on tracks that never split at all (e.g. a binucleated cell that
persists its whole track without dividing — currently invisible, since nothing ever routes it to
Claude unless Cellpose happens to segment its two nuclei as separate masks, making it *look*
like a split topologically). Sending every track to Claude would be cost-prohibitive at ~250
cells/frame, so the plan is a cheap rule-based pre-screen before any vision call, using data
that's already computed for free and currently thrown away: `track.py`'s `link_frames_trackastra`
already calls `skimage.measure.regionprops` per frame (track.py:383) but only keeps `centroid`
and `area` — `eccentricity`, `solidity`, `major_axis_length`/`minor_axis_length`, `extent`,
`perimeter` etc. are computed in that same call and discarded. Two uses for these, both unbuilt:
1. **Screen for review candidates** on non-splitting tracks (shape suddenly gets irregular,
   area collapses before a track ends, etc.) — the deferred "interesting event detector" that
   `docs/investigation_notes.md` (2026-07-04 entry) already flagged.
2. **Possibly also strengthen split detection itself** — review.py's own prompt already
   describes real divisions as "rounds up, elongates along a cleavage plane" (an eccentricity
   signature), so eccentricity trending up before a candidate split might be a free, rule-based
   corroborating signal alongside the existing daughter-persistence confidence, with no added
   Claude cost. Not validated, just an idea worth checking against the existing GT-scored split
   candidates before trusting it.

No existing tool solves this out of the box either — Cell-ACDC (evaluated earlier for this
project, see [[project_cell_split_counter]]) has the same gap, no abnormality classification.
That's the actual reason this project reaches for an LLM instead of pure classical-CV rules:
rules can flag "unusual," but "unusual" vs. "this is binucleation" is a judgment call.

**How to apply:** before starting this, pull the shape descriptors into a real dataset (e.g.
re-run the existing GT-scored split candidates from the golden-set video) and check whether eccentricity/solidity
actually separate real splits from false positives, and separately whether they separate
"anomalous" from "boring" on non-splitting tracks — validate before building a new prompt or
screening heuristic on top of a hunch.
