Twice in the same [[project_cell_split_counter]] session (2026-07-05), a memory note
marked "unconfirmed" or "parked" turned out to be stale the moment it was checked
against the actual repo:

1. The full 2026-07-03 stride-3/wider-review-window fix (recall 51.5%→97% on the
   validation video) wasn't in memory at all — a whole investigation arc had happened and was never
   written back, so I quoted a user a recall/missed-events list that was a week stale.
2. "Suspected real bottleneck: unbatched Cellpose calls... Hypothesis, not yet
   profiled/proven" was contradicted by `git log`/`git show` on `src/segment.py`:
   batching was implemented and tuned the same day the note was written (commits
   `4ba3852`, `8592e25`), with a real (if underwhelming, ~17%) measured result.

**Why:** sessions frequently do real follow-up work on a hypothesis or backlog item
without that work making it back into the persistent memory file — memory lags the
actual repo state more than expected, especially for fast-moving technical projects
with many same-day commits.

**How to apply:** before telling a user an item is "still open," "unconfirmed," or
recommending it as a next step based on a memory note, cross-check against the
actual source of truth first — `git log --oneline -- <file>` / `git show <commit>`
for code repos, or the equivalent for other projects. This is cheap (one or two
tool calls) and prevents confidently giving stale prioritization advice. Applies
most when a memory note itself flags uncertainty ("unconfirmed," "parked," "not yet
profiled") — those are exactly the notes most likely to have been overtaken by
later, unrecorded work.
