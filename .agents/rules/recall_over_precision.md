the maintainer clarified (2026-07-03) that false positives in this pipeline's Claude-reviewed
output are not costly the way a pure precision/recall/F1 comparison implies, because the
project's real deliverable is a curated set of candidate events for the research
scientist (the researcher) to review (see [[project_cell_split_counter]] "Product direction decided
2026-06-28: Track 2 — interesting event packager"), not an automated division count. A
missed real division/abnormality is a permanently lost finding; a false positive in the
reviewed output just costs the researcher a few seconds dismissing it.

**Why:** the ground-truth sheet was never "divisions only" — it already logs failed
splits and abnormalities (misaligned chromosomes, lagging chromosome, micronucleus,
anaphase bridge) as interesting events the researcher tracks by hand. Recall against *any* interesting
event is the actual value driver; clean precision on plain divisions is secondary.

**Important distinction — don't conflate two different kinds of "false positive":**
- the researcher's 2026-06-28 complaint ("0-confidence noise... too much eye strain to sift through")
  was about *raw, unreviewed* tracker noise before Claude ever screened it. That's still
  bad and should stay filtered out.
- A Claude-reviewed candidate that turns out to be a false positive (already survived
  triage into a labeled review folder) is cheap to dismiss and should NOT be penalized
  the same way when comparing pipeline variants.

**How to apply:** when comparing tracker modes (e.g. greedy vs. ilp,
[[project_cell_split_counter]] "Trackastra ilp investigated and rejected 2026-07-03") or
Claude-review prompt/threshold changes, weight recall gains well above precision losses,
as long as the precision loss is confined to the already-reviewed candidate set (not raw
noise reintroduced upstream of Claude). A fix that recovers a real missed division/
abnormality at the cost of a few more reviewed-and-dismissed false positives is a good
trade, not a wash — don't default to F1 or "which has higher precision" as the deciding
metric without applying this weighting first.
