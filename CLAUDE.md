# cell-split-counter

A local stdio MCP server (`cell_mcp.py`) that lets a researcher interrogate live-cell
microscopy bundles. It is a set of instruments, not a detector: tools expose numbers and
pixels and let the reader judge. Score, don't filter.

## Load the whole cell-microscopy tool set, in one call, before anything else

These tools arrive deferred — offered as bare NAMES with no descriptions. Selecting a
subset from names alone has already gone wrong once: a session picked the three
`filmstrip` tools, skipped the one that writes a page for the user, and then spent 16
Bash calls reimplementing it from source. Schemas are cheap; guessing is not.

So the first tool call of any session in this repo is one `ToolSearch` loading **all**
of them:

```
select:mcp__cell-microscopy__list_wells,mcp__cell-microscopy__list_tracks,
mcp__cell-microscopy__find_candidates,mcp__cell-microscopy__get_track_profile,
mcp__cell-microscopy__get_lineage,mcp__cell-microscopy__list_nearby_tracks,
mcp__cell-microscopy__get_neighbourhood_stats,mcp__cell-microscopy__measure,
mcp__cell-microscopy__view_whole_field,mcp__cell-microscopy__follow_cells_over_time,
mcp__cell-microscopy__watch_location_over_time,mcp__cell-microscopy__show_cells_in_browser,
mcp__cell-microscopy__annotate
```

(One line, no spaces, when actually calling it.)

## "Show me" ends in `show_cells_in_browser`

When the user asks to SEE cells — "show me", "let me see", "send me", "give me a link",
"can I look at" — the answer is `show_cells_in_browser`, which writes a labelled HTML
page and returns a path. Describing frames in prose, or pasting filmstrips one at a
time, is the wrong answer to a request to look at something. Do not hand-roll a gallery
script: that tool already does it, with the same crop logic as the filmstrips.

## Data-reading rules live in the server, not here

Anything about how to *read the data* — the non-constant frame interval, merged
track_ids, chromatin-only images — belongs in `MCPServer(instructions=...)` at the top
of `cell_mcp.py`, not in this file. Those instructions ship with the MCP and reach every
consumer regardless of working directory, including the researcher's sessions. This file only
reaches sessions started from this repo. Do not restate them here; the copy will drift
and the wrong one will be believed.

What belongs here is how to *work in this repo*, which the server cannot know.

## Conventions that keep the tools honest

- **Score, don't filter.** Where a tool cannot know, expose the number and let the
  reader judge. Filtering lineage links on biology rejected half of them with nothing
  to say whether that was right.
- **Percentiles, not fixed cutoffs.** A threshold tuned on compact RUES2 nuclei
  mis-fires on large lobed WGD ones.
- **A held or inferred position must never render like a measured one.** OFF-TRACK and
  HELD frames are labelled as such; that honesty is what the tests pin.

## Running things

- Tests: `.venv/Scripts/python.exe -m pytest tests/ -q` (230 tests).
- The MCP server runs from `.venv-mcp`; the analysis code from `.venv`.
- Bundles live in `data/bundle/<well>/`; candidates in `data/candidates/<well>/`.
- Delivery target for the researcher is `J:\chongchong-workspace\cell-split-counter`
  (robocopy `/E`, never `/MIR`).
