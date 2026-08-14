"""The MCP server object, tool-wiring constants, and rendering budgets.

Every other module in this package imports `server` from here and decorates its
tool functions with `@server.tool()`. Split out of the original single-file
cell_mcp.py so the server/constants layer can be imported without pulling in
every tool's implementation.
"""

import os
from pathlib import Path

from mcp.server import MCPServer

server = MCPServer(
    "cell-microscopy",
    version="0.1.0",
    instructions=(
        "Read-only access to time-lapse microscopy of dividing cells. Start with "
        "list_wells(). To find cells worth looking at in a well of thousands, call "
        "find_candidates() -- it is free, scans the whole well, and ranks what the "
        "data already records (recorded splits by how fragment-like they look, tracks "
        "that stop early). list_tracks() is the raw listing when you already know what "
        "you want. Before spending images on a "
        "track, call get_track_profile() -- it's free (no images) and often shows where "
        "to look: a sparkline of solidity, area, and brightness across the track's "
        "frames, plus how close the cell gets to the frame edge (a clipped nucleus has "
        "understated area and brightness, so check that before spending images). "
        "Solidity (area / convex-hull area) dips as a mask rounds up during mitosis, but "
        "how much it is worth depends on the cell line -- strong on compact RUES2 nuclei, "
        "near-useless on the large lobed nuclei of a WGD line, where it has no headroom. "
        "Then use follow_cells_over_time() to watch "
        "the flagged frames closely and measure() for real units. Expect a track to END "
        "at the moment its cell divides, with the daughters carrying new track_ids -- so "
        "if a cell's filmstrip stops abruptly, the event you want is just past it: use "
        "get_lineage() for the daughter ids, and pass start_frame/end_frame beyond the "
        "track's own lifetime, which follow_cells_over_time renders rather than truncating. The "
        "same thing happens in reverse: a track can just as easily BEGIN mid-division, "
        "so a track that looks already mid-event on its very first frame may need frames "
        "from BEFORE first_frame (via the mother, from get_lineage) to see the lead-up. "
        "When a mask-following crop keeps losing the cell, or the object you care about "
        "was never segmented at all and so has no track_id to ask about, switch to "
        "watch_location_over_time() -- it watches a POSITION over time instead of a mask, and "
        "reports the nearest tracked cell per frame so you can tell what you are seeing. "
        "Two more rules that matter: the interval between frames is NOT constant, so "
        "never compute durations from frame counts -- use measure() or time_ms. It "
        "DRIFTS within a single run, it does not just vary between wells: on the nTSC "
        "well it runs 3.0 min early and 14.5 min late, so frames x interval is off by "
        "~25% depending where in the movie you are, and a duration quoted that way is "
        "wrong in the direction that matters for how long a cell spends in mitosis. "
        "manifest.json's interval_ms carries BOTH a median and a mean (4.9 and 6.2 on "
        "that well) -- say which one you are quoting, and note that only the mean "
        "reproduces duration_hours. And some "
        "track_ids are flagged as merged cells, which must not be measured. Images show "
        "chromatin only (H2B-mCherry), so the shapes are nuclei rather than whole cells. "
        "If a cell looks dim or unusual and you can't tell whether that's the cell itself "
        "or the whole field, call get_neighbourhood_stats() before spending more images on "
        "it -- it's free and separates a cell-autonomous change from bleaching/defocus. "
        "When the user asks to SEE something rather than be told about it -- 'show me', "
        "'let me see', 'send me' -- answer with show_cells_in_browser(), which writes a page of "
        "labelled filmstrips they can open, rather than describing frames in prose. "
        "Record human verdicts with annotate(); it is the only file here a human owns."
    ),
)

BUNDLE = Path(os.environ.get("CELL_BUNDLE_DIR", "data/bundle")).expanduser()

# A hard cap, not a default. Every image costs the model a large amount of
# context, and a filmstrip of 40 frames reliably exhausts it mid-task.
MAX_IMAGES = 12

# The same cap for an HTML page, where the images land on disk and in a human's
# browser instead of in the model's context. Nothing is spent per frame there, and
# the thing a researcher asks for over and over is EVERY frame around the event --
# so the token budget has no business shrinking a page it never pays for.
MAX_IMAGES_PAGE = 60

# Auto-window around a membership transition, in MINUTES of real time.
#
# It used to be +/- a fixed number of FRAMES, and that was wrong in a way that
# inverted a whole census. On BeWo M2 the mitotic figure appears up to ~7 frames
# AFTER the frame where lineage.csv records the mother->daughter link (verified on
# track 802: link ends f771, prometaphase f778, two objects by f782) -- so a window
# that stops near the link renders the lead-up and hides the outcome, and every real
# division scored off it reads as an artifact. Blind scoring on 2026-07-31 put 4/5
# `vanishing_daughter` cases as real mitoses and 0/4 `clean` ones.
#
# Frames made it worse: +10 frames is 49 min on RUES2 (4.9 min/frame) and only 30 min
# on BeWo (3.0 min/frame), so the line whose tracker fails hardest got the SHORTEST
# real-time look. Minutes are the units the biology is in, converted per well from its
# own timestamps.
#
# BEFORE_MIN was 30.0 until 2026-08-13, on the assumption "mitosis runs ~30-60 min
# start to finish" -- true for BeWo, false for TSC_batch2_M13_WGD (whole-genome-
# duplicated line, evidently much slower/more irregular): a real division there had
# prophase (f458) 195 minutes before the tracker's own transition frame (f484,
# confirmed by human review), 6.5x past the old window. A short/late-starting mother
# track is NOT evidence against division on this line -- it can mean the object
# simply wasn't segmented/tracked under any id that early, not that nothing happened.
#
# Widened both, kept AFTER_MIN > BEFORE_MIN (the still-valid part of the original
# reasoning -- the visible OUTCOME reliably lands soon after the transition, on
# every line checked so far) without stretching the default alone to cover the full
# 195-minute WGD outlier -- that would 6x the default window for every well to fix
# one cell line's worst case. Catching THAT case is what the mandatory
# find_prophase_onset + wide watch_location_over_time escalation is for (see the
# AOAI-prelim-verification backlog): defaults handle the common case, the explicit
# escalation tools exist precisely because no fixed default covers every outlier.
#
# These are the DISCOVERY defaults, used by the investigation tools
# (follow_cells_over_time, find_prophase_onset's confirm-on-pixels step) --
# deliberately generous, since missing a real division here is the expensive
# direction of error. The separate, narrower _REPORT_WINDOW_BEFORE/AFTER_MIN in
# tools_output.py govern show_cells_in_browser's default instead: report/figure
# pages are built to show a specific, already-identified event cleanly, not to
# search for one, so they keep the old tight default unless a caller deliberately
# widens a specific page.
_WINDOW_BEFORE_MIN = 120.0
_WINDOW_AFTER_MIN = 180.0

# Frames are sampled at this spacing when the caller does not pin max_images, so a
# strip's time resolution stays the same whether the well runs at 3.0 or 4.9 min per
# frame. Roughly anaphase-scale; below this the extra frames mostly repeat.
_STRIDE_MIN = 6.0

# Filmstrip crops are tiny in absolute pixels -- a 60 um crop is 104 px here, and a
# nucleus inside it is about 21 px across. That 21 px is the microscope's limit, not
# ours (the ND2 is natively 1024x1024, and nothing upstream downsamples), so upscaling
# adds no information. It does add legibility: the previous 2x INTER_NEAREST to 160 px
# turned soft chromatin into hard blocks, and chromatin texture is the entire evidence
# for calling a stage. LANCZOS at 3x is visibly better on the same pixels. Ringing is
# not a concern on diffraction-limited fluorescence, which has no sharp edges to ring.
# Cost is negligible: an image this size is ~130 tokens.
_UPSCALE_TO = 312

# Splits a filmstrip header into "what is true of this strip" and "how this tool
# renders". show_cells_in_browser prints the second half once per PAGE instead of once per
# case: a reviewer reading 14 identical paragraphs reads none of them, and the
# caveats live in that half.
_HDR_SEP = "\n\n"

__all__ = [
    "server", "BUNDLE", "MAX_IMAGES", "MAX_IMAGES_PAGE",
    "_WINDOW_BEFORE_MIN", "_WINDOW_AFTER_MIN", "_STRIDE_MIN", "_UPSCALE_TO", "_HDR_SEP",
]
