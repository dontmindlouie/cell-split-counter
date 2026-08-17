"""Output tools: show_cells_in_browser (writes an HTML filmstrip page) and
annotate (the one write path -- appends to a human-owned CSV).

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import base64
import json
import os
import re

import cv2

from .server import server, MAX_IMAGES_PAGE, _STRIDE_MIN, _HDR_SEP
from .tools_filmstrip import _resolve_family

import cell_mcp_server as _cm

# BUNDLE, _manifest, _filmstrip_frames, and _family_filmstrip_frames below go
# through `_cm.` rather than a direct import -- see the note at the top of
# __init__.py and io.py.

# Deliberately NOT importing server.py's _WINDOW_BEFORE_MIN/_WINDOW_AFTER_MIN --
# those are the DISCOVERY defaults for investigation tools (follow_cells_over_time
# etc.), widened 2026-08-13 to be generous about how far back a real division's
# staging can start. This tool is for showing a specific, already-identified event
# cleanly in a report/figure, not for searching -- so it keeps its own, narrower
# default (the pre-2026-08-13 values) unless a caller deliberately widens one page's
# before_min/after_min. A report page auto-inheriting the wide discovery window
# would default to showing hours of mostly-irrelevant lead-up around the one event
# the reader actually wants to see.
_REPORT_WINDOW_BEFORE_MIN = 30.0
_REPORT_WINDOW_AFTER_MIN = 90.0

# Same reasoning as the window split above, applied to _family_filmstrip_frames's
# auto-fit crop sizing: its default floor (25um) is tuned for the 312px inline-tool
# render sent into a conversation (~5.4x upscale on ACTB), where a tight crop is a
# reasonable cost/context tradeoff. On THIS tool's report page -- both the
# filmstrip thumbnail and the 900px lightbox, since a report page is scanned by
# eye, not token-budgeted -- the same 25um floor read as "massively overzoomed" on
# a multi-candidate page (2026-08-13 spot-check feedback) and, at the 900px
# lightbox specifically, as visibly grainy too (Lanczos amplifying noise at the
# resulting ~15.6x stretch). Applied to BOTH tiers here, not just the lightbox.
# 2026-08-16: raised again, 60->90, to match render.py's own family-auto-fit
# floor (also raised to 90 the same session) -- "still way too zoomed in...
# over multiple iterations" was every earlier fix correcting whether the crop
# COVERED every tracked member, never whether it left room to see neighbours.
_REPORT_MIN_CROP_UM = 90.0

# _UPSCALE_TO (server.py, 312) is sized for MCP tools that return ImageContent
# inline -- follow_cells_over_time, watch_location_over_time, list_nearby_tracks --
# where every pixel is a token in this conversation. This tool writes an HTML file
# to disk and returns only its path, never the pixels themselves, so that
# constraint does not apply here. Reported 2026-08-13 (researcher, ACTB_M2): crops
# pulled into presentation/report figures looked visibly softer than an ND2 opened
# directly in Fiji and screenshotted -- the default 312px render, stretched further
# by the browser to fill the lightbox, was the ceiling. This is still Lanczos
# upscaling of the same native pixels (a 60 um crop is ~104 px natively regardless
# of target size -- see the comment on _UPSCALE_TO), not new detail, but a bigger
# target means less additional stretching happens client-side in the browser on
# top of it, which is where most of the visible softness was coming from.
_FIGURE_UPSCALE_TO = 900

@server.tool()
def show_cells_in_browser(well: str, events: list[dict], note: str = "") -> str:
    """Show the user cells -- writes a page of labelled filmstrips they can open.

    QUICK REF (long descriptions truncate ~1200 chars in, no [...] marker --
    read this block first, it may be all you get):
        events[].kind REQUIRED = "track" (needs track_id) / "family" (needs
        track_ids; add hold_centre_after_member_end=True once a vanished member
        is confirmed gone, else re-centring snaps to survivors) / "point" (needs
        x, y, start_frame, end_frame -- no drift-following yet, re-anchor with
        several point events if the target moves). max_images: leave OFF, <60
        frames renders GAP-FREE; the "showing N of M frames" sampling note is
        ONLY in the return text, not on the page. Returns 2 lines: path, file:// URL.

    CALL THIS WHENEVER THE USER SAYS "show me", "let me see", "send me", "put
    together", "can I look at" -- anything meaning they want to LOOK at cells rather
    than read a description of them. Describing twelve frames in prose when the
    person asked to see them is the wrong answer, and it is the most common way to
    get this wrong: the words "show me" should end in this call.

    Also the right tool when you have finished an investigation and want to hand over
    the evidence -- after answering "what happens to these 12 tracks", write one page
    covering all of them instead of pasting filmstrips into chat one at a time. Each
    cell renders with the exact same crop logic as follow_cells_over_time (colour LUT, scale
    bar, OFF-TRACK handling), so the page matches what you already reviewed.

    Pair each entry with a `label` that says what you concluded and why -- the page
    outlives the conversation, and a bare track id tells the reader nothing about
    what they are looking for.

    Args:
        well: well name from list_wells().
        note: one or two sentences at the top of the page saying WHAT THE READER IS
            being asked to do -- "score each case Y/N/unsure for a real division",
            "pick which of these is the metaphase frame". Write it whenever the page
            is a task rather than a report. A reviewer opening 14 filmstrips a day
            later has the labels but not the question, and asking again costs them
            more than writing it costs you.
        events: one dict per cell to show. Each MUST have a `kind` key naming which
            of the three shapes below it is -- this is a declaration, not inferred
            from whichever other keys happen to be present, so a wrong or missing
            id key fails with "kind='track' but no track_id given" instead of a
            generic "needs track_id/track_ids/x/y" that doesn't say which you meant:
            kind="track": ONE mask, rendered as follow_cells_over_time(track_id=...)
                does it. Requires `track_id`.
            kind="family": a member set, as follow_cells_over_time(track_ids=[...])
                does it -- use this for divisions, so the strip follows the mother
                and then the daughters' midpoint in one row instead of losing them
                at the handoff. Requires `track_ids`. A single id here expands to
                that track plus its recorded daughters, the window defaults to 30
                min before / 90 min after the membership transition
                (`before_min`/`after_min` keys), and crop_um defaults to auto-fit
                rather than 60.
            kind="point": a fixed PLACE, rendered as
                watch_location_over_time(x=..., y=...) does it -- no mask, no
                re-centring, just the raw clicked/reported point. Use this for a
                researcher's raw coordinate before it has been snapped to a track,
                or for anything the segmenter never caught. Requires `x`, `y`,
                `start_frame`, and `end_frame` -- all four, since there is no track
                lifetime to default a window from. crop_um defaults to 90.0, wider
                than kind="track"'s default, because a fixed point has no
                re-centring to save it if the crop is too tight and something
                relevant drifts to the edge.
            start_frame, end_frame (optional with kind="track"/"family", REQUIRED
                with kind="point"): as in follow_cells_over_time -- may fall outside
                the track's own lifetime, e.g. to show a division just past its end.
            label (optional): a short heading, e.g. "2036 -- divides, pro/meta/ana
                309/319/321". Defaults to "track <track_id>".
            max_images (optional): pin the frame count. Leave it OFF by default. A
                page costs no context -- the images go to disk and to a human's
                browser -- so frames here are sampled by time and a window under 60
                frames renders GAP-FREE. Capping it is how a researcher ends up
                looking at every third frame of the event they asked to see.
            centre_frame (optional): centre the window on this frame rather than on
                the membership transition. Use it for hand-picked member sets.
            before_min, after_min, stride_min (optional): window and sampling in
                MINUTES, as in follow_cells_over_time.
            crop_um (optional, default 90.0): crop width in micrometres.
            marker (optional, default False): ring the tracked cell. Leave it OFF
                for ordinary review -- the ring is one more shape in an image whose
                shapes ARE the evidence. Turn it on only for the narrow case of
                pointing out one specific cell among several similar-looking ones
                in a crowded or wide crop, e.g. a page someone else will read with
                no other way to say "this one, not that one". With track_ids, rings
                every member present or none. kind="point" never draws a crosshair
                on this page regardless of `marker` (fixed 2026-08-17 -- it read as
                a detection ring even though it isn't one); the crop is centred on
                the requested point regardless. watch_location_over_time keeps the
                crosshair, since it's a live aid while investigating, not a
                handoff report.
            hold_centre_after_member_end (optional, default False, kind="family"
                only): freeze a member's last position into the crop's mean once
                its span ends, instead of letting the mean jump to whoever
                remains. Turn on only once you've confirmed (via
                watch_location_over_time) the member genuinely vanished and you
                want the framing to hold rather than snap to the survivors.

    Returns TWO lines: the absolute path, then the same file as a file:// URL. Give
    the user BOTH, verbatim, each on its own line -- terminals and chat clients
    linkify a URL but not a Windows path, and which one is clickable depends on their
    client. Do not shorten either, do not describe where the file is instead of naming
    it, and do not make them ask twice.
    Images are embedded as base64, so the
    file is portable on its own -- open it directly, or serve it
    (`python -m http.server`) if file:// is blocked in the browser being used.

    Running inside the VS Code extension: tell the user to open the path in their
    regular desktop browser (Chrome, etc.), NOT VS Code's own built-in preview --
    confirmed working end to end via a real desktop browser (2026-08-16); the
    built-in preview's webview sandbox is a different, untested rendering path for
    a self-contained page this size with inline JS (lightbox navigation, the
    figure-mode toggle).
    """
    if not events:
        raise ValueError("events is empty -- nothing to render.")

    sections = []
    shared: list[str] = []
    lb_data = []
    for i, ev in enumerate(events):
        kind = ev.get("kind")
        if kind not in ("track", "family", "point"):
            raise ValueError(
                f"events[{i}] needs a 'kind' of 'track', 'family', or 'point' "
                f"(got {kind!r}) -- declare the shape, don't rely on it being "
                f"guessed from whichever other keys are present: {ev!r}")
        if kind == "point":
            if "x" not in ev or "y" not in ev:
                raise ValueError(
                    f"events[{i}] has kind='point' but no 'x'/'y' given: {ev!r}")
            if ev.get("start_frame") is None or ev.get("end_frame") is None:
                raise ValueError(
                    f"events[{i}] has kind='point' and needs start_frame and "
                    f"end_frame too -- there is no track lifetime to default a "
                    f"window from: {ev!r}")
            mx = ev.get("max_images")
            common = dict(
                max_images=None if mx is None else int(mx),
                crop_um=float(ev.get("crop_um", 90.0)),
                color=True, scale_bar=True,
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=MAX_IMAGES_PAGE,
                # Always off on this page, unconditionally -- not a per-event
                # option. This is a report a researcher opens cold, and the
                # crosshair reads as a detection ring even though it isn't one
                # (2026-08-16 field feedback: "how did the marker get added back?").
                # watch_location_over_time/follow_cells_over_time keep it, since
                # there it's a live aid for figuring out which nucleus a crop is
                # tracking, not a handoff artifact.
                crosshair=False,
            )
            args = (well, int(ev["start_frame"]), int(ev["end_frame"]),
                    float(ev["x"]), float(ev["y"]), None)
            header, thumb_images = _cm._fixed_point_frames(*args, **common)
            _, lb_images = _cm._fixed_point_frames(
                *args, **common, upscale_to=_FIGURE_UPSCALE_TO)
            label = ev.get("label") or f"({ev['x']:.0f}, {ev['y']:.0f})"
        elif kind == "family":
            if "track_ids" not in ev:
                raise ValueError(
                    f"events[{i}] has kind='family' but no 'track_ids' given: {ev!r}")
            members, added = _resolve_family(well, [int(t) for t in ev["track_ids"]])
            crop = ev.get("crop_um")
            mx = ev.get("max_images")
            common = dict(
                start_frame=ev.get("start_frame"), end_frame=ev.get("end_frame"),
                max_images=None if mx is None else int(mx),
                crop_um=None if crop is None else float(crop),
                color=True, scale_bar=True, marker=bool(ev.get("marker", False)),
                before_min=float(ev.get("before_min", _REPORT_WINDOW_BEFORE_MIN)),
                after_min=float(ev.get("after_min", _REPORT_WINDOW_AFTER_MIN)),
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=MAX_IMAGES_PAGE, added=added,
                centre_frame=(None if ev.get("centre_frame") is None
                              else int(ev["centre_frame"])),
                hold_centre_after_member_end=bool(
                    ev.get("hold_centre_after_member_end", False)),
            )
            # Two separate renders, not one shared image at the bigger size: the
            # filmstrip's `image-rendering: pixelated` CSS is only correct when the
            # browser is upscaling a small source (it deliberately avoids blurring
            # tiny native crops) -- fed a 900px source and displayed at 260px, that
            # same rule would nearest-neighbor DOWNSCALE it instead, which is a
            # worse artifact (aliasing/moire) than the softness this was fixing.
            header, thumb_images = _cm._family_filmstrip_frames(
                well, members, **common, min_crop_um=_REPORT_MIN_CROP_UM)
            _, lb_images = _cm._family_filmstrip_frames(
                well, members, **common, upscale_to=_FIGURE_UPSCALE_TO,
                min_crop_um=_REPORT_MIN_CROP_UM)
            label = ev.get("label") or f"tracks {', '.join(str(t) for t in members)}"
        else:  # kind == "track"
            if "track_id" not in ev:
                raise ValueError(
                    f"events[{i}] has kind='track' but no 'track_id' given: {ev!r}")
            track_id = int(ev["track_id"])
            mx = ev.get("max_images")
            common = dict(
                start_frame=ev.get("start_frame"), end_frame=ev.get("end_frame"),
                max_images=None if mx is None else int(mx),
                crop_um=float(ev.get("crop_um", 90.0)),
                color=True, scale_bar=True, marker=bool(ev.get("marker", False)),
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=MAX_IMAGES_PAGE,
            )
            header, thumb_images = _cm._filmstrip_frames(well, track_id, **common)
            _, lb_images = _cm._filmstrip_frames(
                well, track_id, **common, upscale_to=_FIGURE_UPSCALE_TO)
            label = ev.get("label") or f"track {track_id}"

        def _b64_all(imgs):
            out = []
            for img in imgs:
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
            return out

        b64_list = _b64_all(thumb_images)      # filmstrip thumbnails -- small, pixelated-upscale OK
        lb_b64_list = _b64_all(lb_images)      # lightbox only -- large, smooth-scaled
        tiles = [
            f'<img src="data:image/png;base64,{b64}" loading="lazy" '
            f'onclick="openLightbox({i},{j})">'
            for j, b64 in enumerate(b64_list)
        ]
        # Split the per-case facts from the standing how-this-renders text and collect
        # the latter for ONE printing at the top. Repeating it under every case is what
        # made a reviewer stop reading it, and the caveats live in that half.
        spec, _, gen = header.partition(_HDR_SEP)
        if gen and gen not in shared:
            shared.append(gen)
        # The "showing N of M frames..." sampling note is already in `spec` below,
        # but as one clause inside a long grey caption paragraph -- exactly the text
        # a 2026-08-16 field session needed and repeatedly missed ("did you even add
        # frames, why does it look the same" across 4 events on a >60-frame window
        # that got silently thinned). Pull it out as its own small badge next to the
        # title too, where it can't be mistaken for part of the general caption.
        # Lookahead stops at the real sentence-ending period/comma, not a decimal
        # point inside the note itself (e.g. "~2.0 min apart").
        m = re.search(r"showing .*?(?=, (?:fixed at|anchored on)|\.\s*Crop)", spec)
        badge_html = f'<span class=samplebadge>{m.group(0)}</span>' if m else ""
        sections.append(
            f"<section id=sec{i}><h2>{i + 1}. {label} {badge_html}</h2>"
            f"<p class=hdr>{spec}</p>"
            f"<div class=filmstrip>{''.join(tiles)}</div></section>"
        )
        lb_data.append({"label": label, "images": lb_b64_list})

    lb_json = json.dumps(lb_data)
    # Collapsed by default: it is reference, not the task. Open once, then get out of
    # the way of the 14 cases the page actually exists for.
    note_html = f"<p class=task>{note}</p>" if note else ""
    shared_html = "".join(
        f"<details class=howto><summary>How to read these strips</summary>"
        f"<p>{g}</p></details>" for g in shared)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{well} browser</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 2rem; }}
h1 {{ font-weight: 400; }}
section {{ margin-bottom: 2.5rem; border-top: 1px solid #333; padding-top: 1rem; }}
h2 {{ font-size: 1.1rem; font-weight: 600; }}
span.samplebadge {{ font-size: 0.72rem; font-weight: 400; color: #ffb454;
  background: #2a2010; border: 1px solid #5a4620; border-radius: 4px;
  padding: 0.15rem 0.5rem; margin-left: 0.5rem; vertical-align: middle; }}
p.hdr {{ color: #999; font-size: 0.85rem; max-width: 60rem; }}
p.task {{ color: #eee; font-size: 1rem; max-width: 60rem; background: #1d2a35; border-left: 3px solid #7aa7d0; padding: 0.7rem 1rem; }}
details.howto {{ color: #888; font-size: 0.82rem; max-width: 60rem; margin-bottom: 1rem; }}
details.howto summary {{ cursor: pointer; color: #7aa7d0; }}
html {{ scroll-behavior: smooth; }}
section {{ scroll-margin-top: 3.2rem; }}
nav.track-nav {{ position: sticky; top: 0; z-index: 10; background: #111;
  border-bottom: 1px solid #333; padding: 0.6rem 0; margin-bottom: 0.5rem; }}
div.view-controls {{ display: flex; align-items: center; gap: 0.9rem;
  padding: 0 0 0.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid #222;
  font-size: 0.78rem; color: #aaa; }}
div.view-controls label {{ display: flex; align-items: center; gap: 0.3rem; white-space: nowrap; cursor: pointer; }}
div.filmstrip {{ display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 4px; padding-bottom: 4px; }}
div.filmstrip img {{ flex: 0 0 auto; image-rendering: pixelated; max-height: 260px; border: 1px solid #333; cursor: zoom-in; }}
/* The frame/time/coord label (top) and scale bar (bottom) are burned into the PNG
   pixels, not drawn by CSS -- so "hiding" them for a clean figure means clipping
   those strips off the rendered image rather than toggling a layer. Both bands are
   now a single text line each (see _stamp_tile/_scale_bar in render.py), so top and
   bottom take the same inset. Filmstrip and lightbox render the identical base64
   PNG, just at different display sizes -- clip-path % is relative to the image's
   own box either way, so they must use the same values or one under/over-trims. */
body.figure-mode div.filmstrip img,
body.figure-mode .lightbox img {{ clip-path: inset(7% 0 7% 0); }}
.lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); align-items: center; justify-content: center; z-index: 20; }}
.lightbox.open {{ display: flex; }}
.lightbox img {{ max-width: 92vw; max-height: 88vh; }}
.lightbox-caption {{ position: absolute; bottom: 22px; left: 50%; transform: translateX(-50%); color: #ccc; font-size: 12.5px; }}
.lightbox-close {{ position: absolute; top: 14px; right: 20px; color: #fff; font-size: 28px; line-height: 1; cursor: pointer; background: none; border: none; padding: 6px 10px; }}
.lightbox-nav {{ position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.12); color: #fff; border: none; font-size: 28px; line-height: 1; width: 52px; height: 64px; cursor: pointer; border-radius: 8px; }}
.lightbox-nav:hover {{ background: rgba(255,255,255,0.24); }}
.lightbox-nav:disabled {{ opacity: 0.25; cursor: default; }}
.lightbox-nav.prev {{ left: 16px; }}
.lightbox-nav.next {{ right: 16px; }}
</style></head><body>
<h1>{well}</h1>
<nav class="track-nav">
<div class="view-controls">
  <label><input type="checkbox" id="figureCtl" onchange="document.body.classList.toggle('figure-mode', this.checked)"> hide labels + scale bar (figures)</label>
</div>
</nav>
{note_html}
{shared_html}
{''.join(sections)}
<div class="lightbox" id="lightbox">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <button class="lightbox-nav prev" id="lb-prev" onclick="stepLightbox(-1)">&lsaquo;</button>
  <img id="lb-img" alt="">
  <div class="lightbox-caption" id="lb-caption"></div>
  <button class="lightbox-nav next" id="lb-next" onclick="stepLightbox(1)">&rsaquo;</button>
</div>
<script>
var LB_DATA = {lb_json};
var lbSection = 0, lbIdx = 0;
function showLbFrame() {{
  var imgs = LB_DATA[lbSection].images;
  document.getElementById('lb-img').src = 'data:image/png;base64,' + imgs[lbIdx];
  document.getElementById('lb-caption').textContent =
    LB_DATA[lbSection].label + ' -- frame ' + (lbIdx + 1) + ' / ' + imgs.length;
  document.getElementById('lb-prev').disabled = lbIdx === 0;
  document.getElementById('lb-next').disabled = lbIdx === imgs.length - 1;
}}
function openLightbox(section, idx) {{
  lbSection = section; lbIdx = idx;
  showLbFrame();
  document.getElementById('lightbox').classList.add('open');
}}
function closeLightbox() {{ document.getElementById('lightbox').classList.remove('open'); }}
function stepLightbox(delta) {{
  var imgs = LB_DATA[lbSection].images;
  var next = lbIdx + delta;
  if (next < 0 || next >= imgs.length) return;
  lbIdx = next;
  showLbFrame();
}}
document.getElementById('lightbox').addEventListener('click', function(e) {{
  if (e.target.id === 'lightbox') closeLightbox();
}});
document.addEventListener('keydown', function(e) {{
  if (!document.getElementById('lightbox').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') stepLightbox(-1);
  if (e.key === 'ArrowRight') stepLightbox(1);
}});
</script>
</body></html>"""

    out_dir = _cm.BUNDLE / well / "browsers"
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    out_path = out_dir / f"browser_{time.strftime('%Y%m%d_%H%M%S')}.html"
    out_path.write_text(html, encoding="utf-8")
    # ABSOLUTE, always, and as a file:// URL too. _cm.BUNDLE is usually relative, so this
    # used to return "data\bundle\<well>\browsers\browser_....html" -- openable only by
    # someone already sitting in the repo root. The reader is a person in a terminal or
    # a chat window, and they said so twice: first that they had to "hunt down the html
    # link", then that the absolute path still was not clickable and they were pasting
    # it into Chrome by hand. Terminals and chat clients linkify a URL, not a Windows
    # path, so give them both and let whichever one their client understands win.
    full = out_path.resolve()
    return f"{full}\n{full.as_uri()}"


_ANNOTATION_FIELDS = [
    "timestamp", "annotator", "well", "cell_line", "condition", "track_id",
    "event_id", "outcome_class",
    "condensation_frame", "metaphase_frame", "anaphase_frame", "exit_frame",
    "parent_id", "daughter_ids", "notes",
]


@server.tool()
def annotate(
    well: str, track_id: int, outcome_class: str,
    condensation_frame: int | None = None,
    metaphase_frame: int | None = None,
    anaphase_frame: int | None = None,
    exit_frame: int | None = None,
    parent_id: int | None = None,
    daughter_ids: list[int] | None = None,
    event_id: str | None = None,
    notes: str | None = None,
    annotator: str | None = None,
) -> str:
    """Record a human-verified verdict for a cell. Appends a new row -- never edits
    or overwrites a previous one, so nothing is ever silently lost or replaced.

    This is the actual payoff of everything else here: browsing produces nothing
    durable on its own, this is what turns a review session into a labeled dataset.
    Written to a SEPARATE file (<bundle>/<well>/annotations.csv), never mixed into
    events.csv -- that file is machine-generated, gets overwritten on every pipeline
    re-run, and only has a row for events the detector already found, so writing
    human verdicts onto it would silently cap what can ever be recorded at the
    detector's own recall. The most valuable annotation is often one where nothing
    in events.csv corresponds to it at all.

    Only call this after you (or the person you're working with) actually looked at
    the pixels via follow_cells_over_time -- this is a verdict, not a guess from get_track_profile
    numbers alone.

    Args:
        well: well name from list_wells().
        track_id: the cell being annotated.
        outcome_class: e.g. "divides", "dies", "neither" -- free text, but stay
            consistent within a well so later rollups can group on it.
        condensation_frame, metaphase_frame, anaphase_frame, exit_frame: the four
            stage marks (chromatin condensation onset -> metaphase alignment ->
            anaphase separation -> mitotic exit), as frame numbers. Leave any that
            don't apply or weren't determined as None -- durations between whichever
            marks ARE set can still be computed later from manifest.frame_timestamps_ms.
        parent_id: the track this cell was born from, if relevant and known (may
            differ from lineage.csv's own record, if you determined it was wrong).
        daughter_ids: track_ids of the cells born from this one, if it divided.
        event_id: the events.csv row this corresponds to, if any (e.g. "peak_frame"
            value or similar identifier) -- leave None if you found this yourself and
            nothing in events.csv flagged it. Never invent one.
        notes: free text for anything the typed fields don't capture.
        annotator: who determined this. Defaults to the CELL_MCP_ANNOTATOR
            environment variable if set, else "unspecified" -- set that env var once
            rather than passing this every call.
    """
    from datetime import datetime, timezone
    import csv as _csv

    m = _cm._manifest(well)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotator": annotator or os.environ.get("CELL_MCP_ANNOTATOR", "unspecified"),
        "well": well,
        "cell_line": m.get("cell_line") or "",
        "condition": m.get("condition") or "",
        "track_id": track_id,
        "event_id": event_id or "",
        "outcome_class": outcome_class,
        "condensation_frame": condensation_frame if condensation_frame is not None else "",
        "metaphase_frame": metaphase_frame if metaphase_frame is not None else "",
        "anaphase_frame": anaphase_frame if anaphase_frame is not None else "",
        "exit_frame": exit_frame if exit_frame is not None else "",
        "parent_id": parent_id if parent_id is not None else "",
        "daughter_ids": " ".join(str(d) for d in daughter_ids) if daughter_ids else "",
        "notes": notes or "",
    }

    out_path = _cm.BUNDLE / well / "annotations.csv"
    is_new = not out_path.is_file()
    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=_ANNOTATION_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)

    n = sum(1 for _ in open(out_path, encoding="utf-8")) - 1
    return f"Recorded. {out_path} now has {n} annotation(s) for {well}."



__all__ = ["show_cells_in_browser", "_ANNOTATION_FIELDS", "annotate"]
