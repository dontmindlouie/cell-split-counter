"""Output tools: show_cells_in_browser (writes an HTML filmstrip page) and
annotate (the one write path -- appends to a human-owned CSV).

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import base64
import json
import os
import re

import cv2
import numpy as np

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
# Layered pages (channels=...) carry every lightbox frame once PER CHANNEL, so at 900 px
# a 3-channel page ran ~13 MB an event; at 360 the browser's smooth scaling still fills
# the view (a 200 um context crop is ~230 px native, so nothing real is lost).
_LAYER_LB_SIZE = 360
# Filmstrip tiles of a layered page, at the 260 px they are displayed at. 180 px
# saved space but the canvas stretched it ~1.4x, and the filmstrip (with the
# marker) is where researchers actually review -- 2026-10-27 field observation:
# the lightbox goes unused. Still under the 312 px default, which stored
# upscaled noise once per channel.
_LAYER_THUMB_SIZE = 260
# Fewer events per part on layered pages, which also carry 4-5 h windows: a part
# stays a few tens of MB on an external drive.
_LAYER_EVENTS_PER_PAGE = 8
# Noisy far-red channels dominate a layered page's size; at 80 the grain is still
# shot noise, not JPEG blocks, and a part is about a third smaller than at 90.
_LAYER_JPEG_QUALITY = 80

# JPEG, not PNG, for both tiers. Reported 2026-10-17 (researcher, ZO1_STB_P2): a
# 60-event page came out at 2.3 GB and would not open; 10 events was still 417 MB.
# Noisy fluorescence, Lanczos-upscaled, barely compresses losslessly -- measured on a
# real ACTB_M2 page, the 900px lightbox averaged 961 KB as PNG vs 134 KB at q85
# (7.2x) and the 312px thumbnail 156 KB vs 28 KB (5.6x), with no difference visible
# by eye. The lightbox gets the higher quality since it is what gets pulled into
# figures.
_THUMB_JPEG_QUALITY = 85
_LIGHTBOX_JPEG_QUALITY = 90

# Events per page. Same report: one call put every event on one page, and the
# calling session then paged by hand ("page 1 of 6", ten events per call). Pages
# are now split here instead, and each page is rendered, written and released
# before the next is started, so the server's memory is bounded by one page rather
# than growing with the event count (10 events at PNG peaked at ~2.1 GB).
_EVENTS_PER_PAGE = 12

@server.tool()
def show_cells_in_browser(well: str, events: list[dict], note: str = "",
                          review_form: dict | None = None,
                          channels: list[str] | None = None,
                          channels_off: list[str] | None = None,
                          channel_toggles: bool = False,
                          events_per_part: int | None = None,
                          max_frames: int | None = None) -> str:
    """Show the user cells -- writes a page of labelled filmstrips they can open.

    QUICK REF (long descriptions truncate ~1200 chars in, no [...] marker --
    read this block first, it may be all you get):
        events[].kind REQUIRED = "track" (needs track_id) / "family" (needs
        track_ids; add hold_centre_after_member_end=True once a vanished member
        is confirmed gone, else re-centring snaps to survivors) / "point" (needs
        start_frame, end_frame, plus EITHER x+y for a fixed spot OR waypoints=
        [(frame,x,y), ...] for a drifting one -- interpolates the crop centre
        between checkpoints instead of one fixed position; never draws a
        crosshair on this page). max_images: leave OFF, <60 frames renders
        GAP-FREE; a wider window's sampling ratio ("showing N of M frames") is
        also on the page now, as a small badge next to each title. Returns 2
        lines: path, file:// URL.

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
            kind="point": a PLACE, rendered as watch_location_over_time(x=...,
                y=...) does it -- no mask, no ring, just the raw clicked/reported
                point. Use this for a researcher's raw coordinate before it has
                been snapped to a track, or for anything the segmenter never
                caught. Requires `start_frame` and `end_frame`, plus EITHER `x`
                and `y` (a single fixed spot) OR `waypoints` (a drifting one):
                `waypoints`: list of `(frame, x, y)` checkpoints, at least 2,
                    sorted or not. The crop centre is linearly interpolated
                    between the two checkpoints bracketing each rendered frame
                    (clamped to the first/last checkpoint outside their span).
                    For an untracked object that moves -- debris, a fragment --
                    instead of a several-call manual workaround (probe a few
                    checkpoint frames by hand, split into consecutive point
                    events each re-anchored at the next position). Get the
                    checkpoint positions from list_nearby_tracks/measure the
                    same way that manual workaround did.
                crop_um defaults to 90.0, wider than kind="track"'s default,
                because a fixed point has no re-centring to save it if the crop
                is too tight and something relevant drifts to the edge.
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
            context_um (optional): crop width in micrometres for the LIGHTBOX only,
                e.g. 220 to show the neighbours around a 70 um filmstrip -- whether
                the "daughters" are really daughters is often decided by what is
                around them. Default: the same crop as the filmstrip.
            hold_centre_after_member_end (optional, default False, kind="family"
                only): freeze a member's last position into the crop's mean once
                its span ends, instead of letting the mean jump to whoever
                remains. Turn on only once you've confirmed (via
                watch_location_over_time) the member genuinely vanished and you
                want the framing to hold rather than snap to the survivors.

        review_form (optional): marking controls YOU choose for this task, shown
            under every event. Every page already has a notes box per event and an
            "Export CSV" button whether or not this is passed; this adds the
            structured fields. Pick them from what the reader is being asked to
            judge -- there is no fixed list. Keys (all optional):
                checks: list of tick-box labels, any number may be ticked, e.g.
                    ["not a division", "lagging", "bridge", "multipolar",
                     "micronucleus"].
                choices: {name: [options]} -- pick-one groups, e.g.
                    {"HLA-G": ["+", "-", "unsure"], "nuclei after": ["mono", "bi"]}.
                numbers: list of number-box labels, e.g. ["metaphase frame"].
                text: list of short free-text box labels.
            Marks are kept in the reader's browser (shared across all parts of a
            multi-page set) until they press Export, which downloads ONE CSV for
            the whole set: one row per event with its id/frames/position, every
            field, and notes. Ask the user for that file (usually their Downloads
            folder, named review_<well>_<stamp>.csv) and read it -- then use
            annotate() for verdicts worth keeping in the bundle. An event may carry
            an `id` key (any string, e.g. your own candidate number); it becomes the
            CSV's event_id column so rows join back to your list.

        channels (optional, multi-channel wells): channel names from
            list_channels(), e.g. ["AF594", "AF488", "AF647"]. The page then carries
            each as its own grey layer and composes them in the browser, with an
            on/off box per channel and (on by default) contrast stretched to each
            event's crop instead of the whole field -- a dim nucleus next to a bright
            one stays readable, and a marker that washes over the nuclei can be
            switched off. Without it the page shows the bundle's fixed composite.
        channels_off (optional): of those, the ones left out, e.g. ["AF647"] when a
            far-red marker would wash over H2B nuclei.
        channel_toggles (default False): False combines the shown channels here,
            each stretched to the event's own crop, into ONE colour image per
            frame -- the clean look at the data cost of the plain composite, but
            fixed. True embeds each channel as its own grey layer (about 3x the
            data with three channels) and the page gets an on/off box per channel,
            with channels_off starting switched off rather than left out.
        events_per_part (optional): events per page part (default 8 with channels,
            12 without). Fewer events per part is what pays for longer windows: a
            part's size is events x frames, so 4 events with twice the frames load
            like 8 with the default.
        max_frames (optional, default 60): the per-event frame ceiling. A window
            with more frames than this is sampled (with gaps, said on the badge);
            raise it with events_per_part lowered to keep long windows gap-free,
            e.g. 9 h after a 10-min movie's metaphase is ~55 frames, on a 3-min
            movie ~180.

    Returns TWO lines: the absolute path, then the same file as a file:// URL. Give
    the user BOTH, verbatim, each on its own line -- terminals and chat clients
    linkify a URL but not a Windows path, and which one is clickable depends on their
    client. Do not shorten either, do not describe where the file is instead of naming
    it, and do not make them ask twice.
    More than 12 events are split automatically into linked pages of up to 12
    (..._part1of5.html, ...), each with previous/next links. The return is then a
    one-line summary followed by a path + URL pair per page; giving the user part 1
    is enough, but pass all of them through. Pass the whole event list in ONE call --
    do not split it into several calls yourself.
    Keyboard (shown on the page): j/k move between events (crossing into the next
    or previous part), Enter opens the lightbox, 1-9 pick an option of the FIRST
    review_form choices group and move on, n jumps to notes. In the lightbox:
    left/right step frames, space plays, up/down change event, 1-9 mark and move on.
    Put the main verdict first in `choices` so those keys mark it.
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
    # Checked up front for every event, before any rendering: pages are now written
    # as they fill, so a bad event late in the list would otherwise fail the call
    # after earlier pages were already on disk.
    for i, ev in enumerate(events):
        kind = ev.get("kind")
        if kind not in ("track", "family", "point"):
            raise ValueError(
                f"events[{i}] needs a 'kind' of 'track', 'family', or 'point' "
                f"(got {kind!r}) -- declare the shape, don't rely on it being "
                f"guessed from whichever other keys are present: {ev!r}")
        if kind == "point":
            if ev.get("waypoints") is None and ("x" not in ev or "y" not in ev):
                raise ValueError(
                    f"events[{i}] has kind='point' but no 'x'/'y' (or 'waypoints') "
                    f"given: {ev!r}")
            if ev.get("start_frame") is None or ev.get("end_frame") is None:
                raise ValueError(
                    f"events[{i}] has kind='point' and needs start_frame and "
                    f"end_frame too -- there is no track lifetime to default a "
                    f"window from: {ev!r}")
        elif kind == "family" and "track_ids" not in ev:
            raise ValueError(
                f"events[{i}] has kind='family' but no 'track_ids' given: {ev!r}")
        elif kind == "track" and "track_id" not in ev:
            raise ValueError(
                f"events[{i}] has kind='track' but no 'track_id' given: {ev!r}")
    form = _normalize_review_form(review_form)
    layers = []
    if channels:
        known = {c["name"]: c for c in _cm._manifest(well).get("channels") or []}
        missing = [c for c in channels if c not in known]
        if missing:
            raise ValueError(f"{well} has no channel(s) {missing}; it has {sorted(known)}")
        off = set(channels_off or [])
        if off - set(channels):
            raise ValueError(f"channels_off {sorted(off - set(channels))} not in channels {channels}")
        layers = [{"name": c, "rgb": known[c].get("color_rgb") or [255, 255, 255], "on": c not in off}
                  for c in channels]

    def _b64_all(imgs, quality):
        out = []
        for img in imgs:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        return out

    import time
    out_dir = _cm.BUNDLE / well / "browsers"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    # Channel switching that this page does NOT carry is still shown, greyed out, so a
    # reader learns it exists and can ask for it: the combined channels, or for a plain
    # multi-channel page every channel the bundle has.
    fixed_layers = [] if channel_toggles else (layers or [
        {"name": c["name"], "rgb": c.get("color_rgb") or [255, 255, 255], "on": True}
        for c in _cm._manifest(well).get("channels") or [] if c.get("color_rgb")])
    per_page = int((_LAYER_EVENTS_PER_PAGE if layers else _EVENTS_PER_PAGE)
                   if events_per_part is None else events_per_part)
    cap = int(MAX_IMAGES_PAGE if max_frames is None else max_frames)
    if per_page < 1 or cap < 2:
        raise ValueError("events_per_part must be >= 1 and max_frames >= 2")
    n_pages = -(-len(events) // per_page)

    def _page_name(k):
        return (f"browser_{stamp}.html" if n_pages == 1
                else f"browser_{stamp}_part{k}of{n_pages}.html")

    # One row per event for the CSV export. Every part embeds the WHOLE set's list so
    # Export on any part writes every event, marked or not (blank = not reviewed).
    meta = [_event_meta(i, ev) for i, ev in enumerate(events)]
    review = {"key": f"cell-review:{well}:{stamp}", "file": f"review_{well}_{stamp}.csv",
              "form": form, "events": meta}

    written = []
    sections = []
    shared: list[str] = []
    lb_data = []
    def _render(ev, color, lb_size=_FIGURE_UPSCALE_TO, thumb_size=None):
        """(header, label, thumb images, lightbox images) for one event."""
        ts = {} if thumb_size is None else {"upscale_to": thumb_size}
        kind = ev["kind"]
        if kind == "point":
            waypoints = ev.get("waypoints")
            mx = ev.get("max_images")
            common = dict(
                max_images=None if mx is None else int(mx),
                crop_um=float(ev.get("crop_um", 90.0)),
                color=color, scale_bar=True,
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=cap,
                # Always off on this page, unconditionally -- not a per-event
                # option. This is a report a researcher opens cold, and the
                # crosshair reads as a detection ring even though it isn't one
                # (2026-08-16 field feedback: "how did the marker get added back?").
                # watch_location_over_time/follow_cells_over_time keep it, since
                # there it's a live aid for figuring out which nucleus a crop is
                # tracking, not a handoff artifact.
                crosshair=False,
                waypoints=([(int(f), float(wx), float(wy)) for f, wx, wy in waypoints]
                           if waypoints is not None else None),
            )
            args = (well, int(ev["start_frame"]), int(ev["end_frame"]),
                    None if waypoints is not None else float(ev["x"]),
                    None if waypoints is not None else float(ev["y"]), None)
            header, thumb_images = _cm._fixed_point_frames(*args, **common, **ts)
            _, lb_images = _cm._fixed_point_frames(
                *args, **_lightbox_args(common, ev), upscale_to=lb_size)
            label = ev.get("label") or (
                f"drift ({len(waypoints)} waypoints)" if waypoints is not None
                else f"({ev['x']:.0f}, {ev['y']:.0f})")
        elif kind == "family":
            members, added = _resolve_family(well, [int(t) for t in ev["track_ids"]])
            crop = ev.get("crop_um")
            mx = ev.get("max_images")
            common = dict(
                start_frame=ev.get("start_frame"), end_frame=ev.get("end_frame"),
                max_images=None if mx is None else int(mx),
                crop_um=None if crop is None else float(crop),
                color=color, scale_bar=True, marker=bool(ev.get("marker", False)),
                before_min=float(ev.get("before_min", _REPORT_WINDOW_BEFORE_MIN)),
                after_min=float(ev.get("after_min", _REPORT_WINDOW_AFTER_MIN)),
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=cap, added=added,
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
                well, members, **common, min_crop_um=_REPORT_MIN_CROP_UM, **ts)
            _, lb_images = _cm._family_filmstrip_frames(
                well, members, **_lightbox_args(common, ev), upscale_to=lb_size,
                min_crop_um=_REPORT_MIN_CROP_UM)
            label = ev.get("label") or f"tracks {', '.join(str(t) for t in members)}"
        else:  # kind == "track"
            track_id = int(ev["track_id"])
            mx = ev.get("max_images")
            common = dict(
                start_frame=ev.get("start_frame"), end_frame=ev.get("end_frame"),
                max_images=None if mx is None else int(mx),
                crop_um=float(ev.get("crop_um", 90.0)),
                color=color, scale_bar=True, marker=bool(ev.get("marker", False)),
                stride_min=float(ev.get("stride_min", _STRIDE_MIN)),
                cap=cap,
            )
            header, thumb_images = _cm._filmstrip_frames(well, track_id, **common, **ts)
            _, lb_images = _cm._filmstrip_frames(
                well, track_id, **_lightbox_args(common, ev), upscale_to=lb_size)
            label = ev.get("label") or f"track {track_id}"
        return header, label, thumb_images, lb_images

    def _render_layer(ev, name):
        """_render for one channel, uncoloured and bare, with its overlays apart:
        (header, label, thumbs, lightboxes, thumb overlays, lightbox overlays)."""
        tok, ovl = _cm._LAYER.set(name), _cm._OVERLAYS.set([])
        try:
            header, label, t_img, l_img = _render(ev, color=False, lb_size=_LAYER_LB_SIZE,
                                                  thumb_size=_LAYER_THUMB_SIZE)
            drawn = _cm._OVERLAYS.get()
        finally:
            _cm._OVERLAYS.reset(ovl)
            _cm._LAYER.reset(tok)
        assert len(drawn) == len(t_img) + len(l_img), "one overlay per stamped tile"
        return header, label, t_img, l_img, drawn[:len(t_img)], drawn[len(t_img):]

    for i, ev in enumerate(events):
        # Index into THIS page's LB_DATA, not the global event index.
        lb_i = len(lb_data)
        if layers and not channel_toggles:
            # Same per-channel renders, combined here: one colour JPEG per frame.
            shown = [ly for ly in layers if ly["on"]]
            grey_t, grey_l = {}, {}
            for ly in shown:
                header, label, t_img, l_img, t_ovl, l_ovl = _render_layer(ev, ly["name"])
                grey_t[ly["name"]] = [_grey(x) for x in t_img]
                grey_l[ly["name"]] = [_grey(x) for x in l_img]
                del t_img, l_img
            st = {n: _stretch_of(v) for n, v in grey_t.items()}
            b64_list = _b64_all(_overlay(_combine(grey_t, shown, st), t_ovl), _THUMB_JPEG_QUALITY)
            lb_b64_list = _b64_all(_overlay(_combine(grey_l, shown, st), l_ovl), _LIGHTBOX_JPEG_QUALITY)
            del grey_t, grey_l
            tiles = [
                f'<img src="data:image/jpeg;base64,{b64}" loading="lazy" '
                f'onclick="openLightbox({lb_i},{j})">'
                for j, b64 in enumerate(b64_list)
            ]
            lb_entry = {"label": label, "images": lb_b64_list}
        elif layers:
            # One grey JPEG per channel per frame, and the page tints, toggles and
            # stretches them (see compose in the page script). The overlays (label, bar,
            # marker) ride along as one PNG per frame -- lossless, so "nothing drawn"
            # stays exactly black -- and are laid on after the tint.
            thumbs, lbs = {}, {}
            for ly in layers:
                header, label, t_img, l_img, t_ovl, l_ovl = _render_layer(ev, ly["name"])
                thumbs[ly["name"]] = _b64_all([_grey(x) for x in t_img], _LAYER_JPEG_QUALITY)
                lbs[ly["name"]] = _b64_all([_grey(x) for x in l_img], _LAYER_JPEG_QUALITY)
                del t_img, l_img
            n_tiles = len(thumbs[layers[0]["name"]])
            tiles = [f'<canvas data-lb={lb_i} data-j={j} onclick="openLightbox({lb_i},{j})"></canvas>'
                     for j in range(n_tiles)]
            lb_entry = {"label": label, "thumbs": thumbs, "layers": lbs,
                        "thumb_ovl": _png_b64_all(t_ovl), "layer_ovl": _png_b64_all(l_ovl)}
        else:
            header, label, thumb_images, lb_images = _render(ev, color=True)
            b64_list = _b64_all(thumb_images, _THUMB_JPEG_QUALITY)       # filmstrip thumbnails -- small, pixelated-upscale OK
            lb_b64_list = _b64_all(lb_images, _LIGHTBOX_JPEG_QUALITY)    # lightbox only -- large, smooth-scaled
            del thumb_images, lb_images
            tiles = [
                f'<img src="data:image/jpeg;base64,{b64}" loading="lazy" '
                f'onclick="openLightbox({lb_i},{j})">'
                for j, b64 in enumerate(b64_list)
            ]
            lb_entry = {"label": label, "images": lb_b64_list}
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
        #
        # Anchored to the two literal formats _pick_frames emits ("showing all N
        # frames" / "showing N of M frames") -- 2026-08-17 bug found via an actual
        # browser click-test: a bare "showing " match grabbed the WRONG occurrence
        # when a family render's own HELD-frames WARNING (which _family_filmstrip_
        # frames inserts BEFORE the real sampling sentence, and which itself
        # contains the word "showing", e.g. "a frozen crop... showing whatever
        # happens to sit there") preceded it, swallowing the entire warning
        # paragraph into what was supposed to be a short badge.
        m = re.search(r"showing (?:all \d+|\d+ of \d+) frames.*?"
                       r"(?=, (?:fixed at|anchored on|interpolated across)|\.\s*Crop)", spec)
        badge_html = f'<span class=samplebadge>{m.group(0)}</span>' if m else ""
        sections.append(
            f"<section id=sec{i}><h2>{i + 1}. {label} {badge_html}</h2>"
            f"<p class=hdr>{spec}</p>"
            f"<div class=filmstrip>{''.join(tiles)}</div>"
            f"{_review_controls_html(i, form)}</section>"
        )
        lb_data.append(lb_entry)

        if len(sections) == per_page or i == len(events) - 1:
            written.append(_write_page(
                out_dir / _page_name(len(written) + 1), well, note, sections, shared,
                lb_data, len(written) + 1, n_pages, _page_name,
                first_event=i + 2 - len(sections), last_event=i + 1,
                n_events=len(events), review=review, layers=layers if channel_toggles else (),
                per_page=per_page,
                fixed_layers=fixed_layers))
            sections, shared, lb_data = [], [], []

    # ABSOLUTE, always, and as a file:// URL too. _cm.BUNDLE is usually relative, so this
    # used to return "data\bundle\<well>\browsers\browser_....html" -- openable only by
    # someone already sitting in the repo root. The reader is a person in a terminal or
    # a chat window, and they said so twice: first that they had to "hunt down the html
    # link", then that the absolute path still was not clickable and they were pasting
    # it into Chrome by hand. Terminals and chat clients linkify a URL, not a Windows
    # path, so give them both and let whichever one their client understands win.
    full = [p.resolve() for p in written]
    if len(full) == 1:
        return f"{full[0]}\n{full[0].as_uri()}"
    head = (f"{len(events)} events split into {len(full)} pages of up to "
            f"{per_page}; each page links to the previous/next part. "
            f"Open part 1 first:")
    return "\n".join([head] + [f"{p}\n{p.as_uri()}" for p in full])


_FORM_KINDS = ("checks", "choices", "numbers", "text")


def _stretch_of(tiles):
    """(lo, hi) for one channel of one event: its 1st / 99.7th percentile over the
    event's own tiles -- the same rule the page script applies when channels are
    toggled in the browser. Layers are rendered bare (see _OVERLAYS in render.py),
    so no label pixels skew it."""
    px = np.concatenate([t.ravel() for t in tiles])
    lo, hi = np.percentile(px, [1, 99.7])
    return float(lo), float(max(hi, lo + 8))


def _combine(by_ch, shown, st):
    """Per-frame colour images from per-channel grey tiles: stretch, tint, max."""
    out = []
    for j in range(len(by_ch[shown[0]["name"]])):
        acc = None
        for ly in shown:
            lo, hi = st[ly["name"]]
            g = np.clip((by_ch[ly["name"]][j].astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
            r, gg, b = (v / 255.0 for v in ly["rgb"])
            t = np.stack([g * b, g * gg, g * r], axis=-1)
            acc = t if acc is None else np.maximum(acc, t)
        out.append(acc.astype(np.uint8))
    return out


def _overlay(imgs, overlays):
    """Lay each frame's captured overlay over its tinted composite, in the overlay's
    own colour: white text stays white whatever channels are shown."""
    for img, ov in zip(imgs, overlays):
        drawn = ov.any(axis=-1)
        img[drawn] = ov[drawn]
    return imgs


def _png_b64_all(imgs):
    return [base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode("ascii") for img in imgs]


def _grey(img):
    """A rendered tile as one grey plane (layers are rendered uncoloured, so B=G=R
    except for burned-in label colours, which only need to stay visible)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _lightbox_args(common, ev):
    """The lightbox render's arguments: the filmstrip's, with a wider crop if the
    event asks for context_um."""
    return dict(common, crop_um=float(ev["context_um"])) if ev.get("context_um") else common


def _normalize_review_form(form):
    """Validate show_cells_in_browser's review_form up front and return it as
    {checks: [...], choices: {name: [...]}, numbers: [...], text: [...]}."""
    if not form:
        return {"checks": [], "choices": {}, "numbers": [], "text": []}
    unknown = set(form) - set(_FORM_KINDS)
    if unknown:
        raise ValueError(f"review_form has unknown keys {sorted(unknown)}; "
                         f"use any of {list(_FORM_KINDS)}")
    out = {"checks": [str(c) for c in form.get("checks") or []],
           "choices": {str(k): [str(o) for o in v]
                       for k, v in (form.get("choices") or {}).items()},
           "numbers": [str(n) for n in form.get("numbers") or []],
           "text": [str(t) for t in form.get("text") or []]}
    names = (out["checks"] + list(out["choices"]) + out["numbers"] + out["text"])
    reserved = {"notes", "event", "event_id", "label", "kind", "track_ids", "x", "y",
                "start_frame", "end_frame"}
    if len(names) != len(set(names)) or reserved & set(names):
        raise ValueError(f"review_form field names must be unique and not one of "
                         f"{sorted(reserved)} (the export already has those columns): {names}")
    for k, v in out["choices"].items():
        if len(v) < 2:
            raise ValueError(f"review_form choices[{k!r}] needs at least 2 options: {v}")
    return out


def _event_meta(i, ev):
    """The identifying columns one event gets in the review CSV export."""
    ids = ev.get("track_ids") or ([ev["track_id"]] if "track_id" in ev else [])
    default = (f"track {ids[0]}" if ev["kind"] == "track"
               else f"tracks {', '.join(str(t) for t in ids)}" if ev["kind"] == "family"
               else f"point ({ev.get('x', '')}, {ev.get('y', '')})")
    return {"event": i + 1, "event_id": str(ev.get("id", "")),
            "label": ev.get("label") or default, "kind": ev["kind"],
            "track_ids": " ".join(str(t) for t in ids),
            "x": ev.get("x", ""), "y": ev.get("y", ""),
            "start_frame": ev.get("start_frame", ""), "end_frame": ev.get("end_frame", "")}


def _review_controls_html(i, form):
    """The marking row under one event: the review_form fields, then notes."""
    from html import escape
    parts = []
    for c in form["checks"]:
        parts.append(f'<label><input type=checkbox data-ev={i} data-f="{escape(c)}"> '
                     f'{escape(c)}</label>')
    for name, opts in form["choices"].items():
        radios = "".join(
            f'<label><input type=radio name="ev{i}-{escape(name)}" data-ev={i} '
            f'data-f="{escape(name)}" value="{escape(o)}"> {escape(o)}</label>'
            for o in opts)
        parts.append(f'<span class=grp><b>{escape(name)}</b> {radios}</span>')
    for n in form["numbers"]:
        parts.append(f'<label>{escape(n)} <input type=number data-ev={i} '
                     f'data-f="{escape(n)}"></label>')
    for t in form["text"]:
        parts.append(f'<label>{escape(t)} <input type=text data-ev={i} '
                     f'data-f="{escape(t)}"></label>')
    return (f'<div class=review>{"".join(parts)}'
            f'<textarea data-ev={i} data-f="notes" rows=1 placeholder="notes"></textarea></div>')


def _write_page(out_path, well, note, sections, shared, lb_data, part, n_parts,
                page_name, first_event, last_event, n_events, review, layers=(),
                per_page=_EVENTS_PER_PAGE, fixed_layers=()):
    """Write one page of show_cells_in_browser's output and return its path."""
    lb_json = json.dumps(lb_data)
    # "</" escaped so a label containing "</script>" can't end the script block early.
    review_json = json.dumps(review).replace("</", "<\\/")
    layers_json = json.dumps(list(layers))
    layer_ctl = "".join(
        f'<label><input type=checkbox class=layer data-ch="{ly["name"]}"{" checked" if ly["on"] else ""} onchange="redrawAll()">'
        f'<span style="color:rgb({ly["rgb"][0]},{ly["rgb"][1]},{ly["rgb"][2]})">&#9632;</span> {ly["name"]}</label>'
        for ly in layers)
    if fixed_layers:
        why = ("Channel switching is off for this page, to keep it light (about 3x the data with three "
               "channels). Ask Claude to rebuild these cards with channel switching on.")
        layer_ctl = (f'<span class=offctl title="{why}">'
                     + "".join(f'<label><input type=checkbox disabled{" checked" if ly["on"] else ""}>'
                               f'<span style="color:rgb({ly["rgb"][0]},{ly["rgb"][1]},{ly["rgb"][2]})">&#9632;</span> '
                               f'{ly["name"]}</label>' for ly in fixed_layers)
                     + '<label><input type=checkbox disabled checked> auto contrast per event</label>'
                     + '<small>(switching off: ask Claude)</small></span>')
    if layers:
        layer_ctl += ('<label title="Stretch each channel to this event\'s own crop, like Fiji\'s Auto on a selection. '
                      'Off: the bundle\'s field-wide window.">'
                      '<input type=checkbox id=autoCtl checked onchange="redrawAll()"> auto contrast per event</label>')
    if n_parts > 1:
        prev_html = (f'<a href="{page_name(part - 1)}">&lsaquo; part {part - 1}</a>'
                     if part > 1 else "")
        next_html = (f'<a href="{page_name(part + 1)}">part {part + 1} &rsaquo;</a>'
                     if part < n_parts else "")
        # Every part listed, so a reviewer can jump straight to where they left off;
        # the script ticks parts whose events are all marked.
        parts_html = "".join(
            f'<a href="{page_name(k)}" data-part={k}'
            f'{" class=cur" if k == part else ""}>{k}</a>' for k in range(1, n_parts + 1))
        pager_html = (f'<div class="pager"><span>Part {part} of {n_parts} '
                      f'&middot; events {first_event}&ndash;{last_event} of {n_events}</span>'
                      f'{prev_html}{parts_html}{next_html}</div>')
    else:
        pager_html = ""
    # Collapsed by default: it is reference, not the task. Open once, then get out of
    # the way of the 14 cases the page actually exists for.
    note_html = f"<p class=task>{note}</p>" if note else ""
    shared_html = "".join(
        f"<details class=howto><summary>How to read these strips</summary>"
        f"<p>{g}</p></details>" for g in shared)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{well} browser{f" ({part}/{n_parts})" if n_parts > 1 else ""}</title>
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
div.filmstrip canvas {{ flex: 0 0 auto; height: 260px; border: 1px solid #333; cursor: zoom-in; }}
.lightbox canvas {{ width: min(88vh, 92vw); height: auto; }}
/* The frame/time/coord label (top) and scale bar (bottom) are burned into the image
   pixels, not drawn by CSS -- so "hiding" them for a clean figure means clipping
   those strips off the rendered image rather than toggling a layer. Both bands are
   now a single text line each (see _stamp_tile/_scale_bar in render.py), so top and
   bottom take the same inset. Filmstrip and lightbox render the identical base64
   image, just at different display sizes -- clip-path % is relative to the image's
   own box either way, so they must use the same values or one under/over-trims. */
body.figure-mode div.filmstrip img,
body.figure-mode .lightbox img, body.figure-mode div.filmstrip canvas,
body.figure-mode .lightbox canvas {{ clip-path: inset(7% 0 7% 0); }}
.lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); align-items: center; justify-content: center; z-index: 20; }}
.lightbox.open {{ display: flex; }}
.lightbox img {{ width: min(88vh, 92vw); height: auto; }}   /* fill the view whatever size was stored */
.lightbox-caption {{ position: absolute; bottom: 22px; left: 50%; transform: translateX(-50%); color: #ccc; font-size: 12.5px; }}
.lightbox-close {{ position: absolute; top: 14px; right: 20px; color: #fff; font-size: 28px; line-height: 1; cursor: pointer; background: none; border: none; padding: 6px 10px; }}
.lightbox-nav {{ position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.12); color: #fff; border: none; font-size: 28px; line-height: 1; width: 52px; height: 64px; cursor: pointer; border-radius: 8px; }}
.lightbox-nav:hover {{ background: rgba(255,255,255,0.24); }}
.lightbox-nav:disabled {{ opacity: 0.25; cursor: default; }}
.lightbox-nav.prev {{ left: 16px; }}
.lightbox-nav.next {{ right: 16px; }}
div.pager {{ display: flex; align-items: center; gap: 0.9rem; margin-left: auto; font-size: 0.85rem; color: #aaa; }}
div.pager a {{ color: #7aa7d0; text-decoration: none; padding: 0.15rem 0.6rem; border: 1px solid #335; border-radius: 4px; }}
div.pager a:hover {{ background: #1d2a35; }}
div.pager a.cur {{ background: #7aa7d0; color: #111; }}
div.pager a.done::after {{ content: " \\2713"; color: #6fcf7f; }}
section.focus {{ outline: 2px solid #4fa3ff; outline-offset: 6px; border-radius: 4px; }}
p.keys {{ color: #888; font-size: 0.78rem; margin: 0.2rem 0 0; }}
span.offctl {{ display: inline-flex; gap: 0.6rem; align-items: center; opacity: 0.45; cursor: help; }}
span.offctl label {{ cursor: help; }}
.lightbox-play {{ position: absolute; bottom: 48px; left: 50%; transform: translateX(-50%);
  background: rgba(255,255,255,0.12); color: #fff; border: none; border-radius: 6px;
  padding: 4px 14px; cursor: pointer; font: inherit; }}
body > div.pager {{ margin: 2rem 0; }}
div.review {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1rem;
  margin-top: 0.5rem; font-size: 0.85rem; color: #ccc; }}
div.review label {{ display: inline-flex; align-items: center; gap: 0.25rem; cursor: pointer; white-space: nowrap; }}
div.review span.grp {{ display: inline-flex; gap: 0.5rem; align-items: center;
  border: 1px solid #333; border-radius: 4px; padding: 0.1rem 0.5rem; }}
div.review input[type=number] {{ width: 5rem; }}
div.review input[type=number], div.review input[type=text], div.review textarea {{
  background: #1a1a1a; color: #eee; border: 1px solid #444; border-radius: 4px;
  padding: 0.2rem 0.4rem; font: inherit; }}
div.review textarea {{ flex: 1 1 20rem; min-width: 14rem; resize: vertical; }}
section.marked h2::after {{ content: " \\2713"; color: #6fcf7f; }}
div.reviewbar {{ display: flex; align-items: center; gap: 0.6rem; }}
div.reviewbar button {{ background: #1d2a35; color: #cfe3f5; border: 1px solid #335;
  border-radius: 4px; padding: 0.2rem 0.7rem; cursor: pointer; font: inherit; }}
div.reviewbar button:hover {{ background: #26394a; }}
</style></head><body>
<h1>{well}</h1>
<nav class="track-nav">
<div class="view-controls">
  <label><input type="checkbox" id="figureCtl" onchange="document.body.classList.toggle('figure-mode', this.checked)"> hide labels + scale bar (figures)</label>
  {layer_ctl}
  <div class="reviewbar"><span id="rv-count"></span>
    <button onclick="exportReview()" title="Download one CSV with every event in this set (all parts)">Export CSV</button></div>
  {pager_html}
</div>
<p class=keys>Keys: <b>j</b>/<b>k</b> next/previous event &middot; <b>Enter</b> enlarge &middot; <b>1</b>-<b>9</b> pick from the first choice group and move on &middot; <b>n</b> notes.
Enlarged: <b>&larr;</b>/<b>&rarr;</b> frames &middot; <b>space</b> play &middot; <b>&uarr;</b>/<b>&darr;</b> previous/next event &middot; <b>1</b>-<b>9</b> mark and move on &middot; <b>Esc</b> close.</p>
</nav>
{note_html}
{shared_html}
{''.join(sections)}
{pager_html}
<div class="lightbox" id="lightbox">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <button class="lightbox-nav prev" id="lb-prev" onclick="stepLightbox(-1)">&lsaquo;</button>
  <img id="lb-img" alt="">
  <canvas id="lb-canvas" style="display:none"></canvas>
  <div class="lightbox-caption" id="lb-caption"></div>
  <button class="lightbox-nav next" id="lb-next" onclick="stepLightbox(1)">&rsaquo;</button>
  <button class="lightbox-play" id="lb-play" onclick="togglePlay()">play</button>
</div>
<script>
var LB_DATA = {lb_json};
var LAYERS = {layers_json};
var lbSection = 0, lbIdx = 0;
// ---- per-channel layers: each channel is a grey JPEG; tint, toggle and stretch here ----
var decoded = {{}};                       // key -> Image, decoded once
function img(key, b64, done) {{
  var im = decoded[key];
  if (im && im.complete) return done(im);
  if (!im) {{ im = decoded[key] = new Image(); im.src = 'data:image/' + (key.slice(-4) === '_ovl' ? 'png' : 'jpeg') + ';base64,' + b64; }}
  im.addEventListener('load', function() {{ done(im); }}, {{once: true}});
}}
function framesOf(sec) {{ var d = LB_DATA[sec]; return d.images || d.layers[LAYERS[0].name]; }}
function onLayers() {{
  return LAYERS.filter(function(l) {{
    var c = document.querySelector('input.layer[data-ch="' + l.name + '"]'); return !c || c.checked; }});
}}
var stretchCache = {{}}, stretchWaiting = {{}};
function stretchOf(sec, done) {{             // per event, per channel: 1st / 99.7th percentile of its thumbs
  if (stretchCache[sec]) return done(stretchCache[sec]);
  // Every tile of an event asks at once on first draw: compute once, queue the rest.
  if (stretchWaiting[sec]) return stretchWaiting[sec].push(done);
  stretchWaiting[sec] = [done];
  done = function(st) {{ var q = stretchWaiting[sec]; delete stretchWaiting[sec]; q.forEach(function(f) {{ f(st); }}); }};
  var d = LB_DATA[sec], out = {{}}, left = LAYERS.length;
  LAYERS.forEach(function(l) {{
    var hist = new Uint32Array(256), n = 0, arr = d.thumbs[l.name], got = 0;
    arr.forEach(function(b64, j) {{
      img('t' + sec + l.name + j, b64, function(im) {{
        var c = document.createElement('canvas'); c.width = im.width; c.height = im.height;
        var x = c.getContext('2d'); x.drawImage(im, 0, 0);
        var px = x.getImageData(0, 0, im.width, im.height).data;
        for (var p = 0; p < px.length; p += 4) {{ hist[px[p]]++; n++; }}
        if (++got === arr.length) {{
          var lo = 0, hi = 255, acc = 0;
          for (var v = 0; v < 256; v++) {{ acc += hist[v]; if (acc >= n * 0.01) {{ lo = v; break; }} }}
          acc = 0;
          for (var v2 = 0; v2 < 256; v2++) {{ acc += hist[v2]; if (acc >= n * 0.997) {{ hi = v2; break; }} }}
          out[l.name] = [lo, Math.max(hi, lo + 8)];
          if (--left === 0) {{ stretchCache[sec] = out; done(out); }}
        }}
      }});
    }});
  }});
}}
function compose(canvas, sec, key, list, ovl, done) {{   // list: {{name: b64}} for one frame; ovl: its overlay PNG
  var on = onLayers(), auto = (document.getElementById('autoCtl') || {{}}).checked;
  stretchOf(sec, function(st) {{
    var ims = {{}}, left = LAYERS.length;
    LAYERS.forEach(function(l) {{
      img(key + l.name, list[l.name], function(im) {{
        ims[l.name] = im;
        if (--left) return;
        var w = im.width, h = im.height; canvas.width = w; canvas.height = h;
        var ctx = canvas.getContext('2d'), out = ctx.createImageData(w, h), o = out.data;
        for (var p = 3; p < o.length; p += 4) o[p] = 255;
        on.forEach(function(l2) {{
          var c = document.createElement('canvas'); c.width = w; c.height = h;
          var x = c.getContext('2d'); x.drawImage(ims[l2.name], 0, 0);
          var g = x.getImageData(0, 0, w, h).data, lo = auto ? st[l2.name][0] : 0, hi = auto ? st[l2.name][1] : 255;
          var k = 255 / (hi - lo), r = l2.rgb[0] / 255, gg = l2.rgb[1] / 255, b = l2.rgb[2] / 255;
          for (var p = 0; p < g.length; p += 4) {{
            var v = (g[p] - lo) * k; v = v < 0 ? 0 : v > 255 ? 255 : v;
            if (v * r > o[p]) o[p] = v * r; if (v * gg > o[p + 1]) o[p + 1] = v * gg; if (v * b > o[p + 2]) o[p + 2] = v * b;
          }}
        }});
        ctx.putImageData(out, 0, 0);
        if (!ovl) {{ if (done) done(); return; }}
        img(key + '_ovl', ovl, function(oim) {{   // label, bar and marker in their own colour, over the tint
          var c = document.createElement('canvas'); c.width = w; c.height = h;
          var x = c.getContext('2d'); x.drawImage(oim, 0, 0);
          var od = x.getImageData(0, 0, w, h).data, fin = ctx.getImageData(0, 0, w, h), f = fin.data;
          for (var p = 0; p < od.length; p += 4)
            if (od[p] || od[p + 1] || od[p + 2]) {{ f[p] = od[p]; f[p + 1] = od[p + 1]; f[p + 2] = od[p + 2]; }}
          ctx.putImageData(fin, 0, 0); if (done) done();
        }});
      }});
    }});
  }});
}}
function frameLayers(d, kind, j) {{
  var o = {{}}; LAYERS.forEach(function(l) {{ o[l.name] = d[kind][l.name][j]; }}); return o;
}}
function redrawAll() {{
  document.querySelectorAll('div.filmstrip canvas').forEach(function(cv) {{
    var sec = +cv.dataset.lb, j = +cv.dataset.j;
    var d = LB_DATA[sec];
    compose(cv, sec, 't' + sec + '_' + j, frameLayers(d, 'thumbs', j), d.thumb_ovl && d.thumb_ovl[j]);
  }});
  if (document.getElementById('lightbox').classList.contains('open')) showLbFrame();
}}
function showLbFrame() {{
  var imgs = framesOf(lbSection), d = LB_DATA[lbSection];
  var el = document.getElementById('lb-img'), cv = document.getElementById('lb-canvas');
  if (d.layers) {{
    el.style.display = 'none'; cv.style.display = '';
    compose(cv, lbSection, 'l' + lbSection + '_' + lbIdx, frameLayers(d, 'layers', lbIdx),
            d.layer_ovl && d.layer_ovl[lbIdx]);
  }} else {{
    cv.style.display = 'none'; el.style.display = '';
    el.src = 'data:image/jpeg;base64,' + imgs[lbIdx];
  }}
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
function closeLightbox() {{ if (playTimer) stopPlay(); document.getElementById('lightbox').classList.remove('open'); }}
function stepLightbox(delta) {{
  var imgs = framesOf(lbSection);
  var next = lbIdx + delta;
  if (next < 0 || next >= imgs.length) return;
  lbIdx = next;
  showLbFrame();
}}
document.getElementById('lightbox').addEventListener('click', function(e) {{
  if (e.target.id === 'lightbox') closeLightbox();
}});
if (LAYERS.length) redrawAll();
// Keyboard review. Section ids are GLOBAL event numbers (sec0..); LB_DATA is this
// page's events only, so lightbox index k is event FIRST + k.
var FIRST = {first_event - 1}, PAGES = {json.dumps([page_name(k) for k in range(1, n_parts + 1)])},
    PER = {per_page}, focusEv = FIRST, playTimer = null;
function secOf(ev) {{ return document.getElementById('sec' + ev); }}
function setFocus(ev, scroll) {{
  if (ev < FIRST || ev >= FIRST + LB_DATA.length) {{
    // another part: go there, at that event
    if (ev >= 0 && ev < REVIEW.events.length) location.href = PAGES[Math.floor(ev / PER)] + '#sec' + ev;
    return;
  }}
  var old = secOf(focusEv); if (old) old.classList.remove('focus');
  focusEv = ev; var s = secOf(ev); s.classList.add('focus');
  if (scroll) s.scrollIntoView({{block: 'center'}});
}}
function markFirstChoice(ev, n) {{
  var names = Object.keys(REVIEW.form.choices); if (!names.length) return false;
  var opts = REVIEW.form.choices[names[0]]; if (n > opts.length) return false;
  var r = Array.prototype.find.call(document.querySelectorAll('input[type=radio][data-ev="' + ev + '"]'),
    function(x) {{ return x.dataset.f === names[0] && x.value === opts[n - 1]; }});
  if (!r) return false;
  r.checked = true; r.dispatchEvent(new Event('change'));
  return true;
}}
function stopPlay() {{ clearInterval(playTimer); playTimer = null; document.getElementById('lb-play').textContent = 'play'; }}
function togglePlay() {{
  if (playTimer) return stopPlay();
  document.getElementById('lb-play').textContent = 'pause';
  playTimer = setInterval(function() {{
    lbIdx = (lbIdx + 1) % framesOf(lbSection).length; showLbFrame();
  }}, 350);
}}
function lbEvent(d) {{
  stopPlay();
  var k = lbSection + d;
  if (k < 0 || k >= LB_DATA.length) {{ closeLightbox(); setFocus(FIRST + k, true); return; }}
  setFocus(FIRST + k, false); openLightbox(k, 0);
}}
document.addEventListener('keydown', function(e) {{
  var t = e.target.tagName;
  if (t === 'TEXTAREA' || (t === 'INPUT' && e.target.type !== 'checkbox' && e.target.type !== 'radio')) {{
    if (e.key === 'Escape') e.target.blur();
    return;
  }}
  var n = /^[1-9]$/.test(e.key) ? +e.key : 0;
  if (document.getElementById('lightbox').classList.contains('open')) {{
    if (e.key === 'Escape') {{ closeLightbox(); setFocus(focusEv, true); }}   // back to the event you ended on
    else if (e.key === 'ArrowLeft') {{ stopPlay(); stepLightbox(-1); }}
    else if (e.key === 'ArrowRight') {{ stopPlay(); stepLightbox(1); }}
    else if (e.key === ' ') togglePlay();
    else if (e.key === 'ArrowUp') lbEvent(-1);
    else if (e.key === 'ArrowDown') lbEvent(1);
    else if (n) {{ if (markFirstChoice(FIRST + lbSection, n)) lbEvent(1); }}
    else return;
    e.preventDefault(); return;
  }}
  if (e.key === 'j') setFocus(focusEv + 1, true);
  else if (e.key === 'k') setFocus(focusEv - 1, true);
  else if (e.key === 'Enter') openLightbox(focusEv - FIRST, 0);
  else if (e.key === 'n') {{ var ta = secOf(focusEv).querySelector('textarea'); if (ta) ta.focus(); }}
  else if (n) {{ if (markFirstChoice(focusEv, n)) setFocus(focusEv + 1, true); }}
  else return;
  e.preventDefault();
}});

// Marks live in localStorage under one key per page SET, so every part of a
// multi-page set reads and writes the same store and Export from any part has all
// of them. Storage can be unavailable (private window, blocked site data): marking
// still works for this visit, it just won't survive a reload -- said in the counter.
var REVIEW = {review_json};
var rvState = {{}}, rvStored = true;
try {{ rvState = JSON.parse(localStorage.getItem(REVIEW.key) || '{{}}'); }}
catch (e) {{ rvStored = false; }}
function rvSave() {{
  try {{ localStorage.setItem(REVIEW.key, JSON.stringify(rvState)); }} catch (e) {{ rvStored = false; }}
  rvCount();
}}
function rvMarked(ev) {{
  var s = rvState[ev]; if (!s) return false;
  for (var k in s) if (s[k] !== '' && s[k] !== false) return true;
  return false;
}}
function rvCount() {{
  var n = 0;
  for (var i = 0; i < REVIEW.events.length; i++) if (rvMarked(i)) n++;
  document.getElementById('rv-count').textContent = n + ' of ' + REVIEW.events.length +
    ' marked' + (rvStored ? '' : ' (not saved: browser storage blocked, export before closing)');
  document.querySelectorAll('section[id^=sec]').forEach(function(sec) {{
    sec.classList.toggle('marked', rvMarked(+sec.id.slice(3)));
  }});
  document.querySelectorAll('div.pager a[data-part]').forEach(function(a) {{
    var p = +a.dataset.part, done = true;
    for (var i = (p - 1) * PER; i < Math.min(p * PER, REVIEW.events.length); i++) if (!rvMarked(i)) done = false;
    a.classList.toggle('done', done);
  }});
}}
document.querySelectorAll('div.review [data-f]').forEach(function(el) {{
  var s = rvState[el.dataset.ev] || {{}}, v = s[el.dataset.f];
  if (el.type === 'checkbox') el.checked = !!v;
  else if (el.type === 'radio') el.checked = v === el.value;
  else if (v !== undefined) el.value = v;
  el.addEventListener(el.tagName === 'TEXTAREA' || el.type === 'text' || el.type === 'number'
                      ? 'input' : 'change', function() {{
    var st = rvState[el.dataset.ev] = rvState[el.dataset.ev] || {{}};
    st[el.dataset.f] = el.type === 'checkbox' ? el.checked : el.value;
    rvSave();
  }});
}});
function csvCell(v) {{
  v = v === undefined || v === null ? '' : String(v);
  return /[",\\n\\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}}
function exportReview() {{
  var f = REVIEW.form, idCols = ['event', 'event_id', 'label', 'kind', 'track_ids',
    'x', 'y', 'start_frame', 'end_frame'];
  var fields = f.checks.concat(Object.keys(f.choices), f.numbers, f.text, ['notes']);
  var rows = [idCols.concat(['marked'], fields).map(csvCell).join(',')];
  REVIEW.events.forEach(function(m, i) {{
    var s = rvState[i] || {{}};
    var vals = fields.map(function(k) {{
      return f.checks.indexOf(k) >= 0 ? (s[k] ? 1 : '') : s[k];
    }});
    rows.push(idCols.map(function(c) {{ return m[c]; }})
      .concat([rvMarked(i) ? 1 : 0], vals).map(csvCell).join(','));
  }});
  var blob = new Blob(['\\ufeff' + rows.join('\\r\\n')], {{type: 'text/csv'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = REVIEW.file;
  document.body.appendChild(a); a.click(); a.remove();
}}
rvCount();
setFocus(/^#sec[0-9]+$/.test(location.hash) ? +location.hash.slice(4) : FIRST, !!location.hash);
</script>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path


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
