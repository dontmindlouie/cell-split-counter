"""Filmstrip-based tools: watch_location_over_time, list_nearby_tracks,
follow_cells_over_time, and the family-resolution helper they share.

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import cv2
import numpy as np

from .server import server, MAX_IMAGES, _WINDOW_BEFORE_MIN, _WINDOW_AFTER_MIN, _STRIDE_MIN
from .io import _frame_at_offset_min, _minutes_between
from .render import _crop_tile, _stamp_tile, _encode, _display_note

import cell_mcp_server as _cm

# _manifest, _tracks, _hours, _lineage, _filmstrip_frames, and _family_filmstrip_frames
# below go through `_cm.` rather than a direct import -- see the note at the top of io.py.

_FAMILY_NEARBY_UM = 14.0
_FAMILY_NEARBY_BEFORE_MIN = 15.0
_FAMILY_NEARBY_AFTER_MIN = 75.0

def _nearest_detection(well: str, frame: int, x: float, y: float,
                       exclude: int | None = None) -> tuple[int, float] | None:
    """(track_id, distance_um) of the closest tracked cell to a point in one frame.

    `exclude` drops one id from the search -- in anchor mode the anchor is always
    nearest to itself at 0.0 um, which says nothing; the useful answer is what ELSE
    is near the place being watched.
    """
    df = _cm._tracks(well)
    f = df[df.frame == frame]
    if exclude is not None:
        f = f[f.track_id != exclude]
    if f.empty:
        return None
    d = np.hypot(f.cx.to_numpy() - x, f.cy.to_numpy() - y)
    i = int(np.argmin(d))
    return int(f.track_id.iloc[i]), float(d[i]) * _cm._manifest(well)["pixel_size_um"]


@server.tool(structured_output=False)
def watch_location_over_time(
    well: str,
    start_frame: int,
    end_frame: int,
    x: float | None = None,
    y: float | None = None,
    anchor_track_id: int | None = None,
    max_images: int = 8,
    crop_um: float = 45.0,
    color: bool = True,
    scale_bar: bool = True,
) -> list:
    """Watch a PLACE over time, instead of following a cell's own mask.

    follow_cells_over_time answers "what happened to track N". This answers "what happened
    HERE", which is a different question and often the one you actually have.

    Reach for this when:

    - **The thing you care about has no track.** Every other tool addresses cells by
      track_id, so an object the segmenter never caught is otherwise impossible to
      ask about at all -- a small bright body beside a nucleus, debris, anything
      below the detector's threshold. Give its coordinates and look at it.
    - **You are not sure the track is on the object you mean.** A track and its own
      OFF-TRACK walk can sit on two different cells in one crop; a fixed position is
      not confused by that, because it never claims to be following anything.
    - **The mask breaks exactly when the event happens.** Chromatin rounding during
      mitosis and death is precisely when segmentation fails, so a mask-following
      crop tends to lose the cell at the moment of interest. A place does not move.
    - **You want a stable stage.** Pass anchor_track_id to ride a calm neighbouring
      cell's position while the cell you are watching does something violent.

    Coordinates are PIXELS in the full frame, matching cx/cy in tracks.csv and the
    xy columns of list_tracks -- not microns, and not crop-relative.

    Nothing is ringed, because nothing here is claimed to be a detection. Instead
    every frame reports the nearest tracked cell and how far away it is, so you can
    tell "this is track 2036" from "there is nothing tracked within 12 um of here".

    Args:
        well: well name from list_wells().
        start_frame, end_frame: inclusive frame range.
        x, y: the position to watch, in full-frame pixels. Required unless
            anchor_track_id is given.
        anchor_track_id: instead of a fixed point, follow THIS track's centroid --
            useful as a stable vantage on a neighbour. Where the anchor is missing
            from a frame, its closest known position is used.
        max_images: how many frames to show (hard capped at 12).
        crop_um: crop width in micrometres.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
    """
    from mcp.types import TextContent

    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    lo, hi = max(0, int(start_frame)), min(n_frames - 1, int(end_frame))
    if hi < lo:
        raise ValueError(f"empty range {start_frame}-{end_frame}; {well} has 0-{n_frames - 1}.")

    anchor_pos: dict[int, tuple[float, float]] = {}
    if anchor_track_id is not None:
        t = _cm._tracks(well)
        t = t[t.track_id == anchor_track_id]
        if t.empty:
            raise ValueError(f"anchor track {anchor_track_id} not found in {well}.")
        anchor_pos = {int(r.frame): (float(r.cx), float(r.cy)) for r in t.itertuples()}
        known = sorted(anchor_pos)
    elif x is None or y is None:
        raise ValueError("give either x and y (full-frame pixels), or anchor_track_id.")

    def _centre(f: int) -> tuple[float, float]:
        if anchor_track_id is None:
            return float(x), float(y)
        if f in anchor_pos:
            return anchor_pos[f]
        # Nearest frame the anchor was actually seen in.
        nf = min(known, key=lambda k: abs(k - f))
        return anchor_pos[nf]

    avail = list(range(lo, hi + 1))
    n = min(max_images, MAX_IMAGES, len(avail))
    picks = [avail[i] for i in np.linspace(0, len(avail) - 1, n).astype(int)]

    um_px = m["pixel_size_um"]
    half = max(8, int(round(crop_um / um_px / 2)))

    where = (f"anchored on track {anchor_track_id}" if anchor_track_id is not None
             else f"fixed at ({float(x):.0f}, {float(y):.0f}) px")
    # The corner label is the one mark in this toolset a reader cannot decode from the
    # image: a session once grepped cell_mcp.py's source to find out what "~2362 @6um"
    # meant. It is burned in so a frame stays identified if it gets separated from this
    # text, which only works if the text says how to read it.
    lines = [f"{well}: frames {lo}-{hi}, showing {n} of {len(avail)}, {where}. "
             f"Crop {crop_um:g} um wide. The yellow crosshair marks WHERE YOU ASKED to "
             f"look -- nothing is ringed, because this is a place, not a tracked object, "
             f"and a ring would imply something was detected there. The bottom-left label "
             f"reads \"~<track_id> @<distance>um\": the NEAREST tracked cell to the "
             f"crosshair and how far its centre sits from it -- not the thing at the "
             f"crosshair, which may be untracked or nothing at all. It is only drawn when "
             f"that cell is closer than the crop is wide, so a blank corner means nothing "
             f"tracked is even in view. Nearest tracked cell per frame:"]

    images: list[np.ndarray] = []
    for f in picks:
        ccx, ccy = _centre(int(f))
        tile = _crop_tile(well, int(f), ccx, ccy, half, color)
        if tile is None:
            continue
        img = tile.img
        # A crosshair, not a ring: it marks where you asked to look, and must not
        # imply that something was detected there. Gapped in the middle so it never
        # covers the chromatin being judged.
        cxp, cyp = int(tile.cx), int(tile.cy)
        for dx0, dx1 in ((-12, -5), (5, 12)):
            cv2.line(img, (cxp + dx0, cyp), (cxp + dx1, cyp), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (cxp, cyp + dx0), (cxp, cyp + dx1), (0, 255, 255), 1, cv2.LINE_AA)

        near = _nearest_detection(well, int(f), ccx, ccy, exclude=anchor_track_id)
        if near is None:
            note = "no tracked cell in this frame"
        else:
            tid, dum = near
            note = f"track {tid}, {dum:.1f} um away"
        lines.append(f"  f{int(f)} ({_cm._hours(well, int(f)):.2f} h): {note}")
        corner = (f"~{near[0]} @{near[1]:.0f}um"
                  if near is not None and near[1] < crop_um else None)
        images.append(_stamp_tile(tile, f"f{int(f)} t={_cm._hours(well, int(f)):.1f}h",
                                  um_px, scale_bar, corner=corner))

    lines.append(
        "\nA nearest cell many microns away means the thing at this position is NOT "
        "tracked -- which is the usual reason to be here. Distances are centre-to-centre, "
        "so a large nucleus can read several microns away while still overlapping the point."
        + _display_note(well)
    )
    out: list = [TextContent(type="text", text="\n".join(lines))]
    out.extend(_encode(i) for i in images)
    return out


_CHAIN_GAP_MIN = 10.0
_CHAIN_MARGIN_UM = 14.0
_CHAIN_AMBIGUOUS_RATIO = 1.3


def _walk_chain(well: str, track_id: int, direction: str = "forward",
                max_hops: int = 6, seen: set[int] | None = None) -> list[dict]:
    """Walk a chain of id hops from `track_id` -- the same physical object losing and
    regaining its id (segmentation wobble), not a division.

    Applies the COEXISTENCE test list_nearby_tracks hands a reader by hand,
    automatically, one hop at a time: the next link must be NEW near the current
    end of the chain, close in space, and -- the part that tells a hop apart from a
    sister -- must NOT overlap the current track's own span. Two daughters coexist;
    a re-acquisition never does.

    Stops the moment it stops being SURE, not when it runs out of candidates:
    hit max_hops, no candidate within radius/gap, or the two closest candidates
    score too close to call apart (`_CHAIN_AMBIGUOUS_RATIO`). Same reason
    `_resolve_family`'s nearby-track guess is off by default -- BeWo 969 showed a
    silent wrong hop is worse than stopping and asking. `direction="backward"`
    walks the mirror image, from the track's FIRST frame/position backward.

    Returns hops in walk order (excluding the seed), each:
        {track_id, frames: (lo, hi), gap_frames, dist_um, size_ratio}
    Empty if no hop resolves.
    """
    df = _cm._tracks(well)
    if df.empty:
        return []
    um_px = _cm._manifest(well)["pixel_size_um"]
    srt = df.sort_values("frame")
    first = srt.groupby("track_id").first()
    last = srt.groupby("track_id").last()
    has_area = "area_um2" in srt.columns
    med_area = srt.groupby("track_id").area_um2.median() if has_area else None

    seen = set(seen) if seen else set()
    seen.add(int(track_id))
    chain: list[dict] = []
    cur = int(track_id)

    for _ in range(max_hops):
        if cur not in first.index or cur not in last.index:
            break
        cur_lo, cur_hi = int(first.loc[cur, "frame"]), int(last.loc[cur, "frame"])
        a_cur = float(med_area.loc[cur]) if med_area is not None and cur in med_area.index else None
        r_um = (float(np.sqrt(max(a_cur, 1.0) / np.pi)) if a_cur else 0.0) + _CHAIN_MARGIN_UM
        r_px = r_um / um_px

        if direction == "forward":
            f0, x0, y0 = cur_hi, float(last.loc[cur, "cx"]), float(last.loc[cur, "cy"])
            lo, hi = f0 + 1, _frame_at_offset_min(well, f0, _CHAIN_GAP_MIN)
            edge = first
        else:
            f0, x0, y0 = cur_lo, float(first.loc[cur, "cx"]), float(first.loc[cur, "cy"])
            lo, hi = _frame_at_offset_min(well, f0, -_CHAIN_GAP_MIN), f0 - 1
            edge = last
        if lo > hi:
            break

        pool = edge[(edge.frame >= lo) & (edge.frame <= hi)]
        pool = pool[~pool.index.isin(seen)]
        if pool.empty:
            break
        d2 = (pool.cx - x0) ** 2 + (pool.cy - y0) ** 2
        near = pool[d2 <= r_px * r_px]
        if near.empty:
            break

        cands = []
        for t in near.index:
            t = int(t)
            t_lo, t_hi = int(first.loc[t, "frame"]), int(last.loc[t, "frame"])
            if t_lo <= cur_hi and t_hi >= cur_lo:
                continue  # coexists with the current end -- a sister, not a hop
            dist_um = float(np.sqrt(d2.loc[t])) * um_px
            a_t = float(med_area.loc[t]) if med_area is not None and t in med_area.index else None
            size_ratio = (a_t / a_cur) if (a_t and a_cur) else None
            cands.append({
                "track_id": t, "frames": (t_lo, t_hi),
                "gap_frames": (t_lo - cur_hi) if direction == "forward" else (cur_lo - t_hi),
                "dist_um": dist_um, "size_ratio": size_ratio,
            })
        if not cands:
            break
        cands.sort(key=lambda c: c["dist_um"])
        best = cands[0]
        if len(cands) > 1 and cands[1]["dist_um"] < best["dist_um"] * _CHAIN_AMBIGUOUS_RATIO:
            break  # two candidates too close to call -- stop rather than guess

        chain.append(best)
        seen.add(best["track_id"])
        cur = best["track_id"]

    return chain


@server.tool()
def resolve_lineage_chain(well: str, track_id: int, direction: str = "forward",
                          max_hops: int = 6) -> str:
    """Chase a cell through id hops caused by segmentation losing and regaining it --
    NOT a division. Free: no images.

    A track can end mid-life for a reason that has nothing to do with mitosis: the
    mask wobbles at telophase, drops out for a frame, and Cellpose hands the same
    physical cell a new id when it reappears. `get_lineage` will not help here --
    it only reads `lineage.csv`'s division links, and a hop like this was never a
    division. Tracing it today means `list_nearby_tracks` plus `measure` by hand at
    every hop; this does that walk automatically, one hop at a time, and STOPS the
    moment it is no longer sure rather than guessing through an ambiguous one.

    THE TEST IS THE SAME ONE list_nearby_tracks hands you by hand: a hop must be
    NEW near the current end of the chain, close in space, and -- what tells a hop
    apart from a real sister -- must NOT be on screen at the same time as the track
    it follows. Two daughters coexist; a re-acquisition never does. Where two
    candidates are nearly as close as each other this returns what it found so far
    and says why it stopped, rather than pick one silently -- a wrong hop chosen
    quietly is worse than stopping short, same reason `_resolve_family`'s nearby-
    track guess is off by default.

    Args:
        well: well name from list_wells().
        track_id: the track to chase. Usually a daughter whose filmstrip goes
            OFF-TRACK sooner than its sister's.
        direction: "forward" (default) chases id hops after this track's last
            frame -- the usual case, segmentation wobble right after a division.
            "backward" chases hops before its first frame, for a track that starts
            mid-event because the cell was already hard to segment on arrival.
        max_hops: hard cap on chain length, so a bad well cannot walk forever.
    """
    if direction not in ("forward", "backward"):
        raise ValueError('direction must be "forward" or "backward".')
    df = _cm._tracks(well)
    if df.empty or track_id not in df.track_id.values:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    chain = _walk_chain(well, track_id, direction=direction, max_hops=max_hops)
    stitched = [int(track_id), *(h["track_id"] for h in chain)]

    if not chain:
        return (f"{well} track {track_id}: no {direction} hop resolves -- either this "
                f"track's own id covers the cell for the whole window, or the next "
                f"candidate was ambiguous (see list_nearby_tracks to look yourself).")

    out = [f"{well} track {track_id}: {len(chain)} {direction} hop(s) resolved.",
           "", "hop track_id | frames | gap_frames | dist_um | size_ratio"]
    for h in chain:
        sr = f"{h['size_ratio']:.2f}" if h["size_ratio"] is not None else "-"
        out.append(f"{h['track_id']} | {h['frames'][0]}-{h['frames'][1]} | "
                   f"{h['gap_frames']} | {h['dist_um']:.1f} | {sr}")
    out.append(
        f"\nStitched chain: {stitched}. Each hop passed the coexistence test (never "
        f"overlapping the id before it) and had no other candidate within "
        f"{_CHAIN_AMBIGUOUS_RATIO:g}x its distance -- still bookkeeping, not a "
        f"judgement that these are all 'the same cell'; confirm on the pixels via "
        f"follow_cells_over_time(track_ids={stitched})."
        + (f" Stopped before {max_hops} hops -- no further candidate resolved "
           f"(ambiguous or none nearby)." if len(chain) < max_hops else "")
    )
    return "\n".join(out)


def _resolve_family(well: str, track_ids: list[int],
                    include_nearby: bool = False) -> tuple[list[int], list[int]]:
    """Add a track's recorded daughters, then the ones nobody recorded.

    Returns (members, added_positionally).

    The daughters come from lineage.csv, which records what the TRACKER linked, not
    what happened -- and where the tracker fails through a division it links one
    daughter, or none, or a piece of debris. A strip built from that member set then
    follows the mother and one daughter, the real sister drifts out of frame, and the
    reviewer sees half an event. That is the exact complaint this exists to answer:
    "only tracking 1 daughter, would be nice to have midpoint."

    So after the recorded daughters, look for tracks that BEGIN near where the mother
    ended, within a few frames of the link. A sister the tracker never connected still
    has to appear as a new object next to her mother -- that is what makes this
    findable from geometry with no new data. Anything found this way is returned
    separately so the header can say the strip is showing objects the lineage does not
    vouch for, which is a different claim from a recorded daughter.

    OFF BY DEFAULT, because nearest-by-distance is not good enough to decide this and
    a silent wrong guess is worse than no guess. On BeWo 969 it picked tracks 3829 and
    3879 -- which start 2.3 um apart, six frames apart, and NEVER COEXIST. They are one
    object at (250, 412) losing and regaining its id, 22 um north of the mother, and
    the strip centred on a chain of re-acquisitions instead of on a pair of daughters.
    The reviewer spotted it immediately: "I don't think that's the right daughter, and
    it's also not a midpoint -- are there two tracks that instantiated near the same
    place?" There were four.

    Use list_nearby_tracks() instead and choose the members yourself. It reports which
    candidates COEXIST, which is the test this heuristic lacks: two daughters must be
    on screen at the same time, so a chain of non-overlapping tracks at one spot is one
    cell being re-acquired, never a sister pair.

    ALWAYS (not gated by include_nearby): each member is also chain-walked forward
    via _walk_chain, chasing id hops from segmentation wobble rather than division --
    the case that produced the literal complaint "the last couple of tracks snapped
    to 1 daughter" when a daughter's track ended at a wobble hop and dropped out of
    the centring mean. Unlike the nearby-track guess above, this IS the coexistence
    test BeWo 969 was missing -- a hop must never overlap the id before it -- so it
    is on by default; it still stops rather than guesses through an ambiguous hop.
    Resolved hops land in `added`, same as the nearby-heuristic's finds, so the
    header can say which members the lineage record itself does not vouch for.
    """
    ids = [int(t) for t in track_ids]
    if len(track_ids) == 1:
        lin = _cm._lineage(well)
        kids = (lin.get(ids[0]) or {}).get("daughters") or []
        ids = [ids[0], *[int(k) for k in kids]]

    # Only daughters (ids[1:]) are chain-walked, never ids[0]. ids[0] is the anchor
    # (the mother, by this function's own convention -- see the nearby-heuristic
    # below, which reads ids[0] the same way) and her track ending IS the division;
    # chasing her forward would just walk into her own daughters, which is already
    # answered by lineage.csv/the nearby heuristic and would silently reproduce
    # whichever daughter happens to be unlisted -- exactly the wrong-guess failure
    # this whole function exists to avoid.
    chained: list[int] = []
    seen = set(ids)
    for tid in ids[1:]:
        for hop in _walk_chain(well, tid, direction="forward", seen=seen):
            chained.append(hop["track_id"])
            seen.add(hop["track_id"])

    if not include_nearby:
        return [*ids, *chained], chained

    df = _cm._tracks(well)
    mother = df[df.track_id == ids[0]]
    if mother.empty:
        return [*ids, *chained], chained
    mother = mother.sort_values("frame")
    link = int(mother.frame.iloc[-1])
    x0, y0 = float(mother.cx.iloc[-1]), float(mother.cy.iloc[-1])
    a0 = float(mother.area_um2.median()) if "area_um2" in mother else 0.0
    um_px = _cm._manifest(well)["pixel_size_um"]
    r_px = (float(np.sqrt(max(a0, 1.0) / np.pi)) + _FAMILY_NEARBY_UM) / um_px

    # Candidates: tracks whose FIRST frame lands near the link. A sister the tracker
    # lost the mother into appears as a new id right there; a neighbour that merely
    # happens to be close has been on screen for a long time already.
    starts = df.sort_values("frame").groupby("track_id").first()
    lo = _frame_at_offset_min(well, link, -_FAMILY_NEARBY_BEFORE_MIN)
    hi = _frame_at_offset_min(well, link, _FAMILY_NEARBY_AFTER_MIN)
    win = starts[(starts.frame >= lo) & (starts.frame <= hi)]
    d2 = (win.cx - x0) ** 2 + (win.cy - y0) ** 2
    near = win[d2 <= r_px * r_px].assign(d2=d2[d2 <= r_px * r_px])
    extra = [int(t) for t in near.sort_values("d2").index if int(t) not in ids and int(t) not in seen]
    extra = extra[:2]
    return [*ids, *chained, *extra], [*chained, *extra]


@server.tool()
def list_nearby_tracks(
    well: str,
    track_id: int | None = None,
    x: float | None = None, y: float | None = None, frame: int | None = None,
    before_min: float = 15.0, after_min: float = 75.0,
    radius_um: float | None = None,
    new_only: bool = True,
) -> str:
    """Every object segmented near a place and time -- so YOU can work out who the
    daughters are. Free: no images.

    This is the tool to reach for when lineage.csv is wrong or empty about a division,
    which on BeWo is most of the time. Cellpose segments the daughters perfectly well;
    it is the TRACKER that declines to link them, so the objects you need already exist
    in tracks.csv with their own ids. Nothing here needs re-segmenting or a model.

    It deliberately does not pick for you. A first attempt did -- nearest two starts to
    the mother -- and on BeWo 969 it chose tracks 3829 and 3879, which begin 2.3 um and
    six frames apart and are one cell being re-acquired, not two daughters. A wrong
    guess made silently is worse than no guess, so this hands you the evidence instead.

    THE TEST THAT SETTLES IT IS COEXISTENCE. Two daughters must be on screen at the
    same time. A run of tracks at one spot whose spans do not overlap -- 3782 f776-777,
    3806 f778-780, 3829 f781-786, 3879 f787-813 -- is a single object losing and
    regaining its id, however close together they look. The `coexists_with` column is
    there to make that judgement without opening an image, and a candidate coexisting
    with nothing cannot be half of a pair.

    A real sister also has to be NEW: a cell that has been on screen for hours and
    merely happens to be nearby is a neighbour. Hence new_only, and hence the window
    reaching much further forward than back -- the sister appears when the cell divides,
    which on BeWo runs ~20 min past where the tracker's link ends.

    Anchor it either on a track (its last known position and frame, which is where a
    mother was lost) or on an explicit x/y/frame from watch_location_over_time.

    Args:
        well: well name from list_wells().
        track_id: anchor on this track's LAST position and frame. Usually the mother.
        x, y, frame: anchor on an explicit point instead, in pixels.
        before_min, after_min: how far either side of the anchor frame to look, in
            minutes. Forward-heavy by default, for the reason above.
        radius_um: search radius. None = the anchor's own radius + 14 um, so it scales
            with the cell line rather than assuming RUES2-sized nuclei.
        new_only: only tracks that BEGIN inside the window. False also lists tracks
            already running through it, which is what you want when asking "what is
            this thing overlapping with".
    """
    df = _cm._tracks(well)
    m = _cm._manifest(well)
    um_px = m["pixel_size_um"]
    srt = df.sort_values("frame")
    last = srt.groupby("track_id").last()
    first = srt.groupby("track_id").first()
    med = srt.groupby("track_id").area_um2.median()

    if track_id is not None:
        tid = int(track_id)
        if tid not in last.index:
            raise ValueError(f"track {tid} not found in {well}. Use list_tracks().")
        x0, y0 = float(last.loc[tid, "cx"]), float(last.loc[tid, "cy"])
        f0 = int(last.loc[tid, "frame"])
        a0 = float(med.loc[tid])
        anchor = f"track {tid}'s last position, f{f0}"
    elif None not in (x, y, frame):
        x0, y0, f0 = float(x), float(y), int(frame)
        a0 = float(med.median())
        anchor = f"point ({x0:.0f}, {y0:.0f}) at f{f0}"
        tid = None
    else:
        raise ValueError("give either track_id, or all of x, y and frame.")

    r_um = radius_um if radius_um else float(np.sqrt(max(a0, 1.0) / np.pi)) + 14.0
    r_px = r_um / um_px
    lo = _frame_at_offset_min(well, f0, -before_min)
    hi = _frame_at_offset_min(well, f0, after_min)

    if new_only:
        pool = first[(first.frame >= lo) & (first.frame <= hi)]
        near = pool[((pool.cx - x0) ** 2 + (pool.cy - y0) ** 2) <= r_px * r_px]
        ids = [int(t) for t in near.index if t != tid]
    else:
        win = df[(df.frame >= lo) & (df.frame <= hi)]
        win = win[((win.cx - x0) ** 2 + (win.cy - y0) ** 2) <= r_px * r_px]
        ids = [int(t) for t in win.track_id.unique() if t != tid]

    if not ids:
        return (f"{well}: nothing segmented within {r_um:.0f} um of {anchor} "
                f"in f{lo}-{hi}"
                + (" that BEGINS there (new_only=True; pass new_only=False to include "
                   "tracks already running)" if new_only else "") + ".")

    spans = {t: (int(first.loc[t, "frame"]), int(last.loc[t, "frame"])) for t in ids}
    out = [
        f"{well}: {len(ids)} object(s) within {r_um:.0f} um of {anchor}, "
        f"frames {lo}-{hi} ({_minutes_between(well, lo, hi):.0f} min)"
        + (", counting only tracks that BEGIN in the window." if new_only else "."),
        "",
        "track_id | frames | n | area_um2 | dist_um | coexists_with",
    ]
    rows = []
    for t in ids:
        s, e = spans[t]
        co = [u for u in ids
              if u != t and spans[u][0] <= e and spans[u][1] >= s]
        fr = first.loc[t]
        d = float(np.hypot(fr.cx - x0, fr.cy - y0)) * um_px
        rows.append((d, t, s, e, int((df.track_id == t).sum()),
                     float(med.loc[t]), co))
    for d, t, s, e, n, a, co in sorted(rows):
        out.append(f"{t} | {s}-{e} | {n} | {a:.0f} | {d:.1f} | "
                   f"{', '.join(str(u) for u in co) if co else 'NOTHING'}")
    out.append(
        "\nTwo daughters must COEXIST. A candidate whose coexists_with is NOTHING "
        "cannot be half of a pair, and a run of such tracks at one spot with "
        "consecutive spans is one cell losing and regaining its id. Distance alone "
        "will not tell you apart -- on BeWo 969 the two nearest starts were 2.3 um and "
        "six frames apart, and were the same cell. Pick the members yourself and pass "
        "them to follow_cells_over_time(track_ids=[...]) to see the event; nothing here "
        "is a claim that a division happened."
    )
    return "\n".join(out)


@server.tool()
def follow_cells_over_time(
    well: str,
    track_id: int | None = None,
    track_ids: list[int] | None = None,
    start_frame: int | None = None, end_frame: int | None = None,
    centre_frame: int | None = None,
    before_min: float = _WINDOW_BEFORE_MIN, after_min: float = _WINDOW_AFTER_MIN,
    max_images: int | None = None, stride_min: float = _STRIDE_MIN,
    crop_um: float | None = None,
    color: bool = True, scale_bar: bool = True, marker: bool = False,
) -> list:
    """Follow one cell, or a mother and her daughters, over time as close-up images.

    The main tool for judging what a cell is actually doing -- dividing, dying, or
    sitting still. The crop re-centres every frame, so a moving cell stays in view.
    To watch a fixed PLACE rather than a cell, use watch_location_over_time().

    Give EITHER `track_id` (one cell) or `track_ids` (a set), and pick deliberately
    -- they follow different things:

    ONE MASK -- `track_id=N`. Follows that track and nothing else. Frames past the
    track's own lifetime are labelled OFF-TRACK and rendered by walking the nearest
    detected blob from the track's boundary (solid orange ring): usually the same
    physical object under a new id, but never confirmed to be THIS cell rather than
    a neighbour. Where the walk loses the trail the crop freezes at the last
    resolved position (dashed blue ring) instead of guessing further. Read OFF-TRACK
    frames as "this patch of the field", not "this cell".

    A MEMBER SET -- `track_ids=[...]`. The crop centres on the mean position of
    whichever members are present in each frame, so it rides the mother up to the
    handoff and the daughters' midpoint after it. No mode switch, because membership
    does the switching. This is the one to reach for on any event where the cell
    stops being one object: `track_id` follows a single mask and so goes OFF-TRACK
    at exactly the moment the division happens. `track_ids=[N]` -- a ONE-ELEMENT
    list -- means track N plus the daughters recorded in lineage.csv, which is NOT
    the same as `track_id=N`. Pass every id yourself when you do not trust that
    link, or for anything that is not a division: a cell fragmenting during necrosis
    is a member set too, and the strip will stay on the debris field rather than
    chase one shard.

    With a member set the WINDOW is chosen for you around the frame where membership
    changes, measured in MINUTES: 30 before, 90 after. It is lopsided on purpose.
    The transition is the frame where the TRACKER stopped linking, and on BeWo the
    mitotic figure appears up to ~20 min LATER -- so a window that stops near the
    transition shows the lead-up and hides the outcome, and every real division
    scored off it reads as an artifact. Widen `after_min` before concluding a
    candidate is not a division. Give start_frame/end_frame to override with exact
    frames; you may ask for frames outside a track's lifetime, and you often should,
    since a track usually ENDS as its cell divides.

    Frames are sampled by TIME (~`stride_min` apart), not by a fixed count, so a
    strip means the same thing on a well shot every 3.0 min as on one shot every
    4.9. If the whole window fits under the image cap you get EVERY frame -- no
    gaps. Set max_images only to budget context deliberately.

    What it will not do: interpolate. A frame where no member is present holds the
    previous centre and is labelled HELD, because a made-up position rendered like a
    measured one is the failure this whole tool set exists to avoid.

    Args:
        well: well name from list_wells().
        track_id: ONE cell to follow, from list_tracks(). Mutually exclusive with
            track_ids.
        track_ids: the members of a set. One id = that track plus its recorded
            daughters. Mutually exclusive with track_id.
        start_frame, end_frame: inclusive override of the automatic window. Given
            both, the range is rendered gap-free up to the image cap. With
            `track_id` these default to the track's first and last frame.
        centre_frame: put the window around THIS frame instead of around the frame
            where membership changes. Pass it whenever you chose the members
            yourself -- the membership rule only means something for a mother plus
            her recorded daughters, and on a hand-picked set it drifts.
            find_candidates hands you `cond_f`, the frame the chromatin was most
            condensed, which is usually the right thing to centre on. Member sets
            only.
        before_min, after_min: MINUTES either side of the membership transition,
            when the window is automatic. Converted to frames from this well's own
            timestamps, which are not evenly spaced. Member sets only.
        max_images: pin the frame count. None (recommended) samples by TIME: a range
            that fits under the cap of 12 comes back GAP-FREE, and a longer one is
            thinned to ~stride_min spacing rather than to a frame count. A fixed
            count means different time resolution on different wells, and it was
            silently skipping frames inside ranges that were asked for explicitly --
            which is exactly where the evidence is.
        stride_min: target spacing between rendered frames, in minutes.
        crop_um: width of the crop in micrometres. None (recommended) means 60 for a
            single track -- a cell and its immediate neighbours -- and auto-fit for
            a member set, wide enough to hold every member across the sampled
            frames. Do not set it by hand on a member set unless you want a
            particular zoom: guessing is how a sibling drifts out of frame halfway
            along, since separation grows as the daughters move apart.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
        marker: draw a thin ring around the tracked cell -- with track_ids, rings
            every member present in each frame or none at all. Off by default
            because the ring is one more shape in an image whose shapes are the
            evidence. Turn it on for wide crops, where "the one in the middle" stops
            being obvious. Forced on for OFF-TRACK frames, where the ring marks the
            held position rather than a detected cell.
    """
    if (track_id is None) == (track_ids is None):
        raise ValueError(
            "give exactly one of track_id (follow ONE mask) or track_ids (follow a "
            "member set). Note track_ids=[N] is not track_id=N: the list form adds "
            "N's recorded daughters."
        )
    from mcp.types import TextContent

    if track_ids is not None:
        if not track_ids:
            raise ValueError("track_ids is empty; give at least one track.")
        members, added = _resolve_family(well, track_ids)
        header, images = _cm._family_filmstrip_frames(
            well, members, start_frame, end_frame,
            max_images=max_images, crop_um=crop_um,
            color=color, scale_bar=scale_bar, marker=marker,
            before_min=before_min, after_min=after_min, stride_min=stride_min,
            cap=MAX_IMAGES, added=added, centre_frame=centre_frame,
        )
    else:
        header, images = _cm._filmstrip_frames(
            well, int(track_id), start_frame, end_frame,
            max_images=max_images,
            crop_um=60.0 if crop_um is None else crop_um,
            color=color, scale_bar=scale_bar, marker=marker,
            stride_min=stride_min, cap=MAX_IMAGES,
        )
    out: list = [TextContent(type="text", text=header)]
    out.extend(_encode(img) for img in images)
    return out



__all__ = [
    "_FAMILY_NEARBY_UM", "_FAMILY_NEARBY_BEFORE_MIN", "_FAMILY_NEARBY_AFTER_MIN",
    "_CHAIN_GAP_MIN", "_CHAIN_MARGIN_UM", "_CHAIN_AMBIGUOUS_RATIO",
    "_nearest_detection", "watch_location_over_time", "_resolve_family",
    "_walk_chain", "resolve_lineage_chain",
    "list_nearby_tracks", "follow_cells_over_time",
]
