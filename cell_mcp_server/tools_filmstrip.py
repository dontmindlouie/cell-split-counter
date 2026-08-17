"""Filmstrip-based tools: watch_location_over_time, list_nearby_tracks,
follow_cells_over_time, and the family-resolution helper they share.

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import itertools

import numpy as np

from .server import server, MAX_IMAGES, _WINDOW_BEFORE_MIN, _WINDOW_AFTER_MIN, _STRIDE_MIN
from .io import _frame_at_offset_min, _minutes_between
from .render import _encode, _nearest_detection

import cell_mcp_server as _cm

# _manifest, _tracks, _hours, _lineage, _filmstrip_frames, and _family_filmstrip_frames
# below go through `_cm.` rather than a direct import -- see the note at the top of io.py.

_FAMILY_NEARBY_UM = 14.0
_FAMILY_NEARBY_BEFORE_MIN = 15.0
_FAMILY_NEARBY_AFTER_MIN = 75.0

# trace_division's widen-on-weak retry (2026-08-17 field feedback, well
# 20251016_ACTB_M1 track 12): a default-radius/window search finding nothing, or
# finding only short-lived candidates, isn't necessarily a real terminal -- an
# unhealthy/tripolar-looking division can throw off a cluster of 1-2-frame
# tracking artifacts alongside (or well before) the real, eventually long-lived
# daughters, and the daughters can sit outside the default reach. Confirmed on
# real data: track 12's own real answer (828/829, both 330+ frames) is only
# reachable a widened search away from an intermediate hop (558), and the exact
# same short-lived-artifact-cluster signature recurs AT the real split itself
# (827/830/838/839, all 1 frame, born alongside 828/829). See
# _split_candidates' docstring and trace_division's for the full trail.
_TRACE_WIDEN_RADIUS_MULT = 2.2
_TRACE_WIDEN_AFTER_MULT = 2.0

@server.tool(structured_output=False)
def watch_location_over_time(
    well: str,
    start_frame: int,
    end_frame: int,
    x: float | None = None,
    y: float | None = None,
    anchor_track_id: int | None = None,
    waypoints: list[tuple[int, float, float]] | None = None,
    max_images: int = 8,
    crop_um: float = 90.0,
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
    - **The thing you care about DRIFTS.** A single fixed x/y can go off-screen
      partway through (2026-08-16 field feedback: post-fragmentation debris moved
      ~130um over ~100 frames). Give `waypoints` instead -- a list of
      `(frame, x, y)` checkpoints (get their positions the same way you would
      probe by hand: list_nearby_tracks/measure at a few points along the drift)
      -- and the crop centre interpolates between them.

    Coordinates are PIXELS in the full frame, matching cx/cy in tracks.csv and the
    xy columns of list_tracks -- not microns, and not crop-relative.

    Nothing is ringed, because nothing here is claimed to be a detection. Instead
    every frame reports the nearest tracked cell and how far away it is, so you can
    tell "this is track 2036" from "there is nothing tracked within 12 um of here".

    Args:
        well: well name from list_wells().
        start_frame, end_frame: inclusive frame range.
        x, y: the position to watch, in full-frame pixels. Required unless
            anchor_track_id or waypoints is given.
        anchor_track_id: instead of a fixed point, follow THIS track's centroid --
            useful as a stable vantage on a neighbour. Where the anchor is missing
            from a frame, its closest known position is used.
        waypoints: instead of one fixed x/y, a list of `(frame, x, y)` checkpoints
            (at least 2) to interpolate the crop centre between, for a drifting
            untracked object -- clamped to the first/last checkpoint outside
            their frame span. Takes priority over x/y if both are given.
        max_images: how many frames to show (hard capped at 12).
        crop_um: crop width in micrometres. Defaults wide (90) so a neighbour
            cell doesn't fall out of frame or shrink to a sliver at the edge --
            narrow this only once you already know what's nearby.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
    """
    from mcp.types import TextContent

    header, images = _cm._fixed_point_frames(
        well, start_frame, end_frame,
        None if waypoints else x, None if waypoints else y, anchor_track_id,
        max_images=max_images, crop_um=crop_um,
        color=color, scale_bar=scale_bar, cap=MAX_IMAGES,
        waypoints=waypoints,
    )
    out: list = [TextContent(type="text", text=header)]
    out.extend(_encode(img) for img in images)
    return out


_CHAIN_GAP_MIN = 10.0
_CHAIN_MARGIN_UM = 14.0
_CHAIN_AMBIGUOUS_RATIO = 1.3


def _walk_chain(well: str, track_id: int, direction: str = "forward",
                max_hops: int = 6, seen: set[int] | None = None,
                *, first=None, last=None, med_area=None, um_px: float | None = None,
                ) -> list[dict]:
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

    first/last/med_area/um_px: pass these through when the caller already has
    them (they're all a sort_values+groupby over the whole tracks table).
    trace_division calls this repeatedly per generation, and re-deriving them
    from scratch every time was showing up as real, avoidable cost on a large
    well -- None (default) computes them here as before.
    """
    if first is None or last is None or um_px is None:
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


def _new_starts_near(
    first, last, um_px: float, x0: float, y0: float, r_um: float, lo: int, hi: int,
) -> tuple[list[int], dict[int, tuple[int, int]]]:
    """Track ids that BEGIN within [lo, hi] and within r_um of (x0, y0), plus their
    spans -- the candidate pool list_nearby_tracks reports for a human to judge by
    hand, factored out so resolve_division can judge it automatically with the
    same test.

    `first`/`last` are this well's tracks.csv grouped by track_id
    (`_tracks(well).sort_values("frame").groupby("track_id").first()`/`.last()`) --
    passed in rather than recomputed here, since every caller already has them
    (both callers also need the full sort+groupby for their own reporting).
    """
    r_px = r_um / um_px
    pool = first[(first.frame >= lo) & (first.frame <= hi)]
    near = pool[((pool.cx - x0) ** 2 + (pool.cy - y0) ** 2) <= r_px * r_px]
    ids = [int(t) for t in near.index]
    spans = {t: (int(first.loc[t, "frame"]), int(last.loc[t, "frame"])) for t in ids}
    return ids, spans


def _split_candidates(
    well: str, end_id: int, *, before_min: float, after_min: float,
    radius_um: float | None, max_hops: int,
    first, last, med, um_px: float, seen: set[int],
) -> dict:
    """One generation of resolve_division's own logic, factored out so
    trace_division can call it repeatedly without re-deriving track_id's
    position/window bookkeeping each time.

    `seen` is mutated with every id absorbed into a candidate's own chain walk
    (mirrors resolve_division's own `seen` set) so a caller looping across
    generations never re-offers an id already claimed by an earlier one.

    Returns a dict: {"daughters": [...] or None (terminal -- no split here),
    "chains": {cand: [chain]}, "spans": {...}, "lo", "hi", "r_um",
    "n_checked", "extra_groups": [...], "leftover": [...]}.
    """
    x0, y0 = float(last.loc[end_id, "cx"]), float(last.loc[end_id, "cy"])
    f0 = int(last.loc[end_id, "frame"])
    a0 = float(med.loc[end_id]) if med is not None and end_id in med.index else 0.0
    r_um = radius_um if radius_um else float(np.sqrt(max(a0, 1.0) / np.pi)) + _FAMILY_NEARBY_UM
    lo = _frame_at_offset_min(well, f0, -before_min)
    hi = _frame_at_offset_min(well, f0, after_min)
    ids, spans = _new_starts_near(first, last, um_px, x0, y0, r_um, lo, hi)
    ids = [t for t in ids if t not in seen]

    result = dict(daughters=None, chains={}, spans=spans, lo=lo, hi=hi, r_um=r_um,
                  n_checked=len(ids), extra_groups=[], leftover=[], ambiguous={})
    if not ids:
        return result

    chains: dict[int, list[int]] = {}
    absorbed: set[int] = set()
    for t in sorted(ids, key=lambda t: spans[t][0]):
        if t in absorbed:
            continue
        hops = _walk_chain(well, t, direction="forward", max_hops=max_hops, seen=set(seen),
                           first=first, last=last, med_area=med, um_px=um_px)
        full = [t, *(h["track_id"] for h in hops)]
        chains[t] = full
        seen.update(full)
        absorbed.update(full[1:])
        if hops:
            spans[t] = (spans[t][0], hops[-1]["frames"][1])
    ids = [t for t in ids if t not in absorbed]

    parent = {t: t for t in ids}

    def _find(t: int) -> int:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in itertools.combinations(ids, 2):
        (a_lo, a_hi), (b_lo, b_hi) = spans[a], spans[b]
        if a_lo <= b_hi and b_lo <= a_hi:
            _union(a, b)

    groups: dict[int, list[int]] = {}
    for t in ids:
        groups.setdefault(_find(t), []).append(t)
    siblings = [g for g in groups.values() if len(g) >= 2]
    if not siblings:
        result["leftover"] = ids
        return result

    best = max(siblings, key=len)
    result["daughters"] = sorted(best)
    result["chains"] = chains
    result["extra_groups"] = [g for g in siblings if g is not best]
    result["leftover"] = [t for t in ids if not any(t in c for c in siblings)]

    ended_nearby = last[(last.frame >= lo) & (last.frame <= hi)]
    own_chain = {m for t in best for m in chains[t]} | {end_id}
    ambiguous: dict[int, list[tuple[int, float]]] = {}
    for t in sorted(best):
        fx, fy = float(first.loc[t, "cx"]), float(first.loc[t, "cy"])
        for other_id, row in ended_nearby.iterrows():
            oid = int(other_id)
            # Anything already in `seen` is either this branch's own hop-chain
            # ancestry (the id-hops that led here) or another sibling already
            # absorbed elsewhere in the same trace -- not an independent track a
            # daughter could actually be confused with, so flagging it is just
            # noise. Real ambiguity is a track this call has never touched.
            if oid == end_id or oid in own_chain or oid in seen:
                continue
            d_um = float(np.hypot(row.cx - fx, row.cy - fy)) * um_px
            if d_um <= r_um:
                ambiguous.setdefault(t, []).append((oid, d_um))
    result["ambiguous"] = ambiguous
    return result


def _is_weak_split(res: dict, short_lived_frames: int) -> bool:
    """True when a _split_candidates result isn't solid enough to trust as a
    confirmed terminal -- either nothing resolved, or every daughter found is
    short-lived. A cluster of only short-lived candidates is exactly what a
    tracking-noise artifact around a real division looks like too (see
    _TRACE_WIDEN_RADIUS_MULT), so trace_division treats this the same as "found
    nothing" and tries a wider search before giving up.
    """
    if not res["daughters"]:
        return True
    return all(res["spans"][d][1] - res["spans"][d][0] + 1 < short_lived_frames
               for d in res["daughters"])


@server.tool()
def trace_division(
    well: str, track_id: int,
    before_min: float = _FAMILY_NEARBY_BEFORE_MIN, after_min: float = _FAMILY_NEARBY_AFTER_MIN,
    radius_um: float | None = None, max_hops: int = 6, max_generations: int = 6,
    short_lived_frames: int = 3,
) -> str:
    """Walk a division through EVERY hop/fragment generation, not just one --
    resolve_division's own multi-generation extension. Free: no images.

    QUICK REF (long descriptions truncate ~1200 chars in, no [...] marker --
    read this first): a generation only counts as confirmed once at least one
    daughter clears `short_lived_frames` -- if every candidate found is
    short-lived (or nothing is found at all), this automatically retries once
    with a wider radius/window before giving up, tagged "WIDENED SEARCH" in the
    output (lower confidence, verify on the pixels). Fixed 2026-08-17 after well
    20251016_ACTB_M1 track 12 (real answer 828/829) stopped dead at a cluster of
    1-2-frame artifacts (554/558/561/562) -- the SAME short-lived-cluster
    signature turned out to recur right at the real split too (827/830/838/839,
    all 1 frame, alongside the real 828/829). If a branch still looks wrong
    after that, raise max_hops (each generation's own hop-chain walk) or
    hand-walk get_lineage as a cross-check.

    resolve_division answers "who coexists with THIS track's end" for one hop --
    not enough when a real division hides behind a wobbling id-hop chain first
    (see the track-12 case above). This tool repeats resolve_division's own
    coexistence+distance+size test generation after generation: collapse the
    current track's hop chain, look for a real split at its end, and if one
    resolves, repeat from EACH daughter. Stops a branch when no split resolves
    (a genuine terminal -- alive, tracked, or dead, not necessarily missing) or
    max_generations is hit.

    Every terminal branch is returned, including short-lived ones (fewer than
    `short_lived_frames` frames after its own chain collapses) -- flagged as
    LIKELY FRAGMENT but never silently dropped, because a fragment/micronucleus
    daughter carrying part of the outcome is exactly what got missed by hand.

    Args:
        well: well name from list_wells().
        track_id: the track to trace forward from -- usually a mother, or any
            point earlier in a known hop chain.
        before_min, after_min: how far either side of each generation's end
            frame to look for daughters, in minutes. Same defaults as
            resolve_division.
        radius_um: search radius per generation. None = each track's own
            radius + 14 um.
        max_hops: cap on each candidate's own hop-chain walk, per generation.
        max_generations: cap on how many real-split generations to walk forward
            before stopping regardless of what is still splitting -- a runaway
            fragmenting cell (necrosis) could otherwise keep "splitting" for a
            very long chain.
        short_lived_frames: dual role, fixed 2026-08-17. (1) a terminal branch
            with fewer frames than this is flagged LIKELY FRAGMENT rather than
            reported as an ordinary terminal, as before. (2) a GENERATION only
            counts as confirmed once at least one of its daughters clears this
            bar -- if every candidate found there is shorter, that generation is
            treated as unresolved and retried once with a wider search
            (_TRACE_WIDEN_RADIUS_MULT/_TRACE_WIDEN_AFTER_MULT), tagged WIDENED
            SEARCH if that's what it took.
    """
    df = _cm._tracks(well)
    if df.empty or track_id not in df.track_id.values:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    um_px = _cm._manifest(well)["pixel_size_um"]
    srt = df.sort_values("frame")
    first = srt.groupby("track_id").first()
    last = srt.groupby("track_id").last()
    has_area = "area_um2" in srt.columns
    med = srt.groupby("track_id").area_um2.median() if has_area else None

    seen = {int(track_id)}
    # Collapse the seed's own hop chain first, same as resolve_division does.
    hops0 = _walk_chain(well, track_id, direction="forward", max_hops=max_hops, seen=set(seen),
                        first=first, last=last, med_area=med, um_px=um_px)
    chain0 = [int(track_id), *(h["track_id"] for h in hops0)]
    seen.update(chain0)

    generations: list[list[dict]] = []  # per generation: list of split-event dicts
    terminals: list[tuple[int, list[int], tuple[int, int]]] = []  # (parent_end, chain, span)
    frontier = [(int(track_id), chain0)]
    gen = 0
    while frontier and gen < max_generations:
        gen += 1
        next_frontier: list[tuple[int, list[int]]] = []
        gen_events: list[dict] = []
        # Each sibling's OWN current position, for the widen-retry's ownership check
        # below -- computed once per generation from the frontier as it stood before
        # this generation's per-parent loop starts extending chains.
        sibling_pos = {p: (float(last.loc[p, "cx"]), float(last.loc[p, "cy"]))
                       for p, _ in frontier if p in last.index}
        for parent, chain in frontier:
            # A daughter found in the PREVIOUS generation is only chain-collapsed as
            # far as it was walked when first discovered as a sibling candidate --
            # if it then keeps wobbling through further solo re-acquisitions (no
            # coexistence, so _split_candidates alone would call it terminal after
            # one hop), extend it again here before testing for a real split. This
            # is what lets a long id-hop run (562->...->800 on ACTB track 12) get
            # walked all the way to its real split instead of stopping at the first
            # single hop past the previous generation's candidate search.
            more_hops = _walk_chain(well, chain[-1], direction="forward",
                                     max_hops=max_hops, seen=set(seen),
                                     first=first, last=last, med_area=med, um_px=um_px)
            if more_hops:
                chain = [*chain, *(h["track_id"] for h in more_hops)]
                seen.update(h["track_id"] for h in more_hops)
            end_id = chain[-1]
            seen_trial = set(seen)
            res = _split_candidates(
                well, end_id, before_min=before_min, after_min=after_min,
                radius_um=radius_um, max_hops=max_hops,
                first=first, last=last, med=med, um_px=um_px, seen=seen_trial,
            )
            seen.update(seen_trial)
            widened = False
            if _is_weak_split(res, short_lived_frames):
                # Default search found nothing, or found only short-lived candidates --
                # either could be a real terminal, or could be the same short-lived-
                # artifact-cluster signature that surrounds a real division in this kind
                # of data (see _TRACE_WIDEN_RADIUS_MULT above). Try once more, wider,
                # from a FRESH seen copy (not seen_trial) so the widened search sees the
                # full candidate pool at once rather than one already thinned by the
                # first attempt's absorptions -- a candidate the default search grouped
                # as a leftover singleton needs to be visible again to pair up with
                # whatever the wider net additionally finds.
                base_r_um = radius_um if radius_um else res["r_um"]
                seen_wide = set(seen)
                res_wide = _split_candidates(
                    well, end_id, before_min=before_min, after_min=after_min * _TRACE_WIDEN_AFTER_MULT,
                    radius_um=base_r_um * _TRACE_WIDEN_RADIUS_MULT, max_hops=max_hops,
                    first=first, last=last, med=med, um_px=um_px, seen=seen_wide,
                )
                # NOT merged into the live `seen` yet -- only once this widened
                # attempt is actually accepted below. A rejected (stolen) widened
                # attempt still chain-walks and absorbs every candidate it touches
                # into seen_wide; committing that unconditionally would permanently
                # exclude a sibling's own real children from ITS later, legitimate
                # search in this same generation -- confirmed as a second real
                # regression while fixing the first (track 10's rejected widened
                # attempt was starving track 11's correct one).
                # A wider net can reach past this branch's own territory into a
                # SIBLING's -- confirmed as a real regression on the test fixture:
                # widening track 10's empty search (nothing within the default
                # radius) pulled in track 11's own real children, because they sat
                # only 140.8um away against a 154um widened radius. Refuse the
                # widened result if any of its daughters started closer to (or
                # comparably close to) another sibling in THIS generation's own
                # frontier than made sense -- that candidate likely belongs to the
                # sibling, not this branch, and a wrong widened guess is worse than
                # the branch staying an ordinary (still-reported) terminal.
                stolen = False
                if not _is_weak_split(res_wide, short_lived_frames):
                    self_x, self_y = float(last.loc[end_id, "cx"]), float(last.loc[end_id, "cy"])
                    for d in res_wide["daughters"]:
                        dx, dy = float(first.loc[d, "cx"]), float(first.loc[d, "cy"])
                        dist_self = np.hypot(dx - self_x, dy - self_y)
                        for sib, (sx, sy) in sibling_pos.items():
                            if sib == parent:
                                continue
                            if np.hypot(dx - sx, dy - sy) < dist_self:
                                stolen = True
                                break
                        if stolen:
                            break
                    if not stolen:
                        res, widened = res_wide, True
                if not stolen:
                    seen.update(seen_wide)
            if not res["daughters"]:
                terminals.append((parent, chain, (int(first.loc[end_id, "frame"]),
                                                    int(last.loc[end_id, "frame"]))))
                continue
            gen_events.append({"parent": parent, "end": end_id, "widened": widened, **res})
            for d in res["daughters"]:
                next_frontier.append((d, res["chains"][d]))
        if gen_events:
            generations.append(gen_events)
        frontier = next_frontier
    for parent, chain in frontier:  # hit max_generations while still splitting
        end_id = chain[-1]
        terminals.append((parent, chain, (int(first.loc[end_id, "frame"]),
                                            int(last.loc[end_id, "frame"]))))

    lines = [f"{well} track {track_id}: traced {len(generations)} generation(s) of "
             f"real splits, {len(terminals)} terminal branch(es)."]
    for g, events in enumerate(generations, 1):
        lines.append(f"\nGeneration {g}:")
        for ev in events:
            who = ", ".join(str(d) for d in ev["daughters"])
            widened_note = (" -- WIDENED SEARCH: default radius/window found nothing "
                             "solid here, this resolved only after retrying wider "
                             f"({_TRACE_WIDEN_RADIUS_MULT:g}x radius, "
                             f"{_TRACE_WIDEN_AFTER_MULT:g}x after_min) -- lower "
                             "confidence than an ordinary generation, verify on the "
                             "pixels" if ev["widened"] else "")
            lines.append(f"  {ev['parent']} (via chain end {ev['end']}) -> {who}{widened_note}")
            for d in ev["daughters"]:
                chain_note = (f" (chain: {ev['chains'][d]})" if len(ev["chains"][d]) > 1 else "")
                sp = ev["spans"][d]
                lines.append(f"    {d}: f{sp[0]}-{sp[1]}{chain_note}")
            if ev["extra_groups"]:
                lines.append(f"    {len(ev['extra_groups'])} further coexisting group(s) "
                              f"nearby but not included: "
                              + "; ".join(str(sorted(c)) for c in ev["extra_groups"]) + ".")
            if ev["ambiguous"]:
                for t, others in ev["ambiguous"].items():
                    near = ", ".join(f"track {o} ({d:.0f} um away)" for o, d in others)
                    lines.append(f"    OWNERSHIP AMBIGUITY: {t} is also within "
                                  f"{ev['r_um']:.0f} um of {near}'s last position -- "
                                  f"could belong to that track instead of {ev['parent']}.")

    lines.append(f"\nTerminal branches ({len(terminals)}):")
    for parent, chain, span in sorted(terminals, key=lambda x: x[1][-1]):
        n_frames = span[1] - span[0] + 1
        chain_note = f" (chain: {chain})" if len(chain) > 1 else ""
        flag = " -- LIKELY FRAGMENT, verify before trusting" if n_frames < short_lived_frames else ""
        lines.append(f"  {chain[-1]}: f{span[0]}-{span[1]}, {n_frames} frame(s){chain_note}{flag}")

    # `seen` already accumulates every id this walk ever touched -- each generation's
    # candidate chain-collapse updates it regardless of which candidates end up
    # chosen as daughters -- so it's already the full set, no need to re-derive it
    # by re-walking terminals/generations/chains.
    all_members = sorted(seen)
    lines.append(f"\nAll members traced: {all_members}. This is bookkeeping, not a judgement "
                 f"that a division happened at each step -- confirm on the pixels via "
                 f"follow_cells_over_time(track_ids={all_members}).")
    return "\n".join(lines)


@server.tool()
def resolve_division(
    well: str, track_id: int,
    before_min: float = _FAMILY_NEARBY_BEFORE_MIN, after_min: float = _FAMILY_NEARBY_AFTER_MIN,
    radius_um: float | None = None, max_hops: int = 6,
) -> str:
    """Resolve a REAL split -- a track ending in two or more coexisting daughters
    the tracker never linked -- automatically. Free: no images.

    Most divisions found by hand are NOT id hops (that's resolve_lineage_chain's
    job): they are genuine splits into 2+ coexisting daughters that lineage.csv
    simply never recorded, because the tracker's link breaks exactly at the
    division. Finding them by hand means list_nearby_tracks() then judging
    coexistence yourself, often 3-4 hops deep -- this was the single most repeated
    manual workflow in a 2026-08-15 review session. This tool runs that same
    coexistence+distance+size test automatically and hands back the assembled
    mother+daughters set.

    THE TEST IS THE SAME ONE list_nearby_tracks and resolve_lineage_chain use:
    a candidate must be NEW near where this track ends, close in space -- and,
    what makes a sibling pair rather than one object being re-acquired, two
    candidates must COEXIST (on screen at the same time). Candidates are first
    chain-walked forward (resolve_lineage_chain's own logic) so a daughter that
    itself wobbles mid-life collapses into one physical object before the
    coexistence test runs, rather than being mistaken for a second daughter.

    If nothing nearby coexists with anything else, this says so rather than
    guessing -- that pattern (a run of non-overlapping candidates at one spot) is
    itself the signature of an id hop, and resolve_lineage_chain is the right tool
    for it instead.

    Also flags OWNERSHIP AMBIGUITY: a resolved daughter that sits within this same
    search radius of some OTHER track's last position in the window too. On
    2026-08-16 field feedback a real division 40-50 um from one track's last spot
    actually belonged to a different nearby track's granddaughter, and the mixup
    was only caught by a slow manual watch_location_over_time verification loop.
    When this fires, confirm ownership before trusting the attribution.

    Args:
        well: well name from list_wells().
        track_id: the track whose split you want resolved -- usually a mother
            whose own filmstrip goes OFF-TRACK with get_lineage showing no
            daughters recorded.
        before_min, after_min: how far either side of this track's last frame to
            look for daughters, in minutes. Forward-heavy by default (daughters
            appear AFTER the link breaks, and on BeWo the mitotic figure can be
            ~20 min past it) -- same reasoning as list_nearby_tracks.
        radius_um: search radius. None = this track's own radius + 14 um.
        max_hops: cap on each candidate's own hop-chain walk.
    """
    df = _cm._tracks(well)
    if df.empty or track_id not in df.track_id.values:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    um_px = _cm._manifest(well)["pixel_size_um"]
    srt = df.sort_values("frame")
    first = srt.groupby("track_id").first()
    last = srt.groupby("track_id").last()
    has_area = "area_um2" in srt.columns
    med = srt.groupby("track_id").area_um2.median() if has_area else None

    seen = {track_id}
    res = _split_candidates(
        well, track_id, before_min=before_min, after_min=after_min,
        radius_um=radius_um, max_hops=max_hops,
        first=first, last=last, med=med, um_px=um_px, seen=seen,
    )
    lo, hi, r_um = res["lo"], res["hi"], res["r_um"]

    if res["n_checked"] == 0:
        f0 = int(last.loc[track_id, "frame"])
        return (f"{well} track {track_id}: nothing new starts within {r_um:.0f} um of "
                f"its last position (f{f0}) in f{lo}-{hi}. No split resolves here -- if "
                f"the track just loses its id and comes back, try "
                f"resolve_lineage_chain(track_id={track_id}) instead.")

    if not res["daughters"]:
        checked = sorted(set(res["leftover"]))
        return (
            f"{well} track {track_id}: {res['n_checked']} candidate(s) start nearby "
            f"({', '.join(str(t) for t in checked)}) in f{lo}-{hi}, but none of them "
            f"COEXIST with each other -- that pattern is one object being "
            f"re-acquired, not a split. Try "
            f"resolve_lineage_chain(track_id={track_id}) instead, or inspect by "
            f"hand with list_nearby_tracks(track_id={track_id})."
        )

    best, chains = res["daughters"], res["chains"]
    members = [int(track_id), *sorted(m for t in best for m in chains[t])]

    lines = [
        f"{well} track {track_id}: resolved as a split into "
        f"{len(best)} coexisting daughter(s), from {res['n_checked']} candidate(s) checked."
    ]
    for t in sorted(best):
        chain_note = (f" (chain: {chains[t]})" if len(chains[t]) > 1 else "")
        lines.append(f"  {t}: f{res['spans'][t][0]}-{res['spans'][t][1]}{chain_note}")
    if res["ambiguous"]:
        lines.append(
            "\nOWNERSHIP AMBIGUITY -- verify before attributing these to "
            f"{track_id}:")
        for t, others in res["ambiguous"].items():
            near = ", ".join(f"track {o} ({d:.0f} um away)" for o, d in others)
            lines.append(
                f"  {t} is also within {r_um:.0f} um of {near}'s last position in "
                f"f{lo}-{hi} -- could belong to that track instead. Check with "
                f"get_lineage(track_id={t}) or resolve_division on the other "
                f"track before trusting this attribution.")
    if res["extra_groups"]:
        lines.append(
            f"\n{len(res['extra_groups'])} further coexisting group(s) also found nearby "
            f"but not included -- likely unrelated neighbours dividing in the same window: "
            + "; ".join(str(sorted(c)) for c in res["extra_groups"]) + "."
        )
    if res["leftover"]:
        lines.append(
            f"{len(res['leftover'])} candidate(s) did not coexist with anything and were "
            f"dropped: {', '.join(str(t) for t in res['leftover'])}."
        )
    lines.append(
        f"\nStitched set: {members}. This is bookkeeping, not a judgement that a "
        f"division happened -- confirm on the pixels via "
        f"follow_cells_over_time(track_ids={members})."
    )
    return "\n".join(lines)


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
        ids_new, _ = _new_starts_near(first, last, um_px, x0, y0, r_um, lo, hi)
        ids = [t for t in ids_new if t != tid]
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
    hold_centre_after_member_end: bool = False,
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
    a neighbour. The walk refuses to step onto a blob that belongs to an already-
    established track (one that started before this gap) -- fixed 2026-08-17 after
    field feedback found it locking onto a long-lived stable neighbour in 4/4 tries,
    never the real outcome; only a fresh detection (untracked, or a track that
    started at/after this boundary) is accepted. Where the walk loses the trail the
    crop freezes at the last resolved position (dashed blue ring) instead of
    guessing further. Read OFF-TRACK frames as "this patch of the field", not "this
    cell".

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
        crop_um: width of the crop in micrometres. None (recommended) means 90 for a
            single track -- a cell and its immediate neighbours -- and auto-fit
            (floor 90) for a member set, wide enough to hold every member across
            the whole window with room to see neighbours. Do not set it by hand
            on a member set unless you want a
            particular zoom: guessing is how a sibling drifts out of frame halfway
            along, since separation grows as the daughters move apart.
        color: apply the microscope's own display colour.
        scale_bar: burn in a labelled scale bar.
        marker: draw a thin ring around the tracked cell. OFF BY DEFAULT and should
            STAY off for ordinary review -- the ring is one more shape in an image
            whose shapes ARE the evidence, and adding it out of habit is how a real
            feature (a fold, a neighbouring blob) gets mistaken for an annotation.
            Reach for this only in the narrow case where you need to POINT OUT one
            specific cell among several similar-looking ones in a crowded or wide
            crop -- e.g. handing a page to someone else with no other way to say
            "this one, not that one". With track_ids, rings every member present in
            each frame or none at all. Forced on for OFF-TRACK frames regardless of
            this flag, where the ring marks the held position rather than a
            detected cell.
        hold_centre_after_member_end: with a member set, freeze that member's LAST
            position into the crop's mean forever after its span ends, instead of
            dropping it and letting the mean jump to whoever remains. OFF by
            default -- most callers want the mean to move on once a member is
            confirmed gone, e.g. the mother-to-daughters handoff. Turn this on when
            you have already confirmed (via watch_location_over_time) that a member
            genuinely vanished and you want the framing to hold on the spot it left
            rather than snap to the survivors, which otherwise reads as an
            unexplained jump. Member sets only.
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
            hold_centre_after_member_end=hold_centre_after_member_end,
        )
    else:
        header, images = _cm._filmstrip_frames(
            well, int(track_id), start_frame, end_frame,
            max_images=max_images,
            crop_um=90.0 if crop_um is None else crop_um,
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
    "_new_starts_near", "resolve_division", "_split_candidates", "trace_division",
    "list_nearby_tracks", "follow_cells_over_time",
]
