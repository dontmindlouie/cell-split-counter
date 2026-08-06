"""find_candidates and its supporting machinery: condensation scoring, collapse-site
grouping, division strata, sort/pool helpers, and the notes appended to output.

Split out of the original single-file cell_mcp.py's "tools" section.
"""

import numpy as np
import pandas as pd

from .server import server
from .io import _frame_at_offset_min, _minutes_between, _edge_um

import cell_mcp_server as _cm

# BUNDLE, _manifest, and _tracks below go through `_cm.` rather than a direct
# import -- see the note at the top of io.py.

_COND_BEFORE_MIN = 20.0
_COND_AFTER_MIN = 60.0

# The conservation gate: a frame only counts as a condensation candidate if the family
# still holds roughly the DNA the mother had. Asymmetric because the two ends fail
# differently -- below the floor, signal has genuinely gone missing and whatever is
# left is a fragment, not a compacted nucleus; above the ceiling, a neighbour has been
# swept into the family. The ceiling is loose on purpose: once both daughters are
# segmented the family legitimately reads more signal than the mother alone did.
# Margin added to the mother's own radius to make the measuring disc.
_COND_MARGIN_UM = 8.0

# How much of the mother's own recent history forms the baseline. An hour, not her
# whole track: raw signal bleaches over 72 h, so a lifetime median makes a late
# division look like it lost DNA it never lost.
_COND_BASE_MIN = 60.0
_COND_DNA_MIN = 0.75
_COND_DNA_MAX = 2.5


def _condensation(well: str, rows, lin, tracks, um_px: float) -> tuple[list, list, list, list]:
    """Score how strongly each recorded division shows CONDENSING CHROMATIN.

    Every other signal in this file is topological -- who the tracker linked to whom,
    whether a daughter persisted. That is what the 2026-07-31 blind scoring showed to
    be the wrong question on BeWo: where the tracker fails THROUGH the division, the
    recorded daughters are pre-mitotic debris and "daughter persists" is anti-correlated
    with a division having happened. A human scored the same events on morphology --
    is there a metaphase plate -- and disagreed with topology on 7 of 11 BeWo cases.
    This is that morphology question, asked of numbers the bundle already has.

    Mitosis packs the same DNA into a smaller object. So across the transition:

        area          FALLS      chromatin compacts
        mean intensity RISES     same signal, fewer pixels
        integrated     ~FLAT     no DNA was created or destroyed

    The SCORE is the brightness rise, and the conservation is a GATE on it. That
    split is the whole design, and the obvious alternative is wrong: scoring
    (brightness up) x (area down) ranked M12's fragments first, at cond 93 with only
    65% of the DNA still present. Both fragmentation and condensation shrink the
    area, so any factor of a0/area rewards a mask falling apart -- and a fragment
    shrinks the total signal in step with the area, leaving brightness per pixel
    FLAT. That is precisely what tells the two apart:

        condensation  area down, brightness UP, total conserved
        fragment      area down, brightness flat, total DOWN
        death/bleach  brightness DOWN

    So: look only at frames where the family still holds its DNA, and among those
    take the brightest. Area is reported, never multiplied in.

    Brightness is measured RELATIVE TO THE FIELD in the same frame, because the
    signal bleaches over 72 h -- an absolute rise late in a recording is a bigger
    deal than the same rise early, and a mother whose baseline spans hours would
    otherwise be compared against her own brighter past.

    Scored against the mother's OWN history, never an absolute cutoff -- the lesson
    from `solidity`, which is the strongest shape signal on compact RUES2 nuclei and
    the weakest on lobed WGD ones. A ratio to her own median says the same thing on
    both.

    Measured over a NEIGHBOURHOOD, not over the recorded family. That is the second
    thing this got wrong and the more important one. Summing the mother and her
    recorded daughters cannot see the figure at all in the cases that matter: on BeWo
    track 802 the family has rows in only 10 of the 28 window frames, because the
    tracker lost the cell at the link and the condensed object at f778 carries a track
    id nobody linked to anybody. Scoring the family returned NaN there -- on the exact
    event a human called an unmistakable prometaphase figure.

    So the window is a disc around the mother's last known position, and everything
    segmented inside it is summed. That is what makes the score independent of the
    tracking, which is the entire reason to want a morphology signal: if it needed the
    link to be right, it would fail wherever topology already fails, and those are the
    same events. The cost is neighbours drifting into the disc, so the count of objects
    summed is reported rather than hidden.

    Returns three parallel lists (peak score, the frame it peaked at, DNA conservation
    at that frame). NaN where the mother has too little history to have a baseline --
    a one-frame mother has no "normal" to be compared against, and inventing one would
    manufacture the signal this is supposed to measure.

    HOW WELL IT ACTUALLY WORKS, measured against the only human labels this project
    has (maintainer review, 2026-07-31, 26 blind-scored divisions across two lines):

        Line A M12  AUC 0.63   (3 real / 9 not)
        Line B M2   AUC 0.75   (4 real / 10 not)
        combined    AUC 0.68

    That is a weak ranking, and it is stated rather than hidden because the number
    is the point: on BeWo, topology scored BELOW 0.5 on the same events -- the
    `clean` stratum held 0/4 real divisions and `vanishing_daughter` held 4/5. A 0.75
    that is independent of the tracker beats a ranking that is confidently backwards.
    It changes where a reviewer spends images; it does not decide anything.

    n=26 is far too small to tune against, so it has NOT been tuned against them --
    the thresholds are the first physically-argued values. Fitting them to 26 points
    would produce a better-looking number and a worse tool.

    It is a ranking, not a verdict: a condensed-looking object can be a dying cell
    whose chromatin clumped, which is a real and known confusion this cannot resolve.
    Confirm on the pixels.
    """
    need = {"area_um2", "area_px", "intensity_mean", "intensity_integrated"}
    if not need.issubset(tracks.columns):
        n = len(rows)
        return ([float("nan")] * n, [-1] * n, [float("nan")] * n,
                [float("nan")] * n)

    cols = ["track_id", "frame", "cx", "cy", "area_um2", "area_px",
            "intensity_mean", "intensity_integrated"]
    sub = tracks[cols].copy()
    # Bleach correction: every cell's brightness is expressed against the median cell
    # in ITS OWN frame, so a 72 h decay in the illumination or the dye cancels out.
    field = sub.groupby("frame").intensity_mean.median().replace(0, np.nan)
    sub["rel"] = sub.intensity_mean / sub.frame.map(field)
    by_track = {int(t): g for t, g in sub.groupby("track_id")}

    # Per-frame arrays, built once. The inner loop asks "what is near (x, y) in frame
    # f" ~30 times per division across ~1,300 divisions, and a pandas filter per
    # question turns seconds into minutes.
    per_frame: dict[int, tuple] = {}
    for f, g in sub.groupby("frame"):
        per_frame[int(f)] = (
            g.cx.to_numpy(), g.cy.to_numpy(), g.area_um2.to_numpy(),
            g.area_px.to_numpy(), g.rel.to_numpy(),
            g.intensity_integrated.to_numpy(),
        )

    scores, peaks, dnas, areas = [], [], [], []

    def _blank():
        scores.append(float("nan")); peaks.append(-1)
        dnas.append(float("nan")); areas.append(float("nan"))

    for r in rows.itertuples():
        mid = int(r.track_id)
        mrows = by_track.get(mid)
        # Fewer than 5 frames of mother is not a baseline, it is a guess.
        if mrows is None or len(mrows) < 5:
            _blank(); continue
        mrows = mrows.sort_values("frame")
        last = int(mrows.frame.iloc[-1])
        x0, y0 = float(mrows.cx.iloc[-1]), float(mrows.cy.iloc[-1])
        lo = _frame_at_offset_min(well, last, -_COND_BEFORE_MIN)
        hi = _frame_at_offset_min(well, last, _COND_AFTER_MIN)

        # Baseline is LOCAL -- the hour of this mother's own life immediately before
        # the window, not her whole track. It must be, because `intensity_integrated`
        # is raw and the signal bleaches: on BeWo track 793 the whole-track median put
        # every window frame at 0.60 of "baseline", the conservation gate rejected all
        # 28 of them, and a case a human read as prophase scored NaN. A cell is not
        # losing DNA because the recording started brighter than it ended. It also
        # excludes the tail, which is the very thing being measured against it.
        base = mrows[(mrows.frame < lo)
                     & (mrows.frame >= _frame_at_offset_min(well, lo, -_COND_BASE_MIN))]
        if len(base) < 3:
            base = mrows[mrows.frame < lo].tail(10)
        if len(base) < 3:
            base = mrows.iloc[:-3]
        a0 = float(base.area_um2.median())
        i0 = float(base.rel.median())
        s0 = float(base.intensity_integrated.median())
        if not (a0 > 0 and i0 > 0 and s0 > 0):
            _blank(); continue

        # Disc radius: the mother's own equivalent radius plus a fixed margin, so it
        # scales with the cell line. A BeWo nucleus is 2-3x a RUES2 one, and a fixed
        # radius would either clip the daughters apart on one line or sweep in the
        # neighbours on the other.
        r_um = float(np.sqrt(a0 / np.pi)) + _COND_MARGIN_UM
        r_px = r_um / um_px

        best = (float("-inf"), -1, float("nan"), float("nan"))
        for f in range(lo, hi + 1):
            pf = per_frame.get(f)
            if pf is None:
                continue
            cx, cy, a_um, a_px, rel, integ = pf
            near = ((cx - x0) ** 2 + (cy - y0) ** 2) <= r_px * r_px
            if not near.any():
                continue
            area_f = float(a_um[near].sum())
            sig_f = float(integ[near].sum())
            wpx = float(a_px[near].sum())
            if area_f <= 0 or wpx <= 0:
                continue
            # Area-weighted: an unweighted mean would let a 10 px fragment count as
            # much as the nucleus it broke off.
            imean_f = float((rel[near] * a_px[near]).sum()) / wpx
            dna = sig_f / s0
            # The gate. Outside this band the disc no longer holds the same DNA, so
            # whatever its brightness does is not condensation.
            if not (_COND_DNA_MIN <= dna <= _COND_DNA_MAX):
                continue
            val = imean_f / i0
            if val > best[0]:
                best = (val, f, dna, area_f / a0)

        if best[1] < 0:
            _blank(); continue
        scores.append(best[0]); peaks.append(best[1])
        dnas.append(best[2]); areas.append(best[3])

    return scores, peaks, dnas, areas


def _prophase_onset(well: str, track_id: int) -> dict | None:
    """Score condensation walking BACKWARD from a mother's own track start, through
    any predecessor id-hops resolve_lineage_chain resolves, to find where the rise
    actually began -- her first tracked frame is not necessarily where prophase
    started, only where THIS id started.

    Reuses two things that already exist rather than inventing new machinery:
    _walk_chain's backward walk (the same coexistence-tested hop-chasing built for
    a daughter's id hops after a division, applied in the other direction here),
    and _condensation's own DNA-conservation gate and field-normalised brightness
    ratio, scored over the resolved predecessor(s)' own tracked rows instead of a
    disc summed from tracks.csv's whole neighbourhood -- the predecessor IS the
    same physical object by construction, so there is no neighbourhood to sum.

    Returns None, honestly, when no predecessor track resolves. That is not "no
    prophase" -- it means condensation started before segmentation ever produced a
    tracked row here at all (a real case: on nTSC_ZO1_1-4_M1 a mother's rosette-
    pattern chromatin was visible before her track began and only findable by
    manually stepping watch_location_over_time backward). Scoring THAT case needs
    measuring raw label-image blobs with no track backing them, a different and
    larger feature this does not attempt -- it would have to reproduce this
    project's raw intensity calibration outside tracks.csv, and a wrong number
    presented with the same confidence as a real one is worse than admitting the
    gap. See resolve_lineage_chain for the tracked half of this same limit.
    """
    hops = _cm._walk_chain(well, track_id, direction="backward")
    if not hops:
        return None

    tracks = _cm._tracks(well)
    need = {"area_um2", "intensity_mean", "intensity_integrated"}
    if not need.issubset(tracks.columns):
        return None

    chain_ids = [int(track_id), *(h["track_id"] for h in hops)]
    sub = tracks[tracks.track_id.isin(chain_ids)].sort_values("frame")
    if len(sub) < 5:
        return None

    field = tracks.groupby("frame").intensity_mean.median().replace(0, np.nan)
    sub = sub.assign(rel=sub.intensity_mean / sub.frame.map(field))

    mrows = tracks[tracks.track_id == int(track_id)].sort_values("frame")
    first_f = int(mrows.frame.iloc[0])
    base_hi = _frame_at_offset_min(well, first_f, _COND_BASE_MIN)
    base = mrows[mrows.frame <= base_hi]
    if len(base) < 3:
        base = mrows.head(10)
    a0 = float(base.area_um2.median())
    i0 = float((base.intensity_mean / base.frame.map(field)).median())
    s0 = float(base.intensity_integrated.median())
    if not (a0 > 0 and i0 > 0 and s0 > 0):
        return None

    # Only the PREDECESSOR frames -- strictly before this track's own first frame --
    # are candidates. Her own history IS the baseline; scoring it against itself
    # would be circular and would always "find" prophase at her own noise floor.
    window = sub[sub.frame < first_f].copy()
    if window.empty:
        return None
    window["dna"] = window.intensity_integrated / s0
    window = window[(window.dna >= _COND_DNA_MIN) & (window.dna <= _COND_DNA_MAX)]
    if window.empty:
        return None
    window["val"] = window.rel / i0
    peak = window.loc[window.val.idxmax()]

    return {
        "prophase_frame": int(peak.frame), "score": float(peak.val),
        "dna": float(peak.dna), "chain": chain_ids,
    }


@server.tool()
def find_prophase_onset(well: str, track_id: int) -> str:
    """Look for where a mother's chromatin condensation actually BEGAN, earlier
    than her own track -- her first tracked frame is where THIS id started, not
    necessarily where the biology did.

    Chases the same id-hop chain resolve_lineage_chain does, backward from this
    track's first frame, then scores each resolved predecessor frame with the same
    brightness-rise/DNA-conservation test find_candidates uses for cond/cond_f --
    just walked the other direction. A ranking signal, same as cond_f: it names
    where to look, not a verdict that prophase happened there. Confirm on the
    pixels via follow_cells_over_time(track_id=..., centre_frame=<prophase_frame>).

    Give up nothing silently: if no predecessor track resolves at all, that is
    reported as its own case, not folded into "no prophase found" -- it means the
    signal may exist before ANY track here, visible only by stepping
    watch_location_over_time backward through the raw frames by eye.

    Args:
        well: well name from list_wells().
        track_id: the mother track, from a division candidate.
    """
    df = _cm._tracks(well)
    if df.empty or track_id not in df.track_id.values:
        raise ValueError(f"track {track_id} not found in {well}. Use list_tracks().")

    hops = _cm._walk_chain(well, track_id, direction="backward")
    if not hops:
        return (f"{well} track {track_id}: no predecessor track resolves backward, so "
                f"there is nothing here to score. This does NOT mean condensation began "
                f"at this track's own first frame -- it means checking earlier means "
                f"stepping watch_location_over_time backward through the raw frames by "
                f"eye, since no tracked row exists to measure.")

    result = _prophase_onset(well, track_id)
    chain_ids = [int(track_id), *(h["track_id"] for h in hops)]
    if result is None:
        return (f"{well} track {track_id}: {len(hops)} backward hop(s) resolve "
                f"(chain {chain_ids}), but no frame in them passes the same DNA-"
                f"conservation gate find_candidates uses for cond/cond_f -- either "
                f"nothing there was condensing, or the family lost too much signal to "
                f"score. Look at the chain yourself: follow_cells_over_time("
                f"track_ids={chain_ids}).")

    return (
        f"{well} track {track_id}: prophase_frame {result['prophase_frame']} "
        f"(score {result['score']:.2f}, DNA {result['dna']:.2f}x baseline), found by "
        f"walking backward through {len(hops)} resolved predecessor hop(s) "
        f"(chain {chain_ids}).\n"
        f"Ranking signal, not a verdict, same as cond_f -- confirm on the pixels: "
        f"follow_cells_over_time(track_id={chain_ids[-1]}, "
        f"start_frame={result['prophase_frame'] - 4}, "
        f"end_frame={result['prophase_frame'] + 4})."
    )


def _collapse_sites(well: str, rows, tracks) -> dict[int, list[int]]:
    """Group recorded divisions that are the SAME physical event.

    Returns {representative_mother: [the others]}.

    A tracker that fails through a division does not fail once. On BeWo M2, tracks
    1824 (f468-472), 1860 (f475-479) and 1883 (f479-496) are one cell at one spot,
    each recorded as its own division with its own daughters -- and they took three of
    the top five rows of a `cond`-ranked sample. A reviewer asked for five candidates
    and got three copies of one event: the review budget is spent three times on one
    answer, and any rate computed from the pool counts it three times.

    The rule is the one that settled the daughter question, applied to mothers: a
    mother whose track BEGINS where another mother's ENDED, within a short window, is
    the same cell re-acquired. Non-overlapping spans at one place is a broken track,
    not two cells -- BeWo does not divide twice in 84 minutes.

    But proximity alone folds real divisions too, because a genuine daughter also
    begins where her mother ended. What separates them is DURATION: a re-acquisition
    is over in minutes, a lineage takes hours. So a merge is allowed only while the
    whole group stays inside _SITE_MAX_SPAN_MIN.

    Union-find, because the relation chains: 1824 links 1860, 1860 links 1883, and all
    three must land in one group even though 1824 and 1883 are 24 frames apart.

    Nothing is discarded -- the members are returned so the caller can list them. A
    dropped row is a row nobody can audit, which is the failure that made events.csv
    unusable.
    """
    m = _cm._manifest(well)
    # Needs geometry, size and real timestamps. A bundle missing any of them gets no
    # fold rather than a silently wrong one -- every row stays its own site.
    ids0 = [int(t) for t in rows.track_id]
    if (not {"cx", "cy", "area_um2"}.issubset(tracks.columns)
            or not m.get("frame_timestamps_ms")):
        return {t: [] for t in ids0}
    um_px = m["pixel_size_um"]
    srt = tracks.sort_values("frame")
    first = srt.groupby("track_id").first()
    last = srt.groupby("track_id").last()
    med = srt.groupby("track_id").area_um2.median()

    ids = [int(t) for t in rows.track_id if t in last.index]
    parent = {t: t for t in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairs first, then a SPAN-CAPPED merge. Two knobs, and each was forced by a
    # failure:
    #
    # Proximity alone chained f140 to f697 through a whole lineage and folded 81% of
    # the pool, because a genuine daughter also begins where her mother ended -- so
    # A->B->C walks straight through real divisions. Gating on "do two new objects
    # coexist here" then folded almost NOTHING (3 rows of 1253): in a crowded BeWo
    # field some pair always coexists, so the gate fires everywhere.
    #
    # What actually separates the two is DURATION. A re-acquisition of one cell is
    # over in minutes; a lineage takes hours. So the merge is allowed only while the
    # whole group stays inside _SITE_MAX_SPAN_MIN, which is far shorter than any cell
    # cycle and long enough for a tracker to drop and regain an id several times.
    starts = first.loc[ids]
    sx, sy, sf = starts.cx.to_numpy(), starts.cy.to_numpy(), starts.frame.to_numpy()
    sid = np.array(ids)
    links = []
    for t in ids:
        fe = int(last.loc[t, "frame"])
        x0, y0 = float(last.loc[t, "cx"]), float(last.loc[t, "cy"])
        r_px = (float(np.sqrt(max(float(med.loc[t]), 1.0) / np.pi))
                + _SITE_RADIUS_UM) / um_px
        hi = _frame_at_offset_min(well, fe, _SITE_GAP_MIN)
        sel = ((sf >= fe) & (sf <= hi) & (sid != t)
               & (((sx - x0) ** 2 + (sy - y0) ** 2) <= r_px * r_px))
        for u in sid[sel]:
            links.append((fe, t, int(u)))

    span = {t: (int(first.loc[t, "frame"]), int(last.loc[t, "frame"])) for t in ids}
    gspan = {t: span[t] for t in ids}
    for _, a, b in sorted(links):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lo_f = min(gspan[ra][0], gspan[rb][0])
        hi_f = max(gspan[ra][1], gspan[rb][1])
        if _minutes_between(well, lo_f, hi_f) > _SITE_MAX_SPAN_MIN:
            continue
        union(a, b)
        gspan[find(a)] = (lo_f, hi_f)

    groups: dict[int, list[int]] = {}
    for t in ids:
        groups.setdefault(find(t), []).append(t)
    # Representative: the mother with the most frames -- the recording of this event
    # that saw the most of it.
    out: dict[int, list[int]] = {}
    for members in groups.values():
        rep = max(members, key=lambda t: int(last.loc[t, "frame"])
                  - int(first.loc[t, "frame"]))
        out[rep] = [t for t in members if t != rep]
    return out


# How close, and how long after, a mother must re-appear to count as the same cell.
_SITE_RADIUS_UM = 14.0
_SITE_GAP_MIN = 30.0
# A group may not span more than this. Far shorter than any cell cycle, long enough
# for a tracker to drop and regain one cell's id several times over.
_SITE_MAX_SPAN_MIN = 60.0


_STRATA = [
    # (name, blurb) in PRIORITY order -- first match wins, so the counts partition the
    # pool and sum to its total. Measurement defects are tested before biological
    # ambiguity, on the grounds that a clipped or merged mask makes every other number
    # on the row untrustworthy, so it is the more informative label to carry.
    ("merged_id", "mother or a daughter is a track the manifest flags as multiplexed"),
    ("edge_clipped", "mother within 15 um of the frame boundary; area and brightness understated"),
    ("touching", "mother or a daughter shares a mask with another cell in some frame"),
    ("far_link", "daughter linked from >25 px away; most likely an id swap"),
    ("fragment_like", "one daughter tiny while DNA is conserved: micronucleus signature"),
    ("vanishing_daughter", "a daughter lasts <5 frames"),
    ("dim_daughter", "a daughter averages <60% of the well's median brightness"),
    ("clean", "trips none of the above"),
]


def _kid_ids(s) -> list[int]:
    """The daughter ids in a lineage.csv `daughter_ids` cell, which is space-joined text."""
    return [int(k) for k in str(s).split() if k.strip().lstrip("-").isdigit()]


def _fmt2(v) -> str:
    """A score to 2dp, or '-' when it is missing. NaN counts as missing."""
    return "-" if v is None or v != v else f"{v:.2f}"


def _fmt0(v, blank: str = "-") -> str:
    """A whole number, or `blank` when it is missing."""
    return blank if v is None or v != v else f"{v:.0f}"


# A recorded daughter lasting this many frames or fewer is a stub, not an outcome.
# Shared by the vanishing_daughter stratum and by the ratio suppression in
# find_candidates so the two can never disagree about which rows are measurable.
# Counted INCLUSIVELY (last - first + 1), because a daughter that exists cannot last
# zero frames -- the stratum's own `dur < 5` is the same cut on an exclusive span.
_STUB_DAUGHTER_FRAMES = 5


def _daughter_spans(rows, lin) -> tuple[list[str], list[float]]:
    """Per row: how long each recorded daughter lasted, and the shorter of the two.

    Free -- already in lineage.csv. It separated the real divisions from the
    artifacts better than any of the other scored columns on two independent
    samples now: the original 5-case nTSC read (118/159, 77/77 real vs. 6/10,
    4/1, 1/2 artifact) and a second nTSC session 2026-08-06 (divisions 77-159
    frames, non-divisions 1-10). Two sessions, two cell lines' worth of looking,
    same split -- promoted from "printed as a fact" to an actual
    `find_candidates(sort_by="daughter_persistence")` sort on `dau_min`, this
    function's `worst` return. Still topology, not morphology: it inherits the
    same blind spot as fragment_like/dna_anomaly wherever the tracker fails
    THROUGH a division (see `_condensation`'s own docstring) -- it just has not
    been caught failing that way yet.
    """
    li = lin.set_index("track_id")
    dur = (li.last_frame - li.first_frame + 1).to_dict()
    spans, worst = [], []
    for r in rows.itertuples():
        kids = _kid_ids(r.daughter_ids)
        ds = [dur.get(k, float("nan")) for k in kids]
        spans.append("/".join("?" if d != d else f"{int(d)}" for d in ds) or "-")
        finite = [d for d in ds if d == d]
        worst.append(min(finite) if finite else float("nan"))
    return spans, worst


def _division_strata(rows, lin, tracks, m) -> list:
    """Label each recorded division with the FIRST artifact class it trips.

    Classifies; never discards. The reject rate per stratum is itself the finding,
    and a filter that silently drops rows is the exact failure that made events.csv
    unusable. Every signal here comes from tracks.csv + lineage.csv + manifest.json --
    no images, no model, seconds to run.

    The thresholds are UNVALIDATED. They were written from one session's look at one
    RUES2 well; treat each stratum as a hypothesis with a count attached, not as a
    verdict. In particular they are almost certainly wrong per cell line: a
    fragment/micronucleus is segmentation garbage on RUES2 and plausibly the real
    phenotype on a genome-doubled WGD line, so the same bucket can be noise in one
    well and the result in another.
    """
    suspect = {int(t) for t in (m.get("track_multiplicity", {}) or {}).get("suspect_tracks", [])}
    lin_by_id = lin.set_index("track_id")

    dur = (lin_by_id.last_frame - lin_by_id.first_frame).to_dict()
    touching = set(tracks.loc[tracks.get("n_masks_in_frame", 0) > 1, "track_id"].unique().tolist()) \
        if "n_masks_in_frame" in tracks.columns else set()
    bright = tracks.groupby("track_id")["intensity_mean"].mean() \
        if "intensity_mean" in tracks.columns else None
    dim_cut = float(bright.median()) * 0.6 if bright is not None and len(bright) else None

    labels = []
    for r in rows.itertuples():
        kids = _kid_ids(r.daughter_ids)
        fam = [int(r.track_id), *kids]
        if suspect & set(fam):
            labels.append("merged_id")
        elif r.edge_um == r.edge_um and r.edge_um < 15:
            labels.append("edge_clipped")
        elif touching & set(fam):
            labels.append("touching")
        elif r.link_distance_px == r.link_distance_px and r.link_distance_px > 25:
            labels.append("far_link")
        elif (r.size_ratio == r.size_ratio and r.size_ratio < 0.25
              and r.dna_ratio == r.dna_ratio and r.dna_ratio > 0.8):
            labels.append("fragment_like")
        elif kids and min((dur.get(k, np.nan) for k in kids), default=np.nan) < _STUB_DAUGHTER_FRAMES:
            labels.append("vanishing_daughter")
        elif (dim_cut is not None
              and any(k in bright.index and bright[k] < dim_cut for k in kids)):
            labels.append("dim_daughter")
        else:
            labels.append("clean")
    return labels


# The standing explanations printed under a candidate table. They are constants rather
# than inline f-strings because they are the larger half of what find_candidates emits,
# and reading its control flow meant scrolling past them. Every value they interpolate
# is a module constant, so they can be built once at import.
_CENSUS_NOTE = (
    "First match wins, so these partition the pool and sum to its total. "
    "Priority order is a judgement call and the marginal counts move if you "
    "change it. THE THRESHOLDS ARE UNVALIDATED -- each row is a hypothesis "
    "with a count, not a verdict, and they are very likely wrong per cell "
    "line (a micronucleus is garbage on RUES2 and may be the phenotype on WGD). "
    "Nothing here is discarded: pass stratum= to pull rows from one class, "
    "which is how you sample a class to measure how often it is real."
)

_CENSUS_NEXT_STEP = (
    "Every stratum here describes the RECORDED link. When a row's daughters "
    "look like stubs, list_nearby_tracks(well, track_id=...) shows every "
    "object segmented at that spot -- including the ones nothing links to -- "
    "and follow_cells_over_time(track_ids=[...], centre_frame=<cond_f>) renders "
    "whichever of them you decide are the daughters."
)

_FRAGMENT_NOTE = (
    "\nA LOW size ratio with a healthy dna ratio is the fragment signature -- the "
    "big object carries the DNA and the small one is a micronucleus, so the 'split' "
    "is not a division. Confirm on the pixels before believing either way."
)

_STRATUM_NOTE = (
    f"stratum is the first artifact class this row trips -- the same label the "
    f"census counts. It is printed per row because reading a row without it is how "
    f"a known-suspect link gets ranked as the cleanest in the well."
    f"\ndau_frames is how many frames each recorded daughter lasts. Where either is "
    f"{_STUB_DAUGHTER_FRAMES} or fewer, dna and size read 'n/a': they are measured "
    f"ACROSS that link, so a one-frame daughter makes them noise, and a printed "
    f"number would outrank the honest rows. Persistent daughters (e.g. 77/77) are "
    f"the ones whose ratios mean anything. Not a verdict -- the tracker breaking at "
    f"a real division also produces stubs, which is why an 'n/a' row is a reason to "
    f"call list_nearby_tracks, not a reason to reject it."
)

_COND_NOTE = (
    f"cond is the CONDENSATION peak -- brightness per pixel at its highest, "
    f"relative to this mother's own median and to the other cells in the same "
    f"frame (so bleaching cancels). 1.0 is her normal interphase state; mitosis "
    f"packs the same DNA into fewer pixels, so it reads above 1. cond_f is the "
    f"frame it peaked at, and it is usually LATER than last_frame -- the link "
    f"ends where the tracker loses the cell, not where the cell divides. "
    f"cond_area is the family's area there over her baseline (below 1 = compacted) "
    f"and cond_dna is total signal over her baseline. "
    f"Only frames holding {_COND_DNA_MIN}-{_COND_DNA_MAX}x the baseline signal "
    f"were eligible: below that the DNA has genuinely gone and the object is a "
    f"fragment, whatever its brightness does. '-' means no eligible frame, or too "
    f"short a mother to have a baseline. "
    f"This is the only MORPHOLOGY column here -- every other one describes who the "
    f"tracker linked to whom, and on BeWo that topology scored close to "
    f"anti-correlated with a human reading metaphase plates. Unvalidated, and it "
    f"cannot tell condensed chromatin from the clumped chromatin of a dying cell."
)

_BORN_NOTE = (
    "\nborn_dna/born_size describe how this track was BORN, not how it ends -- "
    "it has no daughters by definition. A low born_size that then vanishes is "
    "most likely a fragment appearing and going, not a cell dying."
)

_NOT_DEATHS_NOTE = (
    "This pool is deliberately NOT called deaths. A cell that died and a "
    "division whose daughters were never linked both look like a track that "
    "stops -- topology cannot tell them apart, and 96% of one hand-checked "
    "sample of 'deaths' in this project turned out to still be alive."
)


def _candidate_pool(well: str, lin, pool: str, n_frames: int, m: dict):
    """The rows of one pool, or a string explaining why there are none.

    Returning text rather than raising: an empty or unavailable pool is an answer
    about the well, not a caller error, and the reason is the useful part.
    """
    if pool == "division":
        rows = lin[lin.n_daughters.fillna(0) >= 2].copy()
        # Scores live on the DAUGHTER rows; lift the first daughter's onto the mother.
        by_id = lin.set_index("track_id")
        first_kid = rows.daughter_ids.astype(str).str.split().str[0]
        for col in ("dna_ratio", "size_ratio", "link_distance_px"):
            rows[col] = [
                by_id[col].get(int(k)) if str(k).strip().lstrip("-").isdigit() else None
                for k in first_kid
            ]
    elif pool == "track_end":
        rows = lin[(lin.last_frame < n_frames - 1) & (lin.n_daughters.fillna(0) == 0)].copy()
    elif pool == "contested":
        if "alt_parents" not in lin.columns:
            return (f"{well}'s lineage came from {m.get('lineage', {}).get('source', '?')}, "
                    f"which records no alternatives, so there is no contested pool. "
                    f"alt_parents only exists on geometry-sourced lineage.")
        rows = lin[lin.alt_parents.astype(str).str.strip().replace("nan", "") != ""].copy()
    else:
        raise ValueError(f"unknown pool {pool!r}; use division, track_end, or contested.")
    if rows.empty:
        return f"{well}: nothing in the {pool} pool."
    return rows


def _sort_candidates(rows, sort_by: str, pool: str, seed: int | None):
    """Order the pool, returning (rows, the sort ACTUALLY applied).

    A score-based sort on a pool that carries no scores would silently return the
    rows in file order while the header claimed otherwise, so it falls back
    explicitly and the caller says so. track_end rows have no link scores by
    definition -- they have no link.
    """
    def _has(col: str) -> bool:
        return col in rows.columns and bool(rows[col].notna().any())

    if sort_by == "fragment_like" and _has("size_ratio"):
        return rows.sort_values("size_ratio", na_position="last"), sort_by
    if sort_by == "dna_anomaly" and _has("dna_ratio"):
        return (rows.reindex((rows.dna_ratio - 1.0).abs()
                             .sort_values(ascending=False).index), sort_by)
    if sort_by == "duration":
        return rows.sort_values("duration_f", ascending=False), sort_by
    if sort_by == "frame":
        return rows.sort_values("first_frame"), sort_by
    if sort_by == "condensation" and _has("cond"):
        return rows.sort_values("cond", ascending=False, na_position="last"), sort_by
    if sort_by == "daughter_persistence" and _has("dau_min"):
        return rows.sort_values("dau_min", ascending=False, na_position="last"), sort_by
    if sort_by == "random":
        # Shuffle the WHOLE surviving pool and let limit take the head, so the sample
        # is drawn from every row that qualifies rather than from the top of some
        # other ordering. Sorting by first_frame first makes the draw independent of
        # the row order lineage.csv happened to be written in.
        return rows.sort_values("first_frame").sample(frac=1.0, random_state=seed), sort_by
    applied = "duration" if pool == "track_end" else "frame"
    return ((rows.sort_values("duration_f", ascending=False) if applied == "duration"
             else rows.sort_values("first_frame")), applied)


def _division_table(shown, well: str) -> list[str]:
    """The division pool's row table, its column notes, and a prefilled call per row."""
    out = ["track_id | frames | daughters | dau_frames | stratum | dna | size | "
           "link_px | edge_um | cond | cond_f | cond_dna | cond_area | also"]
    for r in shown.itertuples():
        # Ratios measured across a link whose daughter lasts a frame or two are not
        # weak evidence, they are noise shaped like a measurement -- and worse than a
        # blank, because "1.01 / 1.00" reads as the cleanest row on the page. That is
        # exactly how a reader ranked a vanishing_daughter artifact first on 2026-08-01.
        stub = getattr(r, "dau_min", float("nan"))
        measurable = not (stub == stub and stub <= _STUB_DAUGHTER_FRAMES)
        g = _fmt2 if measurable else (lambda v: "n/a")
        out.append(f"{int(r.track_id)} | {int(r.first_frame)}-{int(r.last_frame)} | "
                   f"{r.daughter_ids} | {getattr(r, 'dau_frames', '-') or '-'} | "
                   f"{getattr(r, 'stratum', '-')} | {g(r.dna_ratio)} | {g(r.size_ratio)} | "
                   f"{_fmt0(r.link_distance_px)} | "
                   f"{_fmt0(r.edge_um, blank='?')} | "
                   f"{_fmt2(getattr(r, 'cond', None))} | "
                   f"{'-' if int(getattr(r, 'cond_frame', -1)) < 0 else int(r.cond_frame)} | "
                   f"{_fmt2(getattr(r, 'cond_dna', None))} | "
                   f"{_fmt2(getattr(r, 'cond_area', None))} | "
                   f"{getattr(r, 'also', '') or '-'}")
    out += [_FRAGMENT_NOTE, _STRATUM_NOTE, _COND_NOTE]
    # Hand-building the next call is where the documented trap gets sprung: centring on
    # the link (last_frame) instead of on the condensation peak shows the frames AFTER
    # the event. Prefill it, so the default is the right one.
    out.append("\nReady to look -- paste one (centred on cond_f, not on the link):")
    for r in shown.itertuples():
        cf_ = int(getattr(r, "cond_frame", -1))
        centre = cf_ if cf_ >= 0 else int(r.last_frame)
        fam = [int(r.track_id), *_kid_ids(r.daughter_ids)]
        out.append(f'  {int(r.track_id)}  follow_cells_over_time(well="{well}", '
                   f'track_ids={fam}, centre_frame={centre})'
                   + ("" if cf_ >= 0 else "   [no cond peak -- centred on the link, "
                                          "which is usually EARLY; widen if empty]"))
    return out


def _birth_table(shown, pool: str) -> list[str]:
    """The non-division pools' row table.

    These scores describe the track's OWN BIRTH link, not an ending -- a track here
    has no daughters by definition. Shown because the combination is genuinely
    informative: something born as a tiny fragment and then vanishing is a different
    story from a full-sized nucleus that stops.
    """
    out = ["track_id | frames | duration | born_dna | born_size | edge_um"]
    for r in shown.itertuples():
        out.append(f"{int(r.track_id)} | {int(r.first_frame)}-{int(r.last_frame)} | "
                   f"{int(r.duration_f)} | {_fmt2(getattr(r, 'dna_ratio', None))} | "
                   f"{_fmt2(getattr(r, 'size_ratio', None))} | "
                   f"{_fmt0(r.edge_um, blank='?')}")
    if pool == "track_end":
        out += [_BORN_NOTE, _NOT_DEATHS_NOTE]
    return out


@server.tool()
def find_candidates(
    well: str,
    pool: str = "division",
    sort_by: str = "fragment_like",
    limit: int = 20,
    exclude_near_edge: bool = True,
    stratum: str | None = None,
    seed: int | None = None,
    collapse: bool = True,
) -> str:
    """Scan a WHOLE WELL and rank what is worth looking at. Free -- no images.

    Every other tool here answers a question about one track you already picked.
    This is the one that answers "where do I even start" on a well with thousands
    of cells, which is the first question anyone actually has.

    It reports what the data ALREADY RECORDS, ranked -- it is not a detector and
    infers nothing new. A row here means "the tracker linked these ids" or "this
    track stops early", not "a division happened" or "this cell died".

    Pools:
      division   mothers with 2+ recorded daughters. The link scores come from
                 lineage.csv: dna_ratio ~1.0 when DNA is conserved across the
                 split, size_ratio ~1.0 when the daughters are comparable objects.
      track_end  tracks that stop before the recording does WITHOUT recorded
                 daughters. Deliberately one pool, not "deaths": a real death and
                 a division whose daughters were never linked look identical from
                 topology, and calling it either would be inventing a verdict.
      contested  links where another mother was equally close, so parent_id was a
                 tie-break. Only exists on geometry-sourced lineage.

    Sorts:
      fragment_like  lowest size_ratio first -- one "daughter" much smaller than
                     the other is the micronucleus/fragment signature, not a
                     division. This is the ranking that finds real problems.
      dna_anomaly    furthest from DNA conservation in EITHER direction; below 1
                     means signal went missing, above means the pair carries more
                     DNA than the mother had.
      condensation   strongest chromatin compaction first -- smaller AND brighter
                     while total signal is conserved, which is what mitosis looks
                     like and what fragmentation, death and bleaching do not. The
                     ONLY morphology-based sort here; every other one ranks on
                     topology, which is what a human reading metaphase plates
                     disagreed with on 7 of 11 BeWo cases. Reach for it when the
                     question is "did this cell divide" rather than "is this row
                     trustworthy".
      duration       longest-lived first (division pool: the MOTHER's lifetime
                     before the link -- not her daughters'. Do not confuse this
                     with daughter_persistence below; a long-lived mother says
                     nothing about whether her recorded daughters were real.)
      daughter_persistence  longest surviving recorded daughter first (the
                     SHORTER of the two, so a row only ranks well if BOTH held
                     up). Division pool only. Reuses `dau_min`/`dau_frames`,
                     already shown in every division row -- promoted to an actual
                     sort 2026-08-06 after a second, independent session found the
                     same split condensation misses: on nTSC, real divisions ran
                     77-159-frame daughters, artifacts 1-10. `_daughter_spans`
                     itself still calls this "not scored" in its own docstring
                     from when n=5 was the only evidence -- that note is now
                     stale, two sessions on two cell lines agree, but it is still
                     TOPOLOGY (the tracker kept linking the id), same blind spot
                     as fragment_like/dna_anomaly/duration: it fails exactly
                     where the tracker fails through a division, which is why
                     `condensation` (morphology, not topology) exists as a
                     separate, independent sort rather than a replacement.
      frame          earliest first.
      random         a seeded shuffle -- THE ONLY SORT THAT MAY BE USED TO ESTIMATE
                     ANYTHING. Every other sort answers "show me the worst", which
                     is triage; a rate, a share, or a per-stratum true-positive rate
                     needs a sample that represents its stratum. `duration` and
                     `daughter_persistence` are both measured in FRAMES, so both
                     favour faster-sampled wells, and on a BeWo draw of 5
                     `vanishing_daughter` cases `duration` plausibly oversampled
                     long-lived cells that never divided. Pass `seed` to make the
                     draw reproducible and quotable; without one it is still
                     random but nobody can redraw it.

    The division pool also comes with a CENSUS: every recorded division is labelled
    with the first artifact class it trips, and the counts partition the pool. That
    is what turns a well into a sampling frame -- review ~20 from a stratum, learn
    how often that class is real, and the whole pool gets a corrected count with an
    error bar for ~150 reviewed events instead of 1,400. Reviewing every division is
    not affordable and never will be; the estimate with a stated uncertainty is the
    better number anyway.

    Use `limit=0` for the census with no rows -- it is a few hundred tokens and is
    usually the right first call on a well. Then `stratum="fragment_like"` (etc.) to
    draw the sample from one class.

    Args:
        well: well name from list_wells().
        pool: "division", "track_end", or "contested".
        sort_by: "fragment_like", "dna_anomaly", "condensation", "duration",
            "frame", or "random".
            Use "random" with a seed for any sample you intend to draw a number
            from; the ranked sorts are for finding problems, not for measuring.
        limit: max rows. Keep small; this is for triage, not export. 0 = census only.
        exclude_near_edge: drop cells within 15 um of the frame boundary, whose
            area and brightness are understated because the nucleus is clipped. They
            are still COUNTED in the census, and asking for stratum="edge_clipped"
            turns this off automatically.
        collapse: fold recorded divisions that are one physical event into a single
            row (division pool only, ON by default). A tracker that fails through a
            division fails repeatedly, so one BeWo mitosis appeared as tracks 1824,
            1860 and 1883 -- three of the top five rows of one ranked sample. The
            folded ids are listed in the `also` column, never dropped. Turn it off to
            see the raw recorded pool.
        stratum: restrict rows to one census class (division pool only).
        seed: fixes the shuffle for sort_by="random", so the same call returns the
            same sample and a result can be checked by someone else. Recorded in
            the header for exactly that reason.
    """
    import pandas as pd

    p = _cm.BUNDLE / well / "lineage.csv"
    if not p.is_file():
        return f"No lineage.csv for {well}, so there is nothing to rank."
    lin = pd.read_csv(p)
    m = _cm._manifest(well)
    n_frames = int(m["n_frames"])
    tracks = _cm._tracks(well)

    pos = tracks.sort_values("frame").groupby("track_id")[["cx", "cy"]].last()

    def _edge(tid: int) -> float:
        if tid not in pos.index:
            return float("nan")
        return _edge_um(well, float(pos.loc[tid, "cx"]), float(pos.loc[tid, "cy"]))

    rows = _candidate_pool(well, lin, pool, n_frames, m)
    if isinstance(rows, str):
        return rows

    rows["edge_um"] = [_edge(int(t)) for t in rows.track_id]
    n_before = len(rows)

    # Strata are computed on the WHOLE pool, before any dropping, so that what
    # exclude_near_edge removes still appears as a counted class rather than
    # vanishing. A number you cannot see is the thing that made events.csv dangerous.
    census = []
    if pool == "division":
        rows["stratum"] = _division_strata(rows, lin, tracks, m)
        rows["dau_frames"], rows["dau_min"] = _daughter_spans(rows, lin)
        cs, cf, cd, ca = _condensation(well, rows, lin, tracks, m["pixel_size_um"])
        rows["cond"], rows["cond_frame"] = cs, cf
        rows["cond_dna"], rows["cond_area"] = cd, ca
        sites = _collapse_sites(well, rows, tracks) if collapse else {}
        if collapse:
            rows["also"] = [" ".join(str(u) for u in sites.get(int(t), []))
                            for t in rows.track_id]
            n_folded = len(rows) - len(sites)
            rows = rows[rows.track_id.astype(int).isin(sites)]
        else:
            rows["also"] = ""
            n_folded = 0
        counts = rows.stratum.value_counts()
        census = [f"{name} | {int(counts.get(name, 0))} | "
                  f"{100 * int(counts.get(name, 0)) / max(n_before, 1):.1f}% | {blurb}"
                  for name, blurb in _STRATA]
        if stratum:
            if stratum not in {n for n, _ in _STRATA}:
                raise ValueError(f"unknown stratum {stratum!r}; one of "
                                 f"{', '.join(n for n, _ in _STRATA)}")
            rows = rows[rows.stratum == stratum]
            # Asking for the edge stratum while the edge filter is on returns nothing
            # and looks like "there are none". Honour the explicit request instead.
            if stratum == "edge_clipped":
                exclude_near_edge = False
    elif stratum:
        return (f"stratum= only applies to the division pool; the {pool} pool has no "
                f"strata defined. Call again without it.")

    n_pool = len(rows)
    if exclude_near_edge:
        rows = rows[~(rows.edge_um < 15)]
    n_edge_dropped = n_pool - len(rows)
    rows["duration_f"] = rows.last_frame - rows.first_frame

    rows, applied = _sort_candidates(rows, sort_by, pool, seed)

    shown = rows.head(limit) if limit > 0 else rows.iloc[:0]
    note = "" if applied == sort_by else (
        f" (asked for {sort_by}, but this pool carries no link scores -- "
        f"sorted by {applied} instead)")
    draw = (f" Random draw of {len(shown)} from the {len(rows)} rows in this pool"
            + (f", seed={seed} (re-callable)." if seed is not None else
               ", UNSEEDED -- pass seed= if this sample will back a number.")
            ) if applied == "random" and limit > 0 else ""
    fold = (f" {n_before} recorded divisions fold into {n_before - n_folded} distinct "
            f"SITES ({n_folded} rows were the same cell re-acquired); the folded ids "
            f"are in the `also` column, not discarded."
            if pool == "division" and n_folded else "")
    out = [f"{well}: {pool} pool, {n_before} total"
           + (f", stratum={stratum}" if stratum else "")
           + (f", showing {len(shown)} sorted by {applied}.{note}{draw}" if limit > 0
              else ", census only (limit=0).") + fold]
    if n_edge_dropped and not census:
        out.append(f"({n_edge_dropped} near-edge rows not shown; "
                   f"exclude_near_edge=False to include them.)")

    if census:
        out.append("\nWhat the pool is made of -- stratum | n | share | why")
        out += census
        out.append(_CENSUS_NOTE)
        out.append(_CENSUS_NEXT_STEP)

    if limit <= 0:
        return "\n".join(out)

    if rows.empty:
        out.append("\nNo rows left after filtering. Widen the stratum or set "
                   "exclude_near_edge=False.")
        return "\n".join(out)

    out += (_division_table(shown, well) if pool == "division"
            else _birth_table(shown, pool))
    out.append("Next: get_track_profile (free) on anything here, then follow_cells_over_time to look.")
    return "\n".join(out)



__all__ = [
    "_COND_BEFORE_MIN", "_COND_AFTER_MIN", "_COND_MARGIN_UM", "_COND_BASE_MIN",
    "_COND_DNA_MIN", "_COND_DNA_MAX", "_condensation",
    "_prophase_onset", "find_prophase_onset",
    "_collapse_sites", "_SITE_RADIUS_UM", "_SITE_GAP_MIN", "_SITE_MAX_SPAN_MIN",
    "_STRATA", "_kid_ids", "_fmt2", "_fmt0", "_STUB_DAUGHTER_FRAMES",
    "_daughter_spans", "_division_strata",
    "_CENSUS_NOTE", "_CENSUS_NEXT_STEP", "_FRAGMENT_NOTE", "_STRATUM_NOTE",
    "_COND_NOTE", "_BORN_NOTE", "_NOT_DEATHS_NOTE",
    "_candidate_pool", "_sort_candidates", "_division_table", "_birth_table",
    "find_candidates",
]
