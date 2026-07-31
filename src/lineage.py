"""Per-frame cell position lookup from the cached Trackastra tracked-mask memmap,
without re-running segmentation/tracking. Built 2026-07-18 as the foundation for
lineage-tracking crops -- a researcher's Claude following a track's full lifetime
(potentially 50-200+ frames between divisions), not just the AI review's fixed
+/-24 frame window. See [[project_cell_split_counter_marker_tracking_backlog]] and
[[project_cell_split_counter_researcher_browser_extend_timeline]].

events.csv only carries a single (cx, cy) snapshot per event, taken at that event's
own peak_frame -- nothing for the frames in between two divisions. Reconstructing a
real per-frame trajectory means going back to frames/_memmap/tracked_masks.dat
(written by src.track's Trackastra call): Trackastra's RAW per-frame CTC label maps.

Critical subtlety: a raw label in tracked_masks.dat is NOT always the same value as
events.csv's track_id. src.track._bridge_track_gaps() remaps raw labels into
canonical lineages after tracking (merging a track across a brief Cellpose
detection gap -- see that function's docstring), and events.csv's track_id reflects
THAT remapping, not the raw pixel values. The remapping itself was never persisted
to disk -- only used in-memory to build TrackNode objects before being collapsed
into events.csv. Rather than reimplementing that gap-bridging heuristic a second
time (a duplication pattern this project has been burned by before -- see
scripts/reports/_crop_shared.py's docstring on review_gpt.py silently drifting from
review.py), this module imports _bridge_track_gaps() itself and feeds it a
begin/end-frame table built from a single scan of tracked_masks.dat.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.track import _bridge_track_gaps

_DTYPE = np.uint16  # must match src/segment.py's label_maps.dat / tracked_masks.dat dtype


def _frame_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "frames").glob("frame_*.png"))


def _open_tracked_masks(run_dir: Path) -> np.memmap:
    frame_paths = _frame_paths(run_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no frame_*.png files under {run_dir / 'frames'}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    T, H, W = len(frame_paths), first.shape[0], first.shape[1]

    path = run_dir / "frames" / "_memmap" / "tracked_masks.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- this run's cached Trackastra output isn't on disk, "
            "can't reconstruct per-frame positions without a pipeline rerun"
        )
    expected = T * H * W * np.dtype(_DTYPE).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"tracked_masks.dat size mismatch -- expected {expected / 1e9:.2f}GB for "
            f"{T} frames at {H}x{W}, found {actual / 1e9:.2f}GB. Stale/wrong run_dir?"
        )
    return np.memmap(path, dtype=_DTYPE, mode="r", shape=(T, H, W))


def _tracking_index_cache_path(run_dir: Path) -> Path:
    return run_dir / "frames" / "_memmap" / "canonical_labels.json"


def _build_tracking_index(
    tracked_masks: np.memmap, run_dir: Path,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """(canonical_of, begin_of, end_of) -- canonical_of maps raw Trackastra CTC
    label -> canonical (post-gap-bridge) label matching events.csv's track_id;
    begin_of/end_of are each RAW label's own first/last-seen frame (pre-bridge --
    a canonical/track_id's true lifetime bounds are the min begin_of/max end_of
    across every raw label that bridges into it, see track_lifetime()).

    Cached to disk since building it requires one full pass over every frame
    (np.unique per frame) -- cheap relative to a pipeline rerun, but not free on a
    long video. Cache keyed on tracked_masks.dat's file size so a stale cache from
    a different run/regeneration doesn't silently get reused.
    """
    import pandas as pd

    cache_path = _tracking_index_cache_path(run_dir)
    masks_path = run_dir / "frames" / "_memmap" / "tracked_masks.dat"
    masks_size = masks_path.stat().st_size
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("_tracked_masks_size") == masks_size and "canonical_of" in cached:
            return (
                {int(k): v for k, v in cached["canonical_of"].items()},
                {int(k): v for k, v in cached["begin_of"].items()},
                {int(k): v for k, v in cached["end_of"].items()},
            )

    T = tracked_masks.shape[0]
    begin_of: dict[int, int] = {}
    end_of: dict[int, int] = {}
    for t in range(T):
        for lbl in np.unique(tracked_masks[t]):
            lbl = int(lbl)
            if lbl == 0:
                continue
            if lbl not in begin_of:
                begin_of[lbl] = t
            end_of[lbl] = t

    df_ctc = pd.DataFrame({
        "label": list(begin_of.keys()),
        "begin": [begin_of[l] for l in begin_of],
        "end": [end_of[l] for l in begin_of],
    })
    canonical_of = _bridge_track_gaps(df_ctc, tracked_masks, "label", "begin", "end")

    cache_path.write_text(
        json.dumps({
            "_tracked_masks_size": masks_size,
            "canonical_of": {str(k): v for k, v in canonical_of.items()},
            "begin_of": {str(k): v for k, v in begin_of.items()},
            "end_of": {str(k): v for k, v in end_of.items()},
        }),
        encoding="utf-8",
    )
    return canonical_of, begin_of, end_of


def track_lifetime(run_dir: Path, track_id: int) -> tuple[int, int] | None:
    """(first_frame, last_frame) this track_id actually appears in tracked_masks.dat,
    across every raw label bridged into it -- None if track_id never appears at all
    (e.g. a typo'd id, or one that classify.py dropped before writing events.csv).
    """
    tracked_masks = _open_tracked_masks(run_dir)
    canonical_of, begin_of, end_of = _build_tracking_index(tracked_masks, run_dir)
    raw_labels = [raw for raw, canon in canonical_of.items() if canon == track_id]
    if not raw_labels:
        return None
    return min(begin_of[r] for r in raw_labels), max(end_of[r] for r in raw_labels)


def per_frame_centroids(
    run_dir: Path, track_id: int, frame_lo: int | None = None, frame_hi: int | None = None,
) -> dict[int, tuple[float, float]]:
    """Real per-frame (cx, cy) centroid for `track_id`. Defaults to the track's own
    observed lifetime (track_lifetime()) rather than the whole video -- correct and
    far cheaper for a short-lived track in a long video. A frame with no data (the
    cell genuinely wasn't detected that frame -- see _bridge_track_gaps's docstring
    on brief Cellpose detection gaps that DIDN'T get bridged, e.g. 2+ plausible
    successors) is simply absent from the returned dict, never interpolated or guessed.
    """
    tracked_masks = _open_tracked_masks(run_dir)
    canonical_of, begin_of, end_of = _build_tracking_index(tracked_masks, run_dir)
    raw_labels = {raw for raw, canon in canonical_of.items() if canon == track_id}
    if not raw_labels:
        return {}

    T = tracked_masks.shape[0]
    if frame_lo is None or frame_hi is None:
        lifetime_lo, lifetime_hi = min(begin_of[r] for r in raw_labels), max(end_of[r] for r in raw_labels)
    lo = lifetime_lo if frame_lo is None else max(0, frame_lo)
    hi = lifetime_hi if frame_hi is None else min(T - 1, frame_hi)

    out: dict[int, tuple[float, float]] = {}
    for t in range(lo, hi + 1):
        frame_arr = np.asarray(tracked_masks[t])
        for raw in raw_labels:
            ys, xs = np.nonzero(frame_arr == raw)
            if len(xs) > 0:
                out[t] = (float(xs.mean()), float(ys.mean()))
                break
    return out


# --------------------------------------------------------------- geometric lineage

# Runs tracked before 2026-07-29 never persisted Trackastra's parent table -- it
# lived only in memory, so canonical_labels.json preserves the label remap but the
# graph itself is gone. Re-running Trackastra to recover it costs a GPU pass per
# well. This rebuilds the same topology from tracks.csv geometry instead, which is
# free, works on every existing run, and uses the same rule that generates division
# candidates, so lineage and candidates can no longer disagree with each other.
#
# Two things this deliberately does NOT do:
#
# 1. It does not filter. A link here means "these two tracks are born adjacent to
#    where that one ended", not "a division happened" -- the same contract
#    src.track._write_lineage_csv already promises. Verdicts are the reviewer's job.
# 2. It does not resolve biology. A micronucleus budding off beside a nucleus is
#    geometrically indistinguishable from a division, and both this rule AND the
#    old events-derived lineage get track 6425 wrong for exactly that reason
#    (recorded mother 4866 is a healthy neighbour -- confirmed by eye 2026-07-30).
#
# What it does instead is SCORE every link, so a reader can discount a weak one:
# dna_ratio and size_ratio would both have flagged 6425. That was the eval's own
# request -- "exposing the underlying overlap would let a session discount weak
# links" -- and it is why the checks are columns rather than conditions.

_MAX_LINK_PX = 40.0


def score_lineage_links(lineage: "pd.DataFrame", tracks: "pd.DataFrame") -> "pd.DataFrame":
    """Add link_distance_px / dna_ratio / size_ratio to a lineage from ANY source.

    Trackastra's own CTC graph is topology with no scores, and a re-track on
    M12_RUES2 (2026-07-30) showed it is strictly the better topology: 100% agreement
    with the geometric graph on the 2,074 links both assign, plus 350 links geometry
    missed because it only spans a one-frame gap. It also recovers track 3908's real
    mother 3288, which geometry could not see.

    But it makes the SAME mistake on track 6425 -- both call the neighbouring cell
    4866 its mother when 6425 is a micronucleus. So a better linker does not remove
    the need to score links; no linking model can separate a fragment budding off
    from a division on topology alone. Scoring is orthogonal to the source, which is
    why it lives in its own function and gets applied to whichever graph is used.

    Unlike the geometric builder, this does not assume a one-frame gap: it measures
    between the mother's last frame and each daughter's first, whatever the spacing.
    """
    import pandas as pd

    t = tracks.drop_duplicates(["track_id", "frame"]).sort_values("frame")
    last = t.groupby("track_id").tail(1).set_index("track_id")
    first = t.groupby("track_id").head(1).set_index("track_id")

    dist: dict[int, float] = {}
    dna: dict[int, float] = {}
    size: dict[int, float] = {}
    for row in lineage.itertuples():
        # Empty daughter_ids reads back from CSV as float nan, not "".
        raw = getattr(row, "daughter_ids", "")
        raw = "" if raw is None or (isinstance(raw, float) and raw != raw) else str(raw)
        kids = [int(k) for k in raw.split() if k.strip().lstrip("-").isdigit()]
        m = int(row.track_id)
        if len(kids) != 2 or m not in last.index:
            continue
        if any(k not in first.index for k in kids):
            continue
        areas = [float(first.loc[k, "area_um2"]) for k in kids]
        mother_dna = float(last.loc[m, "intensity_integrated"])
        kid_dna = sum(float(first.loc[k, "intensity_integrated"]) for k in kids)
        r_dna = kid_dna / mother_dna if mother_dna > 0 else float("nan")
        r_size = min(areas) / max(areas) if max(areas) > 0 else float("nan")
        for k in kids:
            dist[k] = float(np.hypot(first.loc[k, "cx"] - last.loc[m, "cx"],
                                     first.loc[k, "cy"] - last.loc[m, "cy"]))
            dna[k] = r_dna
            size[k] = r_size

    out = lineage.copy()
    # pandas reads an int column with blanks as float, so parent_id round-trips as
    # "127.0" and every downstream int() blows up. Normalise on the way out.
    if "parent_id" in out.columns:
        out["parent_id"] = [
            "" if p is None or (isinstance(p, float) and p != p) or str(p).strip() == ""
            else str(int(float(p)))
            for p in out["parent_id"]
        ]
    out["link_distance_px"] = [round(dist[t_], 1) if t_ in dist else ""
                               for t_ in out.track_id.astype(int)]
    out["dna_ratio"] = [round(dna[t_], 3) if t_ in dna else ""
                        for t_ in out.track_id.astype(int)]
    out["size_ratio"] = [round(size[t_], 3) if t_ in size else ""
                         for t_ in out.track_id.astype(int)]
    return out


def build_lineage_from_tracks(tracks: "pd.DataFrame") -> "pd.DataFrame":
    """Derive the mother/daughter graph from tracks.csv geometry alone.

    A mother ending at frame f is linked to tracks born at f+1 within
    _MAX_LINK_PX of its last centroid. The frame gap is always exactly 1 in this
    pipeline's own splits (median mother->daughter distance 9.6 px, p90 18.9), so
    there is no threshold to tune.

    Each daughter is assigned to its NEAREST eligible mother. Without that step
    two mothers ending at the same frame near each other both claim the same
    births, and which one wins depends on row order -- 70 contested daughters on
    M12_RUES2, silently arbitrary. Nearest-mother resolution drops that to 0 and
    raises agreement with the old events-derived graph from 82.0% to 88.9%.

    Known limitation: only a gap of exactly one frame is linked. A cell whose track
    ends at f and resumes at f+2 or later gets no parent, so "mother none recorded"
    can hide a real chain -- track 5286 (ends 651) -> 6295 (starts 653) -> the real
    division at 654 is one such case on M12_RUES2. Widening the gap would also
    invent links between unrelated neighbours, so it needs its own evidence before
    being changed rather than a guessed tolerance.

    Returns a frame with one row per track and these quality columns on each link,
    all NaN for tracks with no parent:
      link_distance_px  mother's last centroid to this daughter's first
      dna_ratio         sum of both daughters' intensity_integrated at f+1 over
                        the mother's at f. ~1.0 for a real division (DNA is
                        conserved); far below 1 when a fragment budded off.
      size_ratio        smaller daughter's area over larger's. Near 1 for a
                        symmetric division; near 0 when one "daughter" is a
                        micronucleus.
    """
    import pandas as pd

    t = tracks.drop_duplicates(["track_id", "frame"]).sort_values("frame")
    last = t.groupby("track_id").tail(1).set_index("track_id")
    first = t.groupby("track_id").head(1).set_index("track_id")

    births: dict[int, list[int]] = {}
    for tid, r in first.iterrows():
        births.setdefault(int(r.frame), []).append(int(tid))

    # Every candidate (distance, mother, daughter) pair, nearest first.
    pairs: list[tuple[float, int, int]] = []
    for m, r in last.iterrows():
        for c in births.get(int(r.frame) + 1, []):
            d = float(np.hypot(first.loc[c, "cx"] - r.cx, first.loc[c, "cy"] - r.cy))
            if d <= _MAX_LINK_PX:
                pairs.append((d, int(m), int(c)))

    claimed: dict[int, tuple[float, int]] = {}
    contenders: dict[int, list[tuple[float, int]]] = {}
    for d, m, c in sorted(pairs):
        claimed.setdefault(c, (d, m))
        contenders.setdefault(c, []).append((d, m))

    kids: dict[int, list[int]] = {}
    for c, (_d, m) in claimed.items():
        kids.setdefault(m, []).append(c)

    parent: dict[int, int] = {}
    dna: dict[int, float] = {}
    size: dict[int, float] = {}
    dist: dict[int, float] = {}
    for m, cs in kids.items():
        # A single successor is a continuation, not a birth -- that case is what
        # _bridge_track_gaps already merges. Three or more is unresolvable here.
        if len(cs) != 2:
            continue
        areas = [float(first.loc[c, "area_um2"]) for c in cs]
        mother_dna = float(last.loc[m, "intensity_integrated"])
        kid_dna = sum(float(first.loc[c, "intensity_integrated"]) for c in cs)
        r_dna = kid_dna / mother_dna if mother_dna > 0 else float("nan")
        r_size = min(areas) / max(areas) if max(areas) > 0 else float("nan")
        for c in cs:
            parent[c] = m
            dna[c] = r_dna
            size[c] = r_size
            dist[c] = claimed[c][0]

    daughters: dict[int, list[int]] = {}
    for c, m in parent.items():
        daughters.setdefault(m, []).append(c)

    rows = []
    for tid in sorted(set(int(x) for x in first.index)):
        kid = sorted(daughters.get(tid, []))
        # Which OTHER mothers could also have claimed this daughter. Nearest-mother
        # is a tie-break, not a fact, so the runners-up ship alongside the winner
        # rather than being silently discarded -- same reason the biology checks are
        # columns and not filters. Empty for the vast majority; non-empty is the
        # signal that this particular parent_id is a judgement call.
        alts = ""
        if tid in parent:
            alts = " ".join(
                f"{m}:{d:.0f}" for d, m in sorted(contenders.get(tid, []))
                if m != parent[tid]
            )
        rows.append({
            "track_id": tid,
            "parent_id": parent.get(tid, ""),
            "first_frame": int(first.loc[tid, "frame"]),
            "last_frame": int(last.loc[tid, "frame"]),
            "n_daughters": len(kid),
            "daughter_ids": " ".join(str(k) for k in kid),
            "link_distance_px": round(dist[tid], 1) if tid in dist else "",
            "dna_ratio": round(dna[tid], 3) if tid in dna else "",
            "size_ratio": round(size[tid], 3) if tid in size else "",
            "alt_parents": alts,
        })
    return pd.DataFrame(rows)
