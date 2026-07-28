"""Cheap, local, CPU-only triage scoring used to rank candidate events before vision
review -- so the expensive model is spent on the events most likely to be real.

Validated 2026-07-26/27 on 130 human-labeled TSC_batch2_M12_RUES2 events (45 mitoses /
78 artifacts). Used as a pre-filter keeping the top 60% of events, against vision-alone
at the production min_gpt_confidence=0.65 floor:

    recall           71% -> 64%   (3 events, McNemar p=0.25 -- NOT significant)
    junk rejection   +17 events   (McNemar p<0.0001 -- significant)
    vision API calls -40%

Keeping the top 50% instead makes the recall loss real (p=0.016), so 0.6 is the
operating point, not an arbitrary round number.

TWO SIGNALS, both cheap and both independently validated:

1. `solidity_dip_sharpness` -- the maintainer's own observation from the sparklines: a real
   mitosis shows solidity DROP and then RECOVER (prophase chromatin looks sparse,
   metaphase is a line of separate condensed bodies, anaphase splits, then each daughter
   re-solidifies). Prior features all measured the MAGNITUDE of change (max delta, std,
   total variation) and were null; the SHAPE of the change -- a transient dip that
   returns to baseline -- is what carries signal. AUC 0.731 for mitosis, permutation
   p=0.0015, and it survives stratification by density (r=0.12, essentially independent).

2. Local colony density -- direction is the opposite of intuition: DENSE regions predict
   mitosis, SPARSE predict artifact. Healthy proliferating colonies are packed and
   dividing; isolated floating debris, free micronuclei and already-dead nuclei sit in
   empty regions. AUC ~0.73.

Deliberately NOT included: image focus/sharpness. It looked strong (AUC 0.75) but is a
local-density proxy in disguise -- within density strata it collapses to chance
(0.45/0.50). See docs/investigation_notes.md 2026-07-26.

CROSS-WELL TRANSFER WAS TESTED 2026-07-27 AND IT FAILED. Do not enable this by default.

50 feature-blind events from 202660629_Bewop920x_M4 (BeWo, a confluent trophoblast
line, vs RUES2's discrete colonies), all labeled in one sitting. Same 30% mitosis base
rate as M12, so this is not a base-rate effect:

    solidity_dip_sharpness   0.731 -> 0.533   95% CI [0.362, 0.705], includes 0.5
    mask_area_fraction       0.731 -> 0.476   below chance
    n_masks_in_crop          0.749 -> 0.632
    family-wise permutation over all 12 features: p=0.336 -- nothing survives

At keep=0.6 the filter delivers 33% precision vs the 30% no-filter baseline while
discarding a third of the real mitoses (recall 67%). That is not a usable trade.

The density half degrading was PREDICTED (BeWo's density spread is 1.11x vs M12's
1.34x -- in a confluent sheet "is this in a living colony" has no variance to exploit).
The dip half was predicted to HOLD, on the reasoning that chromatin mechanics are
conserved and the score is baseline-normalized. It did not. Ruled out as explanations:
trajectory-walk corruption in the crowded well (label-switch rate 0.089/frame vs M12's
0.070, identical medians, full window coverage in both), the 12% of events whose marker
sits off-screen, and death-kind events (chance in every slice: 0.533 all / 0.579
excluding off-screen / 0.482 splits only).

So `--triage-keep-fraction` should be read as RUES2-specific until re-validated per cell
line. This is the second mask-geometry feature to look strong within a well and die on
held-out data (area_max_delta_z, 0.696 -> 0.530, 2026-07-26) -- the same shape of
failure both times. See docs/investigation_notes.md and
scripts/eval_harness/test_bewo_m4_transfer.py.
"""

import math
from collections import defaultdict

from src.classify import EventType, LineageEvent
from src.track import TrackNode

# Window over which the dip template is measured, in raw frames each side of the event.
# Matches the +/-24 the vision reviewer sees (8 samples x stride 3).
_WINDOW_HALF = 24

# Crop half-width used for the density measurement -- same 192px box the vision model is
# shown, so "how crowded is this" means crowded *in the image the reviewer sees*.
_CROP_RADIUS = 192


def _dip_sharpness(series: list[float]) -> float | None:
    """Transient-dip score: how far below baseline it dips, times how fully it recovers.

    A real division dips AND comes back. A cell that is simply degrading dips and stays
    down (no recovery); a noisy cell wobbles without a coherent dip. Multiplying depth by
    recovery scores only the combination, which is why this separates mitosis from both
    failure modes where a plain magnitude statistic could not.
    """
    if len(series) < 7:
        return None
    n = len(series)
    edge = max(2, n // 5)
    baseline = sorted(series[:edge] + series[-edge:])[(edge * 2) // 2]
    i_min = min(range(n), key=lambda i: series[i])
    depth = baseline - series[i_min]
    if depth <= 1e-9 or abs(baseline) < 1e-9:
        return 0.0
    after = series[i_min + 1:]
    recovery = (max(after) - series[i_min]) / depth if after else 0.0
    recovery = max(0.0, min(2.0, recovery))
    return (depth / abs(baseline)) * recovery


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(var)
    return [(v - mean) / sd for v in values] if sd > 1e-12 else [0.0] * len(values)


def compute_triage_scores(
    events: list[LineageEvent], tracks: list[TrackNode]
) -> dict[tuple[int, int], float]:
    """Score every event; higher = more likely a real mitosis. Keys are (track_id, frame).

    Reads per-frame shape straight off the TrackNode list the pipeline already built --
    each node carries regionprops area/solidity for its own track_id, so no mask reload
    and no re-derivation. This is also strictly better than reconstructing the series by
    nearest-centroid walking (as the offline validation did), which can hop onto a
    neighbouring nucleus in crowded regions.
    """
    by_track: dict[int, list[TrackNode]] = defaultdict(list)
    by_frame: dict[int, list[TrackNode]] = defaultdict(list)
    for node in tracks:
        by_track[node.track_id].append(node)
        by_frame[node.frame].append(node)
    for nodes in by_track.values():
        nodes.sort(key=lambda n: n.frame)

    raw: list[tuple[tuple[int, int], float | None, float]] = []
    for ev in events:
        # For a split the shape story belongs to the PARENT (it is the parent that rounds
        # up and cleaves); event.frame is split_frame, one past the parent's last frame.
        subject = ev.parent_id if (ev.event_type != EventType.DEATH and ev.parent_id) else ev.track_id
        seed = ev.frame - 1 if ev.event_type != EventType.DEATH else ev.frame
        nodes = [n for n in by_track.get(subject, []) if abs(n.frame - seed) <= _WINDOW_HALF]
        dip = _dip_sharpness([n.mask.solidity for n in nodes
                              if n.mask.solidity is not None]) if nodes else None

        # Local density: fraction of the reviewer's crop covered by ANY nucleus mask.
        density = 0.0
        if ev.centroid is not None:
            cx, cy = ev.centroid
            area = 0.0
            for n in by_frame.get(min(seed, max(by_frame) if by_frame else seed), []):
                nx, ny = n.mask.centroid
                if abs(nx - cx) <= _CROP_RADIUS and abs(ny - cy) <= _CROP_RADIUS:
                    area += n.mask.area
            density = area / float((2 * _CROP_RADIUS) ** 2)
        raw.append(((ev.track_id, ev.frame), dip, density))

    # z-score each signal across THIS run's events, then sum. Standardising per-run keeps
    # the two on a comparable scale without needing absolute calibration, and makes the
    # score a within-run ranking -- which is all it is used for.
    usable = [r for r in raw if r[1] is not None]
    dz = dict(zip((r[0] for r in usable), _zscore([r[1] for r in usable])))
    nz = dict(zip((r[0] for r in raw), _zscore([r[2] for r in raw])))
    # An event with too short a trajectory to score gets density only, rather than being
    # silently ranked last -- absence of a dip measurement is not evidence against it.
    return {k: dz.get(k, 0.0) + nz.get(k, 0.0) for k, _, _ in raw}


def select_for_review(
    events: list[LineageEvent], tracks: list[TrackNode], keep_fraction: float
) -> tuple[list[LineageEvent], list[LineageEvent]]:
    """Split events into (send to vision, skip) by triage rank.

    Returns them in the original order within each group so downstream output ordering
    stays stable.
    """
    if keep_fraction >= 1.0 or not events:
        return list(events), []
    scores = compute_triage_scores(events, tracks)
    ranked = sorted(events, key=lambda e: scores.get((e.track_id, e.frame), 0.0), reverse=True)
    n_keep = max(1, round(len(events) * keep_fraction))
    keep_ids = {(e.track_id, e.frame) for e in ranked[:n_keep]}
    send = [e for e in events if (e.track_id, e.frame) in keep_ids]
    skip = [e for e in events if (e.track_id, e.frame) not in keep_ids]
    return send, skip
