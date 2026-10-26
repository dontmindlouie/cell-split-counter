"""Cell-fate profiling of mitosis_scan candidates: what happened AFTER each metaphase-like event.

Classifying an event as "failed" depends on the hypothesis (a sample-health study and a
cytokinesis study draw the line differently), so this module does not classify. It measures,
per event, the variables a cell biologist would read off an H2B movie, under their usual names:

  plate_aspect              metaphase plate elongation (major/minor axis at the brightest frame)
  mitotic_duration_min      first detection of the condensed plate -> first frame with >= 2 daughter nuclei
                            (H2B proxy for metaphase-to-anaphase; NaN if no segregation was seen)
  n_daughters               most daughter nuclei seen at once after the plate (2 bipolar, 3+ multipolar)
  daughter_area_ratio       smaller / larger daughter area (1 = symmetric division)
  daughter_intensity_ratio  dimmer / brighter daughter mean H2B (DNA content asymmetry proxy)
  separation_max_um         furthest the two main daughters got from each other in the follow-up window
  settle_min                minutes from segregation until the daughters stop separating
                            (pair separation speed < settle_um_per_min, the maintainer's definition)
  separation_at_settle_um   their distance when they stopped: small = stayed together (binucleation-like)
  motion_symmetry           min/max displacement of the two daughters from the plate centre 30 min after
                            segregation (1 = moved apart equally; ~0 = one cell moved, the other did not)
  post_area_fold            no segregation seen: area of the single nucleus at the end of follow-up vs the
                            interphase nucleus before (~1.5-2 = mitotic slippage / tetraploidisation)
  micronuclei               small (< micro_frac x mother) objects near the daughters for >= 2 frames
  fragments                 most small objects at once near the event (death in mitosis / catastrophe)
  signal_retained           total H2B signal near the event at the end of follow-up vs before
                            (low = nuclear loss: death or the cell left)
  followup_min              minutes of movie available after the plate (short = unresolved)
  near_edge                 within fate_radius_um of the image border

Lagging chromosomes and chromatin bridges are NOT measured: at ~0.9 um/px and 10-min frames they
are usually below resolution or between frames. Micronuclei, their persistent consequence, are.

All distances are um and all times minutes, set in the profile, so a profile for another cell type
or imaging setup carries over by changing those numbers (nucleus size, expected mitotic timing).

    uv run python -m src.mitosis_fate <nd2> <scan out dir> [--channel 1] [--position 0]

Writes <scan out dir>/fate.csv, one row per candidate (same order and ids as candidates.csv).
"""
import argparse, json
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from src.mitosis_scan import Profile, segment

PRE = (30, 60)   # interphase reference, minutes before the first detection (as the scan's baseline)


def _props(lab, im, cx, cy, r_px, um_px):
    """Objects whose centroid is within r_px of (cx, cy)."""
    y0, y1 = max(0, int(cy - r_px)), min(lab.shape[0], int(cy + r_px) + 1)
    x0, x1 = max(0, int(cx - r_px)), min(lab.shape[1], int(cx + r_px) + 1)
    out = []
    for rp in regionprops(lab[y0:y1, x0:x1], intensity_image=im[y0:y1, x0:x1]):
        y, x = rp.centroid[0] + y0, rp.centroid[1] + x0
        if np.hypot(x - cx, y - cy) > r_px:
            continue
        out.append({"label": rp.label, "x": x, "y": y, "area": rp.area * um_px * um_px,
                    "mean": rp.intensity_mean, "sum": rp.intensity_mean * rp.area,
                    "aspect": rp.axis_major_length / max(rp.axis_minor_length, 1e-6),
                    "orient": rp.orientation})
    return out


def profile_event(ev, get, minutes, um_px, p, shape):
    """`get(t)` -> (image, labels). Returns a dict of fate variables for one candidate."""
    n = len(minutes)
    r_px = p.fate_radius_um / um_px
    t0, t1 = int(ev["first"]), int(ev["last"])
    cx, cy = ev["x"], ev["y"]
    end = int(np.searchsorted(minutes, minutes[t0] + p.fate_window_min, side="right")) - 1
    f = {"followup_min": round(float(minutes[min(n - 1, end)] - minutes[t0]), 1),
         "near_edge": int(min(cx, cy, shape[1] - 1 - cx, shape[0] - 1 - cy) < r_px)}

    # interphase mother: the object at the event centre 30-60 min before
    pre_t = [t for t in range(t0) if PRE[0] <= minutes[t0] - minutes[t] <= PRE[1]]
    mother, pre_sum = [], []
    for t in pre_t:
        obs = _props(*get(t)[::-1], cx, cy, r_px, um_px)
        pre_sum.append(sum(o["sum"] for o in obs))
        near = min(obs, key=lambda o: np.hypot(o["x"] - cx, o["y"] - cy), default=None)
        if near and np.hypot(near["x"] - cx, near["y"] - cy) * um_px < p.fate_radius_um / 2:
            mother.append(near["area"])
    m_area = float(np.median(mother)) if mother else np.nan
    small = p.micro_frac * (m_area if np.isfinite(m_area) else 80.0)

    # plate: brightest object at the centre during the detection
    plate = None
    for t in range(t0, t1 + 1):
        for o in _props(*get(t)[::-1], cx, cy, p.fate_radius_um / 2 / um_px, um_px):
            if plate is None or o["mean"] > plate["mean"]:
                plate = o | {"t": t}
    f["plate_aspect"] = round(plate["aspect"], 2) if plate else np.nan
    pcx, pcy = (plate["x"], plate["y"]) if plate else (cx, cy)

    # follow-up. Daughters are chosen, not assumed: in a crowd "every new nucleus near the plate" also
    # picks up neighbours (median 3 "daughters" on clicked bipolar divisions, first version). Neighbours
    # present just before the plate are followed by position and excluded; among what is left, the
    # daughter pair is the one that best fits a bipolar division -- opposite sides of the plate centre,
    # split along the plate's SHORT axis (skipped for a round, face-on plate), similar size, midpoint at
    # the centre -- and is then followed frame to frame. Further nuclei at the pair's distance from the
    # centre count as extra daughters (multipolar).
    neigh = [(o["x"], o["y"]) for o in _props(*get(max(0, t0 - 1))[::-1], pcx, pcy, r_px, um_px)
             if np.hypot(o["x"] - pcx, o["y"] - pcy) * um_px > p.fate_radius_um / 3]
    axis = None
    if plate and plate["aspect"] >= p.plate_axis_min_aspect:
        axis = np.array([np.sin(plate["orient"]), np.cos(plate["orient"])])   # long axis as (x, y)
    c = np.array([pcx, pcy])
    xy = lambda o: np.array([o["x"], o["y"]])

    def pair_cost(a, b):
        va, vb = xy(a) - c, xy(b) - c
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        opposite = 1.0 if min(na, nb) < 1e-6 else (1 + va @ vb / (na * nb)) / 2
        ab = vb - va
        across = abs(ab @ axis) / np.linalg.norm(ab) if axis is not None and np.linalg.norm(ab) > 0 else 0.0
        size = 1 - min(a["area"], b["area"]) / max(a["area"], b["area"])
        centred = np.linalg.norm((xy(a) + xy(b)) / 2 - c) / r_px
        return opposite + across + size + centred

    step = p.fate_radius_um / 2 / um_px
    seg_t, pair, n_d, frag, micro_hits, pairs, post_single = None, None, 0, 0, 0, [], None
    for t in range(t0 + 1, min(n, end + 1)):
        obs = _props(*get(t)[::-1], pcx, pcy, r_px, um_px)
        keep, nxt = [], []
        for o in obs:
            j = min(range(len(neigh)), key=lambda k: np.hypot(o["x"] - neigh[k][0], o["y"] - neigh[k][1]), default=None)
            if j is not None and np.hypot(o["x"] - neigh[j][0], o["y"] - neigh[j][1]) * um_px < p.split_um * 1.5:
                nxt.append((o["x"], o["y"]))      # follow the neighbour as it moves
            else:
                keep.append(o)
        neigh = nxt or neigh
        big = [o for o in keep if o["area"] >= small]
        tiny = [o for o in keep if o["area"] < small]
        frag = max(frag, len(tiny))
        if pair is None:
            if len(big) >= 2:
                cost, a, b = min(((pair_cost(a, b), a, b) for i, a in enumerate(big) for b in big[i + 1:]),
                                 key=lambda z: z[0])
                if cost <= p.pair_max_cost:
                    seg_t, pair = t, (a, b)
                    rr = (np.linalg.norm(xy(a) - c) + np.linalg.norm(xy(b) - c)) / 2
                    extra = [o for o in big if o is not a and o is not b
                             and abs(np.linalg.norm(xy(o) - c) - rr) < rr / 2]
                    n_d = 2 + len(extra)
                    pairs.append((t, a, b))
                    continue
            if len(big) == 1 and np.linalg.norm(xy(big[0]) - c) < r_px / 2:
                post_single = big[0]
            n_d = max(n_d, min(1, len(big)))
            continue
        # follow each daughter to the nearest nucleus within one step; both landing on one object = merged
        near = lambda d: min(big, key=lambda o: np.linalg.norm(xy(o) - xy(d)), default=None)
        a, b = near(pair[0]), near(pair[1])
        if a is None or b is None or max(np.linalg.norm(xy(a) - xy(pair[0])), np.linalg.norm(xy(b) - xy(pair[1]))) > step:
            break                                  # lost in the crowd: stop rather than guess
        pair = (a, b)
        pairs.append((t, a, b))
        if any(min(np.linalg.norm(xy(o) - xy(a)), np.linalg.norm(xy(o) - xy(b))) * um_px < p.fate_radius_um / 3
               for o in tiny):
            micro_hits += 1
    f["n_daughters"] = n_d
    f["fragments"] = frag
    f["micronuclei"] = int(micro_hits >= 2)
    f["mitotic_duration_min"] = round(float(minutes[seg_t] - minutes[t0]), 1) if seg_t is not None else np.nan
    if pairs:
        t, a, b = pairs[min(2, len(pairs) - 1)]          # 20 min in: past the anaphase blur
        f["daughter_area_ratio"] = round(min(a["area"], b["area"]) / max(a["area"], b["area"]), 2)
        f["daughter_intensity_ratio"] = round(min(a["mean"], b["mean"]) / max(a["mean"], b["mean"]), 2)
        sep = [(minutes[t], np.hypot(a["x"] - b["x"], a["y"] - b["y"]) * um_px) for t, a, b in pairs]
        f["separation_max_um"] = round(max(s for _, s in sep), 1)
        settle = next((i for i in range(1, len(sep))
                       if (sep[i][1] - sep[i - 1][1]) / max(sep[i][0] - sep[i - 1][0], 1e-6) < p.settle_um_per_min), None)
        if settle is not None:
            f["settle_min"] = round(float(sep[settle][0] - minutes[seg_t]), 1)
            f["separation_at_settle_um"] = round(sep[settle][1], 1)
        k = next((i for i, (tt, _, _) in enumerate(pairs) if minutes[tt] - minutes[seg_t] >= 30), len(pairs) - 1)
        _, a, b = pairs[k]
        da, db = np.hypot(a["x"] - pcx, a["y"] - pcy), np.hypot(b["x"] - pcx, b["y"] - pcy)
        f["motion_symmetry"] = round(min(da, db) / max(da, db, 1e-6), 2)
    elif post_single is not None and np.isfinite(m_area):
        f["post_area_fold"] = round(post_single["area"] / m_area, 2)
    post = _props(*get(min(n - 1, end))[::-1], pcx, pcy, r_px, um_px)
    if pre_sum and np.mean(pre_sum) > 0:
        f["signal_retained"] = round(sum(o["sum"] for o in post) / np.mean(pre_sum), 2)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2"); ap.add_argument("scan")
    ap.add_argument("--profile", help="defaults to the profile recorded in the scan's report.json")
    ap.add_argument("--channel", type=int); ap.add_argument("--position", type=int, default=0)
    a = ap.parse_args()
    scan = Path(a.scan)
    rep = json.loads((scan / "report.json").read_text())
    p = Profile.load(a.profile) if a.profile else Profile(**rep["profile"])
    if a.channel is not None:
        p.nucleus_channel = a.channel
    cand = pd.read_csv(scan / "candidates.csv")

    f = nd2.ND2File(a.nd2)
    dims, dask = list(f.sizes), f.to_dask()
    um_px, n = f.voxel_size().x, f.sizes["T"]
    shape = (f.sizes["Y"], f.sizes["X"])
    t_s = pd.DataFrame(f.events()).drop_duplicates("T Index")["Time [s]"].to_numpy()[:n]
    minutes = (t_s - t_s[0]) / 60
    pick = {"P": a.position, "C": p.nucleus_channel}
    cache = {}

    def get(t):
        if t not in cache:
            if len(cache) > 48:
                cache.pop(min(cache))
            im = np.asarray(dask[tuple(t if d == "T" else pick[d] for d in dims if d not in ("Y", "X"))])
            cache[t] = (im, segment(im, p, um_px))
        return cache[t]

    rows = []
    for i, ev in cand.sort_values("first").iterrows():     # time order keeps the frame cache warm
        rows.append({"cand": i} | profile_event(ev, get, minutes, um_px, p, shape))
    f.close()
    out = cand.reset_index(names="cand").merge(pd.DataFrame(rows), on="cand")
    out.to_csv(scan / "fate.csv", index=False)
    # denominator for rates: counts depend on how many cells were there, so report per 100 cell-hours
    # (nuclei per frame x frame interval, from the scan's own segmentation; edge objects included)
    per_frame = pd.read_csv(scan / "objects.csv.gz", usecols=["frame"]).frame.value_counts().sort_index()
    dt_h = np.diff(minutes, append=minutes[-1] + np.median(np.diff(minutes))) / 60
    cell_h = float((per_frame.reindex(range(n), fill_value=0).to_numpy() * dt_h).sum())
    summary = {"nuclei_per_frame_median": float(per_frame.median()), "cell_hours": round(cell_h, 1),
               "candidates": len(out), "candidates_per_100_cell_hours": round(100 * len(out) / cell_h, 2),
               "note": "candidates are not confirmed mitoses; multiply by measured precision, or count reviewed events"}
    (scan / "fate_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(out)} events -> {scan / 'fate.csv'}; {summary}")


if __name__ == "__main__":
    main()
