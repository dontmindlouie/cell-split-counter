"""Scan a whole movie for mitoses by chromatin condensation, with every knob in a profile.

Why this exists next to find_candidates' `_condensation`: that one only scores divisions
the tracker already RECORDED, so a metaphase plate the segmenter dropped (the commonest
miss on H2B movies -- 9 of 12 misses on 20260822 EVT crop1_D4_1) is never asked about.
This scans every segmented object in every frame instead:

    segment  -> smooth, threshold, watershed-split touching nuclei (sizes in um)
    link     -> each object's parent = the previous-frame object it overlaps most
    score    -> mean signal and area against the SAME lineage's median BASE_MIN_LO..HI
                minutes earlier; a candidate is brighter by >= cond AND smaller by <= area
    fold     -> candidates within fold_um and fold_min of each other are one event

Nothing is in frames or pixels: a 4-min movie and a 10-min movie, or a 0.3 and a 0.86
um/px objective, read the same profile. The profile is where cell-type knowledge goes --
how long prophase-to-metaphase takes, how big an interphase nucleus is -- and it carries
a `rationale` per value so the next reader (or Claude) can tell a measured number from a
first guess.

Calibration (`--clicks`) is the other half: a guessed profile is a starting point, one
clicked field turns it into a measured one. It grid-searches cond x area and picks the
fewest candidates that still reach `target_recall` of the clicks.

usage:
    python -m src.mitosis_scan <nd2> --profile profiles/mitosis/h2b_tsc.json --out DIR
        [--clicks fiji_clicks.csv] [--frames N]
"""
import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import nd2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


@dataclass
class Profile:
    name: str
    nucleus_channel: int | None      # None = single-channel file (no C axis)
    smooth_um: float = 1.0           # gaussian sigma before thresholding
    threshold_scale: float = 0.5     # x Otsu. The biggest lever: at 1.0 dim interphase nuclei drop out
                                     # and a plate has no baseline (23/35 -> 33/35 on crop1_D4_1)
    local_bg_frac: float = 0.3       # also require bg + frac x (Otsu - mean bg); 0 = global floor only
    bg_window_um: float = 87.0       # local background box (her 101 px at 0.865 um/px)
    dist_smooth_px: float = 1.5      # smooths the distance map so one nucleus gives one seed
    split_um: float = 5.0            # min distance between watershed seeds ~ nucleus radius
    min_area_um2: float = 8.0        # drops debris; a metaphase plate is still >~15 um2
    base_min_lo: float = 30.0        # baseline window, minutes before the object
    base_min_hi: float = 60.0
    signal: str = "mean"             # "mean" or "top" (brightest point; survives merges)
    baseline: str = "lineage"        # "lineage" (overlap chain) or "nearest" (position only)
    base_radius_um: float = 6.0      # "nearest": how far a cell may have moved in that hour
    cond: float = 1.3                # mean signal >= cond x baseline
    area: float = 0.7                # area <= area x baseline
    fold_um: float = 20.0
    fold_min: float = 40.0
    edge_um: float = 7.0             # ignore objects centred this close to the image border: a cell
                                     # leaving the field shrinks and looks like a plate (6/78 of the
                                     # unclicked crop8_D0_1 candidates; 0 clicks lost on 3 fields)
    # cell-fate profiling (src/mitosis_fate.py); per cell type like the rest
    fate_radius_um: float = 25.0     # where to look for daughters: TSC daughters end ~10-20 um apart
    fate_window_min: float = 180.0   # follow-up after the plate; normal M phase + daughter settling < 2 h
    settle_um_per_min: float = 0.1   # daughters "stop moving apart" below this (1 um per 10-min frame)
    micro_frac: float = 0.15         # objects smaller than this x the interphase nucleus = micronucleus/fragment
    plate_axis_min_aspect: float = 1.3   # rounder plates are face-on: no usable division axis
    pair_max_cost: float = 1.5       # worst acceptable daughter-pair fit (0 = textbook bipolar; terms in mitosis_fate)
    grid_cond: list = field(default_factory=lambda: [1.15, 1.2, 1.3, 1.4, 1.6])
    grid_area: list = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    target_recall: float = 0.95
    rationale: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path) -> "Profile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def segment(img: np.ndarray, p: Profile, um_px: float) -> np.ndarray:
    sm = ndi.gaussian_filter(img.astype(np.float32), p.smooth_um / um_px)
    th = threshold_otsu(sm)
    floor = th * p.threshold_scale
    if p.local_bg_frac > 0:
        # her segment_field.py (20260822 EVT): also stay above the LOCAL background, so a
        # bright patch of field cannot pass on the global floor alone
        bg = ndi.uniform_filter(sm, int(p.bg_window_um / um_px) | 1)
        floor = np.maximum(floor, bg + p.local_bg_frac * (th - bg.mean()))
    fg = ndi.binary_fill_holes(ndi.binary_opening(sm > floor, iterations=1))
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(fg), p.dist_smooth_px)
    seeds = peak_local_max(dist, min_distance=max(1, int(p.split_um / um_px)), threshold_abs=1.5,
                           labels=ndi.label(fg)[0], exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    lab = watershed(-dist, markers, mask=fg)
    # blobs too small to hold a seed keep their own label instead of vanishing
    rest = ndi.label(fg & (lab == 0))[0]
    lab = np.where(rest > 0, rest + lab.max(), lab)
    areas = np.bincount(lab.ravel())
    small = np.nonzero(areas * um_px * um_px < p.min_area_um2)[0]
    lab[np.isin(lab, small[small > 0])] = 0
    return lab


def objects(stack: np.ndarray, labels_of, um_px: float) -> pd.DataFrame:
    """One row per object per frame: centroid (px), mean and peak signal, area (um2), parent label.

    `top` is the brightest pixel after a 1 px blur (one hot pixel is not a peak). It is kept
    for plates the watershed merges with a neighbour: the merged object's mean and area are
    dragged toward interphase, but its brightest point is still the plate."""
    rows, prev = [], None
    for t in range(len(stack)):
        L = labels_of(t).astype(np.int64)
        idx = np.unique(L); idx = idx[idx > 0]
        if len(idx):
            mean = ndi.mean(stack[t], L, idx)
            top = ndi.maximum(ndi.gaussian_filter(stack[t].astype(np.float32), 1), L, idx)
            area = ndi.sum(np.ones_like(L), L, idx) * um_px * um_px
            cy, cx = zip(*ndi.center_of_mass(np.ones_like(L), L, idx))
            par = {}
            if prev is not None:
                m = (prev > 0) & (L > 0)
                pr = pd.DataFrame({"b": L[m], "a": prev[m]}).value_counts().reset_index()
                par = pr.sort_values("count", ascending=False).drop_duplicates("b").set_index("b")["a"].to_dict()
            rows += [(t, l, cx[i], cy[i], mean[i], top[i], area[i], par.get(l, 0)) for i, l in enumerate(idx)]
        prev = L
    return pd.DataFrame(rows, columns="frame label x y mean top area parent".split())


def score(obj: pd.DataFrame, minutes: np.ndarray, p: Profile, um_px: float) -> pd.DataFrame:
    """Add cond/aratio: each object against its own past BASE_MIN_LO..HI minutes back."""
    if p.baseline == "nearest":
        return _score_nearest(obj, minutes, p, um_px)
    key = {(f, l): i for i, (f, l) in enumerate(zip(obj.frame, obj.label))}
    fr, par, mn, ar = obj.frame.values, obj.parent.values, obj[p.signal].values, obj.area.values
    bh, ba = np.full(len(obj), np.nan), np.full(len(obj), np.nan)
    for i in range(len(obj)):
        j, hh, aa = i, [], []
        while par[j]:
            j2 = key.get((fr[j] - 1, par[j]))
            if j2 is None:
                break
            j = j2
            dt = minutes[fr[i]] - minutes[fr[j]]
            if dt > p.base_min_hi:
                break
            if dt >= p.base_min_lo:
                hh.append(mn[j]); aa.append(ar[j])
        if hh:
            bh[i], ba[i] = np.median(hh), np.median(aa)
    return obj.assign(cond=mn / bh, aratio=ar / ba)


def _score_nearest(obj: pd.DataFrame, minutes: np.ndarray, p: Profile, um_px: float) -> pd.DataFrame:
    """Baseline WITHOUT the overlap chain: in each frame BASE_MIN_LO..HI back, take the
    object nearest this one's position (within base_radius_um) and use the median of those.

    The lineage chain breaks exactly where it is needed -- a cell rounding up for mitosis
    gets merged with a neighbour or re-split by the watershed, and on 20260822 crop1_D4_1
    that left 10 of 35 clicked mitoses with no baseline at all. A position-only baseline
    does not care what the tracker did; the cost is a neighbour standing in for a cell that
    moved more than base_radius_um in an hour, which the median damps."""
    from scipy.spatial import cKDTree
    by_f = {f: (cKDTree(g[["x", "y"]].to_numpy()), g[p.signal].to_numpy(), g.area.to_numpy())
            for f, g in obj.groupby("frame")}
    r_px = p.base_radius_um / um_px
    bh, ba = np.full(len(obj), np.nan), np.full(len(obj), np.nan)
    fr, xy = obj.frame.to_numpy(), obj[["x", "y"]].to_numpy()
    for f in np.unique(fr):
        back = [g for g in range(f) if p.base_min_lo <= minutes[f] - minutes[g] <= p.base_min_hi and g in by_f]
        if not back:
            continue
        rows = np.nonzero(fr == f)[0]
        hh = np.full((len(rows), len(back)), np.nan); aa = hh.copy()
        for k, g in enumerate(back):
            tree, mn, ar = by_f[g]
            d, j = tree.query(xy[rows], distance_upper_bound=r_px)
            ok = np.isfinite(d)
            hh[ok, k], aa[ok, k] = mn[j[ok]], ar[j[ok]]
        bh[rows], ba[rows] = np.nanmedian(hh, axis=1), np.nanmedian(aa, axis=1)
    return obj.assign(cond=obj[p.signal].to_numpy() / bh, aratio=obj.area.to_numpy() / ba)


def events(obj: pd.DataFrame, minutes: np.ndarray, um_px: float, cond: float, area: float,
           p: Profile) -> list[dict]:
    c = obj[(obj.cond >= cond) & (obj.aratio <= area)].sort_values("frame")
    ev = []
    for r in c.itertuples():
        e = next((e for e in reversed(ev)
                  if 0 <= minutes[r.frame] - minutes[e["last"]] <= p.fold_min
                  and np.hypot(r.x - e["x"], r.y - e["y"]) * um_px < p.fold_um), None)
        if e:
            e["last"] = r.frame; e["peak"] = max(e["peak"], r.cond)
        else:
            ev.append({"first": r.frame, "last": r.frame, "x": r.x, "y": r.y, "peak": r.cond})
    return ev


def recall(ev: list[dict], clicks: pd.DataFrame, minutes: np.ndarray, um_px: float, p: Profile) -> int:
    return sum(any(abs(minutes[e["first"]] - minutes[c.frame]) <= p.fold_min * 2
                   and np.hypot(e["x"] - c.x, e["y"] - c.y) * um_px < p.fold_um * 1.3 for e in ev)
               for c in clicks.itertuples())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2"); ap.add_argument("--profile", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--clicks"); ap.add_argument("--frames", type=int)
    ap.add_argument("--channel", help="nucleus channel index, or 'none' for a single-channel file "
                                      "(overrides the profile -- a fact about the file, not a tuning knob)")
    ap.add_argument("--position", type=int, help="P index of a multi-position ND2")
    ap.add_argument("--fixed", action="store_true",
                    help="with --clicks: score the profile's own cond/area instead of picking the best of the grid "
                         "(the honest test of a profile on new data)")
    a = ap.parse_args()
    p, out = Profile.load(a.profile), Path(a.out)
    if a.channel is not None:
        p.nucleus_channel = None if a.channel == "none" else int(a.channel)
    out.mkdir(parents=True, exist_ok=True)
    cl = None
    if a.clicks:
        cl = pd.read_csv(a.clicks)
    f = nd2.ND2File(a.nd2)
    dims, dask = list(f.sizes), f.to_dask()
    um_px = f.voxel_size().x
    n = min(f.sizes["T"], a.frames or f.sizes["T"])
    t_s = pd.DataFrame(f.events()).drop_duplicates("T Index")["Time [s]"].to_numpy()[:n]
    minutes = (t_s - t_s[0]) / 60
    pick = {"P": a.position or 0, "C": p.nucleus_channel}

    class _Frames:
        """Frame-at-a-time view of the nucleus plane: a 1,273-frame position would be ~3 GB of
        pixels plus as much again of labels if held whole. Caches the last frame read."""
        last = (None, None)
        def __len__(self):
            return n
        def __getitem__(self, t):
            if self.last[0] != t:
                idx = tuple(t if d == "T" else pick[d] for d in dims if d not in ("Y", "X"))
                self.last = (t, np.asarray(dask[idx]))
            return self.last[1]

    frames, at_clicks = _Frames(), {}
    want = {} if cl is None else cl[cl.frame < n].groupby("frame")[["y", "x"]].apply(lambda g: g.to_numpy()).to_dict()

    def labels_of(t):
        lab = segment(frames[t], p, um_px)
        if t in want:
            at_clicks[t] = lab[want[t][:, 0], want[t][:, 1]]
        return lab

    obj = score(objects(frames, labels_of, um_px), minutes, p, um_px)
    H, W = f.sizes["Y"], f.sizes["X"]
    f.close()
    obj.to_csv(out / "objects.csv.gz", index=False)
    e = p.edge_um / um_px
    inside = obj[(obj.x >= e) & (obj.y >= e) & (obj.x <= W - 1 - e) & (obj.y <= H - 1 - e)]
    report = {"profile": asdict(p), "um_px": um_px, "frames": n,
              "objects_per_frame": round(len(obj) / n)}
    cond, area = p.cond, p.area
    if cl is not None:
        cl = cl[cl.frame < n]
        report["clicks"] = len(cl)
        report["clicks_in_mask"] = int(sum((v > 0).sum() for v in at_clicks.values()))
        if a.fixed:
            ev = events(inside, minutes, um_px, cond, area, p)
            lo, hi = cl.frame.min(), cl.frame.max()
            report["chosen"] = {"cond": cond, "area": area, "candidates": len(ev),
                                "found": int(recall(ev, cl, minutes, um_px, p)), "met_target": None}
            # clicks may cover only part of a movie; candidates outside that span cannot be judged
            report["candidates_in_clicked_span"] = int(sum(lo <= e["first"] <= hi for e in ev))
        grid = []
        for C in p.grid_cond:
            for A in p.grid_area:
                ev = events(inside, minutes, um_px, C, A, p)
                grid.append({"cond": C, "area": A, "candidates": len(ev),
                             "found": int(recall(ev, cl, minutes, um_px, p))})
        report["grid"] = grid
        if not a.fixed:
            ok = [g for g in grid if g["found"] >= p.target_recall * len(cl)]
            best = min(ok, key=lambda g: g["candidates"]) if ok else max(grid, key=lambda g: (g["found"], -g["candidates"]))
            cond, area = best["cond"], best["area"]
            report["chosen"] = best | {"met_target": bool(ok)}
    ev = events(inside, minutes, um_px, cond, area, p)
    pd.DataFrame(ev).assign(minutes=lambda d: minutes[d["first"]].round(1)).to_csv(out / "candidates.csv", index=False)
    report["candidates"] = len(ev)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("profile", "grid")}))


if __name__ == "__main__":
    main()
