"""Check each mitosis_scan candidate for the SEQUENCE a human clicks, not one bright frame.

mitosis_scan finds "a nucleus brighter and no bigger than 30-60 min ago". That is the metaphase
plate alone, and on its own it is ~3-4 candidates per real mitosis. A person scoring the movie
(the maintainer, 2026-10-25) recognises a division by what happens AROUND the plate:

    plate     smaller, brighter, ELONGATED object near where a round nucleus was
    daughters two objects appear 0-3 frames later (the plate often vanishes for a frame first)
    axis      they sit off the plate's FLAT side -- along its short axis, not off its pointy ends
    motion    they move apart in opposite directions at similar speed; one fast mover alone is
              not a division

So for every candidate: find its plate (the peak frame), measure its long axis from the pixel
cloud, then look for a daughter pair in the next frames whose two displacement vectors from the
plate centre point opposite ways, have similar lengths, and lie across the long axis. A face-on
plate is round and has no axis, so the axis test only applies above `min_aspect`.

usage: python -m src.mitosis_verify <nd2> --profile P --scan DIR [--clicks CSV]
  DIR is a mitosis_scan --out dir (objects.csv.gz). Writes DIR/verified.csv; with --clicks,
  prints how many clicked mitoses survive vs how many candidates are dropped.
"""
import argparse
from pathlib import Path

import nd2
import numpy as np
import pandas as pd

from src.mitosis_scan import Profile, segment, score, events, recall

DAUGHTER_UM = 18.0      # daughters land within this of the plate centre
LOOK_AHEAD = 3          # frames after the plate to look for them (10-min frames skip anaphase)
OPPOSITE_COS = -0.5     # displacement vectors at least 120 deg apart
DIST_RATIO = 0.4        # shorter / longer displacement; one fast mover alone fails
AREA_RATIO = 0.4        # smaller / larger daughter
MIN_ASPECT = 1.4        # below this a plate is face-on: no axis to test
AXIS_COS = 0.71         # |cos| between daughter axis and plate long axis: must be >= 45 deg off


def _axis(mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Aspect ratio and unit long-axis vector (x, y) of a pixel cloud, from its covariance."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return 1.0, np.array([1.0, 0.0])
    w, v = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    return float(np.sqrt(w[1] / max(w[0], 1e-6))), v[:, 1]


def verify(ev: list[dict], obj: pd.DataFrame, labels, um_px: float, p: Profile) -> pd.DataFrame:
    by_f = {f: g for f, g in obj.groupby("frame")}
    r_px = DAUGHTER_UM / um_px
    out = []
    for e in ev:
        near = obj[obj.frame.between(e["first"], e["last"])
                   & (np.hypot(obj.x - e["x"], obj.y - e["y"]) * um_px < p.fold_um)]
        row = dict(e, plate_frame=-1, aspect=np.nan, daughter_frame=-1, opp_cos=np.nan,
                   dist_ratio=np.nan, axis_cos=np.nan, verdict="no plate")
        if near.cond.notna().any():
            pl = near.loc[near.cond.idxmax()]
            pf, cx, cy = int(pl.frame), pl.x, pl.y
            aspect, ax = _axis(labels(pf) == int(pl.label))
            row.update(plate_frame=pf, aspect=round(aspect, 2), verdict="no daughters")
            best = None
            for f in range(pf, pf + LOOK_AHEAD + 1):
                g = by_f.get(f)
                if g is None:
                    continue
                g = g[(np.hypot(g.x - cx, g.y - cy) < r_px) & ~((f == pf) & (g.label == pl.label))]
                pts, ar = g[["x", "y"]].to_numpy() - [cx, cy], g.area.to_numpy()
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        di, dj = np.linalg.norm(pts[i]), np.linalg.norm(pts[j])
                        if min(di, dj) * um_px < 1.5:
                            continue
                        oc = float(pts[i] @ pts[j] / (di * dj))
                        dr = min(di, dj) / max(di, dj)
                        arr = min(ar[i], ar[j]) / max(ar[i], ar[j])
                        pair = pts[i] - pts[j]
                        acos = abs(float(pair @ ax / np.linalg.norm(pair)))
                        ok = oc <= OPPOSITE_COS and dr >= DIST_RATIO and arr >= AREA_RATIO
                        axis_ok = aspect < MIN_ASPECT or acos <= AXIS_COS
                        cand = (ok and axis_ok, -oc + dr, f, oc, dr, acos)
                        if best is None or cand[:2] > best[:2]:
                            best = cand
            if best:
                passed, _, f, oc, dr, acos = best
                row.update(daughter_frame=f, opp_cos=round(oc, 2), dist_ratio=round(dr, 2),
                           axis_cos=round(acos, 2),
                           verdict="division" if passed else "daughters fail geometry")
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2"); ap.add_argument("--profile", required=True); ap.add_argument("--scan", required=True)
    ap.add_argument("--clicks")
    a = ap.parse_args()
    p, d = Profile.load(a.profile), Path(a.scan)
    with nd2.ND2File(a.nd2) as f:
        um_px = f.voxel_size().x
        t_s = pd.DataFrame(f.events()).drop_duplicates("T Index")["Time [s]"].to_numpy()
        lab_path = d / "labels.npy"
        if not lab_path.exists():
            stack = f.to_dask()[:, p.nucleus_channel]
            labs = np.zeros(stack.shape, np.uint16)
            for t in range(stack.shape[0]):
                labs[t] = segment(np.asarray(stack[t]), p, um_px)
            np.save(lab_path, labs)
    labs = np.load(lab_path, mmap_mode="r")
    minutes = (t_s - t_s[0]) / 60
    obj = pd.read_csv(d / "objects.csv.gz")[["frame", "label", "x", "y", "mean", "top", "area", "parent"]]
    obj = score(obj, minutes, p, um_px)
    ev = events(obj, minutes, um_px, p.cond, p.area, p)
    v = verify(ev, obj, lambda t: labs[t], um_px, p)
    v.to_csv(d / "verified.csv", index=False)
    print(v.verdict.value_counts().to_dict())
    if a.clicks:
        cl = pd.read_csv(a.clicks)
        keep = v[v.verdict == "division"].to_dict("records")
        print(f"clicks found: all candidates {recall(ev, cl, minutes, um_px, p)}/{len(cl)} of {len(ev)}; "
              f"verified {recall(keep, cl, minutes, um_px, p)}/{len(cl)} of {len(keep)}")


if __name__ == "__main__":
    main()
