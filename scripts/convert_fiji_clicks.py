"""Fiji multipoint Results rows -> one row per mitosis (0-based frame, pixel x/y), checked against the ND2.

Paste the Results rows (Analyze > Measure on a multipoint selection) into a .txt file as they are:
    n  Area  Mean  Min  Max  X  Y  Ch  Frame
X/Y are calibrated (um) and sub-pixel in Fiji, so px = round(X / um_per_px); Frame and Ch are 1-based.
Every click's Mean is checked against the raw pixel (px_ok), which catches a wrong file or channel.
Clicks on the same division twice (within 15 px and 6 frames) are folded into the first.

    uv run python scripts/convert_fiji_clicks.py <rows.txt> <nd2> <out.csv>

Assumes a T, C, Y, X ND2 (the 20260822 EVT crops). Single-channel or multi-position files need the
indexing changed.
"""
import sys

import nd2
import numpy as np
import pandas as pd

rows = pd.read_csv(sys.argv[1], sep=r"\s+", header=None, names="n area mean min max X Y ch frame".split())
with nd2.ND2File(sys.argv[2]) as f:
    um = f.voxel_size().x; d = f.to_dask()
    out = []
    for r in rows.itertuples():
        t, c, x, y = r.frame - 1, r.ch - 1, int(round(r.X / um)), int(round(r.Y / um))
        img = np.asarray(d[t, c, max(0, y - 1):y + 2, max(0, x - 1):x + 2])
        out.append(dict(click=r.n, frame=t, x=x, y=y, mean=r.mean, px_ok=bool((img == r.mean).any())))
c = pd.DataFrame(out)
keep, dup_of = [], {}
for r in c.itertuples():
    m = [k for k in keep if abs(c.loc[k, "frame"] - r.frame) <= 6 and np.hypot(c.loc[k, "x"] - r.x, c.loc[k, "y"] - r.y) < 15]
    if m: dup_of[r.click] = int(c.loc[m[0], "click"])
    else: keep.append(r.Index)
print("pixel-verified", int(c.px_ok.sum()), "/", len(c), "| duplicates", dup_of)
c.loc[keep, ["click", "frame", "x", "y"]].to_csv(sys.argv[3], index=False)
print("unique", len(keep), "->", sys.argv[3])
