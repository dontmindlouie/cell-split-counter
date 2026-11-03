"""Estimate each mitosis_scan candidate's metaphase frame from the scan's own measurements.

    uv run python scripts/metaphase_frame.py data/mitosis_scan/<field> [<field dir> ...]

Writes <field dir>/metaphase.csv: cand, metaphase_frame, split_frame, basis.

The rule: the LAST frame before the daughters separate in which the plate is still
>= BRIGHT_FRAC of its own peak brightness. "Daughters separate" is mitosis_fate's
segregation frame (first + mitotic_duration_min); with no segregation seen (slippage,
binucleation, a crowd), the scan's last condensed frame stands in for it.
The plate's brightness per frame is the brightest object within RADIUS_UM of the
candidate in objects.csv.gz -- no image is re-read.

MEASURED 2026-11-02 against the researcher's metaphase frames ("f number on image")
on 902 real EVT mitoses over 11 fields of 20260822 EVT:

    rule                              exact   +-1 frame   +-2 frames
    scan's last condensed frame        24%       48%         77%
    this rule                          43%       65%         82%
    this rule, segregation seen (81%)  51%       70%         86%

BRIGHT_FRAC 0.80-0.90 all score within a point (a plateau, not a tuned peak). Tried
and NOT better on crop1_D4_4 (129 cards): stepping back past dumbbell-shaped frames
(at 0.865 um/px and 10-min frames anaphase rarely shows as one), and the most
elongated plate frame (elongation peaks on her frame on average but is noisy per
cell). Jonathan's check of 28 cards sided with her frame on 17 of 20 disagreements
with the scan's last frame, and put metaphase at 1-2 frames, so +-1 frame is the
realistic tolerance.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BRIGHT_FRAC = 0.85
RADIUS_UM = 12.0
UM_PX = 0.865       # 20260822 EVT; read from report.json when it records one


def estimate(cand: pd.DataFrame, fate: pd.DataFrame, obj: pd.DataFrame, frame_min: float,
             um_px: float = UM_PX, bright_frac: float = BRIGHT_FRAC) -> pd.DataFrame:
    """One row per candidate: metaphase_frame, split_frame (NaN if none seen), basis."""
    r_px = RADIUS_UM / um_px
    dur = fate.set_index("cand")["mitotic_duration_min"] if fate is not None else pd.Series(dtype=float)
    by_frame = dict(tuple(obj.groupby("frame")))
    rows = []
    for c, ev in cand.iterrows():
        first, last = int(ev["first"]), int(ev["last"])
        d = dur.get(c, np.nan)
        split = first + int(round(d / frame_min)) if pd.notna(d) else None
        hi = split - 1 if split is not None else last
        bright = {}
        for f in range(max(0, first - 1), hi + 1):
            o = by_frame.get(f)
            if o is None:
                continue
            o = o[np.hypot(o.x - ev["x"], o.y - ev["y"]) < r_px]
            if len(o):
                bright[f] = float(o["mean"].max())
        if bright:
            top = max(bright.values())
            meta = max(f for f, v in bright.items() if v >= bright_frac * top)
        else:
            meta = hi
        rows.append({"cand": c, "metaphase_frame": meta,
                     "split_frame": split if split is not None else np.nan,
                     "basis": "before segregation" if split is not None else "no segregation seen"})
    return pd.DataFrame(rows)


def run(scan: Path) -> Path:
    cand = pd.read_csv(scan / "candidates.csv")
    if "cand" in cand:
        cand = cand.set_index("cand")
    fate_p = scan / "fate.csv"
    fate = pd.read_csv(fate_p) if fate_p.exists() else None
    obj = pd.read_csv(scan / "objects.csv.gz", usecols=["frame", "x", "y", "mean"])
    k = cand["first"] > 0
    frame_min = float((cand["minutes"][k] / cand["first"][k]).median()) if k.any() else 10.0
    out = estimate(cand, fate, obj, frame_min)
    p = scan / "metaphase.csv"
    out.to_csv(p, index=False)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="+", help="data/mitosis_scan/<field> folders")
    for s in ap.parse_args().scan:
        print(run(Path(s)))
