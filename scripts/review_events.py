"""Merge events from any sources into one list of review cards, one card per real event.

A field can have the same mitosis clicked by two people and found by the detector too.
Reviewing each once means merging them first: events closer than --merge-um and
--merge-min become one card that keeps every source and every source id.

    uv run python scripts/review_events.py <well> --out events.csv \
        --clicks clicker_a=<clicks.csv> --clicks clicker_b=<clicks.csv> \
        --candidates <mitosis_scan out dir>

Clicks CSVs are scripts/convert_fiji_clicks.py output (click, frame 0-based, x, y px).
A candidates dir is src.mitosis_scan output; its fate.csv (src.mitosis_fate), if present,
adds the fate variables and SUGGESTIONS -- hints for the reviewer to confirm, never
filters: multipolar? (3+ daughters), disruption? (fragments and lost H2B),
long mitosis? (>= 80 min plate to segregation, which precedes cytokinesis failure).

Output: one row per card, ordered for review -- suggested errors first, then clicked
events, then detector-only by peak. Columns: event_id, frame (0-based, bundle),
fiji_frame (1-based), minutes (real time from the ND2 clock), x, y (px), x_um, y_um,
source, n_sources, ids, suggest, then any passthrough columns. Times come from the
bundle's frame timestamps, so 3-min, 10-min and drifting intervals all merge correctly.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import DATA_DIR  # noqa: E402

FATE_COLS = ["n_daughters", "mitotic_duration_min", "fragments", "signal_retained",
             "daughter_area_ratio", "motion_symmetry", "separation_at_settle_um", "followup_min"]


def suggestions(r):
    s = []
    if r.get("n_daughters", 0) >= 3:
        s.append("multipolar?")
    if r.get("fragments", 0) >= 2 and r.get("signal_retained", 1) < 0.6:
        s.append("disruption?")
    if r.get("mitotic_duration_min", 0) >= 80:
        s.append("long mitosis?")
    return " ".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("well"); ap.add_argument("--out", required=True)
    ap.add_argument("--clicks", action="append", default=[], help="NAME=path, repeatable")
    ap.add_argument("--candidates", action="append", default=[], help="mitosis_scan out dir, repeatable")
    # The same rule mitosis_scan uses to say a click was "found" (fold_um x 1.3, fold_min x 2):
    # a click sits on anaphase or the daughters, up to ~an hour after the plate the detector
    # flags. 20 um / 40 min left 13 of 33 crop8_D0_1 clicks as separate cards from their own
    # detection; this leaves 6, the ones the detector did not find.
    ap.add_argument("--merge-um", type=float, default=26.0)
    ap.add_argument("--merge-min", type=float, default=80.0)
    ap.add_argument("--bundle", default=str(REPO / DATA_DIR / "bundle"))
    a = ap.parse_args()
    man = json.loads((Path(a.bundle) / a.well / "manifest.json").read_text(encoding="utf-8"))
    ts = np.asarray(man["frame_timestamps_ms"], float)
    minutes = (ts - ts[0]) / 60000.0
    um = float(man["pixel_size_um"])

    rows = []
    for spec in a.clicks:
        name, path = spec.split("=", 1)
        for r in pd.read_csv(path).itertuples():
            rows.append({"frame": int(r.frame), "x": float(r.x), "y": float(r.y), "source": name,
                         "id": f"{name}:{r.click}", "clicked": True, "peak": np.nan})
    for d in a.candidates:
        d = Path(d)
        c = pd.read_csv(d / "fate.csv") if (d / "fate.csv").exists() else \
            pd.read_csv(d / "candidates.csv").reset_index(names="cand")
        for r in c.to_dict("records"):
            rows.append({"frame": int(r["first"]), "x": float(r["x"]), "y": float(r["y"]),
                         "source": "detector", "id": f"detector:{r['cand']}", "clicked": False,
                         "peak": float(r["peak"]), **{k: r[k] for k in FATE_COLS if k in r}})
    if not rows:
        raise SystemExit("no events given")
    ev = pd.DataFrame(rows)
    ev["minutes"] = minutes[ev.frame.clip(0, len(minutes) - 1)]

    # Greedy merge in time order. A card's anchor is its first CLICK if it has one (a
    # person put the click where the event is), else its first detection.
    cards = []
    for r in ev.sort_values(["minutes", "clicked"], ascending=[True, False]).to_dict("records"):
        hit = next((c for c in reversed(cards)
                    if abs(r["minutes"] - c["minutes"]) <= a.merge_min
                    and np.hypot(r["x"] - c["x"], r["y"] - c["y"]) * um <= a.merge_um), None)
        if hit is None:
            cards.append({**r, "members": [r]})
            continue
        hit["members"].append(r)
        if r["clicked"] and not hit["clicked"]:
            hit.update(frame=r["frame"], x=r["x"], y=r["y"], minutes=r["minutes"], clicked=True)
    out = []
    for c in cards:
        m = pd.DataFrame(c["members"])
        det = m[m.source == "detector"]
        row = {"frame": c["frame"], "fiji_frame": c["frame"] + 1, "minutes": round(c["minutes"], 1),
               "x": round(c["x"], 1), "y": round(c["y"], 1),
               "x_um": round(c["x"] * um, 2), "y_um": round(c["y"] * um, 2),
               "source": "+".join(sorted(m.source.unique())), "n_sources": m.source.nunique(),
               "ids": " ".join(m["id"]), "clicked": bool(m.clicked.any()),
               "peak": round(det.peak.max(), 2) if len(det) else np.nan}
        if len(det):
            best = det.loc[det.peak.idxmax()].to_dict()
            row.update({k: best[k] for k in FATE_COLS if k in best})
        row["suggest"] = suggestions(row)
        out.append(row)
    o = pd.DataFrame(out)
    o["_order"] = np.select([o.suggest != "", o.clicked], [0, 1], 2)
    o = o.sort_values(["_order", "peak", "minutes"], ascending=[True, False, True]).drop(columns="_order")
    o.insert(0, "event_id", [f"{a.well}#{i + 1:04d}" for i in range(len(o))])
    o.to_csv(a.out, index=False)
    print(f"{len(ev)} source rows -> {len(o)} cards: "
          + ", ".join(f"{k} {v}" for k, v in o.source.value_counts().items())
          + f"; {int((o.suggest != '').sum())} with suggestions -> {a.out}")


if __name__ == "__main__":
    main()
