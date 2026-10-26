"""Turn a review_events.py list into show_cells_in_browser pages, set up by a config file.

The page is the same reusable browser for every analysis; what changes per analysis is the
config: the window (MINUTES, converted per event through the bundle's real frame clock),
crop sizes, which channels are shown and which start hidden, and the marking form.

    uv run python scripts/review_pages.py <well> <events.csv> <config.json> [--first N] [--limit N]

Config keys (all optional except review_form):
    note            what the reader is asked to do (shown at the top of every part)
    before_min, after_min   window around each event's anchor frame (default 60 / 300)
    crop_um, context_um     filmstrip crop and the wider lightbox crop (default 70 / 200)
    channels, channels_off  as in show_cells_in_browser
    review_form     as in show_cells_in_browser -- put the main verdict first in
                    "choices" so the 1-9 keys mark it
Each card's label carries its source, Fiji frame and any suggestion; its id is the
event_id, which the CSV export keeps, so marks join back to events.csv.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cell_mcp_server as m  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("well"); ap.add_argument("events"); ap.add_argument("config")
    ap.add_argument("--first", type=int, default=0); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ev = pd.read_csv(a.events).iloc[a.first:]
    if a.limit:
        ev = ev.iloc[:a.limit]
    ts = np.asarray(m._manifest(a.well)["frame_timestamps_ms"], float)
    mins = (ts - ts[0]) / 60000.0
    before, after = float(cfg.get("before_min", 60)), float(cfg.get("after_min", 300))
    events = []
    for r in ev.itertuples():
        t = mins[int(r.frame)]
        start = int(np.searchsorted(mins, t - before, side="left"))
        end = int(np.searchsorted(mins, t + after, side="right")) - 1
        sug = f" -- suggested: {r.suggest}" if isinstance(r.suggest, str) and r.suggest else ""
        events.append({"kind": "point", "x": float(r.x), "y": float(r.y),
                       "start_frame": start, "end_frame": max(end, start + 1),
                       "crop_um": float(cfg.get("crop_um", 70)), "context_um": float(cfg.get("context_um", 200)),
                       "id": r.event_id,
                       "label": f"{r.event_id} &middot; {r.source} &middot; Fiji frame {r.fiji_frame}{sug}"})
    print(m.show_cells_in_browser(a.well, events, note=cfg.get("note", ""), review_form=cfg["review_form"],
                                  channels=cfg.get("channels"), channels_off=cfg.get("channels_off")))


if __name__ == "__main__":
    main()
