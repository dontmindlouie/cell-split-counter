"""Turn a review_events.py list into show_cells_in_browser pages, set up by a config file.

The page is the same reusable browser for every analysis; what changes per analysis is the
config: the window (MINUTES, converted per event through the bundle's real frame clock),
crop sizes, which channels are shown and which start hidden, and the marking form.

    uv run python scripts/review_pages.py <well> <events.csv> <config.json>
        [--first N] [--limit N] [--ids ID,ID,...]
        [--before-min M] [--after-min M] [--events-per-part N] [--max-frames N]

MORE FRAMES is one knob: raise --after-min (or after_min in the config). Unless set
explicitly, the per-card frame cap follows the window (every frame, no sampling gaps) and
cards per part shrink so a part stays within frames_per_part frames in total -- longer
windows cost more parts, not heavier ones. To extend only some cards (the reviewer asked
for "more frames after" on them), rebuild just those: --ids <event_id,...> --after-min 720.

Config keys (all optional except review_form; the command-line flags override them):
    note            what the reader is asked to do (shown at the top of every part)
    before_min, after_min   window around each event's anchor frame (default 60 / 300)
    frames_per_part         page-weight budget, frames summed over a part's cards (default 240)
    events_per_part, max_frames   set either to pin it instead of deriving it
    crop_um, context_um     filmstrip crop and the wider lightbox crop (default 70 / 200)
    channels, channels_off  as in show_cells_in_browser
    channel_toggles         default false: the shown channels are combined into one image per
                            frame (contrast per card). true / --toggles: an on/off box per channel
                            in the browser, at about 3x the data with three channels
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
    ap.add_argument("--ids", help="comma-separated event_ids: build only these cards")
    ap.add_argument("--before-min", type=float); ap.add_argument("--after-min", type=float)
    ap.add_argument("--events-per-part", type=int); ap.add_argument("--max-frames", type=int)
    ap.add_argument("--toggles", action="store_true",
                    help="per-channel on/off in the browser (about 3x the data with three channels)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ev = pd.read_csv(a.events)
    if a.ids:
        want = [s.strip() for s in a.ids.split(",") if s.strip()]
        missing = sorted(set(want) - set(ev.event_id))
        if missing:
            raise SystemExit(f"not in {a.events}: {missing}")
        ev = ev[ev.event_id.isin(want)]
    ev = ev.iloc[a.first:]
    if a.limit:
        ev = ev.iloc[:a.limit]
    pick = lambda flag, key, default: flag if flag is not None else cfg.get(key, default)
    before = float(pick(a.before_min, "before_min", 60))
    after = float(pick(a.after_min, "after_min", 300))
    ts = np.asarray(m._manifest(a.well)["frame_timestamps_ms"], float)
    mins = (ts - ts[0]) / 60000.0
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
                       # plain text: the label is also the lightbox caption, set as text, so an
                       # HTML entity would show literally there
                       "label": f"{r.event_id} | {r.source} | Fiji frame {r.fiji_frame}{sug}"})
    if not events:
        raise SystemExit("no cards selected")
    longest = max(e["end_frame"] - e["start_frame"] + 1 for e in events)
    max_frames = int(pick(a.max_frames, "max_frames", None) or longest)
    per_part = pick(a.events_per_part, "events_per_part", None)
    if per_part is None:
        per_part = max(1, int(cfg.get("frames_per_part", 240)) // min(longest, max_frames))
    print(f"{len(events)} cards, window -{before:g}/+{after:g} min, up to {min(longest, max_frames)} frames "
          f"a card{' (sampled: raise --max-frames)' if longest > max_frames else ''}, {per_part} cards a part")
    print(m.show_cells_in_browser(a.well, events, note=cfg.get("note", ""), review_form=cfg["review_form"],
                                  channels=cfg.get("channels"), channels_off=cfg.get("channels_off"),
                                  channel_toggles=bool(a.toggles or cfg.get("channel_toggles", False)),
                                  events_per_part=int(per_part), max_frames=max_frames))


if __name__ == "__main__":
    main()
