"""Search all 5 Bewo wells' unflagged death pools for candidate real-necrosis events,
using the recalibrated death-review prompt (see rereview_deaths_recalibrated.py,
83.7% accuracy vs. 63.3% baseline on the 49-event Batch B golden set) as a
high-precision filter: it only said "genuine death" for 7/49 golden-set events, each
grounded in specific fragmentation language, not a coin flip -- so surfacing every
event it still calls death despite the strong dropout-default bias should be a much
higher-yield candidate pool for human review than raw keyword matching (tested
earlier this session at ~53% accuracy, worse than a trivial baseline).

Restricted to the "unflagged" pool (likely_division_dropout != 1) per Batch B's own
finding that the one real-death example lived there, not in the flagged pool.

Checkpointed: writes each result to CSV as it completes (learned from
project_cell_split_counter_checkpoint_backlog -- a prior M4 run got killed at 21% and
lost all progress). Safe to re-run; skips track_ids already in the output CSV.

Usage:
    python scripts/eval_harness/find_necrosis_candidates.py --dry-run
    python scripts/eval_harness/find_necrosis_candidates.py
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv; load_dotenv()

from openai import AzureOpenAI

from scripts.eval_harness.death_golden_set import load_death_golden_set
from scripts.eval_harness.rereview_deaths_recalibrated import review_recalibrated
from src.classify import EventType, LineageEvent

REPO = Path(__file__).resolve().parents[2]
WELLS = ["M2", "M3", "M4", "M5", "M6"]
OUT_CSV = Path("G:/Projects/cell-split-counter-shared-data/human_review/necrosis_candidates_2026-07-23.csv")
MAX_WORKERS = 10

FIELDNAMES = ["well", "track_id", "parent_id", "peak_frame", "recalibrated_dropout", "confidence", "description"]


def load_well_events(well: str) -> dict[int, LineageEvent]:
    """Every unflagged death-topology row for a well."""
    events_csv = REPO / f"data/output/202660629_Bewop920x_{well}/events.csv"
    rows = list(csv.DictReader(events_csv.open(encoding="utf-8")))
    out = {}
    for r in rows:
        if r["split_topology"] != "death" or r.get("likely_division_dropout") == "1":
            continue
        tid = int(r["track_id"])
        out[tid] = LineageEvent(
            track_id=tid,
            parent_id=int(float(r["parent_id"])) if r.get("parent_id") else None,
            frame=int(r["peak_frame"]),
            event_type=EventType.DEATH,
            classification_source="find_necrosis_candidates",
            confidence=0.0,
            centroid=(float(r["centroid_x"]), float(r["centroid_y"])),
            cell_area_px=float(r["cell_area_px"]) if r.get("cell_area_px") else None,
            neighbor_distance_px=float(r["neighbor_distance_px"]) if r.get("neighbor_distance_px") else None,
            neighbor_area_px=float(r["neighbor_area_px"]) if r.get("neighbor_area_px") else None,
        )
    return out


def load_already_done() -> set[tuple[str, int]]:
    if not OUT_CSV.exists():
        return set()
    rows = list(csv.DictReader(OUT_CSV.open(encoding="utf-8")))
    return {(r["well"], int(r["track_id"])) for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    golden_tracks = set(load_death_golden_set().labels)  # M4's 49 already scored, skip
    already_done = load_already_done()

    work: list[tuple[str, int, LineageEvent, Path]] = []
    for well in WELLS:
        events = load_well_events(well)
        frame_dir = REPO / f"data/output/202660629_Bewop920x_{well}/frames"
        for tid, ev in events.items():
            if well == "M4" and tid in golden_tracks:
                continue
            if (well, tid) in already_done:
                continue
            work.append((well, tid, ev, frame_dir))

    print(f"{len(work)} events to review ({len(already_done)} already done, resuming)")
    for well in WELLS:
        n = sum(1 for w, *_ in work if w == well)
        print(f"  {well}: {n} to review")

    if args.dry_run:
        return

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview", max_retries=8, timeout=90.0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    file_exists = OUT_CSV.exists()
    out_f = OUT_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    usage_log: list = []
    n_death_calls = 0
    done = 0

    def _call(well: str, tid: int, ev: LineageEvent, frame_dir: Path):
        # review_recalibrated hardcodes FRAME_DIR at import time in the source module;
        # temporarily override via a module-level patch instead of duplicating the function.
        import scripts.eval_harness.rereview_deaths_recalibrated as recal_mod
        original_frame_dir = recal_mod.FRAME_DIR
        recal_mod.FRAME_DIR = frame_dir
        try:
            result = review_recalibrated(client, ev, usage_log)
        except Exception as exc:
            result = {"error": str(exc), "likely_division_dropout": None}
        finally:
            recal_mod.FRAME_DIR = original_frame_dir
        return well, tid, ev, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(_call, well, tid, ev, frame_dir) for well, tid, ev, frame_dir in work]
        for future in as_completed(futures):
            well, tid, ev, result = future.result()
            done += 1
            pred = result.get("likely_division_dropout")
            row = {
                "well": well, "track_id": tid, "parent_id": ev.parent_id, "peak_frame": ev.frame,
                "recalibrated_dropout": pred, "confidence": result.get("confidence"),
                "description": result.get("description", result.get("error", "")),
            }
            with write_lock:
                writer.writerow(row)
                out_f.flush()
            if pred is False:
                n_death_calls += 1
                print(f"[{done}/{len(work)}] {well} track {tid:>6}: CANDIDATE (predicted death) -- {row['description'][:100]}")
            elif done % 50 == 0:
                print(f"[{done}/{len(work)}] progress checkpoint, {n_death_calls} candidates so far")

    out_f.close()
    from src.review import _estimate_cost_usd
    from src.config import GPT_DEPLOYMENT
    cost = sum(_estimate_cost_usd(GPT_DEPLOYMENT, u["input_tokens"], u["output_tokens"]) or 0 for u in usage_log)
    print(f"\ndone: {done} reviewed, {n_death_calls} candidate necrosis events found")
    print(f"cost: ${cost:.4f}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
