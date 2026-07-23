"""Score a set of {track_id: likely_division_dropout} predictions against the death
golden set (death_golden_set.py). Shared by the zero-cost baseline (existing flag
already in events.csv) and any new re-review config (dense window, new prompt, etc).

Note on metrics: the golden set is heavily skewed (48 True / 1 False in Batch B) --
plain accuracy is dominated by the majority class and a trivial "always predict True"
baseline scores 48/49 = 98%. Report the confusion matrix alongside accuracy so a config
that just predicts the majority class isn't mistaken for a working necrosis detector.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_harness.death_golden_set import DeathGoldenSet, load_death_golden_set

REPO = Path(__file__).resolve().parents[2]


def score(predictions: dict[int, bool | None], golden: DeathGoldenSet) -> dict:
    tp = tn = fp = fn = unscored = 0
    misses = []
    for track_id, expected in golden.labels.items():
        pred = predictions.get(track_id)
        if pred is None:
            unscored += 1
            continue
        if expected and pred:
            tp += 1
        elif not expected and not pred:
            tn += 1
        elif expected and not pred:
            fn += 1
            misses.append((track_id, expected, pred))
        else:
            fp += 1
            misses.append((track_id, expected, pred))
    scored = tp + tn + fp + fn
    accuracy = (tp + tn) / scored if scored else None
    return {
        "scored": scored, "unscored": unscored,
        "tp_alive_correctly_flagged": tp, "tn_death_correctly_flagged": tn,
        "fp_alive_called_death": fn,  # expected True (alive), predicted False -- costly, hides a real division
        "fn_death_called_alive": fp,  # expected False (death), predicted True -- the necrosis-detection miss
        "accuracy": accuracy,
        "misses": misses,
    }


def baseline_predictions_from_events_csv(
    events_csv: Path, golden: DeathGoldenSet
) -> dict[int, bool | None]:
    """Zero-cost baseline: the likely_division_dropout flag already stored in a run's
    events.csv, for exactly the tracks in the golden set."""
    rows = list(csv.DictReader(events_csv.open(encoding="utf-8")))
    by_track = {int(r["track_id"]): r for r in rows if r["split_topology"] == "death"}
    preds: dict[int, bool | None] = {}
    for track_id in golden.labels:
        row = by_track.get(track_id)
        if row is None:
            preds[track_id] = None
            continue
        flag = row.get("likely_division_dropout")
        preds[track_id] = {"1": True, "0": False, "": None}.get(flag, None)
    return preds


def print_report(label: str, result: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"scored={result['scored']} unscored={result['unscored']}")
    print(f"accuracy: {result['accuracy']:.1%}" if result["accuracy"] is not None else "accuracy: n/a")
    print(f"confusion: TP(alive/alive)={result['tp_alive_correctly_flagged']}  "
          f"TN(death/death)={result['tn_death_correctly_flagged']}  "
          f"FN(alive predicted death)={result['fp_alive_called_death']}  "
          f"FP(death predicted alive)={result['fn_death_called_alive']}")
    if result["misses"]:
        print("misses:")
        for track_id, expected, pred in result["misses"]:
            print(f"  track {track_id}: expected dropout={expected}  predicted dropout={pred}")


if __name__ == "__main__":
    golden = load_death_golden_set()
    print(f"Golden set: {len(golden)} labels ({golden.stats['label_counts']})")

    m4_events = REPO / "data/output/202660629_Bewop920x_M4/events.csv"
    baseline = baseline_predictions_from_events_csv(m4_events, golden)
    result = score(baseline, golden)
    print_report("Baseline: existing likely_division_dropout flag (sparse stride-3 window)", result)
