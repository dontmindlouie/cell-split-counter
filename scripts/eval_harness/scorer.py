"""Score a run's events.csv against the Tier A golden set.

Deterministic, no API calls, no human involved -- this is what makes Tier A sweeps
hands-off. Usage:

    python scripts/eval_harness/scorer.py data/output/<run_dir>/events.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_harness.golden_set import GoldenSet, load_golden_set


def effective_verdict(row: dict) -> int:
    """Real/false_positive as events.csv actually shipped it -- see
    scripts/reports/spot_check_review.py's _effective_verdict for the same rule.
    Fail-open (review_error=1) always ships as "real" regardless of confidence.
    """
    if row.get("review_error") == "1":
        return 1
    confidence = float(row["ai_confidence"]) if row["ai_confidence"] else 0.0
    return 1 if confidence > 0 else 0


def score_events_csv(events_csv: Path, golden: GoldenSet) -> dict:
    rows = list(csv.DictReader(events_csv.open(encoding="utf-8")))
    splits_by_key = {}
    for r in rows:
        if r["split_topology"] == "death":
            continue
        key = (r["parent_id"], int(r["peak_frame"]))
        splits_by_key[key] = r

    tp = fp = fn = tn = 0
    unmatched_golden_keys = []
    for key, gold_label in golden.labels.items():
        row = splits_by_key.get(key)
        if row is None:
            unmatched_golden_keys.append(key)
            continue
        pred = effective_verdict(row)
        if gold_label == 1 and pred == 1:
            tp += 1
        elif gold_label == 1 and pred == 0:
            fn += 1
        elif gold_label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1

    n_scored = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None

    return {
        "events_csv": str(events_csv),
        "n_golden_total": len(golden),
        "n_scored": n_scored,
        "n_unmatched_golden_keys": len(unmatched_golden_keys),
        "unmatched_golden_keys": unmatched_golden_keys,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/eval_harness/scorer.py <path-to-events.csv>")
        sys.exit(1)

    events_csv = Path(sys.argv[1])
    golden = load_golden_set()
    result = score_events_csv(events_csv, golden)

    print(f"Scored {result['n_scored']}/{result['n_golden_total']} golden events against {events_csv}")
    if result["n_unmatched_golden_keys"]:
        print(
            f"WARNING: {result['n_unmatched_golden_keys']} golden keys had no matching row in this "
            "events.csv -- this run's candidate set doesn't match the golden set's. Only expected "
            "for Tier B (frame-sampling) configs; for Tier A configs this signals a mismatched "
            "--frame-dir/--reuse-masks setup, not a real config difference."
        )
        print(f"  {result['unmatched_golden_keys']}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}")
    p = f"{result['precision']:.3f}" if result["precision"] is not None else "n/a"
    r = f"{result['recall']:.3f}" if result["recall"] is not None else "n/a"
    f1 = f"{result['f1']:.3f}" if result["f1"] is not None else "n/a"
    print(f"precision={p} recall={r} f1={f1}")
