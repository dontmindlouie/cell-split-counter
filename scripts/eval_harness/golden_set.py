"""Load the frozen Tier A golden set: human-labeled split verdicts for M4 full + M4
neighborfix, keyed by (parent_id, peak_frame).

Tier A variables (marker geometry, marker-motion-tracking, prompt wording, confidence
threshold, vision backend) never change which candidate split events exist -- only how
the model judges a fixed picture. --reuse-masks reuses cached Cellpose segmentation
(skips the neural net); trackastra linking still re-runs each time but is deterministic
given the same segmentation labels, so (parent_id, peak_frame) identities are stable
and reproducible across Tier A configs, and this key-based join against a fixed
human-reviewed set is valid. Tier B (frame sampling) changes segmentation/tracking
topology itself and does NOT reuse this key scheme -- see scripts/eval_harness/README.md.

Ground truth source: cell-split-counter-shared-data/human_review/human_review_compiled_2026-07-13.csv,
restricted to run_labels reviewed with clean tri-state verdicts (spot_check_review.py's
real/false_positive/unsure), not researcher_browser.py's freeform-note rows.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import HUMAN_REVIEW_DIR

COMPILED_CSV = HUMAN_REVIEW_DIR / "human_review_compiled_2026-07-13.csv"

DEFAULT_RUN_LABELS = ("M4_full_848frame", "M4_neighborfix_200frame_subsample")

# human_verdict -> binary label. Anything else (freeform notes, "unsure") is excluded.
VERDICT_LABEL = {"real": 1, "false_positive": 0}


class GoldenSet:
    def __init__(self, labels: dict[tuple[str, int], int], excluded: list[dict], stats: dict):
        self.labels = labels  # {(parent_id, peak_frame): 0|1}
        self.excluded = excluded  # rows dropped (unsure / conflicting duplicate / freeform)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.labels)


def load_golden_set(
    run_labels: tuple[str, ...] = DEFAULT_RUN_LABELS,
    csv_path: Path = COMPILED_CSV,
) -> GoldenSet:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found at {csv_path}. Is cell-split-counter-shared-data "
            "checked out as a sibling directory? See src/config.py SHARED_DATA_DIR."
        )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows = [r for r in rows if r["run_label"] in run_labels]

    by_key: dict[tuple[str, int], list[str]] = defaultdict(list)
    for r in rows:
        if r["human_verdict"] not in VERDICT_LABEL:
            continue  # unsure / freeform note -- not a usable binary label
        key = (r["parent_id"], int(r["peak_frame"]))
        by_key[key].append(r["human_verdict"])

    labels: dict[tuple[str, int], int] = {}
    excluded: list[dict] = []
    for key, verdicts in by_key.items():
        unique = set(verdicts)
        if len(unique) > 1:
            excluded.append({"key": key, "reason": "conflicting_duplicate_review", "verdicts": verdicts})
            continue
        labels[key] = VERDICT_LABEL[verdicts[0]]

    stats = {
        "run_labels": run_labels,
        "raw_rows": len(rows),
        "usable_labels": len(labels),
        "excluded_unsure_or_freeform": len(rows) - sum(len(v) for v in by_key.values()),
        "excluded_conflicting": len(excluded),
        "label_counts": dict(Counter(labels.values())),
    }
    return GoldenSet(labels, excluded, stats)


if __name__ == "__main__":
    gs = load_golden_set()
    print(f"Golden set: {len(gs)} usable labels")
    for k, v in gs.stats.items():
        print(f"  {k}: {v}")
    if gs.excluded:
        print(f"Excluded (conflicting duplicate reviews): {len(gs.excluded)}")
        for e in gs.excluded:
            print(f"  {e['key']}: {e['verdicts']}")
