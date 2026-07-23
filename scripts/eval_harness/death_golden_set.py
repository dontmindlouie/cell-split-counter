"""Load the death/dropout golden set: Batch B's 50 human-labeled M4 death events,
keyed by track_id.

Companion to golden_set.py's Tier A split-review golden set. This one is Tier B-shaped
(the question is which frames the model gets to see), so it doesn't share that file's
(parent_id, peak_frame) key scheme -- track_id alone is the natural key here since a
death event has no daughter to disambiguate against.

Ground truth source: cell-split-counter-shared-data/human_review/
batch_b_death_vs_dropout_verdicts_2026-07-21.csv, built via scripts/reports/
batch_review_viewer.py, hand-labeled by the maintainer 2026-07-21. See
docs/investigation_notes.md's 2026-07-21 entry for the full write-up.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import HUMAN_REVIEW_DIR

BATCH_B_CSV = HUMAN_REVIEW_DIR / "batch_b_death_vs_dropout_verdicts_2026-07-21.csv"

# human verdict -> expected likely_division_dropout value. "unsure" is excluded.
VERDICT_LABEL = {
    "cell still alive (tracking lost it)": True,
    "real death": False,
}


class DeathGoldenSet:
    def __init__(self, labels: dict[int, bool], notes: dict[int, str], stats: dict):
        self.labels = labels  # {track_id: expected likely_division_dropout}
        self.notes = notes  # {track_id: human freeform note}
        self.stats = stats

    def __len__(self) -> int:
        return len(self.labels)


def load_death_golden_set(csv_path: Path = BATCH_B_CSV) -> DeathGoldenSet:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found at {csv_path}. Is cell-split-counter-shared-data "
            "checked out as a sibling directory? See src/config.py SHARED_DATA_DIR."
        )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    labels: dict[int, bool] = {}
    notes: dict[int, str] = {}
    excluded_unsure = 0
    for r in rows:
        verdict = r["verdict"]
        if verdict not in VERDICT_LABEL:
            excluded_unsure += 1
            continue
        track_id = int(r["track_id"])
        labels[track_id] = VERDICT_LABEL[verdict]
        notes[track_id] = r.get("notes", "")

    stats = {
        "raw_rows": len(rows),
        "usable_labels": len(labels),
        "excluded_unsure": excluded_unsure,
        "label_counts": dict(Counter(labels.values())),
    }
    return DeathGoldenSet(labels, notes, stats)


if __name__ == "__main__":
    gs = load_death_golden_set()
    print(f"Death golden set: {len(gs)} usable labels")
    for k, v in gs.stats.items():
        print(f"  {k}: {v}")
