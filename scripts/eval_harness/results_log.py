"""Append-only results log for eval_harness sweeps -- the "reliability over time" report
from the human-review-ground-truth backlog, made real. One row per scored config.

Shared across worktrees at cell-split-counter-shared-data/eval_harness/results_log.csv
(same convention as human_review/ -- see src/config.py SHARED_DATA_DIR).
"""

import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import SHARED_DATA_DIR

RESULTS_LOG_DIR = SHARED_DATA_DIR / "eval_harness"
RESULTS_LOG_CSV = RESULTS_LOG_DIR / "results_log.csv"

FIELDS = [
    "timestamp", "config_id", "git_commit", "run_labels_scored",
    "vision_backend", "gpt_reasoning_effort", "min_gpt_confidence", "extra_flags",
    "n_golden_total", "n_scored", "n_unmatched_golden_keys",
    "tp", "fp", "fn", "tn", "precision", "recall", "f1",
    "events_csv", "notes",
]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def append_result(
    config_id: str,
    score: dict,
    run_labels_scored: tuple[str, ...],
    vision_backend: str = "",
    gpt_reasoning_effort: str = "",
    min_gpt_confidence: str = "",
    extra_flags: str = "",
    notes: str = "",
) -> None:
    RESULTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RESULTS_LOG_CSV.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_id": config_id,
        "git_commit": _git_commit(),
        "run_labels_scored": "|".join(run_labels_scored),
        "vision_backend": vision_backend,
        "gpt_reasoning_effort": gpt_reasoning_effort,
        "min_gpt_confidence": min_gpt_confidence,
        "extra_flags": extra_flags,
        "n_golden_total": score["n_golden_total"],
        "n_scored": score["n_scored"],
        "n_unmatched_golden_keys": score["n_unmatched_golden_keys"],
        "tp": score["tp"],
        "fp": score["fp"],
        "fn": score["fn"],
        "tn": score["tn"],
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
        "events_csv": score["events_csv"],
        "notes": notes,
    }

    with RESULTS_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
