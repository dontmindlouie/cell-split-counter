"""Shared path constants and model defaults for scripts and src modules.

All paths are relative to the project root (the directory containing main.py).
Scripts that override these via argparse should import these as defaults only.
"""

import os
from pathlib import Path

DATA_DIR     = Path("data")
FRAME_DIR    = DATA_DIR / "frames"
OUTPUT_DIR   = DATA_DIR / "output"
EVENTS_CSV   = OUTPUT_DIR / "events.csv"
DEBUG_DIR    = DATA_DIR / "debug" / "crops"
PACKAGES_DIR = DATA_DIR / "packages"

# This project is checked out as multiple sibling git worktrees under the same parent
# directory (cell-split-counter, cell-split-counter-agy-analysis, etc.). Anything that
# needs to persist and be shared across all of them -- not duplicated/diverging inside
# each worktree's gitignored data/ -- lives in cell-split-counter-shared-data, a sibling
# of this worktree root rather than inside it. Override via env var if the project is
# ever checked out somewhere without that sibling layout (e.g. a single clone, CI).
_WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SHARED_DATA_DIR = Path(os.environ.get(
    "CELL_SPLIT_SHARED_DATA_DIR",
    _WORKTREE_ROOT.parent / "cell-split-counter-shared-data",
))
HUMAN_REVIEW_DIR = SHARED_DATA_DIR / "human_review"

CLAUDE_MODEL = "claude-haiku-4-5"

# Azure OpenAI deployment name for the "gpt" vision review backend (see review.py's
# backend= param). Recall parity with Claude but lower precision as of the 2026-07-06
# spike -- offered as a cost/credit tradeoff, not a quality upgrade.
GPT_DEPLOYMENT = "gpt-5-mini"

# Confidence floor used by scripts that backfill/re-review already-confirmed events.
HIGH_CONFIDENCE = 0.5
