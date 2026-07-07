"""Shared path constants and model defaults for scripts and src modules.

All paths are relative to the project root (the directory containing main.py).
Scripts that override these via argparse should import these as defaults only.
"""

from pathlib import Path

DATA_DIR     = Path("data")
FRAME_DIR    = DATA_DIR / "frames"
OUTPUT_DIR   = DATA_DIR / "output"
EVENTS_CSV   = OUTPUT_DIR / "events.csv"
DEBUG_DIR    = DATA_DIR / "debug" / "crops"
PACKAGES_DIR = DATA_DIR / "packages"

CLAUDE_MODEL = "claude-haiku-4-5"

# Azure OpenAI deployment name for the "gpt" vision review backend (see review.py's
# backend= param). Recall parity with Claude but lower precision as of the 2026-07-06
# spike -- offered as a cost/credit tradeoff, not a quality upgrade.
GPT_DEPLOYMENT = "gpt-5-mini"

# Confidence floor used by scripts that backfill/re-review already-confirmed events.
HIGH_CONFIDENCE = 0.5
