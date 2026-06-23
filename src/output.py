"""Write lineage events and summary metadata to CSV/JSON."""

import json
from pathlib import Path

from src.classify import LineageEvent


def write_events_csv(events: list[LineageEvent], out_path: Path) -> None:
    raise NotImplementedError


def write_summary_json(events: list[LineageEvent], video_metadata: dict, out_path: Path) -> None:
    raise NotImplementedError
