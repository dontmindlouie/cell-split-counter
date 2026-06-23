"""Claude-vision review of ambiguous lineage events flagged by classify.py."""

from pathlib import Path

from src.classify import LineageEvent


def review_ambiguous(events: list[LineageEvent], frame_dir: Path) -> list[LineageEvent]:
    """Send the frames around each AMBIGUOUS event to Claude for a verdict.

    Returns updated events with event_type resolved, classification_source="claude",
    and a confidence score from the model's response.
    """
    raise NotImplementedError
