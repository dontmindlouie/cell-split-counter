"""Trial: run review_ambiguous() against the latest pipeline output, capped at 5 splits."""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.classify import EventType, LineageEvent
from src.review import review_ambiguous

EVENTS_CSV = Path(r"G:\Projects\cell-split-counter\data\output\events.csv")
FRAME_DIR  = Path(r"G:\Projects\cell-split-counter\data\frames")

TYPE_MAP = {
    "normal_split":    EventType.NORMAL_SPLIT,
    "multi_way_split": EventType.MULTI_WAY_SPLIT,
    "failed_split":    EventType.FAILED_SPLIT,
    "roi_exit":        EventType.ROI_EXIT,
    "death":           EventType.DEATH,
    "ambiguous":       EventType.AMBIGUOUS,
}

def load_events(path: Path) -> list[LineageEvent]:
    events = []
    with path.open() as f:
        for row in csv.DictReader(f):
            events.append(LineageEvent(
                track_id=int(row["track_id"]),
                parent_id=int(row["parent_id"]) if row["parent_id"] else None,
                frame=int(row["peak_frame"]),
                event_type=TYPE_MAP[row["division_type"]],
                classification_source=row["classification_source"],
                confidence=float(row["confidence"]),
            ))
    return events

events = load_events(EVENTS_CSV)
low = [e for e in events if e.confidence < 1.0]
print(f"Total events: {len(events)}  |  Low-confidence: {len(low)}")

reviewed = review_ambiguous(
    events,
    FRAME_DIR,
    confidence_threshold=1.0,
    model="claude-haiku-4-5",
    max_reviews=5,       # cap: 5 unique split points = 5 API calls
)

changed = [e for e in reviewed if e.classification_source == "claude"]
print(f"\nReviewed by Claude: {len(changed)} events ({len({(e.parent_id, e.frame) for e in changed})} unique splits)\n")

for e in changed:
    verdict = "REAL" if e.confidence > 0 else "FALSE POSITIVE"
    print(f"  frame={e.frame:3d}  parent={e.parent_id}  track={e.track_id}  "
          f"confidence={e.confidence:.2f}  -> {verdict}")
