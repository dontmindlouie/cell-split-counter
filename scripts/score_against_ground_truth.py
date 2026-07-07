"""Compare detected split events against the research scientist's ground-truth sheet.

Usage:
  python scripts/score_against_ground_truth.py [tolerance_frames] [--min-conf N]

  tolerance_frames  frame window for matching (default 5)
  --min-conf N      only count detected events with confidence >= N (default 0)

Ground-truth frames are 1-indexed raw video frame numbers; our pipeline's peak_frame
is a 0-indexed frame index (frame_step=1), so we add 1 before comparing.

Detected events are deduplicated by (parent_id, peak_frame) — each unique split
contributes one detected peak, not one per daughter track.

KNOWN LIMITATION (2026-07-05): matching is by frame proximity ONLY -- there is no
spatial (x/y) check, because the ground-truth sheet has no coordinates at all (frame
ranges + freeform text only). This means "false credit" is possible and has been
confirmed multiple times by manual Fiji cross-reference: a detected event that is
spatially wrong (a different, unrelated cell) can still count as a "match" here purely
because its frame number happens to fall within tolerance of a real GT event elsewhere
in the same crowded field. Confirmed false-credit cases so far: GT frame 56 (matched
detection was 738px from the real cell), GT frames 514/520 (the real division, tracked
correctly at 5px from ground truth, was rejected by Claude at confidence=0.0 and excluded;
a spatially-wrong detection 132px away was what actually satisfied the match). Treat any
recall number from this script as an upper bound, not a verified number, unless spatial
correctness has been separately confirmed for the specific events in question.
"""

import os
import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVENTS_CSV

# Overridable via env vars so real dataset identifiers never need to be hardcoded
# into this tracked (public) file -- set these in your shell/.env for local runs.
GROUND_TRUTH_XLSX = Path(os.environ.get("GT_XLSX_PATH", "data/ground_truth/ground_truth.xlsx"))
GROUND_TRUTH_SHEET = os.environ.get("GT_SHEET_NAME", "Sheet1")

GT_ROW_START = int(os.environ.get("GT_ROW_START", 19))  # first data row (1-indexed)
GT_ROW_END = int(os.environ.get("GT_ROW_END", 51))      # last data row (inclusive)


def parse_ground_truth_peaks(sheet) -> tuple[list[int], list[tuple[int, str]]]:
    """Returns (peaks, excluded) where excluded is [(peak_frame, note), ...] for rows
    with a non-blank Division Type/notes column (column D) -- these are annotator
    uncertainty notes (e.g. "2 to 2 non-division") or non-bipolar outcomes (e.g.
    "1-1 can not separate", a failed split) that the current binary 1->2 split
    detector was never built to catch, so scoring them as plain misses inflates
    the miss count for reasons unrelated to detector quality.
    """
    peaks = []
    excluded = []
    for row in sheet.iter_rows(min_row=GT_ROW_START, max_row=GT_ROW_END, values_only=True):
        value = row[2] if len(row) > 2 else None
        note = row[3] if len(row) > 3 else None
        if value is None:
            continue
        peak = None
        if isinstance(value, (int, float)):
            peak = int(value)
        elif isinstance(value, str):
            match = re.match(r"(\d+)\s*-\s*(\d+)", value.strip())
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                peak = (start + end) // 2
        if peak is None:
            continue
        if note is not None and str(note).strip():
            excluded.append((peak, str(note).strip()))
        else:
            peaks.append(peak)
    return peaks, excluded


def parse_detected_peaks(csv_path: Path, min_conf: float = 0.0) -> list[int]:
    import csv

    seen: set[tuple] = set()
    peaks = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            # death/roi_exit rows are track ends, not splits -- exclude so they can't
            # inflate/false-credit recall against a ground truth of division events.
            if row["split_topology"] not in ("normal_split", "multi_way_split"):
                continue
            conf = float(row["claude_confidence"])
            if conf < min_conf:
                continue
            # skip Claude-confirmed false positives -- review.py forces confidence=0.0
            # whenever verdict != "real", regardless of whether notes are populated
            # (notes are written unconditionally as of the 2026-07-02 fix, so checking
            # for empty claude_notes here no longer distinguishes real vs false_positive)
            if row["classification_source"] == "claude" and conf == 0.0:
                continue
            key = (row["parent_id"], row["peak_frame"])
            if key in seen:
                continue
            seen.add(key)
            peaks.append(int(row["peak_frame"]) + 1)  # 0-indexed -> 1-indexed
    return peaks


def main() -> None:
    args = sys.argv[1:]
    tolerance = 5
    min_conf = 0.0
    for i, a in enumerate(args):
        if a == "--min-conf" and i + 1 < len(args):
            min_conf = float(args[i + 1])
        elif a.lstrip("-").isdigit() and not a.startswith("--"):
            tolerance = int(a)

    wb = openpyxl.load_workbook(GROUND_TRUTH_XLSX, data_only=True)
    ground_truth_peaks, excluded = parse_ground_truth_peaks(wb[GROUND_TRUTH_SHEET])
    detected_peaks = parse_detected_peaks(EVENTS_CSV, min_conf=min_conf)

    matched_gt, missed_gt = [], []
    for gt in ground_truth_peaks:
        if any(abs(gt - d) <= tolerance for d in detected_peaks):
            matched_gt.append(gt)
        else:
            missed_gt.append(gt)

    true_positives = [d for d in detected_peaks if any(abs(d - gt) <= tolerance for gt in ground_truth_peaks)]
    false_positives = [d for d in detected_peaks if not any(abs(d - gt) <= tolerance for gt in ground_truth_peaks)]

    recall = len(matched_gt) / len(ground_truth_peaks) if ground_truth_peaks else 0
    precision = len(true_positives) / len(detected_peaks) if detected_peaks else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"tolerance: +/-{tolerance} frames  |  min_conf: {min_conf}")
    print(f"ground-truth events : {len(ground_truth_peaks)}")
    print(f"detected events     : {len(detected_peaks)} unique splits")
    print(f"recall    : {len(matched_gt)}/{len(ground_truth_peaks)} = {recall:.1%}")
    print(f"precision : {len(true_positives)}/{len(detected_peaks)} = {precision:.1%}")
    print(f"F1        : {f1:.3f}")
    print(f"missed GT frames: {sorted(missed_gt)}")
    print(f"false positive detected frames (sample): {sorted(false_positives)[:20]}")
    if excluded:
        print(f"excluded from GT denominator ({len(excluded)} rows, non-blank Division Type/notes column):")
        for peak, note in excluded:
            print(f"  frame {peak}: {note!r}")


if __name__ == "__main__":
    main()
