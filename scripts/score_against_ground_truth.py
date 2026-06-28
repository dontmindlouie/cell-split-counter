"""Compare detected split events against the research scientist's ground-truth sheet.

Usage:
  python scripts/score_against_ground_truth.py [tolerance_frames] [--min-conf N]

  tolerance_frames  frame window for matching (default 5)
  --min-conf N      only count detected events with confidence >= N (default 0)

Ground-truth frames are 1-indexed raw video frame numbers; our pipeline's peak_frame
is a 0-indexed frame index (frame_step=1), so we add 1 before comparing.

Detected events are deduplicated by (parent_id, peak_frame) — each unique split
contributes one detected peak, not one per daughter track.
"""

import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUND_TRUTH_XLSX = Path("data/ground_truth/ACD_analysis.xlsx")
GROUND_TRUTH_SHEET = "iPSC_nTSC_Tom20_ACTB_ZO1"
EVENTS_CSV = Path("data/output/events.csv")

GT_ROW_START = 19
GT_ROW_END = 51


def parse_ground_truth_peaks(sheet) -> list[int]:
    peaks = []
    for row in sheet.iter_rows(min_row=GT_ROW_START, max_row=GT_ROW_END, values_only=True):
        value = row[2] if len(row) > 2 else None
        if value is None:
            continue
        if isinstance(value, (int, float)):
            peaks.append(int(value))
        elif isinstance(value, str):
            match = re.match(r"(\d+)\s*-\s*(\d+)", value.strip())
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                peaks.append((start + end) // 2)
    return peaks


def parse_detected_peaks(csv_path: Path, min_conf: float = 0.0) -> list[int]:
    import csv

    seen: set[tuple] = set()
    peaks = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            conf = float(row["confidence"])
            if conf < min_conf:
                continue
            # skip Claude-confirmed false positives (confidence=0, no notes)
            if row["classification_source"] == "claude" and not row.get("claude_notes"):
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
    ground_truth_peaks = parse_ground_truth_peaks(wb[GROUND_TRUTH_SHEET])
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


if __name__ == "__main__":
    main()
