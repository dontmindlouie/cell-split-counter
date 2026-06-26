"""Compare detected split events against the research scientist's ground-truth sheet.

Usage: python scripts/score_against_ground_truth.py [tolerance_frames]

Ground-truth frames are 1-indexed raw video frame numbers; our pipeline's peak_frame
is a 0-indexed frame index over every extracted frame (frame_step=1), so we add 1
before comparing.
"""

import re
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUND_TRUTH_XLSX = Path("data/ground_truth/ACD_analysis.xlsx")
GROUND_TRUTH_SHEET = "iPSC_nTSC_Tom20_ACTB_ZO1"  # matches the ACTB_Tom20 video (575 frames)
EVENTS_CSV = Path("data/output/events.csv")

# Sheet has multiple stacked blocks for different videos.
# Rows 19-51: 33-event Tom20 block (the one we have video for — use this).
# Rows 2-17:  original 15-event Tom20 block (superseded by the 33-event block).
# Rows 70+:   ACTB_Ntsc block (different video, not in data/raw/).
GT_ROW_START = 19
GT_ROW_END = 51


def parse_ground_truth_peaks(sheet) -> list[int]:
    """Column C ('metaphase-anaphase') is either a single frame number or a 'start-end'
    range string; for ranges, use the midpoint as the representative peak frame."""
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


def parse_detected_peaks(csv_path: Path) -> list[int]:
    import csv

    peaks = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            peaks.append(int(row["peak_frame"]) + 1)  # 0-indexed -> 1-indexed
    return peaks


def main() -> None:
    tolerance = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    wb = openpyxl.load_workbook(GROUND_TRUTH_XLSX, data_only=True)
    ground_truth_peaks = parse_ground_truth_peaks(wb[GROUND_TRUTH_SHEET])
    detected_peaks = parse_detected_peaks(EVENTS_CSV)

    matched = []
    missed = []
    for gt in ground_truth_peaks:
        if any(abs(gt - d) <= tolerance for d in detected_peaks):
            matched.append(gt)
        else:
            missed.append(gt)

    unmatched_detections = [d for d in detected_peaks if not any(abs(d - gt) <= tolerance for gt in ground_truth_peaks)]

    print(f"tolerance: +/-{tolerance} frames")
    print(f"ground-truth events: {len(ground_truth_peaks)}")
    print(f"detected events: {len(detected_peaks)}")
    print(f"recall: {len(matched)}/{len(ground_truth_peaks)} = {len(matched) / len(ground_truth_peaks):.1%}")
    print(f"missed ground-truth frames: {sorted(missed)}")
    print(f"detected events with no nearby ground truth (informational, not precision-scored): {len(unmatched_detections)}")


if __name__ == "__main__":
    main()
