"""Compare Trackastra confidence thresholds against Tom20 ground truth.

Ground truth: rows 17-49 of iPSC_nTSC_Tom20_ACTB_ZO1 sheet (the updated section).
We filter to events whose frame window starts at or before frame 20 (1-indexed),
matching the 20-frame smoke test.

Run: python scripts/compare_gt.py [--rerun]
  --rerun  force re-run of Trackastra pipeline (default: load cached events_trackastra.csv)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GT_XLSX = ROOT / "data/ground_truth/ACD_analysis.xlsx"
EVENTS_TRACKASTRA = ROOT / "data/output/events_trackastra.csv"
SMOKE_AVI = ROOT / "data/smoke_test.avi"
FRAME_DIR = ROOT / "data/frames"
OUTPUT_DIR = ROOT / "data/output"

SMOKE_FRAMES = 20  # 0-indexed: frames 0..19, 1-indexed: frames 1..20


def load_gt_events(max_frame_1indexed: int = SMOKE_FRAMES) -> list[dict]:
    """Load Tom20 GT divisions that START within the smoke test window.

    The updated section is rows 17-49 (0-indexed) of the sheet.
    A GT event counts if its frame range start <= max_frame_1indexed.
    """
    df = pd.read_excel(GT_XLSX, sheet_name="iPSC_nTSC_Tom20_ACTB_ZO1", header=0)

    # Updated section: rows 17-49 (0-indexed pandas), the second block of Tom20 data
    section = df.iloc[17:50].reset_index(drop=True)

    events = []
    for _, row in section.iterrows():
        frame_str = str(row.iloc[1]).strip()  # Frame column
        if not frame_str or frame_str in ("nan", "NaN", ""):
            continue
        # Parse "start-end" or "start-" or just "start"
        parts = frame_str.replace(" ", "").split("-")
        try:
            start = int(parts[0])
        except ValueError:
            continue
        try:
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        except ValueError:
            end = None

        # Metaphase-anaphase peak frame
        peak_str = str(row.iloc[2]).strip()
        try:
            peak = int(float(peak_str))
        except (ValueError, TypeError):
            peak = start  # fall back to start if unspecified

        division_type = str(row.iloc[3]).strip()

        if start <= max_frame_1indexed:
            events.append({
                "cell_id": int(float(str(row.iloc[0]))) if str(row.iloc[0]).strip() not in ("nan", "") else None,
                "frame_start": start,
                "frame_end": end,
                "peak_frame_1indexed": peak,
                # Convert to 0-indexed for comparison with pipeline output
                "peak_frame_0indexed": peak - 1,
                "division_type": division_type,
            })
    return events


def run_trackastra_pipeline() -> pd.DataFrame:
    """Run Trackastra on smoke_test.avi and save events_trackastra.csv."""
    from src.ingest import IngestConfig, extract_frames
    from src.pipeline import run

    config = IngestConfig(video_path=SMOKE_AVI, frame_step=1, roi=None)
    # Save to a dedicated file by temporarily naming it
    EVENTS_TRACKASTRA.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline into a temp output dir then rename
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    run(config, frame_dir=FRAME_DIR, output_dir=tmp, tracker="trackastra")
    shutil.copy(tmp / "events.csv", EVENTS_TRACKASTRA)
    shutil.rmtree(tmp)

    return pd.read_csv(EVENTS_TRACKASTRA)


def precision_recall_at_threshold(
    detected: pd.DataFrame,
    gt_events: list[dict],
    threshold: float,
    frame_tolerance: int = 3,
) -> dict:
    """Compute TP/FP/FN for detected events at a given confidence threshold.

    A detected event matches a GT event if its peak_frame is within
    frame_tolerance frames of the GT peak (0-indexed).
    """
    above = detected[detected["confidence"] >= threshold]

    matched_gt = set()
    tp = 0
    fp = 0

    for _, ev in above.iterrows():
        det_frame = int(ev["peak_frame"])
        matched = False
        for i, gt in enumerate(gt_events):
            if i in matched_gt:
                continue
            if abs(det_frame - gt["peak_frame_0indexed"]) <= frame_tolerance:
                matched_gt.add(i)
                matched = True
                tp += 1
                break
        if not matched:
            fp += 1

    fn = len(gt_events) - len(matched_gt)
    total = len(above)

    precision = tp / total if total > 0 else 0.0
    recall = tp / len(gt_events) if gt_events else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "detected": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true", help="Re-run Trackastra pipeline")
    parser.add_argument("--tolerance", type=int, default=3, help="Frame match tolerance (default 3)")
    args = parser.parse_args()

    # Load or generate Trackastra events
    if args.rerun or not EVENTS_TRACKASTRA.exists():
        print("Running Trackastra pipeline on smoke_test.avi...")
        events_df = run_trackastra_pipeline()
        print(f"  Saved {len(events_df)} events to {EVENTS_TRACKASTRA}")
    else:
        events_df = pd.read_csv(EVENTS_TRACKASTRA)
        print(f"Loaded {len(events_df)} cached Trackastra events from {EVENTS_TRACKASTRA}")

    # Load ground truth
    gt_events = load_gt_events(max_frame_1indexed=SMOKE_FRAMES)
    print(f"\nGround truth: {len(gt_events)} division events in frames 1-{SMOKE_FRAMES}")
    for ev in gt_events:
        print(f"  Cell {ev['cell_id']}: frames {ev['frame_start']}-{ev['frame_end']}, "
              f"peak={ev['peak_frame_1indexed']} (0-idx: {ev['peak_frame_0indexed']}), "
              f"type={ev['division_type']}")

    # Show all detected events
    print(f"\nTrackastra detected {len(events_df)} events:")
    print(events_df[["event_id", "peak_frame", "confidence", "division_type"]].to_string(index=False))

    # Sweep confidence thresholds
    thresholds = sorted(events_df["confidence"].unique(), reverse=True)
    if 0.1 not in thresholds:
        thresholds.append(0.1)
    thresholds = sorted(set(thresholds), reverse=True)

    print(f"\nPrecision/Recall sweep (frame tolerance ±{args.tolerance}):")
    print(f"{'Threshold':>10}  {'Detected':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    print("-" * 72)

    best = None
    for t in thresholds:
        r = precision_recall_at_threshold(events_df, gt_events, t, args.tolerance)
        print(f"{r['threshold']:>10.2f}  {r['detected']:>8}  {r['tp']:>4}  {r['fp']:>4}  "
              f"{r['fn']:>4}  {r['precision']:>10.3f}  {r['recall']:>8.3f}  {r['f1']:>6.3f}")
        if best is None or r["f1"] > best["f1"]:
            best = r

    if best:
        print(f"\nBest F1 at threshold {best['threshold']:.2f}: "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  F1={best['f1']:.3f}")
        print(f"  -> use confidence >= {best['threshold']:.2f} as the detection cutoff")

    # Separate analysis: only GT events whose peak falls within the smoke test window
    gt_in_window = [ev for ev in gt_events if ev["peak_frame_0indexed"] < SMOKE_FRAMES]
    gt_out_window = [ev for ev in gt_events if ev["peak_frame_0indexed"] >= SMOKE_FRAMES]
    if gt_out_window:
        print(f"\nNote: {len(gt_out_window)} GT event(s) start in frame window but peak OUTSIDE it "
              f"(peaks at 0-idx frames {[ev['peak_frame_0indexed'] for ev in gt_out_window]})")
        print(f"Re-evaluating against {len(gt_in_window)} GT events with peaks inside window:")
        print(f"{'Threshold':>10}  {'Detected':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
              f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}")
        print("-" * 72)
        best2 = None
        for t in thresholds:
            r = precision_recall_at_threshold(events_df, gt_in_window, t, args.tolerance)
            print(f"{r['threshold']:>10.2f}  {r['detected']:>8}  {r['tp']:>4}  {r['fp']:>4}  "
                  f"{r['fn']:>4}  {r['precision']:>10.3f}  {r['recall']:>8.3f}  {r['f1']:>6.3f}")
            if best2 is None or r["f1"] > best2["f1"]:
                best2 = r
        if best2:
            print(f"\nBest F1 (in-window GT) at threshold {best2['threshold']:.2f}: "
                  f"P={best2['precision']:.3f}  R={best2['recall']:.3f}  F1={best2['f1']:.3f}")


if __name__ == "__main__":
    main()
