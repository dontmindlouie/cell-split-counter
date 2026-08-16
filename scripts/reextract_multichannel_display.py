"""Re-run frame extraction only (not Cellpose/Trackastra) for the 5 multi-channel
wells, against their archived run dirs on D:, to add the new frames/display/
composite (src/ingest.py, 2026-08-05) without repaying the expensive
segment+track stage. Masks/tracks are unaffected since the segmentation-input
frame_*.png bytes are unchanged by this fix -- only the new display/ composite
is new output; frame_*.png/_display_color.json/_display_windows.json get
rewritten in place with identical or corrected-metadata-only content.

frame_step=1, roi=None match main.py's defaults, which is what these 5 wells
were run with (no --frame-step/--roi override recorded anywhere for this batch).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest import IngestConfig, extract_frames

ARCHIVE = Path(r"D:\cell-split-counter-output-archive")

WELLS = [
    (ARCHIVE / "nTSC_ZO1_1-4_M1", r"G:\Projects\nd2_raw\ZO1_nTSC\nTSC_ZO1_1-4_M1.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M2", r"G:\Projects\nd2_raw\ZO1_nTSC\nTSC_ZO1_1-4_M2.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M3", r"G:\Projects\nd2_raw\ZO1_nTSC\nTSC_ZO1_1-4_M3.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M4", r"G:\Projects\nd2_raw\ZO1_nTSC\nTSC_ZO1_1-4_M4.nd2"),
    (ARCHIVE / "20251016_ACTB_M3", r"G:\Projects\nd2_raw\2025_1016_TSC_ACTB_Tom20\20251016_ACTB_M3.nd2"),
]


def main() -> None:
    for run_dir, nd2_path in WELLS:
        stem = run_dir.name
        print(f"\n{'='*70}\n>>> RE-EXTRACT {stem}\n{'='*70}", flush=True)
        cfg = IngestConfig(video_path=Path(nd2_path), frame_step=1, roi=None)
        paths = extract_frames(cfg, run_dir / "frames")
        n_display = len(list((run_dir / "frames" / "display").glob("*.png")))
        print(f"  {len(paths)} frames, {n_display} display composites", flush=True)
    print("ALL_REEXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
