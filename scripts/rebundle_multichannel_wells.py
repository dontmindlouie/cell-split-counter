"""Re-run build_bundle.py for the 5 multi-channel wells whose first bundle attempt
crashed (regionprops_table shape mismatch, fixed 2026-08-04 in build_bundle.py's
write_labels_and_tracks). main.py's output for all 5 already ran successfully and
was archived to D: before the crash was noticed, so this points build_bundle.py
directly at the D: archive copies instead of re-running the whole pipeline.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ARCHIVE = Path(r"D:\cell-split-counter-output-archive")
BUNDLE_OUT = ROOT / "data" / "bundle"

WELLS = [
    (ARCHIVE / "nTSC_ZO1_1-4_M1", r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M1.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M2", r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M2.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M3", r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M3.nd2"),
    (ARCHIVE / "nTSC_ZO1_1-4_M4", r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M4.nd2"),
    (ARCHIVE / "20251016_ACTB_M3", r"G:\Projects\2025_1016_TSC_ACTB_Tom20\20251016_ACTB_M3.nd2"),
]


def main() -> None:
    ok, failed = [], []
    for run_dir, nd2_path in WELLS:
        stem = run_dir.name
        print(f"\n{'='*70}\n>>> REBUNDLE {stem}\n{'='*70}", flush=True)
        cmd = [str(PY), str(ROOT / "scripts" / "build_bundle.py"), str(run_dir),
               "--nd2", nd2_path, "--out", str(BUNDLE_OUT), "--overwrite"]
        p = subprocess.run(cmd, cwd=str(ROOT))
        (ok if p.returncode == 0 else failed).append(stem)

    print("\n" + "=" * 70)
    print(f"done: {len(ok)} ok, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    print("ALL_REBUNDLE_DONE", flush=True)


if __name__ == "__main__":
    main()
