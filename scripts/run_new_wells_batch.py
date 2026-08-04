"""One-off sequential driver for the 2026-08-02 new-wells batch (ZO1/ACTB-Tom20/nTSC-cherry).

Runs main.py then build_bundle.py for each well, one at a time (single 8GB GPU,
no concurrent heavy jobs). Skips the 5 known multi-channel wells (ZO1 M1-M4,
ACTB_M3) pending a channel-selection decision -- see
project_cell_split_counter_new_wells_2026_08_02 memory note.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

WELLS = [
    (r"G:\Projects\2025_1016_TSC_ACTB_Tom20\20251016_ACTB_M2_red.nd2", None),
    (r"G:\Projects\2025_1016_TSC_ACTB_Tom20\20251016_Tom20_cy5.nd2", None),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M9_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M10_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M11_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M12_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M13_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M14_size medium.nd2", "nTSC"),
    (r"G:\Projects\nTSC_living image\TSC size cherry_Live Imaging4min002_M15_size medium.nd2", "nTSC"),
]

BUNDLE_OUT = ROOT / "data" / "bundle"


def main() -> None:
    ok, failed = [], []
    for nd2_path, cell_line in WELLS:
        nd2_path = Path(nd2_path)
        stem = nd2_path.stem
        out_dir = ROOT / "data" / "output" / stem
        print(f"\n{'='*70}\n>>> RUN {stem}\n{'='*70}", flush=True)

        run_cmd = [str(PY), str(ROOT / "main.py"), str(nd2_path)]
        p = subprocess.run(run_cmd, cwd=str(ROOT))
        if p.returncode != 0:
            print(f"!!! main.py FAILED for {stem}", flush=True)
            failed.append(stem)
            continue

        print(f">>> BUNDLE {stem}", flush=True)
        bundle_cmd = [str(PY), str(ROOT / "scripts" / "build_bundle.py"), str(out_dir),
                      "--nd2", str(nd2_path), "--out", str(BUNDLE_OUT), "--overwrite"]
        if cell_line:
            bundle_cmd += ["--cell-line", cell_line]
        p2 = subprocess.run(bundle_cmd, cwd=str(ROOT))
        (ok if p2.returncode == 0 else failed).append(stem)

    print("\n" + "=" * 70)
    print(f"done: {len(ok)} ok, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    print("ALL_WELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
