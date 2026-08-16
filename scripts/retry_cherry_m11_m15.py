"""Retry driver for the 5 wells that failed 2026-08-03 when G: filled to 0 bytes free
(OSError: No space left on device) partway through run_new_wells_batch.py.
G: now has ~290GB free after archiving already-bundled wells' output to D:.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

WELLS = [
    (r"G:\Projects\nd2_raw\nTSC_living image\TSC size cherry_Live Imaging4min002_M11_size medium.nd2", "nTSC"),
    (r"G:\Projects\nd2_raw\nTSC_living image\TSC size cherry_Live Imaging4min002_M12_size medium.nd2", "nTSC"),
    (r"G:\Projects\nd2_raw\nTSC_living image\TSC size cherry_Live Imaging4min002_M13_size medium.nd2", "nTSC"),
    (r"G:\Projects\nd2_raw\nTSC_living image\TSC size cherry_Live Imaging4min002_M14_size medium.nd2", "nTSC"),
    (r"G:\Projects\nd2_raw\nTSC_living image\TSC size cherry_Live Imaging4min002_M15_size medium.nd2", "nTSC"),
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

        # Move this well's raw output to D: archive right after bundling, to avoid
        # refilling G: across a 5-well retry the way the original batch did.
        archive_dst = Path(r"D:\cell-split-counter-output-archive") / stem
        if out_dir.is_dir():
            subprocess.run(["robocopy", str(out_dir), str(archive_dst), "/E", "/MOVE", "/MT:8", "/NFL", "/NDL", "/NP"])

    print("\n" + "=" * 70)
    print(f"done: {len(ok)} ok, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    print("ALL_RETRY_WELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
