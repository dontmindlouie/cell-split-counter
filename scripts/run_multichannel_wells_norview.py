"""Run the 5 wells blocked on multi-channel ingest (ZO1_nTSC M1-M4, ACTB_M3),
now that _extract_frames_nd2 combines channels via max-projection (2026-08-03).

No vision review (--no-review-splits --no-review-deaths): cell_mcp.py never reads
candidates.csv (where review verdicts land) -- only tracks.csv/lineage.csv/
annotations.csv -- so vision review is pure cost/time with zero effect on the MCP
bundle. Same rationale as retry_cherry_m12_m15_norview.py.

cell_line left unset for all 5, matching run_new_wells_batch.py's convention for
this same batch of non-cherry wells (build_bundle infers/leaves it unset).
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

WELLS = [
    r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M1.nd2",
    r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M2.nd2",
    r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M3.nd2",
    r"G:\Projects\ZO1_nTSC\nTSC_ZO1_1-4_M4.nd2",
    r"G:\Projects\2025_1016_TSC_ACTB_Tom20\20251016_ACTB_M3.nd2",
]

BUNDLE_OUT = ROOT / "data" / "bundle"
ARCHIVE = Path(r"D:\cell-split-counter-output-archive")


def main() -> None:
    ok, failed = [], []
    for nd2_path in WELLS:
        nd2_path = Path(nd2_path)
        stem = nd2_path.stem
        out_dir = ROOT / "data" / "output" / stem
        print(f"\n{'='*70}\n>>> RUN {stem} (multi-channel, no vision review)\n{'='*70}", flush=True)

        run_cmd = [str(PY), str(ROOT / "main.py"), str(nd2_path),
                   "--no-review-splits", "--no-review-deaths"]
        p = subprocess.run(run_cmd, cwd=str(ROOT))
        if p.returncode != 0:
            print(f"!!! main.py FAILED for {stem}", flush=True)
            failed.append(stem)
            continue

        print(f">>> BUNDLE {stem}", flush=True)
        bundle_cmd = [str(PY), str(ROOT / "scripts" / "build_bundle.py"), str(out_dir),
                      "--nd2", str(nd2_path), "--out", str(BUNDLE_OUT), "--overwrite"]
        p2 = subprocess.run(bundle_cmd, cwd=str(ROOT))
        (ok if p2.returncode == 0 else failed).append(stem)

        if out_dir.is_dir():
            subprocess.run(["robocopy", str(out_dir), str(ARCHIVE / stem), "/E", "/MOVE", "/MT:8", "/NFL", "/NDL", "/NP"])

    print("\n" + "=" * 70)
    print(f"done: {len(ok)} ok, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    print("ALL_MULTICHANNEL_WELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
