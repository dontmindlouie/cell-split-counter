"""Driver for the 2026-08-14 batch: ACTB_M1 (new), ACTB_M2 + ACTB_M3 + Tom20
(rebuilt from their full 2-channel J: exports, superseding the earlier
single-channel-only / 3-channel builds).

Segmentation now runs on the NUCLEUS CHANNEL ALONE (--nucleus-channel), not a
max-projection of both channels -- decided 2026-08-14: mixing in the membrane/ACTB
or mitochondrial/Tom20 marker would inflate the mask past the nucleus. Confirmed
per-well nucleus channel (nd2 channel *names*, not targets -- only the imaging lab
knows which dye was conjugated to which protein):
  - ACTB wells (M1/M2/M3): AF555 = nucleus, AF488 = membrane (ACTB), display-only.
  - Tom20 well: AF647 (cy5) = nucleus, AF555 = Tom20 (mitochondrial), display-only.

No vision review (main.py's default is already OFF as of 2026-08-05) -- cell_mcp.py
never reads candidates.csv, so review is pure cost with zero effect on the bundle.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RAW = ROOT.parent / "2025_1016_TSC_ACTB_Tom20"

WELLS = [
    # (nd2 path, canonical bundle name, nucleus channel substring, cell_line, stale bundle it supersedes)
    (RAW / "20251016_ACTB_M1_green and red.nd2", "20251016_ACTB_M1", "AF555", None, None),
    (RAW / "20251016_ACTB_M2_green and red.nd2", "20251016_ACTB_M2", "AF555", None, "20251016_ACTB_M2_red"),
    (RAW / "20251016_ACTB_M3_green and red.nd2", "20251016_ACTB_M3", "AF555", None, "20251016_ACTB_M3"),
    (RAW / "20251016_Tom20_red and cy5.nd2", "20251016_Tom20", "AF647", None, "20251016_Tom20_cy5"),
]

BUNDLE_OUT = ROOT / "data" / "bundle"
ARCHIVE = Path(r"H:\cell-split-counter-output-archive")


def main() -> None:
    ok, failed = [], []
    for nd2_path, stem, nucleus_channel, cell_line, stale_bundle in WELLS:
        out_dir = ROOT / "data" / "output" / stem
        print(f"\n{'='*70}\n>>> RUN {stem} (nucleus channel: {nucleus_channel})\n{'='*70}", flush=True)

        run_cmd = [str(PY), str(ROOT / "main.py"), str(nd2_path),
                   "--nucleus-channel", nucleus_channel, "--output-dir", str(out_dir)]
        p = subprocess.run(run_cmd, cwd=str(ROOT))
        if p.returncode != 0:
            print(f"!!! main.py FAILED for {stem}", flush=True)
            failed.append(stem)
            continue

        print(f">>> BUNDLE {stem}", flush=True)
        bundle_cmd = [str(PY), str(ROOT / "scripts" / "build_bundle.py"), str(out_dir),
                      "--nd2", str(nd2_path), "--nucleus-channel", nucleus_channel,
                      "--out", str(BUNDLE_OUT), "--overwrite"]
        if cell_line:
            bundle_cmd += ["--cell-line", cell_line]
        p2 = subprocess.run(bundle_cmd, cwd=str(ROOT))
        (ok if p2.returncode == 0 else failed).append(stem)

        if p2.returncode == 0 and stale_bundle and stale_bundle != stem:
            stale_dir = BUNDLE_OUT / stale_bundle
            if stale_dir.is_dir():
                import shutil
                print(f">>> removing superseded partial-channel bundle {stale_bundle}", flush=True)
                shutil.rmtree(stale_dir)

        if out_dir.is_dir():
            subprocess.run(["robocopy", str(out_dir), str(ARCHIVE / stem), "/E", "/MOVE", "/MT:8", "/NFL", "/NDL", "/NP"])

    print("\n" + "=" * 70)
    print(f"done: {len(ok)} ok, {len(failed)} failed")
    for f in failed:
        print("  FAILED:", f)
    print("ALL_ACTB_TOM20_NUCLEUS_WELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
