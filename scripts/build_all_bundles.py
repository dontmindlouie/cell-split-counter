"""Run build_bundle.py across every real well, resolving ND2 path + cell line.

Skips smoke/fixture/derived runs -- only the 21 wells that correspond to a real
acquisition. Cell-line mapping is per reference_cell_split_counter_well_cell_lines:
TSC batch2 M1-M16 are FOUR different lines (not replicates), and "M4" collides
between TSC batch2 (nTSC) and Bewo (BeWo), so the run name is always qualified.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSC_DIR = Path(r"G:\Projects\nd2_raw\20260709_TSC_Batch2")
BEWO_DIR = Path(r"G:\Projects\nd2_raw\Bewo_nd2")

TSC_LINES = {1: None, 2: "nTSC", 3: "nTSC", 4: "nTSC",
             5: "pTSC", 6: "pTSC", 7: "pTSC", 8: "pTSC", 9: "pTSC",
             10: "RUES2", 11: "RUES2", 12: "RUES2",
             13: "WGD", 14: "WGD", 15: "WGD", 16: "WGD"}


def resolve(run: Path) -> tuple[Path, str | None] | None:
    n = run.name
    m = re.fullmatch(r"TSC_batch2_M(\d+)(?:_\w+)?", n)
    if m:
        w = int(m.group(1))
        hits = sorted(TSC_DIR.glob(f"*__M{w} *.nd2")) + sorted(TSC_DIR.glob(f"*__M{w}.nd2"))
        return (hits[0], TSC_LINES.get(w)) if hits else None
    m = re.fullmatch(r"(202660629_Bewop920x_M\d+)", n)
    if m:
        p = BEWO_DIR / f"{m.group(1)}.nd2"
        return (p, "BeWo") if p.is_file() else None
    return None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "bundle"
    runs = sorted((ROOT / "data" / "output").iterdir())
    ok, skipped, failed = [], [], []

    for run in runs:
        if not run.is_dir():
            continue
        r = resolve(run)
        if r is None:
            skipped.append(run.name)
            continue
        nd2, line = r
        if not (run / "frames" / "_memmap" / "tracked_masks.dat").is_file():
            skipped.append(f"{run.name} (no tracked_masks.dat)")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "build_bundle.py"), str(run),
               "--nd2", str(nd2), "--out", str(out), "--overwrite"]
        if line:
            cmd += ["--cell-line", line]
        print(f"\n>>> {run.name}  [{line or 'unlabeled'}]", flush=True)
        p = subprocess.run(cmd)
        (ok if p.returncode == 0 else failed).append(run.name)

    print("\n" + "=" * 60)
    print(f"built {len(ok)} | failed {len(failed)} | skipped {len(skipped)}")
    for f in failed:
        print("  FAILED:", f)
    for s in skipped:
        print("  skipped:", s)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundle total: {total/1e9:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
