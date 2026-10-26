"""Run src.mitosis_scan on every 20260822 EVT field in a folder of ND2s.

    uv run python scripts/run_mitosis_scan_evt_0822.py <folder with the crop ND2s> [--workers 1]

Writes data/mitosis_scan/20260822_EVT_crop{c}_{field}/ (candidates.csv, report.json, objects.csv.gz)
with profiles/mitosis/h2b_tsc_evt.json, the profile measured on this dataset's clicked fields.
One worker by default: a 16 GB machine ran out of memory with three 1.8 GB fields in parallel
alongside other work.
"""
import argparse, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "mitosis_scan"
LOG = REPO / "data" / "run_mitosis_scan_evt_0822.log"
PROFILE = "profiles/mitosis/h2b_tsc_evt.json"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}\n")


def run(nd2):
    c, fld = re.search(r"crop_(\d+)_ZO1_EVT_(D\d_\d)\.nd2$", nd2.name).groups()
    name = f"20260822_EVT_crop{c}_{fld}"
    cmd = [sys.executable, "-m", "src.mitosis_scan", str(nd2), "--profile", PROFILE, "--out", str(OUT / name)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout or r.stderr).strip()
    log(f"{'OK' if r.returncode == 0 else 'FAIL'} {name}: {out.splitlines()[-1] if out else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nd2_dir"); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    # the folder may also hold a "- Denoised" copy of a field: only the plain crop files are fields
    nd2s = sorted(p for p in Path(a.nd2_dir).glob("*.nd2") if re.search(r"crop_\d+_ZO1_EVT_D\d_\d\.nd2$", p.name))
    log(f"start {len(nd2s)} fields")
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, nd2s))
    log("ALL_DONE")
