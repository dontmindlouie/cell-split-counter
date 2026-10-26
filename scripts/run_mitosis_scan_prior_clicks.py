"""Score src.mitosis_scan against every earlier set of hand clicks, with the EVT-calibrated profile UNCHANGED.

The question is whether the detector generalizes: different cell lines (nTSC/pTSC/RUES2/BeWo), a
3-min instead of 10-min interval, 0.57 instead of 0.86 um/px, single-channel files and a 13-position
ND2. Only facts read from the files are changed per experiment (which channel is the nucleus, which
position); every threshold stays as calibrated on 20260822 EVT. Times and sizes are in minutes and um,
so the profile carries over without new numbers.

Writes data/mitosis_scan_prior/<name>/ and a summary table data/mitosis_scan_prior/summary.csv.
To compare another profile: run_mitosis_scan_prior_clicks.py [profile.json] [out dir]
"""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO.parent / "nd2_raw"
HR = REPO.parent / "cell-split-counter-shared-data" / "human_review"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "data" / "mitosis_scan_prior"
LOG = OUT / "run.log"
PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "profiles" / "mitosis" / "h2b_tsc.json"
B2 = str(RAW / "20260709_TSC_Batch2" / "20260709_nTSCP20_PTSCP7_RUES2P42_WGDP2__{}.nd2")
B2_WELLS = {"M1": "M1", "M2": "M2 nTSC", "M3": "M3 nTSC", "M4": "M4 nTSC", "M5": "M5 pTSC",
            "M6": "M6 pTSC", "M7": "M7 pTSC", "M8": "M8 pTSC", "M9": "M9 pTSC"}

# name: (nd2, channel or None for single-channel files, position or None, clicks csv)
JOBS = {f"TSC_batch2_{w}": (B2.format(s), None, None, HR / f"fiji_clicks_TSC_batch2_{w}_2026-10-19.csv")
        for w, s in B2_WELLS.items()}
JOBS["Bewo_M5"] = (RAW / "Bewo_nd2" / "202660629_Bewop920x_M5.nd2", None, None, HR / "fiji_clicks_Bewo_M5_2026-10-19.csv")
for pos, day in [(1, "15"), (2, "16"), (3, "16"), (4, "16"), (5, "15"), (6, "16"), (7, "16"), (8, "16"), (9, "16")]:
    JOBS[f"ZO1_STB_P{pos}"] = (RAW / "20260925_ZO1_STB" / "20260925_ZO1_STB001.nd2", 1, pos - 1,
                               HR / f"fiji_clicks_20260925_ZO1_STB_P{pos}_2026-10-{day}.csv")


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}\n")


def run(item):
    name, (nd2, ch, pos, clicks) = item
    cl = pd.read_csv(clicks)
    cl = cl.rename(columns={"frame0": "frame", "x_px": "x", "y_px": "y"})[["click", "frame", "x", "y"]]
    if not len(cl):
        log(f"SKIP {name}: no clicks"); return
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cl.to_csv(out / "clicks.csv", index=False)
    cmd = [sys.executable, "-m", "src.mitosis_scan", str(nd2), "--profile", str(PROFILE), "--out", str(out),
           "--clicks", str(out / "clicks.csv"), "--fixed"]
    if ch is not None:
        cmd += ["--channel", str(ch)]
    else:
        cmd += ["--channel", "none"]
    if pos is not None:
        cmd += ["--position", str(pos)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or r.stderr).strip().splitlines()
    log(f"{'OK' if r.returncode == 0 else 'FAIL'} {name}: {tail[-1] if tail else ''}")


OUT.mkdir(parents=True, exist_ok=True)
log(f"start {len(JOBS)} click sets")
# ZO1_STB positions are 1,273 frames each. Two at a time: three plus other work ran a 16 GB machine out of memory
with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, sorted(JOBS.items(), key=lambda kv: not kv[0].startswith("ZO1"))))
rows = []
for name in JOBS:
    f = OUT / name / "report.json"
    if f.exists():
        r = json.loads(f.read_text())
        rows.append({"name": name, "frames": r["frames"], "clicks": r["clicks"], "clicks_in_mask": r["clicks_in_mask"],
                     "found": r["chosen"]["found"], "candidates": r["candidates"],
                     "candidates_in_clicked_span": r.get("candidates_in_clicked_span")})
pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
log("ALL_DONE")
