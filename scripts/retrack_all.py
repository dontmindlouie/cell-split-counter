"""Re-track every run that still lacks a CTC lineage, verifying each before it lands.

Settled on M12_RUES2 (2026-07-30): Trackastra's own graph agrees with the geometric
one on 100% of the links both assign and adds ~17% more, including links across
multi-frame gaps that geometry cannot span by construction. So every well is worth
re-tracking once. ~7 min each on CUDA reusing existing Cellpose masks; no vision spend.

The safety property that matters: re-tracking OVERWRITES tracked_masks.dat, and the
bundle's tracks.csv was measured against the ORIGINAL one. If a re-track produced
different canonical ids, the new lineage would reference cells that tracks.csv does not
have, and silently rebuilding the bundle from it would corrupt the well. So each run is
verified -- the lineage's id set must equal the bundle's -- and the bundle is only
rebuilt on a match. A mismatch is reported and skipped, leaving that well untouched.

  python scripts/retrack_all.py --dry-run
  python scripts/retrack_all.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def candidates(out_dir: Path, bundle: Path) -> list[Path]:
    ready = []
    for d in sorted(out_dir.iterdir()):
        mm = d / "frames" / "_memmap"
        if not d.is_dir() or not (mm / "frames.dat").is_file() or not (mm / "labels.dat").is_file():
            continue
        if (mm / "ctc_lineage.csv").is_file():
            continue
        if not (bundle / d.name / "tracks.csv").is_file():
            continue
        ready.append(d)
    return ready


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/output")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/bundle")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    from src.lineage import score_lineage_links

    runs = candidates(args.out_dir, args.bundle)
    print(f"{len(runs)} runs to re-track\n", flush=True)
    if args.dry_run:
        for d in runs:
            print("  would re-track", d.name)
        return

    ok = mismatch = failed = 0
    t_start = time.time()
    for i, d in enumerate(runs, 1):
        print(f"\n===== [{i}/{len(runs)}] {d.name} =====", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/retrack_only.py"), str(d)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
            failed += 1
            continue

        ctc_path = d / "frames" / "_memmap" / "ctc_lineage.csv"
        if not ctc_path.is_file():
            print("  FAILED: no ctc_lineage.csv written", flush=True)
            failed += 1
            continue

        bdir = args.bundle / d.name
        ctc = pd.read_csv(ctc_path)
        tracks = pd.read_csv(bdir / "tracks.csv")
        ctc_ids = set(ctc.track_id.astype(int))
        trk_ids = set(tracks.track_id.astype(int))
        if ctc_ids != trk_ids:
            print(f"  ID MISMATCH -- lineage has {len(ctc_ids):,} ids, tracks.csv has "
                  f"{len(trk_ids):,} ({len(ctc_ids ^ trk_ids):,} differ). "
                  f"NOT rebuilding this bundle; its lineage is unchanged.", flush=True)
            mismatch += 1
            continue

        lin = score_lineage_links(ctc, tracks)
        lin.to_csv(bdir / "lineage.csv", index=False)
        man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        man["lineage"] = {"coverage": "complete", "source": "ctc", "n_tracks": len(lin)}
        (bdir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        n_scored = int((lin.dna_ratio.astype(str) != "").sum())
        print(f"  OK in {(time.time()-t0)/60:.1f} min: {len(lin):,} tracks, "
              f"{n_scored:,} scored links -> bundle rebuilt", flush=True)
        ok += 1

    print(f"\n\n===== DONE in {(time.time()-t_start)/60:.1f} min ====="
          f"\n  rebuilt   {ok}\n  mismatch  {mismatch}\n  failed    {failed}", flush=True)


if __name__ == "__main__":
    main()
