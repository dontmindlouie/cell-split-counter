"""Re-track every run and REBUILD its bundle from scratch.

Two-stage per well: `retrack_only.py` recovers Trackastra's own CTC graph from the
existing Cellpose masks (~7-18 min on CUDA, no vision spend, segmentation untouched),
then `build_bundle.py --overwrite` regenerates the whole bundle from the result.

## Why this script had to change (2026-07-31)

It used to re-track, then patch the new `lineage.csv` into the EXISTING bundle and
call that "rebuilt". It also refused to do even that unless the new lineage's track-id
set exactly equalled the old `tracks.csv`'s, on the reasoning that a mismatch meant the
canonical ids had moved and the bundle would be corrupted by the patch.

That guard was correct for a patch and is wrong for a rebuild, and after `03df4b4`
(the `_bridge_track_gaps` over-merge fix) it became actively dangerous: that fix
deliberately changes canonical ids, so the id sets now legitimately disagree for every
well. The old script would have hit ID MISMATCH on all 21, skipped all 21, and printed
a clean summary -- a silent no-op that reads like success.

The fix is not a better guard. It is to stop patching. `tracked_masks.dat`,
`tracks.csv`, `labels/*.png` and `lineage.csv` all key off canonical track ids, so once
those ids move, *every* one of them is stale and only a full rebuild is coherent.
Partial recreation is the whole problem.

The old skip-if-already-re-tracked behaviour is likewise off by default: a run that was
re-tracked BEFORE the bridging fix carries pre-fix lineage and is exactly the case that
most needs redoing. `--only-missing` restores it.

## What is checked

Verification moved to AFTER the build, where it can be meaningful: the freshly written
lineage must reference only ids the freshly written tracks.csv has, and the manifest
must carry a provenance block. A mismatch there is a real bug, not a bookkeeping one.

The ND2 is required for calibration and is resolved by BASENAME from the well's
existing manifest (`source_nd2` stores a bare filename, not a path). Every well is
resolved up front, so an unreachable ND2 is reported in seconds rather than after
hours of GPU time.

  python scripts/retrack_all.py --dry-run
  python scripts/retrack_all.py
  python scripts/retrack_all.py --wells TSC_batch2_M12_RUES2
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def nd2_index(root: Path, depth: int = 3) -> dict[str, Path]:
    """basename -> path for every ND2 under `root`, so manifests can name a file
    without pinning the directory it lived in when the run was made."""
    out: dict[str, Path] = {}
    for p in root.glob("/".join(["*"] * (depth - 1) + ["*.nd2"])):
        out.setdefault(p.name, p)
    for d in range(depth - 1, 0, -1):
        for p in root.glob("/".join(["*"] * (d - 1) + ["*.nd2"])):
            out.setdefault(p.name, p)
    return out


def plan(out_dir: Path, bundle: Path, index: dict[str, Path],
         only_missing: bool, wells: list[str] | None) -> list[dict]:
    """One entry per well, carrying everything the rebuild needs -- read from the OLD
    manifest before it is overwritten, since cell_line/condition live nowhere else."""
    rows = []
    for d in sorted(out_dir.iterdir()):
        if not d.is_dir() or (wells and d.name not in wells):
            continue
        mm = d / "frames" / "_memmap"
        if not (mm / "frames.dat").is_file() or not (mm / "labels.dat").is_file():
            continue
        man_p = bundle / d.name / "manifest.json"
        if not man_p.is_file():
            continue
        if only_missing and (mm / "ctc_lineage.csv").is_file():
            continue
        man = json.loads(man_p.read_text(encoding="utf-8"))
        name = man.get("source_nd2") or ""
        rows.append({
            "run": d, "name": d.name,
            "cell_line": man.get("cell_line"), "condition": man.get("condition"),
            "nd2_name": name, "nd2": index.get(Path(name).name),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/output")
    ap.add_argument("--bundle", type=Path, default=ROOT / "data/bundle")
    ap.add_argument("--candidates", type=Path, default=ROOT / "data/candidates")
    ap.add_argument("--nd2-root", type=Path, default=ROOT.parent,
                    help="directory tree searched for the ND2s named in each manifest")
    ap.add_argument("--wells", nargs="*", default=None, help="limit to these well names")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip runs that already have a ctc_lineage.csv. OFF by default: "
                         "a run re-tracked before the bridging fix carries pre-fix "
                         "lineage and is precisely what needs redoing.")
    ap.add_argument("--skip-retrack", action="store_true",
                    help="rebuild bundles from the lineage already on disk, no GPU pass")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    index = nd2_index(args.nd2_root)
    runs = plan(args.out_dir, args.bundle, index, args.only_missing, args.wells)
    unresolved = [r for r in runs if r["nd2"] is None]

    print(f"{len(runs)} wells planned; {len(index)} ND2s found under {args.nd2_root}\n")
    for r in runs:
        print(f"  {r['name']:32s} {r['cell_line'] or '?':6s} "
              f"{r['nd2'] or 'NO ND2 FOUND: ' + (r['nd2_name'] or '(manifest has none)')}")
    if unresolved:
        # Refuse rather than proceed: build_bundle exits on a missing ND2 anyway, and
        # discovering that per-well after each GPU pass wastes hours before the first
        # bundle lands. Calibration is not optional -- a bundle without real pixel size
        # and timestamps has fabricated units.
        print(f"\n{len(unresolved)} well(s) have no reachable ND2. Point --nd2-root at "
              f"the drive holding them, or pass --wells to do the rest. Nothing run.")
        sys.exit(1)
    if args.dry_run:
        print("\n(dry run -- nothing executed)")
        return

    ok = failed = 0
    t_start = time.time()
    for i, r in enumerate(runs, 1):
        print(f"\n===== [{i}/{len(runs)}] {r['name']} =====", flush=True)
        t0 = time.time()

        if not args.skip_retrack:
            p = subprocess.run(
                [sys.executable, str(ROOT / "scripts/retrack_only.py"), str(r["run"])],
                cwd=ROOT, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"  RETRACK FAILED (exit {p.returncode})\n"
                      f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}", flush=True)
                failed += 1
                continue
            print(f"  re-tracked in {(time.time()-t0)/60:.1f} min", flush=True)

        cmd = [sys.executable, str(ROOT / "scripts/build_bundle.py"), str(r["run"]),
               "--nd2", str(r["nd2"]), "--out", str(args.bundle),
               "--candidates", str(args.candidates), "--overwrite"]
        if r["cell_line"]:
            cmd += ["--cell-line", r["cell_line"]]
        if r["condition"]:
            cmd += ["--condition", r["condition"]]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"  BUILD FAILED (exit {p.returncode})\n"
                  f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}", flush=True)
            failed += 1
            continue

        # Post-build coherence, on the files that were just written together. Unlike
        # the id-set equality this replaces, a failure here means something is
        # genuinely inconsistent rather than merely newer.
        bdir = args.bundle / r["name"]
        lin = pd.read_csv(bdir / "lineage.csv")
        trk = pd.read_csv(bdir / "tracks.csv")
        orphan = set(lin.track_id.astype(int)) - set(trk.track_id.astype(int))
        man = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        if orphan:
            print(f"  INCONSISTENT: {len(orphan):,} lineage ids are absent from the "
                  f"tracks.csv written in the same build.", flush=True)
            failed += 1
            continue
        if not man.get("provenance", {}).get("built_at"):
            print("  INCONSISTENT: no provenance block written.", flush=True)
            failed += 1
            continue
        print(f"  OK in {(time.time()-t0)/60:.1f} min: {len(lin):,} tracks, "
              f"lineage source {man.get('lineage', {}).get('source')}, "
              f"built {man['provenance']['built_at']}", flush=True)
        ok += 1

    print(f"\n\n===== DONE in {(time.time()-t_start)/60:.1f} min ====="
          f"\n  rebuilt {ok}\n  failed  {failed}", flush=True)
    if ok:
        print("\nNext: re-sync the researcher's copies (robocopy /MT:8, never Explorer). "
              "Any count taken from a pre-rebuild bundle is not comparable to these.")


if __name__ == "__main__":
    main()
