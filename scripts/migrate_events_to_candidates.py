"""Move every already-built bundle's events.csv out to data/candidates/<well>/candidates.csv.

One-off migration for bundles built before 2026-07-30, when the detector's candidate
list lived inside the bundle as events.csv. See scripts/build_bundle.write_candidates
for why it moved and what the schema change is; this applies the same transform to
bundles that already exist, so nobody has to rebuild 20 wells to get it.

The bundle copy is only REMOVED once the candidates file has been written and read
back with the expected event count. The pipeline's own data/output/<run>/events.csv is
never touched, so the original is always recoverable.

  python scripts/migrate_events_to_candidates.py --dry-run
  python scripts/migrate_events_to_candidates.py
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_bundle import write_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("data/bundle"))
    ap.add_argument("--candidates", type=Path, default=Path("data/candidates"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would move; write and delete nothing")
    args = ap.parse_args()

    wells = sorted(d for d in args.bundle.iterdir() if (d / "manifest.json").is_file())
    moved = kept = 0
    for well in wells:
        src = well / "events.csv"
        hidden = well / "_events.csv.hidden"
        # The eval blinds one well by renaming rather than deleting; migrate that too,
        # so the hiding ritual can retire along with the file.
        actual = src if src.is_file() else (hidden if hidden.is_file() else None)
        if actual is None:
            print(f"[{well.name}] no events.csv -- nothing to move")
            kept += 1
            continue

        if args.dry_run:
            n = sum(1 for _ in open(actual, encoding="utf-8", errors="replace")) - 1
            print(f"[{well.name}] would move {n:,} rows -> {args.candidates / well.name}")
            moved += 1
            continue

        # write_candidates reads <run_dir>/events.csv, so point it at a shim whose
        # events.csv is the bundle's copy.
        class _Shim:
            name = well.name

            def __truediv__(self, other):
                return actual if other == "events.csv" else well / other

        info = write_candidates(_Shim(), args.candidates / well.name)
        out = args.candidates / well.name / "candidates.csv"
        if not out.is_file():
            print(f"[{well.name}] REFUSING to delete: {out} was not written")
            kept += 1
            continue
        n_read = sum(1 for _ in csv.DictReader(open(out, encoding="utf-8")))
        if n_read != info["n_events"]:
            print(f"[{well.name}] REFUSING to delete: wrote {info['n_events']} "
                  f"events but read back {n_read}")
            kept += 1
            continue

        actual.unlink()
        print(f"[{well.name}] moved, bundle copy removed")
        moved += 1

    print(f"\n{moved} moved, {kept} skipped"
          + (" (dry run -- nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
