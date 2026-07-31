"""Re-run ONLY the tracking step on a finished run, to recover Trackastra's own lineage.

Runs tracked before 2026-07-29 never persisted Trackastra's parent table -- it lived
in memory and the graph survived only as far as classify_events. src.track now writes
`_memmap/ctc_lineage.csv` (see _write_lineage_csv), so re-tracking recovers the
complete graph for an old run.

Why a separate script rather than `main.py --reuse-masks`: a full pipeline run would
also redo classification and the vision review, which costs real money and produces
nothing this needs. Tracking reads the existing Cellpose masks and writes the lineage;
nothing else here touches an API.

Cellpose is NOT re-run either -- labels.dat is read as-is, so the segmentation is
identical to the original run and the resulting lineage is comparable to the geometric
one built from the same masks. That is the whole point: it isolates the linking model
as the only variable.

  python scripts/retrack_only.py data/output/TSC_batch2_M12_RUES2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.track import link_frames_trackastra  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--mode", choices=["greedy", "ilp"], default="greedy")
    args = ap.parse_args()

    mm = args.run_dir / "frames" / "_memmap"
    for name in ("frames.dat", "labels.dat"):
        if not (mm / name).is_file():
            sys.exit(f"missing {mm / name} -- this run's memmaps were cleaned up, "
                     "so tracking cannot be re-run without re-segmenting.")

    # Shape is not stored beside the memmaps; recover it from the exported PNGs.
    import cv2
    pngs = sorted((args.run_dir / "frames").glob("frame_*_raw*.png"))
    if not pngs:
        sys.exit(f"no exported frames under {args.run_dir / 'frames'}")
    h, w = cv2.imread(str(pngs[0]), cv2.IMREAD_GRAYSCALE).shape
    T = len(pngs)
    print(f"{args.run_dir.name}: {T} frames of {w}x{h}, mode={args.mode}")

    frames = np.memmap(mm / "frames.dat", dtype=np.uint8, mode="r", shape=(T, h, w))
    labels = np.memmap(mm / "labels.dat", dtype=np.uint16, mode="r", shape=(T, h, w))

    # Pass the memmaps THEMSELVES, not np.asarray() copies. src.track derives its
    # output directory from `frames.filename`, which a plain ndarray does not have --
    # wrapping them silently sent ctc_lineage.csv, tracked_masks.dat and a 2.9GB
    # float32 scratch file to the default data/frames/ instead of this run's dir.
    t0 = time.time()
    tracks = link_frames_trackastra(frames, labels, mode=args.mode)
    dt = time.time() - t0

    out = mm / "ctc_lineage.csv"
    if not out.is_file():
        sys.exit(f"tracking finished ({len(tracks)} nodes, {dt/60:.1f} min) but "
                 f"{out} was not written -- is src.track._write_lineage_csv wired in?")
    n = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"\ndone in {dt/60:.1f} min: {len(tracks):,} track nodes, "
          f"lineage for {n:,} canonical tracks -> {out}")
    print("\nNext: rebuild the bundle for this well so build_bundle picks ctc_lineage.csv "
          "up as the 'ctc' lineage source, then compare against the geometric graph.")
    (args.run_dir / "retrack_summary.json").write_text(json.dumps({
        "mode": args.mode, "n_nodes": len(tracks), "n_lineage_tracks": n,
        "minutes": round(dt / 60, 1),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
