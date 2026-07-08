"""QC check: what does the confidence distribution actually look like for a run?

Why this exists: a healthy run should show a clearly bimodal confidence distribution
(most candidates near 0.0, confirmed splits clustered high) -- a flat or unimodal
spread is a sign the review step isn't discriminating real splits from noise.

Usage:
  python scripts/reports/confidence_histogram.py data/output/<run_dir>
  python scripts/reports/confidence_histogram.py data/output/<run_dir> --bins 30

This is the reference example for src/reports/html_chart.py's histogram_bins() +
render_bar_html() combo -- copy this file to histogram any other numeric column
(cell_size_um2, tracker_persistence_score, etc.).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reports.html_chart import histogram_bins, render_bar_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--out", default=None, help="output .html path (default: <run_dir>/reports/confidence_histogram.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = list(csv.DictReader(open(run_dir / "events.csv")))

    # one row per daughter track -- dedupe by parent_id so each split event counts once.
    seen: dict[str, float] = {}
    for r in rows:
        if r["ai_confidence"]:
            seen[r["parent_id"]] = float(r["ai_confidence"])
    values = list(seen.values())

    labels, counts = histogram_bins(values, n_bins=args.bins)
    n_confirmed = sum(1 for v in values if v >= 0.5)

    out_path = Path(args.out) if args.out else run_dir / "reports" / "confidence_histogram.html"
    render_bar_html(
        labels, counts,
        out_path=out_path,
        title="ai_confidence distribution",
        subtitle=f"{run_dir.name} · {len(values)} split events, bin start shown per bar",
        y_label="events",
        stats={
            "events": str(len(values)),
            "confirmed (>=0.5)": f"{n_confirmed} ({n_confirmed / len(values):.0%})" if values else "n/a",
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
