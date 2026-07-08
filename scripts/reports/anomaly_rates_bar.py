"""QC check: how common is each abnormality flag among confirmed splits?

Why this exists: micronucleus / lagging_chromosome / anaphase_bridge / misaligned_
chromosomes / binucleation are the scientifically interesting signal the researcher cares about
-- this gives a quick per-run read of their rates without opening the CSV by hand.
Only counts rows with ai_confidence >= 0.5 (near_edge excluded by default, since
a partially-visible cell at the frame boundary makes abnormality judgment unreliable
-- see events.csv's near_edge column doc in generate_package_readme.py).

Usage:
  python scripts/reports/anomaly_rates_bar.py data/output/<run_dir>
  python scripts/reports/anomaly_rates_bar.py data/output/<run_dir> --include-near-edge

This is the reference example for src/reports/html_chart.py's render_bar_html() on
plain categorical (non-histogram) data -- copy this file to chart acd_division_type
counts or any other categorical column the same way.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reports.html_chart import render_bar_html

_FLAGS = ["misaligned_chromosomes", "lagging_chromosome", "anaphase_bridge", "micronucleus", "binucleation"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv")
    parser.add_argument("--include-near-edge", action="store_true", help="include near_edge=1 events (default: excluded)")
    parser.add_argument("--out", default=None, help="output .html path (default: <run_dir>/reports/anomaly_rates_bar.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = list(csv.DictReader(open(run_dir / "events.csv")))

    seen: dict[str, dict] = {}
    for r in rows:
        if not r["ai_confidence"] or float(r["ai_confidence"]) < 0.5:
            continue
        if not args.include_near_edge and r["near_edge"] == "1":
            continue
        seen[r["parent_id"]] = r
    confirmed = list(seen.values())

    counts = [sum(1 for r in confirmed if r[flag] == "1") for flag in _FLAGS]
    n = len(confirmed)
    rates = [c / n if n else 0.0 for c in counts]

    out_path = Path(args.out) if args.out else run_dir / "reports" / "anomaly_rates_bar.html"
    render_bar_html(
        _FLAGS, [round(r, 4) for r in rates],
        out_path=out_path,
        title="Abnormality flag rates among confirmed splits",
        subtitle=f"{run_dir.name} · {n} confirmed events" + ("" if args.include_near_edge else " (near_edge excluded)"),
        y_label="rate",
        stats={flag: str(c) for flag, c in zip(_FLAGS, counts)},
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
