"""QC check: does bleach_risk carry any information beyond frame position?

Why this exists: bleach_risk is computed in src/pipeline.py as `frame / total_frames`
-- a proxy, not a measurement (see src/classify.py's field comment). Plotted against
peak_frame it will always be a near-perfect diagonal line; this script exists to make
that visible and quantify it (Pearson r), rather than trusting the column name.

Usage:
  python scripts/reports/bleach_risk_scatter.py data/output/<run_dir>
  python scripts/reports/bleach_risk_scatter.py data/output/<run_dir> --out my_report.html

This is also the reference example for src/reports/html_chart.py's scatter renderer --
copy this file and swap the two CSV columns you read to check a different pair of
fields for a spurious/expected relationship.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.reports.html_chart import render_scatter_html


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    return cov / ((var_x * var_y) ** 0.5) if var_x and var_y else float("nan")


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = cov / var_x if var_x else 0.0
    return slope, mean_y - slope * mean_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv")
    parser.add_argument("--out", default=None, help="output .html path (default: <run_dir>/reports/bleach_risk_scatter.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = list(csv.DictReader(open(run_dir / "events.csv")))

    # events.csv has one row per daughter track -- dedupe to one point per split
    # event (grouped by parent_id) so points aren't double-counted.
    seen: dict[str, tuple[float, float]] = {}
    for r in rows:
        if not r["bleach_risk"]:
            continue
        seen[r["parent_id"]] = (float(r["peak_frame"]), float(r["bleach_risk"]))
    points = sorted(seen.values())

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    r = _pearson_r(xs, ys)
    slope, intercept = _least_squares(xs, ys)

    callout = None
    if abs(r) > 0.99:
        callout = (
            f"<strong>Not an independent measurement.</strong> Pearson r between "
            f"<code>bleach_risk</code> and <code>peak_frame</code> is <strong>{r:.7f}</strong> "
            f"&mdash; consistent with <code>bleach_risk = frame / total_frames</code> "
            f"(src/pipeline.py). This field can't tell you which candidates are "
            f"trustworthy, only how late in the movie they occurred."
        )

    out_path = Path(args.out) if args.out else run_dir / "reports" / "bleach_risk_scatter.html"
    render_scatter_html(
        points,
        out_path=out_path,
        title="bleach_risk vs. peak_frame",
        subtitle=f"{run_dir.name} · {len(points)} split events",
        x_label="peak_frame",
        y_label="bleach_risk",
        callout_html=callout,
        fit_line=(slope, intercept),
        fit_label=f"y = {slope:.6f}x + {intercept:.4f}",
        stats={
            "events plotted": str(len(points)),
            "pearson r": f"{r:.7f}",
            "slope": f"{slope:.6f}",
            "frame range": f"{int(min(xs))} – {int(max(xs))}" if xs else "n/a",
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
