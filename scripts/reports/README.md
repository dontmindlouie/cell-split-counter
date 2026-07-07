# QC report scripts

Small standalone scripts that turn a run's `events.csv` into one self-contained
`.html` file — open it by double-clicking, no server, no install beyond this repo's
existing Python environment. Each one is meant to be **copied and edited**, not
extended into a framework — that's why they're short and read top-to-bottom.

## Run one

```
python scripts/reports/bleach_risk_scatter.py data/output/<your_run>
python scripts/reports/confidence_histogram.py data/output/<your_run>
python scripts/reports/anomaly_rates_bar.py data/output/<your_run>
python scripts/reports/spot_check_review.py data/output/<your_run>
```

Each writes into `data/output/<your_run>/reports/` by default (`--out` to change it).

## What each one answers

- **`bleach_risk_scatter.py`** — is `bleach_risk` actually informative, or does it
  just track frame position? (Spoiler on most runs: it's `frame / total_frames`, a
  proxy, not a measurement — the chart's Pearson r makes this obvious at a glance.)
- **`confidence_histogram.py`** — is the review step actually discriminating real
  splits from noise? A healthy run looks bimodal (mass near 0, mass near 1), not flat.
- **`anomaly_rates_bar.py`** — how often does each abnormality flag
  (misaligned_chromosomes / lagging_chromosome / anaphase_bridge / micronucleus /
  binucleation) show up among confirmed splits this run?
- **`spot_check_review.py`** — is a different shape from the other three: not a
  chart, an interactive blind-review tool. Samples events across risk buckets
  (fail-open reviews, GPT-confidence-floor downgrades, anomaly-flagged, near-edge,
  auto-confirmed, obvious rejects), shows you the crop sequence with no pipeline
  verdict visible, lets you call it, then reveals what the pipeline said and
  live-scores agreement per bucket. This is a QA/validation tool for whoever is
  deciding whether to trust a run's output, not a chart-writing example — read it
  separately from the pattern below if you're adapting it.

## Writing your own

All three follow the same shape:

1. Read `events.csv` with `csv.DictReader`.
2. **Dedupe by `parent_id`** — every split event has 2 rows (one per daughter track)
   with identical `peak_frame`/`bleach_risk`/`claude_confidence`/etc. If you don't
   dedupe, every chart double-counts.
3. Build either a list of `(x, y)` points or parallel `categories`/`values` lists.
4. Call `render_scatter_html(...)` or `render_bar_html(...)` from
   `src/reports/html_chart.py`, pass `title`/`subtitle`/labels, get a `.html` path back.

That's the whole pattern — no new library to learn beyond those two functions. If a
question needs a chart type these two don't cover (e.g. a true time series with a
line, not a scatter), look at `render_scatter_html`'s SVG/JS as a starting template
rather than reaching for a new charting dependency — this project deliberately
doesn't use matplotlib/pandas/plotly, so anything you add either stays inside this
self-contained-HTML pattern or needs a separate conversation about adding a real
plotting stack.
