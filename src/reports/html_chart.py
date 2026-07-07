"""Self-contained, interactive HTML chart generators for one-off pipeline QC questions.

Why this exists: QC questions on a run's events.csv ("is this field real, or does it
just track frame number?" / "does confidence look sane over time?" / "how common is
each abnormality flag?") come up constantly and don't warrant a dashboard -- they
warrant one throwaway HTML page you can double-click open, with the data baked in.
Two functions are the whole pattern: pull (x, y) points or (category, value) pairs
out of a CSV in a small script, call one function, get a page. See scripts/reports/
for worked examples against real run output -- copy one and change what it reads.

Design constraints (deliberate -- keep them if you extend this):
- Self-contained: CSS/JS/data all inlined into one .html file. No CDN, no build step,
  no server. Works from a double-click in any browser, including handed to someone
  outside this repo.
- No new dependencies: pure stdlib (json). This project doesn't use pandas/matplotlib
  elsewhere and a one-off QC page doesn't justify adding them.
- Light/dark aware, colorblind-checked palette, taken from the project's dataviz
  design system (categorical slot 1 "blue" + status "critical" -- see the dataviz
  skill's references/palette.md if picking new colors; don't pick by eye).
"""

import json
from pathlib import Path

_CSS = """
:root {
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --text-muted:     #898781;
  --gridline:       #e1e0d9;
  --baseline:       #c3c2b7;
  --series-1:       #2a78d6;
  --series-1-wash:  rgba(42,120,214,0.12);
  --border:         rgba(11,11,11,0.10);
  --critical:       #d03b3b;
  --critical-wash:  #fbeceb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --series-1:       #3987e5;
    --series-1-wash:  rgba(57,135,229,0.16);
    --border:         rgba(255,255,255,0.10);
    --critical:       #e66767;
    --critical-wash:  #2a1717;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0;
  padding: 28px 20px 40px;
}
.report { max-width: 880px; margin: 0 auto; }
h1 { font-size: 18px; font-weight: 600; margin: 0 0 2px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }
.callout {
  background: var(--critical-wash);
  border: 1px solid var(--border);
  border-left: 3px solid var(--critical);
  border-radius: 6px;
  padding: 14px 16px;
  margin-bottom: 22px;
  font-size: 13.5px;
  line-height: 1.55;
}
.callout strong { color: var(--critical); }
.callout code { background: var(--border); padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 18px 8px;
  overflow-x: auto;
}
svg { display: block; }
.axis-label { fill: var(--text-muted); font-size: 11px; }
.tick-text { fill: var(--text-muted); font-size: 10px; }
.gridline { stroke: var(--gridline); stroke-width: 1; }
.baseline { stroke: var(--baseline); stroke-width: 1; }
.fit-line { stroke: var(--series-1); stroke-width: 2; fill: none; opacity: 0.9; }
.fit-label { fill: var(--text-secondary); font-size: 11px; }
.dot { fill: var(--series-1); opacity: 0.55; }
.bar { fill: var(--series-1); }
.bar:hover, .bar.hovered { opacity: 0.8; }
.crosshair { stroke: var(--text-muted); stroke-width: 1; opacity: 0; pointer-events: none; }
.hover-dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; opacity: 0; pointer-events: none; }
.hit-layer { fill: transparent; cursor: crosshair; }
.tooltip {
  position: absolute;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
  line-height: 1.5;
  pointer-events: none;
  opacity: 0;
  transform: translate(-50%, -110%);
  box-shadow: 0 4px 16px rgba(0,0,0,0.15);
  white-space: nowrap;
}
.tooltip .val { font-weight: 600; color: var(--text-primary); }
.tooltip .lbl { color: var(--text-secondary); }
.stats-row {
  display: flex; gap: 28px; margin-top: 14px; padding-top: 14px;
  border-top: 1px solid var(--border); font-size: 12px; flex-wrap: wrap;
}
.stat-label { color: var(--text-muted); display: block; }
.stat-value { color: var(--text-primary); font-weight: 600; font-size: 14px; }
details { margin-top: 18px; }
summary { cursor: pointer; font-size: 12.5px; color: var(--text-secondary); padding: 6px 0; }
.table-wrap { max-height: 260px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { text-align: right; padding: 4px 12px; font-variant-numeric: tabular-nums; }
th { position: sticky; top: 0; background: var(--surface-1); color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }
tr:nth-child(even) { background: var(--series-1-wash); }
"""


def _stats_row_html(stats: dict[str, str] | None) -> str:
    if not stats:
        return ""
    items = "".join(
        f'<div><span class="stat-label"></span><span class="stat-value"></span></div>'
        for _ in stats
    )
    # build with real text via textContent-equivalent escaping is unnecessary here --
    # stats are author-supplied strings from our own scripts, not untrusted CSV cells.
    rows = "".join(
        f'<div><span class="stat-label">{k}</span><span class="stat-value">{v}</span></div>'
        for k, v in stats.items()
    )
    return f'<div class="stats-row">{rows}</div>'


def _callout_html(callout_html: str | None) -> str:
    if not callout_html:
        return ""
    return f'<div class="callout">{callout_html}</div>'


def _page(title: str, subtitle: str, callout_html: str | None, card_title: str,
          card_sub: str, chart_html: str, stats: dict[str, str] | None,
          table_html: str, script: str) -> str:
    return f"""<title>{title}</title>
<style>{_CSS}</style>
<div class="report">
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  {_callout_html(callout_html)}
  <div class="card">
    <p style="font-size:13px;font-weight:600;margin:0 0 2px;">{card_title}</p>
    <p style="font-size:12px;color:var(--text-muted);margin:0 0 10px;">{card_sub}</p>
    {chart_html}
    {_stats_row_html(stats)}
    {table_html}
  </div>
</div>
<script>{script}</script>
"""


def _table_html(headers: list[str], rows: list[tuple], table_id: str) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    return f"""<details>
    <summary>View underlying data (table)</summary>
    <div class="table-wrap">
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody id="{table_id}"></tbody>
      </table>
    </div>
  </details>"""


def render_scatter_html(
    points: list[tuple[float, float]],
    *,
    out_path: str | Path,
    title: str,
    subtitle: str = "",
    x_label: str = "x",
    y_label: str = "y",
    callout_html: str | None = None,
    fit_line: tuple[float, float] | None = None,
    fit_label: str | None = None,
    stats: dict[str, str] | None = None,
    card_title: str = "",
    card_sub: str = "",
) -> Path:
    """Render a scatter plot (+ optional fit line, crosshair tooltip, data table) to out_path.

    points: list of (x, y) pairs, any numeric range -- axes auto-scale to the data.
    fit_line: optional (slope, intercept) drawn as y = slope*x + intercept across the
      x range, in the same series color -- use this to make a suspected deterministic
      relationship visually obvious (see scripts/reports/bleach_risk_scatter.py).
    stats: small key -> display-string dict shown as a row of stat tiles below the
      chart (e.g. {"n": "587", "pearson r": "0.9999995"}).
    """
    out_path = Path(out_path)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = (min(xs), max(xs)) if xs else (0, 1)
    y_min, y_max = (min(ys), max(ys)) if ys else (0, 1)
    x_pad = (x_max - x_min) * 0.02 or 1
    y_pad = (y_max - y_min) * 0.05 or 1
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    chart_html = """
    <div style="position: relative;">
      <svg id="chart" viewBox="0 0 820 380" width="100%" height="380"></svg>
      <div class="tooltip" id="tooltip">
        <span class="lbl" id="tt-xlabel"></span> <span class="val" id="tt-x"></span><br>
        <span class="lbl" id="tt-ylabel"></span> <span class="val" id="tt-y"></span>
      </div>
    </div>"""

    table_html = _table_html([x_label, y_label], [], "data-table")

    fit_json = json.dumps(fit_line) if fit_line is not None else "null"
    script = f"""
(function () {{
  var points = {json.dumps(points)};
  var xLo = {x_lo}, xHi = {x_hi}, yLo = {y_lo}, yHi = {y_hi};
  var fit = {fit_json};
  var fitLabel = {json.dumps(fit_label or "")};
  var xLabel = {json.dumps(x_label)}, yLabel = {json.dumps(y_label)};

  var svg = document.getElementById('chart');
  var W = 820, H = 380;
  var M = {{ top: 16, right: 20, bottom: 40, left: 54 }};
  var plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  function xScale(x) {{ return M.left + ((x - xLo) / (xHi - xLo)) * plotW; }}
  function yScale(y) {{ return M.top + plotH - ((y - yLo) / (yHi - yLo)) * plotH; }}

  var ns = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {{
    var e = document.createElementNS(ns, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }}

  function niceTicks(lo, hi, n) {{
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = (span / n) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var ticks = [];
    var start = Math.ceil(lo / step) * step;
    for (var v = start; v <= hi; v += step) ticks.push(Math.round(v / step) * step);
    return ticks;
  }}

  niceTicks(yLo, yHi, 5).forEach(function (v) {{
    var y = yScale(v);
    svg.appendChild(el('line', {{ class: 'gridline', x1: M.left, x2: W - M.right, y1: y, y2: y }}));
    var t = el('text', {{ class: 'tick-text', x: M.left - 8, y: y + 3, 'text-anchor': 'end' }});
    t.textContent = v;
    svg.appendChild(t);
  }});
  niceTicks(xLo, xHi, 6).forEach(function (v) {{
    var x = xScale(v);
    var t = el('text', {{ class: 'tick-text', x: x, y: H - M.bottom + 18, 'text-anchor': 'middle' }});
    t.textContent = v;
    svg.appendChild(t);
  }});

  svg.appendChild(el('line', {{ class: 'baseline', x1: M.left, x2: W - M.right, y1: M.top + plotH, y2: M.top + plotH }}));
  var xl = el('text', {{ class: 'axis-label', x: M.left + plotW / 2, y: H - 6, 'text-anchor': 'middle' }});
  xl.textContent = xLabel;
  svg.appendChild(xl);
  var yl = el('text', {{ class: 'axis-label', x: -(M.top + plotH / 2), y: 14, 'text-anchor': 'middle', transform: 'rotate(-90)' }});
  yl.textContent = yLabel;
  svg.appendChild(yl);

  if (fit) {{
    var slope = fit[0], intercept = fit[1];
    var y0 = slope * xLo + intercept, y1v = slope * xHi + intercept;
    var path = 'M ' + xScale(xLo) + ' ' + yScale(y0) + ' L ' + xScale(xHi) + ' ' + yScale(y1v);
    svg.appendChild(el('path', {{ class: 'fit-line', d: path }}));
    if (fitLabel) {{
      var fl = el('text', {{ class: 'fit-label', x: xScale(xHi) - 6, y: yScale(y1v) - 8, 'text-anchor': 'end' }});
      fl.textContent = fitLabel;
      svg.appendChild(fl);
    }}
  }}

  var dotsGroup = el('g', {{}});
  points.forEach(function (p) {{
    dotsGroup.appendChild(el('circle', {{ class: 'dot', cx: xScale(p[0]), cy: yScale(p[1]), r: 3 }}));
  }});
  svg.appendChild(dotsGroup);

  var crosshair = el('line', {{ class: 'crosshair', y1: M.top, y2: M.top + plotH }});
  svg.appendChild(crosshair);
  var hoverDot = el('circle', {{ class: 'hover-dot', r: 5 }});
  svg.appendChild(hoverDot);
  var hit = el('rect', {{ class: 'hit-layer', x: M.left, y: M.top, width: plotW, height: plotH }});
  svg.appendChild(hit);

  var tooltip = document.getElementById('tooltip');
  var ttX = document.getElementById('tt-x'), ttY = document.getElementById('tt-y');
  document.getElementById('tt-xlabel').textContent = xLabel;
  document.getElementById('tt-ylabel').textContent = yLabel;
  var sorted = points.slice().sort(function (a, b) {{ return a[0] - b[0]; }});

  function nearest(x) {{
    var lo = 0, hi = sorted.length - 1;
    while (lo < hi) {{
      var mid = (lo + hi) >> 1;
      if (sorted[mid][0] < x) lo = mid + 1; else hi = mid;
    }}
    if (lo > 0 && Math.abs(sorted[lo - 1][0] - x) < Math.abs(sorted[lo][0] - x)) lo -= 1;
    return sorted[lo];
  }}

  function handleMove(evt) {{
    if (!sorted.length) return;
    var rect = svg.getBoundingClientRect();
    var scaleX = W / rect.width;
    var px = (evt.clientX - rect.left) * scaleX;
    var x = xLo + ((px - M.left) / plotW) * (xHi - xLo);
    var p = nearest(x);
    var cx = xScale(p[0]), cy = yScale(p[1]);
    crosshair.setAttribute('x1', cx); crosshair.setAttribute('x2', cx); crosshair.setAttribute('opacity', 1);
    hoverDot.setAttribute('cx', cx); hoverDot.setAttribute('cy', cy); hoverDot.setAttribute('opacity', 1);
    var wrapRect = svg.parentElement.getBoundingClientRect();
    var svgRect = svg.getBoundingClientRect();
    tooltip.style.left = ((svgRect.left - wrapRect.left) + cx / scaleX) + 'px';
    tooltip.style.top = ((svgRect.top - wrapRect.top) + cy / scaleX) + 'px';
    tooltip.style.opacity = 1;
    ttX.textContent = (Math.round(p[0] * 1000) / 1000);
    ttY.textContent = (Math.round(p[1] * 1000) / 1000);
  }}
  function hide() {{ crosshair.setAttribute('opacity', 0); hoverDot.setAttribute('opacity', 0); tooltip.style.opacity = 0; }}
  hit.addEventListener('pointermove', handleMove);
  hit.addEventListener('pointerleave', hide);

  var tbody = document.getElementById('data-table');
  var frag = document.createDocumentFragment();
  sorted.forEach(function (p) {{
    var tr = document.createElement('tr');
    [p[0], p[1]].forEach(function (v) {{
      var td = document.createElement('td');
      td.textContent = v;
      tr.appendChild(td);
    }});
    frag.appendChild(tr);
  }});
  tbody.appendChild(frag);
}})();
"""
    html = _page(title, subtitle, callout_html, card_title or f"{y_label} vs. {x_label}",
                 card_sub, chart_html, stats, table_html, script)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_bar_html(
    categories: list[str],
    values: list[float],
    *,
    out_path: str | Path,
    title: str,
    subtitle: str = "",
    y_label: str = "count",
    callout_html: str | None = None,
    stats: dict[str, str] | None = None,
    card_title: str = "",
    card_sub: str = "",
) -> Path:
    """Render a categorical bar chart (per-bar hover tooltip, data table) to out_path.

    categories/values: parallel lists, one bar per pair, in the order given (not
    resorted -- sort/bin before calling this if order matters, e.g. histogram edges
    low to high). Good for anomaly-flag rates, acd_division_type counts, per-bucket
    confidence histograms.
    """
    out_path = Path(out_path)
    assert len(categories) == len(values), "categories and values must be parallel lists"

    chart_html = """
    <div style="position: relative;">
      <svg id="chart" viewBox="0 0 820 380" width="100%" height="380"></svg>
      <div class="tooltip" id="tooltip">
        <span class="lbl" id="tt-cat"></span><br>
        <span class="val" id="tt-val"></span> <span class="lbl" id="tt-ylabel"></span>
      </div>
    </div>"""
    table_html = _table_html(["category", y_label], [], "data-table")

    script = f"""
(function () {{
  var categories = {json.dumps(categories)};
  var values = {json.dumps(values)};
  var yLabel = {json.dumps(y_label)};

  var svg = document.getElementById('chart');
  var W = 820, H = 380;
  var M = {{ top: 16, right: 20, bottom: 54, left: 54 }};
  var plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  var yMax = Math.max.apply(null, values.concat([0])) * 1.08 || 1;
  var n = values.length;
  var slot = plotW / Math.max(n, 1);
  var barW = Math.min(24, slot * 0.6);

  function yScale(v) {{ return M.top + plotH - (v / yMax) * plotH; }}

  var ns = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {{
    var e = document.createElementNS(ns, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }}

  [0, 0.25, 0.5, 0.75, 1.0].forEach(function (f) {{
    var v = yMax * f;
    var y = yScale(v);
    svg.appendChild(el('line', {{ class: 'gridline', x1: M.left, x2: W - M.right, y1: y, y2: y }}));
    var t = el('text', {{ class: 'tick-text', x: M.left - 8, y: y + 3, 'text-anchor': 'end' }});
    t.textContent = Math.round(v * 100) / 100;
    svg.appendChild(t);
  }});
  svg.appendChild(el('line', {{ class: 'baseline', x1: M.left, x2: W - M.right, y1: M.top + plotH, y2: M.top + plotH }}));
  var yl = el('text', {{ class: 'axis-label', x: -(M.top + plotH / 2), y: 14, 'text-anchor': 'middle', transform: 'rotate(-90)' }});
  yl.textContent = yLabel;
  svg.appendChild(yl);

  var tooltip = document.getElementById('tooltip');
  var ttCat = document.getElementById('tt-cat'), ttVal = document.getElementById('tt-val');
  document.getElementById('tt-ylabel').textContent = yLabel;

  categories.forEach(function (cat, i) {{
    var v = values[i];
    var cx = M.left + slot * (i + 0.5);
    var barX = cx - barW / 2;
    var barY = yScale(v);
    var barH = (M.top + plotH) - barY;
    var bar = el('rect', {{
      class: 'bar', x: barX, y: barY, width: barW, height: Math.max(barH, 0), rx: 3,
    }});
    var hitW = Math.max(barW, 24);
    var hit = el('rect', {{
      class: 'hit-layer', x: cx - hitW / 2, y: M.top, width: hitW, height: plotH,
    }});
    hit.addEventListener('pointerenter', function () {{ bar.classList.add('hovered'); }});
    hit.addEventListener('pointerleave', function () {{ bar.classList.remove('hovered'); tooltip.style.opacity = 0; }});
    hit.addEventListener('pointermove', function (evt) {{
      var rect = svg.getBoundingClientRect();
      var scaleX = W / rect.width;
      var wrapRect = svg.parentElement.getBoundingClientRect();
      var svgRect = svg.getBoundingClientRect();
      tooltip.style.left = ((svgRect.left - wrapRect.left) + cx / scaleX) + 'px';
      tooltip.style.top = ((svgRect.top - wrapRect.top) + barY / scaleX) + 'px';
      tooltip.style.opacity = 1;
      ttCat.textContent = cat;
      ttVal.textContent = v;
    }});
    svg.appendChild(bar);
    svg.appendChild(hit);

    var lbl = el('text', {{ class: 'tick-text', x: cx, y: H - M.bottom + 18, 'text-anchor': 'middle' }});
    lbl.textContent = cat;
    svg.appendChild(lbl);
  }});

  var tbody = document.getElementById('data-table');
  var frag = document.createDocumentFragment();
  categories.forEach(function (cat, i) {{
    var tr = document.createElement('tr');
    var td1 = document.createElement('td'); td1.textContent = cat;
    var td2 = document.createElement('td'); td2.textContent = values[i];
    tr.appendChild(td1); tr.appendChild(td2);
    frag.appendChild(tr);
  }});
  tbody.appendChild(frag);
}})();
"""
    html = _page(title, subtitle, callout_html, card_title or title, card_sub,
                 chart_html, stats, table_html, script)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def histogram_bins(values: list[float], n_bins: int = 20) -> tuple[list[str], list[int]]:
    """Bin raw values into (label, count) pairs, ready for render_bar_html.

    Convenience helper -- most QC histograms (confidence distribution, cell size,
    frame timing) start as a flat list of numbers, not pre-binned data.
    """
    if not values:
        return [], []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [f"{lo:.3g}"], [len(values)]
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / width), n_bins - 1)
        counts[idx] += 1
    labels = [f"{lo + i * width:.2g}" for i in range(n_bins)]
    return labels, counts
