"""Blind human spot-check of review_crops against the pipeline's own verdicts.

Why this exists: events.csv and verdict.txt tell you what the model concluded, but
not whether the model was right. This builds a self-contained HTML page that samples
events across the risk categories found during the 2026-07-08 verdict.txt/events.csv
audit (no-verdict fail-opens, GPT-floor downgrades, anomaly-flagged, high-confidence,
near-edge, obvious false-positives), shows you only the raw crop image sequence for
each one -- no pipeline verdict visible -- lets you call it, then reveals what the
pipeline said and tallies agreement live as you go. No separate scoring step: the
page already has both answers once you've made your call.

Sampling is stratified, not random-uniform, because a uniform sample would mostly
just re-confirm "the model is usually right on easy cases" -- the buckets below are
specifically the ones flagged as most likely to hide real disagreement.

Usage:
  python scripts/reports/spot_check_review.py data/output/<run_dir>
  python scripts/reports/spot_check_review.py data/output/<run_dir> --n-per-bucket 8 --seed 1

Note: pipeline verdict/confidence/notes for every sampled event ARE present in the
page's embedded JSON (so the live scoring/reveal can work client-side, no server) --
"blind" here means the UI doesn't render them until after you submit a call, not that
they're cryptographically hidden. Fine for a self spot-check; don't reuse this for a
study where a participant might peek at dev tools.
"""

import argparse
import csv
import json
import random
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Must match src/review.py's _CROP_RADIUS -- the crop is [cy-192:cy+192, cx-192:cx+192]
# clamped to the frame, so the centroid sits at pixel (192, 192) in the saved PNG
# UNLESS the crop was clipped by a frame edge (near_edge events), in which case the
# centroid shifts toward whichever edge got clamped. _centroid_in_crop below computes
# the real position either way instead of assuming dead-center.
_CROP_RADIUS = 192

# Must match src/review.py's _FRAME_STRIDE. review_crops/ holds every consecutive frame
# (src/review.py's _build_dense_debug_window, added 2026-07-10) but the AI only ever saw
# every _FRAME_STRIDE-th one -- this tells us which saved frames were actually reviewed
# vs. extra context saved purely for this tool's "show every frame" toggle.
_FRAME_STRIDE = 3
_CROP_NAME_RE = re.compile(r"^\d+_(?:before|split|after)_(\d+)\.png$")

_FLAG_COLS = ["misaligned_chromosomes", "lagging_chromosome", "anaphase_bridge", "micronucleus", "binucleation"]

# Priority order: an event that matches an earlier bucket never falls into a later
# one, so sampling stays balanced instead of everything piling into "confirmed_high".
_BUCKETS = [
    ("no_verdict", "review_crops folder has no verdict.txt (API call failed open)"),
    ("gpt_floor_downgrade", "verdict.txt says real, but events.csv confidence is 0.0 (min_gpt_confidence floor)"),
    ("anomaly_flagged", "confirmed split with an abnormality flag set (micronucleus/lagging/etc.)"),
    ("near_edge", "centroid near the frame boundary"),
    ("confirmed_high", "confidence >= 0.85, auto-confirmed"),
    ("false_positive", "confidence 0.0, model called it false_positive"),
]


def _png_size(path: Path) -> tuple[int, int]:
    """Width/height from a PNG's IHDR chunk -- no imaging library needed."""
    with open(path, "rb") as f:
        header = f.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _centroid_in_crop_pct(img_path: Path, cx: float, cy: float) -> tuple[float, float]:
    """Where the tracked centroid sits within this crop, as a 0-100% (left, top) pair.

    src/review.py crops [cy-R:cy+R, cx-R:cx+R], clamped to the frame at 0 on the low
    side (`max(0, cx - R)`). So the centroid's distance from the crop's own left edge
    is exactly `cx - max(0, cx - R)` = `min(cx, R)` -- R (dead center) when the left
    side wasn't clamped, or less than R (shifted toward that edge) when it was. This
    holds regardless of right-side clamping, which only affects the crop's width, not
    where its left edge starts. Same logic for y. Dividing by the crop's *actual*
    saved width/height (read from the PNG itself, since we don't have the source
    frame's dimensions) turns that pixel offset into a percentage CSS can position
    against, correct for interior, edge-clamped, and corner-clamped crops alike.
    """
    w, h = _png_size(img_path)
    offset_x = min(cx, _CROP_RADIUS)
    offset_y = min(cy, _CROP_RADIUS)
    return (offset_x / w) * 100, (offset_y / h) * 100


def _parse_verdict(path: Path) -> dict | None:
    if not path.exists():
        return None
    d: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    return d


def _bucket_for(row: dict, verdict: dict | None) -> str:
    confidence = float(row["ai_confidence"]) if row["ai_confidence"] else 0.0
    has_anomaly = any(row.get(c) == "1" for c in _FLAG_COLS)
    if verdict is None:
        return "no_verdict"
    if verdict.get("verdict") == "real" and confidence == 0.0:
        return "gpt_floor_downgrade"
    if confidence >= 0.5 and has_anomaly:
        return "anomaly_flagged"
    if row.get("near_edge") == "1":
        return "near_edge"
    if confidence >= 0.85:
        return "confirmed_high"
    return "false_positive"


def _effective_verdict(verdict: dict | None, csv_confidence: float) -> str:
    """What events.csv actually shipped for this event -- NOT verdict.txt's raw
    pre-floor call. These differ exactly on the gpt_floor_downgrade bucket by
    construction (that's its whole definition: raw verdict real, CSV confidence 0),
    and scoring against the raw call there silently inverts agree/disagree for that
    bucket -- this is the exact bug found and fixed 2026-07-08. Fail-open (no
    verdict.txt) always ends up "real" in the CSV regardless of confidence value --
    see src/review.py's exception handler, which hardcodes verdict="real" and never
    zeroes confidence.
    """
    if verdict is None:
        return "real"
    return "real" if csv_confidence > 0 else "false_positive"


def _build_manifest(run_dir: Path, n_per_bucket: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(run_dir / "events.csv")))
    by_parent: dict[str, dict] = {}
    for r in rows:
        by_parent.setdefault(r["parent_id"], r)  # first row per event; daughter rows are identical for our purposes

    crops_dir = run_dir / "review_crops"
    folder_re = re.compile(r"^frame_(\d+)_parent_(\d+)$")
    folder_by_parent: dict[str, Path] = {}
    for d in crops_dir.iterdir() if crops_dir.exists() else []:
        m = folder_re.match(d.name)
        if m:
            folder_by_parent[m.group(2)] = d

    buckets: dict[str, list[dict]] = {name: [] for name, _ in _BUCKETS}
    for parent_id, row in by_parent.items():
        folder = folder_by_parent.get(parent_id)
        if folder is None:
            continue  # no crops at all for this event -- nothing to show, skip
        verdict = _parse_verdict(folder / "verdict.txt")
        bucket = _bucket_for(row, verdict)
        all_names = sorted(p.name for p in folder.glob("*.png"))
        if not all_names:
            continue
        peak_frame = int(row["peak_frame"])

        def _idx_of(name: str) -> int | None:
            m = _CROP_NAME_RE.match(name)
            return int(m.group(1)) if m else None

        sampled_names = [n for n in all_names if (i := _idx_of(n)) is not None and (i - peak_frame) % _FRAME_STRIDE == 0]
        if not sampled_names:
            sampled_names = all_names  # older runs / unrecognized names: nothing to filter down to

        # Same centroid, same crop window for every frame in this event's sequence --
        # one representative image's dimensions are enough to place the crosshair.
        cx, cy = float(row["centroid_x"]), float(row["centroid_y"])
        crosshair_x_pct, crosshair_y_pct = _centroid_in_crop_pct(folder / all_names[0], cx, cy)
        csv_confidence = float(row["ai_confidence"]) if row["ai_confidence"] else 0.0
        effective_verdict = _effective_verdict(verdict, csv_confidence)
        buckets[bucket].append({
            "parent_id": parent_id,
            "bucket": bucket,
            "frame": row["peak_frame"],
            "centroid_x": cx,
            "centroid_y": cy,
            "images": [f"../review_crops/{folder.name}/{name}" for name in sampled_names],
            "dense_images": [
                {"src": f"../review_crops/{folder.name}/{name}", "sampled": name in set(sampled_names)}
                for name in all_names
            ],
            "has_dense": len(all_names) > len(sampled_names),
            "crosshair_x_pct": round(crosshair_x_pct, 2),
            "crosshair_y_pct": round(crosshair_y_pct, 2),
            "pipeline_verdict": effective_verdict,
            "pipeline_confidence": csv_confidence,
            "raw_verdict": verdict.get("verdict") if verdict else None,
            "raw_verdict_confidence": float(verdict["confidence"]) if verdict and "confidence" in verdict else None,
            "pipeline_notes": (verdict.get("notes") if verdict else row.get("ai_notes")) or "",
        })

    rng = random.Random(seed)
    sample: list[dict] = []
    for name, _ in _BUCKETS:
        pool = buckets[name]
        rng.shuffle(pool)
        sample.extend(pool[:n_per_bucket])
    rng.shuffle(sample)
    return sample


_CSS = """
:root {
  --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #898781; --border: rgba(11,11,11,0.10); --series-1: #2a78d6;
  --good: #0ca30c; --good-wash: #eaf7ea; --warning: #b8860b; --warning-wash: #fbf3e0;
  --critical: #d03b3b; --critical-wash: #fbeceb;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --text-muted: #898781; --border: rgba(255,255,255,0.10); --series-1: #3987e5;
    --good: #0ca30c; --good-wash: #12251a; --warning: #fab219; --warning-wash: #2a2311;
    --critical: #e66767; --critical-wash: #2a1717;
  }
}
* { box-sizing: border-box; }
body { background: var(--page); color: var(--text-primary); font-family: system-ui,-apple-system,"Segoe UI",sans-serif; margin: 0; padding: 24px 20px 48px; }
.wrap { max-width: 920px; margin: 0 auto; }
h1 { font-size: 18px; font-weight: 600; margin: 0 0 2px; }
.subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 18px; }
.progress { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
.filmstrip { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.crop-wrap { position: relative; display: inline-block; line-height: 0; }
.filmstrip .crop-wrap { border-radius: 4px; overflow: hidden; border: 1px solid var(--border); cursor: zoom-in; }
.crop-thumb { display: block; width: 150px; height: 150px; background-repeat: no-repeat; }
/* dense ("every frame") mode: green border = AI actually reviewed this frame, dimmed
   gray = extra consecutive context saved only for this tool, never sent to the model */
.filmstrip .crop-wrap.dense-sampled { border-color: var(--good); border-width: 2px; }
.filmstrip .crop-wrap.dense-skipped { opacity: 0.55; }
.dense-toggle { margin: 0 0 12px; }
.dense-legend { font-size: 11.5px; color: var(--text-muted); margin: 0 0 10px; }
.dense-legend .sw { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px; vertical-align: middle; border-radius: 2px; }
.dense-legend .sw.reviewed { border: 2px solid var(--good); }
.dense-legend .sw.context { background: var(--border); opacity: 0.55; }
.lightbox .crop-wrap img { max-width: 92vw; max-height: 92vh; display: block; }
/* Thin reticle with a gap at the center -- marks the tracked centroid without a line
   drawn straight through it, since a solid crosshair was obscuring the small cell it's
   supposed to help locate. Doubled stroke (white halo behind red) for contrast on both
   bright and dark microscopy backgrounds. */
.crosshair { position: absolute; width: 30px; height: 30px; transform: translate(-50%, -50%); pointer-events: none; overflow: visible; }
.lightbox .crosshair { width: 46px; height: 46px; }
.crosshair .halo { stroke: rgba(255,255,255,0.85); stroke-width: 1.5; stroke-linecap: round; }
.crosshair .mark { stroke: #ff2d55; stroke-width: 0.6; stroke-linecap: round; }
.crosshair-note { font-size: 11.5px; color: var(--text-muted); margin: 0 0 14px; }
.crosshair-note .swatch { display: inline-block; width: 10px; height: 2px; background: #ff2d55; box-shadow: 0 0 0 1px rgba(255,255,255,0.9); vertical-align: middle; margin: 0 4px; }
.controls { display: flex; gap: 10px; margin-bottom: 14px; }
button { font: inherit; cursor: pointer; border-radius: 8px; padding: 10px 18px; font-size: 13.5px; font-weight: 600; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary); }
button:hover { filter: brightness(0.97); }
.btn-real { border-color: var(--good); color: var(--good); }
.btn-fp { border-color: var(--critical); color: var(--critical); }
.btn-unsure { border-color: var(--warning); color: var(--warning); }
.btn-primary { background: var(--series-1); color: white; border-color: var(--series-1); }
.hint { font-size: 11.5px; color: var(--text-muted); margin-bottom: 18px; }
.reveal { display: none; border-radius: 8px; padding: 14px 16px; margin-top: 6px; font-size: 13.5px; line-height: 1.55; }
.reveal.agree { display: block; background: var(--good-wash); border: 1px solid var(--good); }
.reveal.disagree { display: block; background: var(--critical-wash); border: 1px solid var(--critical); }
.reveal.unsure { display: block; background: var(--warning-wash); border: 1px solid var(--warning); }
.reveal .verdict-line { font-weight: 600; margin-bottom: 4px; }
.tally { margin-top: 24px; }
.tally table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
.tally th, .tally td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
.tally th:not(:first-child), .tally td:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
.lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); align-items: center; justify-content: center; z-index: 10; }
.lightbox.open { display: flex; }
.lightbox-caption { position: absolute; top: 18px; left: 50%; transform: translateX(-50%); color: white; font-size: 13px; background: rgba(0,0,0,0.5); padding: 4px 12px; border-radius: 6px; }
.lightbox-close { position: absolute; top: 14px; right: 20px; color: white; font-size: 26px; line-height: 1; cursor: pointer; background: none; border: none; padding: 6px 10px; }
.lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.12); color: white; border: none; font-size: 28px; line-height: 1; width: 52px; height: 64px; cursor: pointer; border-radius: 8px; }
.lightbox-nav:hover { background: rgba(255,255,255,0.24); }
.lightbox-nav:disabled { opacity: 0.25; cursor: default; }
.lightbox-nav.prev { left: 16px; }
.lightbox-nav.next { right: 16px; }
.done { text-align: center; padding: 40px 0; }
.done .big { font-size: 40px; font-weight: 700; }
textarea { width: 100%; font: inherit; border-radius: 6px; border: 1px solid var(--border); background: var(--page); color: var(--text-primary); padding: 8px; margin-bottom: 14px; resize: vertical; }
"""


def _render_html(manifest: list[dict], title: str, subtitle: str, storage_key: str, thumb_zoom: int) -> str:
    return f"""<title>{title}</title>
<style>{_CSS}</style>
<div class="wrap">
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
  <div id="app"></div>
  <div id="tally" class="tally"></div>
</div>
<div class="lightbox" id="lightbox">
  <button class="lightbox-close" id="lightbox-close" title="Close (Esc)">&times;</button>
  <button class="lightbox-nav prev" id="lightbox-prev" title="Previous (&larr;)">&lsaquo;</button>
  <div class="crop-wrap"><img id="lightbox-img"><span id="lightbox-crosshair-slot"></span></div>
  <button class="lightbox-nav next" id="lightbox-next" title="Next (&rarr;)">&rsaquo;</button>
  <div class="lightbox-caption" id="lightbox-caption"></div>
</div>
<script>
(function () {{
  var manifest = {json.dumps(manifest)};
  var storageKey = {json.dumps(storage_key)};
  var thumbZoom = {json.dumps(thumb_zoom)};
  var app = document.getElementById('app');
  var tallyEl = document.getElementById('tally');
  var lightbox = document.getElementById('lightbox');
  var lightboxImg = document.getElementById('lightbox-img');
  var lightboxCrosshairSlot = document.getElementById('lightbox-crosshair-slot');
  var lightboxCaption = document.getElementById('lightbox-caption');

  function crosshairHtml(xPct, yPct) {{
    // Short ticks pulled out to the very edge of a 24-unit box (center at 12,12) --
    // leaves a wide, genuinely empty gap around the actual target pixel, rather than
    // ticks that reach most of the way to center.
    var ticks = '<line x1="12" y1="0" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="24"/>' +
      '<line x1="0" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="24" y2="12"/>';
    return '<svg class="crosshair" viewBox="0 0 24 24" style="left:' + xPct + '%;top:' + yPct + '%;">' +
      '<g class="halo">' + ticks + '</g>' +
      '<g class="mark">' + ticks + '</g></svg>';
  }}
  var lightboxPrevBtn = document.getElementById('lightbox-prev');
  var lightboxNextBtn = document.getElementById('lightbox-next');
  var lightboxCloseBtn = document.getElementById('lightbox-close');
  var lightboxImages = [];   // current event's image list, while lightbox is open
  var lightboxIdx = 0;

  var saved = null;
  try {{ saved = JSON.parse(localStorage.getItem(storageKey) || 'null'); }} catch (e) {{}}
  var results = (saved && saved.results) || {{}};
  var idx = (saved && saved.idx) || 0;
  var judged = false;
  var denseMode = false;  // "show every frame" toggle -- resets to off on navigation

  function currentFrames(event) {{
    if (denseMode && event.has_dense) return event.dense_images;
    return event.images.map(function (src) {{ return {{ src: src, sampled: true }}; }});
  }}

  function save() {{
    localStorage.setItem(storageKey, JSON.stringify({{ results: results, idx: idx }}));
  }}

  function openLightbox(frames, startIdx) {{
    lightboxImages = frames;  // array of {{src, sampled}}
    lightboxIdx = startIdx;
    showLightboxImage();
    lightbox.classList.add('open');
  }}
  function showLightboxImage() {{
    var event = manifest[idx];
    var frame = lightboxImages[lightboxIdx];
    lightboxImg.src = frame.src;
    lightboxCrosshairSlot.outerHTML = crosshairHtml(event.crosshair_x_pct, event.crosshair_y_pct).replace('<svg ', '<svg id="lightbox-crosshair-slot" ');
    lightboxCrosshairSlot = document.getElementById('lightbox-crosshair-slot');
    var name = frame.src.split('/').pop();
    var tag = frame.sampled ? '' : ' \\u2014 extra context, not seen by the model';
    lightboxCaption.textContent = (lightboxIdx + 1) + ' / ' + lightboxImages.length + ' \\u2014 ' + name + tag;
    lightboxPrevBtn.disabled = lightboxIdx === 0;
    lightboxNextBtn.disabled = lightboxIdx === lightboxImages.length - 1;
  }}
  function lightboxPrev() {{ if (lightboxIdx > 0) {{ lightboxIdx--; showLightboxImage(); }} }}
  function lightboxNext() {{ if (lightboxIdx < lightboxImages.length - 1) {{ lightboxIdx++; showLightboxImage(); }} }}
  function closeLightbox() {{ lightbox.classList.remove('open'); }}

  lightboxPrevBtn.addEventListener('click', function (e) {{ e.stopPropagation(); lightboxPrev(); }});
  lightboxNextBtn.addEventListener('click', function (e) {{ e.stopPropagation(); lightboxNext(); }});
  lightboxCloseBtn.addEventListener('click', function (e) {{ e.stopPropagation(); closeLightbox(); }});
  lightbox.addEventListener('click', closeLightbox);
  lightboxImg.addEventListener('click', function (e) {{ e.stopPropagation(); }});

  function agreement(human, event) {{
    if (human === 'unsure') return 'unsure';
    var pipelineReal = event.pipeline_verdict === 'real';
    var humanReal = human === 'real';
    return pipelineReal === humanReal ? 'agree' : 'disagree';
  }}

  function renderTally() {{
    var byBucket = {{}};
    Object.keys(results).forEach(function (pid) {{
      var r = results[pid];
      byBucket[r.bucket] = byBucket[r.bucket] || {{ agree: 0, disagree: 0, unsure: 0 }};
      byBucket[r.bucket][r.outcome]++;
    }});
    var buckets = Object.keys(byBucket).sort();
    var totalAgree = 0, totalDisagree = 0, totalUnsure = 0;
    var rows = buckets.map(function (b) {{
      var c = byBucket[b];
      var n = c.agree + c.disagree + c.unsure;
      totalAgree += c.agree; totalDisagree += c.disagree; totalUnsure += c.unsure;
      var rate = (c.agree + c.disagree) ? Math.round(100 * c.agree / (c.agree + c.disagree)) + '%' : 'n/a';
      return '<tr><td>' + b + '</td><td>' + n + '</td><td>' + c.agree + '</td><td>' + c.disagree + '</td><td>' + c.unsure + '</td><td>' + rate + '</td></tr>';
    }}).join('');
    var totalN = totalAgree + totalDisagree + totalUnsure;
    var overallRate = (totalAgree + totalDisagree) ? Math.round(100 * totalAgree / (totalAgree + totalDisagree)) + '%' : 'n/a';
    if (!totalN) {{ tallyEl.innerHTML = ''; return; }}
    tallyEl.innerHTML = '<h2 style="font-size:14px;margin:0 0 8px;">Live scoring (' + totalN + ' reviewed, ' + overallRate + ' agreement overall)</h2>' +
      '<table><thead><tr><th>bucket</th><th>n</th><th>agree</th><th>disagree</th><th>unsure</th><th>rate</th></tr></thead><tbody>' + rows +
      '<tr><td><strong>total</strong></td><td>' + totalN + '</td><td>' + totalAgree + '</td><td>' + totalDisagree + '</td><td>' + totalUnsure + '</td><td>' + overallRate + '</td></tr>' +
      '</tbody></table>' +
      '<div style="margin-top:12px;"><button id="export-btn">Export results (CSV)</button> <button id="reset-btn">Reset progress</button></div>';
    document.getElementById('export-btn').addEventListener('click', exportCsv);
    document.getElementById('reset-btn').addEventListener('click', function () {{
      if (confirm('Clear all recorded judgments?')) {{ localStorage.removeItem(storageKey); location.reload(); }}
    }});
  }}

  function exportCsv() {{
    var lines = ['parent_id,frame,bucket,human_verdict,pipeline_verdict,pipeline_confidence,outcome,note'];
    Object.keys(results).forEach(function (pid) {{
      var r = results[pid];
      var note = (r.note || '').replace(/"/g, '""');
      lines.push([pid, r.frame, r.bucket, r.human, r.pipeline_verdict, r.pipeline_confidence, r.outcome, '"' + note + '"'].join(','));
    }});
    var blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'spot_check_results.csv';
    a.click();
  }}

  function renderDone() {{
    app.innerHTML = '<div class="card done"><div class="big">Done</div><p>' + manifest.length + ' events reviewed. See the live scoring table below, or export a CSV.</p>' +
      '<button id="prev-link">&larr; Back to last event</button></div>';
    document.getElementById('prev-link').addEventListener('click', prev);
    renderTally();
  }}

  function renderEvent() {{
    if (idx >= manifest.length) {{ renderDone(); return; }}
    var event = manifest[idx];
    judged = !!results[event.parent_id];

    var reticle = crosshairHtml(event.crosshair_x_pct, event.crosshair_y_pct);
    var frames = currentFrames(event);
    var showingDense = denseMode && event.has_dense;
    var imgs = frames.map(function (frame, i) {{
      // Thumbnails are zoomed in around the tracked centroid (background-position/size,
      // not the img itself) -- the crop the pipeline saved is far larger than the cell,
      // so a 1:1 thumbnail mostly shows empty field. Click through to the lightbox for
      // the untouched, un-zoomed full crop.
      var cls = 'crop-wrap' + (showingDense ? (frame.sampled ? ' dense-sampled' : ' dense-skipped') : '');
      return '<span class="' + cls + '" data-idx="' + i + '">' +
        '<span class="crop-thumb" style="background-image:url(' + frame.src + ');background-position:' +
        event.crosshair_x_pct + '% ' + event.crosshair_y_pct + '%;background-size:' + thumbZoom + '%;"></span>' +
        reticle +
        '</span>';
    }}).join('');

    var denseToggleHtml = event.has_dense ?
      '<div class="dense-toggle"><button id="dense-toggle-btn">' +
        (showingDense ? 'Show only AI-reviewed frames' : 'Show every frame (' + event.dense_images.length + ')') +
      '</button></div>' +
      (showingDense ? '<p class="dense-legend"><span class="sw reviewed"></span>AI reviewed this frame' +
        '<span class="sw context"></span>extra context, not seen by the model</p>' : '')
      : '';

    var revealHtml = '';
    if (judged) {{
      var r = results[event.parent_id];
      var rawLine = (event.raw_verdict !== null && event.raw_verdict !== event.pipeline_verdict)
        ? '<div style="margin-top:2px;">(raw model call before the confidence floor: ' + event.raw_verdict +
          (event.raw_verdict_confidence !== null ? ' at ' + event.raw_verdict_confidence.toFixed(2) : '') + ')</div>'
        : '';
      revealHtml = '<div class="reveal ' + r.outcome + '">' +
        '<div class="verdict-line">Pipeline shipped: ' + event.pipeline_verdict + ' (confidence ' + event.pipeline_confidence.toFixed(2) + ') &mdash; ' +
        (r.outcome === 'agree' ? 'you agreed' : r.outcome === 'disagree' ? 'you disagreed' : 'you were unsure') + '</div>' +
        rawLine +
        '<div>bucket: ' + event.bucket + '</div>' +
        (event.pipeline_notes ? '<div style="margin-top:6px;color:var(--text-secondary);">"' + event.pipeline_notes + '"</div>' : '') +
        '<div class="controls" style="margin-top:12px;"><button id="rejudge-btn">Re-judge this event</button></div>' +
        '</div>';
    }}

    var navHtml = '<p class="progress">Event ' + (idx + 1) + ' / ' + manifest.length + ' &middot; frame ' + event.frame +
      (idx > 0 ? ' &middot; <a href="#" id="prev-link">&larr; Previous</a>' : '') + '</p>';

    app.innerHTML =
      navHtml +
      '<div class="card">' +
      denseToggleHtml +
      '<div class="filmstrip">' + imgs + '</div>' +
      '<p class="crosshair-note"><span class="swatch"></span>marks the tracked centroid &mdash; click any frame to step through the sequence with &larr;/&rarr;</p>' +
      (judged ? '' : '<div class="controls">' +
        '<button class="btn-real" data-v="real">Real division</button>' +
        '<button class="btn-fp" data-v="false_positive">False positive</button>' +
        '<button class="btn-unsure" data-v="unsure">Unsure</button>' +
        '</div>' +
        '<textarea id="note" rows="2" placeholder="optional note"></textarea>' +
        '<p class="hint">No pipeline verdict shown yet -- call it from the images only.</p>') +
      revealHtml +
      (judged ? '<div class="controls" style="margin-top:14px;"><button class="btn-primary" id="next-btn">Next (Enter)</button></div>' : '') +
      '</div>';

    app.querySelectorAll('.filmstrip .crop-wrap').forEach(function (wrap) {{
      wrap.addEventListener('click', function () {{
        openLightbox(frames, parseInt(wrap.getAttribute('data-idx'), 10));
      }});
    }});

    var denseToggleBtn = document.getElementById('dense-toggle-btn');
    if (denseToggleBtn) {{
      denseToggleBtn.addEventListener('click', function () {{
        denseMode = !denseMode;
        renderEvent();
      }});
    }}

    var prevLink = document.getElementById('prev-link');
    if (prevLink) {{
      prevLink.addEventListener('click', function (e) {{ e.preventDefault(); prev(); }});
    }}

    if (!judged) {{
      app.querySelectorAll('.controls button').forEach(function (btn) {{
        btn.addEventListener('click', function () {{ judge(btn.getAttribute('data-v')); }});
      }});
    }} else {{
      document.getElementById('next-btn').addEventListener('click', next);
      document.getElementById('rejudge-btn').addEventListener('click', function () {{
        delete results[event.parent_id];
        judged = false;
        save();
        renderEvent();
      }});
    }}

    renderTally();
  }}

  function judge(human) {{
    var event = manifest[idx];
    var note = (document.getElementById('note') || {{}}).value || '';
    var outcome = agreement(human, event);
    results[event.parent_id] = {{
      human: human, note: note, outcome: outcome, bucket: event.bucket, frame: event.frame,
      pipeline_verdict: event.pipeline_verdict, pipeline_confidence: event.pipeline_confidence,
    }};
    judged = true;
    save();
    renderEvent();
  }}

  function next() {{
    idx++;
    judged = false;
    denseMode = false;
    save();
    renderEvent();
  }}

  function prev() {{
    if (idx === 0) return;
    idx--;
    judged = !!results[manifest[idx].parent_id];
    denseMode = false;
    save();
    renderEvent();
  }}

  document.addEventListener('keydown', function (e) {{
    if (lightbox.classList.contains('open')) {{
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'ArrowLeft') lightboxPrev();
      if (e.key === 'ArrowRight') lightboxNext();
      return;
    }}
    if (!judged) {{
      // R/F/S judgment hotkeys removed -- they fired while typing a note (e.g. typing
      // "r" in a sentence), causing accidental mis-judgments. Buttons only now.
    }} else {{
      if (e.key === 'Enter' && document.activeElement.tagName !== 'TEXTAREA') next();
    }}
  }});

  renderEvent();
}})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv and review_crops/")
    parser.add_argument("--n-per-bucket", type=int, default=5, help="max events sampled per risk bucket (default 5, ~30 total across 6 buckets)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed, change to get a different sample")
    parser.add_argument("--thumb-zoom", type=int, default=280, help="thumbnail zoom %% around the tracked centroid (default 280; the saved crop is far larger than a typical cell, so thumbnails zoom in -- click through to the lightbox for the untouched full crop)")
    parser.add_argument("--out", default=None, help="output .html path (default: <run_dir>/reports/spot_check.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = _build_manifest(run_dir, args.n_per_bucket, args.seed)
    if not manifest:
        raise SystemExit("No events with crop folders found -- check --run-dir points at a run with review_crops/")

    bucket_counts = {}
    for e in manifest:
        bucket_counts[e["bucket"]] = bucket_counts.get(e["bucket"], 0) + 1
    subtitle = f"{run_dir.name} · {len(manifest)} events sampled: " + ", ".join(f"{k}={v}" for k, v in bucket_counts.items())

    out_path = Path(args.out) if args.out else run_dir / "reports" / "spot_check.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    storage_key = f"spotcheck_{run_dir.name}_{args.seed}_{args.n_per_bucket}"
    html = _render_html(manifest, "Spot-check review", subtitle, storage_key, args.thumb_zoom)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(manifest)} events across {len(bucket_counts)} buckets)")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
