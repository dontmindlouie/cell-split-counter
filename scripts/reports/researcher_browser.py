"""Open-book filmstrip browser for stem-cell researchers.

Shows all confirmed split events from a pipeline run, sorted by biological interest
(anomaly-flagged events first, then abnormal geometry, then normal confirmed), with
AI verdict and notes visible from the start -- unlike spot_check_review.py, which is
a blind QC tool. Primary audience is a researcher (or an AI assistant helping a
researcher) exploring what happened in a video.

Annotations (free-text researcher notes + flag-for-followup) are stored in browser
localStorage and survive page reloads. Use the Export button to download
researcher_notes.csv -- a clean machine-readable patch file an AI assistant can use
to write annotations back into events.csv.

Usage:
    python scripts/reports/researcher_browser.py data/output/<run_dir>
    python scripts/reports/researcher_browser.py data/output/<run_dir> --min-conf 0.5
    python scripts/reports/researcher_browser.py data/output/<run_dir> --include-fps

Handles both old (claude_confidence/claude_notes) and new (ai_confidence/ai_notes)
events.csv column names.
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_CROP_RADIUS = 192  # must match src/review.py's _CROP_RADIUS

_FLAG_COLS = [
    "misaligned_chromosomes",
    "lagging_chromosome",
    "anaphase_bridge",
    "micronucleus",
    "binucleation",
]

_FLAG_LABELS = {
    "misaligned_chromosomes": "misaligned chr",
    "lagging_chromosome":     "lagging chr",
    "anaphase_bridge":        "anaphase bridge",
    "micronucleus":           "micronucleus",
    "binucleation":           "binucleation",
}


def _conf_col(row: dict) -> str:
    """Handle both old (claude_confidence) and new (ai_confidence) column names."""
    return "ai_confidence" if "ai_confidence" in row else "claude_confidence"


def _notes_col(row: dict) -> str:
    return "ai_notes" if "ai_notes" in row else "claude_notes"


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _centroid_in_crop_pct(img_path: Path, cx: float, cy: float) -> tuple[float, float]:
    w, h = _png_size(img_path)
    offset_x = min(cx, _CROP_RADIUS)
    offset_y = min(cy, _CROP_RADIUS)
    return (offset_x / w) * 100, (offset_y / h) * 100


def _is_split_type_mismatch(row: dict) -> bool:
    """Tracker topology said normal_split (2 children) but the model visually saw 3+
    daughters -- see docs/output_schema.md's multi_way undercounting gotcha (2026-07-09)."""
    return row.get("split_type") == "multi_way" and row.get("split_topology") == "normal_split"


def _interest_score(row: dict) -> tuple[int, float]:
    """Return (tier_score, confidence) for sorting. Higher = more interesting."""
    conf_col = _conf_col(row)
    conf = float(row.get(conf_col) or 0)
    acd = (row.get("acd_division_type") or "").lower()
    near = row.get("near_edge") == "1"
    has_anomaly = any(row.get(f) == "1" for f in _FLAG_COLS) or bool((row.get("anomaly_notes") or "").strip())
    is_failed = row.get("split_topology") == "failed_split"
    mismatch = _is_split_type_mismatch(row)

    if conf <= 0:
        score = 5          # false positive / unconfirmed
    elif has_anomaly and conf >= 0.5:
        score = 40         # Tier 1: anomaly-flagged + confirmed
    elif is_failed or mismatch:
        score = 35         # Tier 1b: failed division, or tracker undercounted a multi-way split
                            # (both added 2026-07-09 -- biologically/correctness interesting on
                            # their own, independent of confidence tier or ACD geometry)
    elif acd in ("tripolar", "multipolar"):
        score = 30         # Tier 2: abnormal geometry
    elif conf >= 0.5:
        score = 20         # Tier 3: normal confirmed
    else:
        score = 10         # Tier 4: low confidence
    if near:
        score -= 3         # deprioritize near-edge within tier
    return score, conf


def _build_manifest(
    run_dir: Path,
    min_conf: float,
    include_fps: bool,
    thumb_zoom: int,
) -> list[dict]:
    rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))

    # Deduplicate to one row per unique split point
    by_split: dict[tuple, dict] = {}
    for r in rows:
        # failed_split included 2026-07-09 -- previously excluded entirely, meaning a real,
        # confirmed failed division was invisible in this tool despite being a distinct,
        # biologically interesting event type. Still excluded from the "confirmed splits"
        # count in main() below, since it's not a completed division.
        if r.get("split_topology") not in ("normal_split", "multi_way_split", "failed_split"):
            continue
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        if key not in by_split:
            by_split[key] = r

    crops_dir = run_dir / "review_crops"
    import re
    folder_re = re.compile(r"^frame_(\d+)_parent_(\d+)$")
    folder_by_parent: dict[str, Path] = {}
    if crops_dir.exists():
        for d in crops_dir.iterdir():
            m = folder_re.match(d.name)
            if m:
                folder_by_parent[m.group(2)] = d

    events = []
    for (parent_id, peak_frame), row in by_split.items():
        conf_col = _conf_col(row)
        notes_col = _notes_col(row)
        conf = float(row.get(conf_col) or 0)

        if conf < min_conf and not include_fps:
            continue

        folder = folder_by_parent.get(parent_id)
        images: list[str] = []
        crosshair_x_pct = 50.0
        crosshair_y_pct = 50.0

        if folder is not None:
            imgs = sorted(folder.glob("*.png"))
            if imgs:
                try:
                    crosshair_x_pct, crosshair_y_pct = _centroid_in_crop_pct(
                        imgs[0],
                        float(row.get("centroid_x") or _CROP_RADIUS),
                        float(row.get("centroid_y") or _CROP_RADIUS),
                    )
                except Exception:
                    pass
                images = [f"../review_crops/{folder.name}/{p.name}" for p in imgs]

        flags = [_FLAG_LABELS[f] for f in _FLAG_COLS if row.get(f) == "1"]
        acd = row.get("acd_division_type") or ""
        score, _ = _interest_score(row)

        events.append({
            "parent_id": parent_id,
            "peak_frame": peak_frame,
            "confidence": conf,
            "raw_ai_confidence": row.get("raw_ai_confidence") or None,
            "acd_division_type": acd,
            "flags": flags,
            "near_edge": row.get("near_edge") == "1",
            "bleach_risk": row.get("bleach_risk") or None,
            "classification_source": row.get("classification_source") or "",
            "ai_notes": row.get(notes_col) or "",
            "anomaly_notes": row.get("anomaly_notes") or "",
            "review_error": row.get("review_error") == "1",
            "split_topology": row.get("split_topology") or "",
            "split_type": row.get("split_type") or "",
            "is_failed_split": row.get("split_topology") == "failed_split",
            "split_type_mismatch": _is_split_type_mismatch(row),
            "images": images,
            "has_crops": len(images) > 0,
            "crosshair_x_pct": round(crosshair_x_pct, 2),
            "crosshair_y_pct": round(crosshair_y_pct, 2),
            "interest_score": score,
        })

    events.sort(key=lambda e: (-e["interest_score"], -e["confidence"]))
    return events


_CSS = """
:root {
  --surface-1:#fcfcfb;--page:#f4f4f2;--text-primary:#0b0b0b;--text-secondary:#52514e;
  --text-muted:#898781;--border:rgba(11,11,11,0.10);--series-1:#2a78d6;
  --good:#0ca30c;--good-wash:#eaf7ea;--warning:#b8860b;--warning-wash:#fbf3e0;
  --critical:#d03b3b;--critical-wash:#fbeceb;--tier1:#7c3aed;--tier1-wash:#f3eeff;
  --tier2:#0369a1;--tier2-wash:#e0f2fe;
}
@media (prefers-color-scheme:dark){
  :root{
    --surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#ffffff;--text-secondary:#c3c2b7;
    --text-muted:#898781;--border:rgba(255,255,255,0.10);
    --good:#0ca30c;--good-wash:#12251a;--warning:#fab219;--warning-wash:#2a2311;
    --critical:#e66767;--critical-wash:#2a1717;--tier1:#a78bfa;--tier1-wash:#1e1030;
    --tier2:#38bdf8;--tier2-wash:#0c1e2a;
  }
}
*{box-sizing:border-box;}
body{background:var(--page);color:var(--text-primary);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:0;}
.sidebar{position:fixed;top:0;left:0;width:220px;height:100vh;background:var(--surface-1);border-right:1px solid var(--border);overflow-y:auto;padding:16px 14px;z-index:5;}
.sidebar h2{font-size:13px;font-weight:600;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);}
.sidebar .run-name{font-size:12px;color:var(--text-secondary);margin:0 0 14px;word-break:break-all;}
.filter-group{margin-bottom:16px;}
.filter-group label{display:block;font-size:12.5px;margin-bottom:4px;}
.filter-group input[type=range]{width:100%;}
.filter-group input[type=checkbox]{margin-right:5px;}
.filter-group select{width:100%;font:inherit;font-size:12.5px;border-radius:5px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);padding:4px 6px;}
.export-btn{width:100%;font:inherit;font-size:12.5px;font-weight:600;cursor:pointer;border-radius:7px;padding:8px 12px;background:var(--series-1);color:white;border:none;margin-bottom:8px;}
.export-btn:hover{filter:brightness(1.1);}
.stats{font-size:11.5px;color:var(--text-muted);line-height:1.6;}
.main{margin-left:220px;padding:20px 24px;}
.main-header{margin-bottom:16px;}
.main-header h1{font-size:18px;font-weight:600;margin:0 0 2px;}
.main-header .subtitle{font-size:13px;color:var(--text-secondary);margin:0;}
.grid{display:flex;flex-direction:column;gap:14px;}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px;}
.card.tier1{border-left:3px solid var(--tier1);}
.card.tier2{border-left:3px solid var(--tier2);}
.card-header{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap;}
.frame-label{font-size:13px;font-weight:600;}
.conf-badge{font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:4px;white-space:nowrap;}
.conf-high{background:var(--good-wash);color:var(--good);}
.conf-mid{background:var(--warning-wash);color:var(--warning);}
.conf-low{background:var(--critical-wash);color:var(--critical);}
.acd-badge{font-size:11.5px;padding:2px 8px;border-radius:4px;background:var(--tier2-wash);color:var(--tier2);white-space:nowrap;}
.flag-chip{display:inline-block;font-size:11px;padding:2px 7px;border-radius:4px;background:var(--tier1-wash);color:var(--tier1);margin-right:4px;margin-bottom:4px;white-space:nowrap;}
.near-edge-chip{background:var(--warning-wash);color:var(--warning);}
.error-chip{background:var(--critical-wash);color:var(--critical);}
.filmstrip{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.crop-wrap{position:relative;display:inline-block;line-height:0;border-radius:4px;overflow:hidden;border:1px solid var(--border);cursor:zoom-in;flex-shrink:0;}
.crop-thumb{display:block;width:120px;height:120px;background-repeat:no-repeat;background-color:var(--border);}
.crop-thumb.loading{background-image:none!important;}

.no-crops-note{font-size:12px;color:var(--text-muted);padding:8px 0;}
.ai-notes{font-size:13px;color:var(--text-secondary);line-height:1.55;margin:6px 0 10px;font-style:italic;}
.ai-notes:empty{display:none;}
.anomaly-notes{font-size:13px;color:var(--tier1);line-height:1.55;margin:0 0 10px;padding:6px 10px;border-radius:6px;background:var(--tier1-wash);}
.meta-row{font-size:11.5px;color:var(--text-muted);margin-bottom:10px;}
.annotation-area textarea{width:100%;font:inherit;font-size:13px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);padding:7px 9px;resize:vertical;min-height:60px;}
.annotation-area label{font-size:12.5px;display:flex;align-items:center;gap:6px;margin-top:6px;cursor:pointer;}
.annotation-area label input[type=checkbox]{width:15px;height:15px;cursor:pointer;}
.saved-note{font-size:11.5px;color:var(--text-muted);margin-top:4px;min-height:1em;}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.88);align-items:center;justify-content:center;z-index:20;}
.lightbox.open{display:flex;}
.lightbox-img-wrap{position:relative;line-height:0;}
.lightbox-img-wrap img{max-width:92vw;max-height:88vh;display:block;}
.lightbox-caption{position:absolute;bottom:-28px;left:50%;transform:translateX(-50%);color:white;font-size:12px;white-space:nowrap;}
.lightbox-close{position:absolute;top:14px;right:20px;color:white;font-size:26px;line-height:1;cursor:pointer;background:none;border:none;padding:6px 10px;z-index:21;}
.lightbox-nav{position:absolute;top:50%;transform:translateY(-50%);background:rgba(255,255,255,0.12);color:white;border:none;font-size:28px;line-height:1;width:52px;height:64px;cursor:pointer;border-radius:8px;z-index:21;}
.lightbox-nav:hover{background:rgba(255,255,255,0.24);}
.lightbox-nav:disabled{opacity:0.25;cursor:default;}
.lightbox-nav.prev{left:16px;}
.lightbox-nav.next{right:16px;}
.hidden{display:none!important;}
"""


def _render_html(
    manifest: list[dict],
    run_name: str,
    total_confirmed: int,
    thumb_zoom: int,
) -> str:
    storage_key = f"researcher_{run_name}"
    subtitle = f"{run_name} · {total_confirmed} confirmed splits · {len(manifest)} shown"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Researcher browser — {run_name}</title>
<style>{_CSS}</style>
</head>
<body>

<div class="sidebar" id="sidebar">
  <h2>Filters</h2>
  <div class="run-name" id="run-name-label">{run_name}</div>

  <div class="filter-group">
    <label>ACD type</label>
    <select id="filter-acd">
      <option value="">All types</option>
      <option value="bipolar">Bipolar</option>
      <option value="tripolar">Tripolar</option>
      <option value="multipolar">Multipolar</option>
      <option value="unknown">Unknown</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Anomaly flags</label>
    <label><input type="checkbox" class="flag-filter" data-flag="misaligned_chromosomes"> misaligned chr</label>
    <label><input type="checkbox" class="flag-filter" data-flag="lagging_chromosome"> lagging chr</label>
    <label><input type="checkbox" class="flag-filter" data-flag="anaphase_bridge"> anaphase bridge</label>
    <label><input type="checkbox" class="flag-filter" data-flag="micronucleus"> micronucleus</label>
    <label><input type="checkbox" class="flag-filter" data-flag="binucleation"> binucleation</label>
  </div>

  <div class="filter-group">
    <label>Min confidence: <span id="conf-val">0.0</span></label>
    <input type="range" id="filter-conf" min="0" max="1" step="0.05" value="0">
  </div>

  <div class="filter-group">
    <label><input type="checkbox" id="filter-annotated-only"> Annotated only</label>
    <label><input type="checkbox" id="filter-flagged-only"> Flagged for follow-up only</label>
    <label><input type="checkbox" id="filter-hide-near-edge"> Hide near-edge</label>
    <label><input type="checkbox" id="filter-hide-fps" checked> Hide false positives</label>
  </div>

  <button class="export-btn" id="export-btn">Export researcher_notes.csv</button>

  <div class="stats" id="stats"></div>
</div>

<div class="main">
  <div class="main-header">
    <h1>Researcher browser</h1>
    <p class="subtitle">{subtitle}</p>
  </div>
  <div class="grid" id="grid"></div>
</div>

<div class="lightbox" id="lightbox">
  <button class="lightbox-close" id="lightbox-close">&times;</button>
  <button class="lightbox-nav prev" id="lightbox-prev">&lsaquo;</button>
  <div class="lightbox-img-wrap">
    <img id="lightbox-img" alt="">
    <div class="lightbox-caption" id="lightbox-caption"></div>
  </div>
  <button class="lightbox-nav next" id="lightbox-next">&rsaquo;</button>
</div>

<script>
(function(){{
  var manifest = {json.dumps(manifest)};
  var storageKey = {json.dumps(storage_key)};
  var thumbZoom = {json.dumps(thumb_zoom)};

  // --- localStorage ---
  function loadAnnotations() {{
    try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch(e) {{ return {{}}; }}
  }}
  function saveAnnotations(a) {{
    try {{ localStorage.setItem(storageKey, JSON.stringify(a)); }} catch(e) {{}}
  }}
  var annotations = loadAnnotations();

  // --- Lightbox ---
  var lightbox = document.getElementById('lightbox');
  var lbImg = document.getElementById('lightbox-img');
  var lbCaption = document.getElementById('lightbox-caption');
  var lbPrev = document.getElementById('lightbox-prev');
  var lbNext = document.getElementById('lightbox-next');
  var lbClose = document.getElementById('lightbox-close');
  var lbImages = [], lbIdx = 0;

  function openLightbox(images, startIdx) {{
    lbImages = images; lbIdx = startIdx;
    showLbFrame();
    lightbox.classList.add('open');
  }}
  function showLbFrame() {{
    lbImg.src = lbImages[lbIdx];
    lbCaption.textContent = (lbIdx + 1) + ' / ' + lbImages.length + '  \u2014  ' + lbImages[lbIdx].split('/').pop();
    lbPrev.disabled = lbIdx === 0;
    lbNext.disabled = lbIdx === lbImages.length - 1;
  }}
  lbPrev.addEventListener('click', function(e) {{ e.stopPropagation(); if(lbIdx>0){{lbIdx--;showLbFrame();}} }});
  lbNext.addEventListener('click', function(e) {{ e.stopPropagation(); if(lbIdx<lbImages.length-1){{lbIdx++;showLbFrame();}} }});
  lbClose.addEventListener('click', function(e) {{ e.stopPropagation(); lightbox.classList.remove('open'); }});
  lightbox.addEventListener('click', function() {{ lightbox.classList.remove('open'); }});
  lbImg.addEventListener('click', function(e) {{ e.stopPropagation(); }});
  document.addEventListener('keydown', function(e) {{
    if (!lightbox.classList.contains('open')) return;
    if (e.key==='Escape') lightbox.classList.remove('open');
    if (e.key==='ArrowLeft' && lbIdx>0) {{ lbIdx--; showLbFrame(); }}
    if (e.key==='ArrowRight' && lbIdx<lbImages.length-1) {{ lbIdx++; showLbFrame(); }}
  }});

  // --- Filter state ---
  var filterAcd = '';
  var filterFlags = [];
  var filterMinConf = 0;
  var filterAnnotatedOnly = false;
  var filterFlaggedOnly = false;
  var filterHideNearEdge = false;
  var filterHideFps = true;

  function passesFilter(ev) {{
    if (filterHideFps && ev.confidence <= 0) return false;
    if (filterHideNearEdge && ev.near_edge) return false;
    if (filterMinConf > 0 && ev.confidence < filterMinConf) return false;
    if (filterAcd && ev.acd_division_type !== filterAcd) return false;
    if (filterFlags.length > 0) {{
      // event must have ALL checked flags in its flags array
      // flags array contains human labels; map back via FLAG_LABELS
      var flagLabels = {{
        'misaligned_chromosomes':'misaligned chr',
        'lagging_chromosome':'lagging chr',
        'anaphase_bridge':'anaphase bridge',
        'micronucleus':'micronucleus',
        'binucleation':'binucleation'
      }};
      for (var i=0;i<filterFlags.length;i++) {{
        if (ev.flags.indexOf(flagLabels[filterFlags[i]]) === -1) return false;
      }}
    }}
    if (filterAnnotatedOnly) {{
      var a = annotations[ev.parent_id];
      if (!a || !a.notes) return false;
    }}
    if (filterFlaggedOnly) {{
      var a2 = annotations[ev.parent_id];
      if (!a2 || !a2.followup) return false;
    }}
    return true;
  }}

  function confClass(c) {{
    if (c <= 0) return 'conf-low';
    if (c >= 0.75) return 'conf-high';
    if (c >= 0.4) return 'conf-mid';
    return 'conf-low';
  }}

  function tierClass(ev) {{
    if ((ev.flags.length > 0 || ev.anomaly_notes) && ev.confidence >= 0.5) return 'tier1';
    if (ev.is_failed_split || ev.split_type_mismatch) return 'tier1';
    if (ev.acd_division_type === 'tripolar' || ev.acd_division_type === 'multipolar') return 'tier2';
    return '';
  }}

  function renderCard(ev) {{
    var ann = annotations[ev.parent_id] || {{}};
    var tier = tierClass(ev);
    var confBadge = '<span class="conf-badge ' + confClass(ev.confidence) + '">' +
      (ev.confidence > 0 ? ev.confidence.toFixed(2) : 'FP') + '</span>';
    var acdBadge = ev.acd_division_type ? '<span class="acd-badge">' + ev.acd_division_type + '</span>' : '';
    var flagChips = ev.flags.map(function(f) {{ return '<span class="flag-chip">' + f + '</span>'; }}).join('');
    var nearChip = ev.near_edge ? '<span class="flag-chip near-edge-chip">near edge</span>' : '';
    var errChip = ev.review_error ? '<span class="flag-chip error-chip">review error</span>' : '';
    var failedChip = ev.is_failed_split ? '<span class="flag-chip near-edge-chip">failed division</span>' : '';
    var mismatchChip = ev.split_type_mismatch ? '<span class="flag-chip error-chip">split_type mismatch: model saw multi_way</span>' : '';
    var anomalyChip = ev.anomaly_notes ? '<span class="flag-chip">anomaly noted</span>' : '';

    var filmstrip = '';
    if (ev.images.length > 0) {{
      filmstrip = ev.images.map(function(src, i) {{
        var label = src.split('/').pop().replace(/^\d+_/, '').replace(/_\d+\.png$/, '');
        return '<span class="crop-wrap" data-idx="' + i + '" title="' + label + '">' +
          '<span class="crop-thumb loading" data-bg="' +
          'background-image:url(' + src + ');' +
          'background-position:' + ev.crosshair_x_pct + '% ' + ev.crosshair_y_pct + '%;' +
          'background-size:' + thumbZoom + '%;" style=""></span>' +
          '</span>';
      }}).join('');
    }} else {{
      filmstrip = '<p class="no-crops-note">No crop images available for this event.</p>';
    }}

    var rawConf = ev.raw_ai_confidence ? ' (raw: ' + parseFloat(ev.raw_ai_confidence).toFixed(2) + ')' : '';
    var bleach = ev.bleach_risk ? ' · bleach risk: ' + parseFloat(ev.bleach_risk).toFixed(2) : '';
    var meta = 'frame ' + ev.peak_frame + ' · parent ' + ev.parent_id +
      ' · ' + (ev.classification_source || 'rule') + rawConf + bleach;

    var savedNote = ann.notes ? '<span style="color:var(--text-secondary);font-style:italic;">Saved: ' +
      ann.notes.substring(0, 80) + (ann.notes.length > 80 ? '…' : '') + '</span>' : '';

    return '<div class="card ' + tier + '" data-parent="' + ev.parent_id + '">' +
      '<div class="card-header">' +
        '<span class="frame-label">Frame ' + ev.peak_frame + '</span>' +
        confBadge + acdBadge +
        '<span>' + flagChips + anomalyChip + failedChip + mismatchChip + nearChip + errChip + '</span>' +
      '</div>' +
      '<div class="filmstrip">' + filmstrip + '</div>' +
      (ev.ai_notes ? '<div class="ai-notes">&ldquo;' + ev.ai_notes + '&rdquo;</div>' : '') +
      (ev.anomaly_notes ? '<div class="anomaly-notes">&#9888; ' + ev.anomaly_notes + '</div>' : '') +
      '<div class="meta-row">' + meta + '</div>' +
      '<div class="annotation-area">' +
        '<textarea placeholder="Researcher notes…" data-parent="' + ev.parent_id + '">' +
          (ann.notes ? ann.notes.replace(/</g,'&lt;') : '') + '</textarea>' +
        '<label>' +
          '<input type="checkbox" class="followup-check" data-parent="' + ev.parent_id + '"' +
          (ann.followup ? ' checked' : '') + '> Flag for follow-up' +
        '</label>' +
        '<div class="saved-note" id="saved-' + ev.parent_id + '">' + savedNote + '</div>' +
      '</div>' +
    '</div>';
  }}

  function renderGrid() {{
    var visible = manifest.filter(passesFilter);
    var grid = document.getElementById('grid');
    grid.innerHTML = visible.map(renderCard).join('');

    // filmstrip click → lightbox
    grid.querySelectorAll('.crop-wrap[data-idx]').forEach(function(wrap) {{
      wrap.addEventListener('click', function() {{
        var card = wrap.closest('.card');
        var parentId = card.getAttribute('data-parent');
        var ev = manifest.find(function(e) {{ return e.parent_id === parentId; }});
        if (ev && ev.images.length) {{
          openLightbox(ev.images, parseInt(wrap.getAttribute('data-idx'), 10));
        }}
      }});
    }});

    // lazy-load filmstrip thumbnails: swap in background-image only when card scrolls
    // near the viewport -- prevents the browser from fetching all ~16k images at once.
    var thumbObserver = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (!entry.isIntersecting) return;
        var card = entry.target;
        card.querySelectorAll('.crop-thumb.loading').forEach(function(thumb) {{
          var bg = thumb.getAttribute('data-bg');
          if (bg) {{ thumb.style.cssText = bg; thumb.classList.remove('loading'); }}
        }});
        thumbObserver.unobserve(card);
      }});
    }}, {{ rootMargin: '300px' }});
    grid.querySelectorAll('.card').forEach(function(card) {{
      thumbObserver.observe(card);
    }});

    // annotation textarea → auto-save on change
    grid.querySelectorAll('textarea[data-parent]').forEach(function(ta) {{
      var pid = ta.getAttribute('data-parent');
      var timer = null;
      ta.addEventListener('input', function() {{
        clearTimeout(timer);
        timer = setTimeout(function() {{
          if (!annotations[pid]) annotations[pid] = {{}};
          annotations[pid].notes = ta.value;
          saveAnnotations(annotations);
          var el = document.getElementById('saved-' + pid);
          if (el) el.textContent = ta.value ? 'Saved.' : '';
        }}, 600);
      }});
    }});

    // follow-up checkboxes
    grid.querySelectorAll('.followup-check').forEach(function(cb) {{
      cb.addEventListener('change', function() {{
        var pid = cb.getAttribute('data-parent');
        if (!annotations[pid]) annotations[pid] = {{}};
        annotations[pid].followup = cb.checked;
        saveAnnotations(annotations);
      }});
    }});

    updateStats(visible.length);
  }}

  function updateStats(visibleCount) {{
    var annotated = Object.keys(annotations).filter(function(k) {{ return annotations[k] && annotations[k].notes; }}).length;
    var flagged = Object.keys(annotations).filter(function(k) {{ return annotations[k] && annotations[k].followup; }}).length;
    document.getElementById('stats').innerHTML =
      visibleCount + ' events shown<br>' +
      annotated + ' annotated · ' + flagged + ' flagged';
  }}

  // --- Filter wiring ---
  document.getElementById('filter-acd').addEventListener('change', function() {{
    filterAcd = this.value; renderGrid();
  }});
  document.querySelectorAll('.flag-filter').forEach(function(cb) {{
    cb.addEventListener('change', function() {{
      filterFlags = Array.from(document.querySelectorAll('.flag-filter:checked')).map(function(el) {{ return el.getAttribute('data-flag'); }});
      renderGrid();
    }});
  }});
  var confSlider = document.getElementById('filter-conf');
  confSlider.addEventListener('input', function() {{
    filterMinConf = parseFloat(this.value);
    document.getElementById('conf-val').textContent = filterMinConf.toFixed(2);
    renderGrid();
  }});
  document.getElementById('filter-annotated-only').addEventListener('change', function() {{
    filterAnnotatedOnly = this.checked; renderGrid();
  }});
  document.getElementById('filter-flagged-only').addEventListener('change', function() {{
    filterFlaggedOnly = this.checked; renderGrid();
  }});
  document.getElementById('filter-hide-near-edge').addEventListener('change', function() {{
    filterHideNearEdge = this.checked; renderGrid();
  }});
  document.getElementById('filter-hide-fps').addEventListener('change', function() {{
    filterHideFps = this.checked; renderGrid();
  }});

  // --- Export ---
  document.getElementById('export-btn').addEventListener('click', function() {{
    var lines = ['parent_id,peak_frame,researcher_notes,flagged_for_followup,ai_confidence,acd_division_type,anomaly_flags'];
    manifest.forEach(function(ev) {{
      var ann = annotations[ev.parent_id] || {{}};
      if (!ann.notes && !ann.followup) return;
      var notes = (ann.notes || '').replace(/"/g, '""');
      var flags = ev.flags.join('; ');
      lines.push([
        ev.parent_id, ev.peak_frame,
        '"' + notes + '"',
        ann.followup ? '1' : '0',
        ev.confidence, ev.acd_division_type,
        '"' + flags + '"'
      ].join(','));
    }});
    var blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'researcher_notes.csv';
    a.click();
  }});

  renderGrid();
}})();
</script>
</body></html>
"""


def generate(
    run_dir: Path,
    min_conf: float = 0.0,
    include_fps: bool = False,
    thumb_zoom: int = 280,
    out: Path | None = None,
) -> Path | None:
    """Build and write the researcher browser HTML for a run. Returns the output path,
    or None if there was nothing to show (no events.csv, or no events matched).

    Callable directly (e.g. from src/pipeline.py to auto-generate at the end of a run)
    as well as via this script's CLI -- see main() below.
    """
    if not (run_dir / "events.csv").exists():
        print(f"  [researcher_browser] no events.csv found in {run_dir}, skipping")
        return None

    all_rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))
    splits = [r for r in all_rows if r.get("split_topology") in ("normal_split", "multi_way_split")]
    by_split: dict[tuple, dict] = {}
    for r in splits:
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        by_split.setdefault(key, r)

    conf_col = _conf_col(splits[0]) if splits else "ai_confidence"
    confirmed = sum(1 for r in by_split.values() if float(r.get(conf_col) or 0) > 0)

    manifest = _build_manifest(run_dir, min_conf, include_fps, thumb_zoom)
    if not manifest:
        print(f"  [researcher_browser] no events matched the filter criteria in {run_dir}, skipping")
        return None

    out_path = out if out else run_dir / "reports" / "researcher_browser.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = _render_html(manifest, run_dir.name, confirmed, thumb_zoom)
    out_path.write_text(html, encoding="utf-8")

    anomaly_count = sum(1 for e in manifest if e["flags"] or e["anomaly_notes"])
    abnormal_geom = sum(1 for e in manifest if e["acd_division_type"] in ("tripolar", "multipolar"))
    failed_count = sum(1 for e in manifest if e["is_failed_split"])
    mismatch_count = sum(1 for e in manifest if e["split_type_mismatch"])
    print(f"  [researcher_browser] wrote {out_path}")
    print(f"    {len(manifest)} events · {confirmed} confirmed splits total")
    print(f"    {anomaly_count} anomaly-flagged · {abnormal_geom} tripolar/multipolar · "
          f"{failed_count} failed_split · {mismatch_count} split_type mismatch")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv and review_crops/")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="minimum ai_confidence to include (default 0 = include FPs; use 0.01 to exclude)")
    parser.add_argument("--include-fps", action="store_true",
                        help="explicitly include false positives (confidence=0) even when --min-conf > 0")
    parser.add_argument("--thumb-zoom", type=int, default=280,
                        help="thumbnail zoom %% around tracked centroid (default 280)")
    parser.add_argument("--out", default=None,
                        help="output .html path (default: <run_dir>/reports/researcher_browser.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else None
    result = generate(run_dir, args.min_conf, args.include_fps, args.thumb_zoom, out)
    if result is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
