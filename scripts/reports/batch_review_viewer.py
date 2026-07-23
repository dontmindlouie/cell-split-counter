"""Standalone gallery viewer for a small hand-picked batch of events (e.g. the
2026-07-19 Batch A / Batch B targeted review CSVs), as a faster alternative to
opening the raw .nd2 in Fiji for each one.

Not a general-purpose tool -- reads a small sample CSV (produced ad hoc for a
specific review batch, one row per event) plus each event's review_crops/ folder,
and renders one self-contained HTML page with a filmstrip + lightbox per event and
a verdict/notes form (localStorage-persisted, exportable to CSV), same UX pattern
as researcher_browser.py's annotation feature.

Usage:
    python scripts/reports/batch_review_viewer.py <run_dir> <sample_csv> --out <out.html> \\
        --title "Batch A" --folder-pattern parent --verdict-options "yes,no,unsure"
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reports._crop_shared import CROP_RADIUS as _CROP_RADIUS
from scripts.reports._crop_shared import centroid_in_crop_pct as _centroid_in_crop_pct
from src.review import adaptive_radius as _adaptive_radius

# Same 2026-07-13/07-13 real-usage-feedback marker as researcher_browser.py: a human
# reviewer finds crowded frames ambiguous too, not just the AI, but baking a marker into
# the saved review_crops/ PNGs would ruin a clean copy for a researcher's own
# reports/slides -- so this is a display-only CSS overlay, reusing the AI's own
# neighbor-aware adaptive_radius() but pushed further out (a human just needs
# orientation, not a radius tight enough to disambiguate crowded neighbors).
_DISPLAY_MARKER_SCALE = 1.7


def _find_crop_folder(run_dir: Path, track_id: str, parent_id: str, peak_frame: str, pattern: str) -> Path | None:
    crops_dir = run_dir / "review_crops"
    if pattern == "parent":
        folder = crops_dir / f"frame_{int(peak_frame):05d}_parent_{int(float(parent_id))}"
    else:
        folder = crops_dir / f"frame_{int(peak_frame):05d}_track_{int(track_id)}"
    return folder if folder.is_dir() else None


def _marker_radius_pct(row: dict) -> float:
    def _f(key: str) -> float | None:
        v = row.get(key)
        return float(v) if v not in (None, "") else None

    radius_px = _adaptive_radius(
        _f("neighbor_distance_px"), cell_area_px=_f("cell_area_px"), neighbor_area_px=_f("neighbor_area_px"),
    )
    return (radius_px * _DISPLAY_MARKER_SCALE / (2 * _CROP_RADIUS)) * 100


def _build_manifest(run_dir: Path, sample_csv: Path, folder_pattern: str, extra_cols: list[str]) -> list[dict]:
    rows = list(csv.DictReader(open(sample_csv, encoding="utf-8", errors="replace")))
    manifest = []
    for r in rows:
        track_id, parent_id, peak_frame = r["track_id"], r.get("parent_id", ""), r["peak_frame"]
        folder = _find_crop_folder(run_dir, track_id, parent_id, peak_frame, folder_pattern)
        images = []
        if folder is not None:
            for p in sorted(folder.glob("*.png"), key=lambda p: int(p.name.split("_", 1)[0])):
                images.append(f"../review_crops/{folder.name}/{p.name}")

        crosshair_x_pct = crosshair_y_pct = 50.0
        cx, cy = r.get("centroid_x"), r.get("centroid_y")
        if folder is not None and images and cx not in (None, "") and cy not in (None, ""):
            crosshair_x_pct, crosshair_y_pct = _centroid_in_crop_pct(
                folder / Path(images[0]).name, float(cx), float(cy)
            )

        manifest.append({
            "entry_key": f"{track_id}_{peak_frame}",
            "track_id": track_id,
            "parent_id": parent_id or "",
            "peak_frame": peak_frame,
            "extra": {c: r.get(c, "") for c in extra_cols},
            "images": images,
            "crosshair_x_pct": round(crosshair_x_pct, 2),
            "crosshair_y_pct": round(crosshair_y_pct, 2),
            "marker_radius_pct": round(_marker_radius_pct(r), 2),
        })
    return manifest


_CSS = """
:root {
  --surface-1:#fcfcfb;--page:#f4f4f2;--text-primary:#0b0b0b;--text-secondary:#52514e;
  --text-muted:#898781;--border:rgba(11,11,11,0.10);--accent:#2a78d6;--accent-wash:#e8f0fb;
  --done:#2a8a4a;--done-wash:#e6f5ea;
}
@media (prefers-color-scheme:dark){
  :root{
    --surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#ffffff;--text-secondary:#c3c2b7;
    --text-muted:#898781;--border:rgba(255,255,255,0.10);--accent:#5b9bea;--accent-wash:#152438;
    --done:#4fc274;--done-wash:#0f2415;
  }
}
*{box-sizing:border-box;}
body{background:var(--page);color:var(--text-primary);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:0;}
.sidebar{position:fixed;top:0;left:0;width:220px;height:100vh;background:var(--surface-1);border-right:1px solid var(--border);overflow-y:auto;padding:16px 14px;z-index:5;}
.sidebar h2{font-size:13px;font-weight:600;margin:0 0 10px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);}
.sidebar .title{font-size:15px;font-weight:600;margin:0 0 4px;}
.sidebar .subtitle{font-size:12px;color:var(--text-secondary);margin:0 0 14px;}
.stats{font-size:12px;color:var(--text-muted);line-height:1.7;}
.export-btn{width:100%;margin-top:14px;padding:8px 10px;font:inherit;font-size:12.5px;font-weight:600;border-radius:6px;border:1px solid var(--accent);background:var(--accent-wash);color:var(--accent);cursor:pointer;}
.export-btn:hover{opacity:0.85;}
.main{margin-left:220px;padding:20px 24px;}
.main h1{font-size:18px;font-weight:600;margin:0 0 16px;}
.grid{display:flex;flex-direction:column;gap:14px;}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px;}
.card.done{border-left:3px solid var(--done);}
.card-header{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap;}
.frame-label{font-size:13px;font-weight:600;}
.meta-chip{font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:4px;background:var(--accent-wash);color:var(--accent);}
.filmstrip{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;max-height:200px;overflow-y:auto;}
.crop-wrap{position:relative;display:inline-block;line-height:0;border-radius:4px;overflow:hidden;border:1px solid var(--border);cursor:zoom-in;flex-shrink:0;}
.crop-thumb{display:block;width:64px;height:64px;object-fit:cover;}
.no-crops{font-size:12.5px;color:var(--text-muted);font-style:italic;margin-bottom:10px;}
.verdict-row{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap;}
.verdict-row label{font-size:12.5px;color:var(--text-secondary);}
.verdict-row select{font:inherit;font-size:12.5px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);padding:5px 8px;}
.notes-area textarea{width:100%;font:inherit;font-size:13px;border-radius:6px;border:1px solid var(--border);background:var(--page);color:var(--text-primary);padding:7px 9px;resize:vertical;min-height:44px;}
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
.marker-tick{position:absolute;width:10px;height:10px;border-color:rgba(230,170,60,0.75);border-style:solid;border-width:0;pointer-events:none;}
.marker-tick.tl{border-top-width:1.5px;border-left-width:1.5px;}
.marker-tick.tr{border-top-width:1.5px;border-right-width:1.5px;transform:translateX(-100%);}
.marker-tick.bl{border-bottom-width:1.5px;border-left-width:1.5px;transform:translateY(-100%);}
.marker-tick.br{border-bottom-width:1.5px;border-right-width:1.5px;transform:translate(-100%,-100%);}
.marker-toggle{font-size:12px;color:var(--text-secondary);display:block;margin:6px 0 14px;}
"""


def _render_html(
    manifest: list[dict], title: str, subtitle: str, run_name: str,
    extra_cols: list[str], verdict_options: list[str],
) -> str:
    verdict_opts_html = "".join(
        f'<option value="{v}">{v}</option>' for v in ["", *verdict_options]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>

<div class="sidebar">
  <div class="title">{title}</div>
  <div class="subtitle">{subtitle}</div>
  <div class="stats" id="stats"></div>
  <label class="marker-toggle"><input type="checkbox" id="filter-show-marker" checked> Show position marker</label>
  <button class="export-btn" id="export-btn">Export verdicts CSV</button>
</div>

<div class="main">
  <h1>{title}</h1>
  <div class="grid" id="grid"></div>
</div>

<div class="lightbox" id="lightbox">
  <button class="lightbox-close" id="lightbox-close">&times;</button>
  <button class="lightbox-nav prev" id="lightbox-prev">&lsaquo;</button>
  <div class="lightbox-img-wrap">
    <img id="lightbox-img" alt="">
    <div class="lightbox-marker" id="lightbox-marker"></div>
    <div class="lightbox-caption" id="lightbox-caption"></div>
  </div>
  <button class="lightbox-nav next" id="lightbox-next">&rsaquo;</button>
</div>

<script>
(function(){{
  var manifest = {json.dumps(manifest)};
  var extraCols = {json.dumps(extra_cols)};
  var verdictOptsHtml = {json.dumps(verdict_opts_html)};
  var storageKey = 'batch_review_' + {json.dumps(run_name)} + '_' + {json.dumps(title)};

  function loadAnnotations() {{
    try {{ return JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch(e) {{ return {{}}; }}
  }}
  function saveAnnotations(a) {{
    try {{ localStorage.setItem(storageKey, JSON.stringify(a)); }} catch(e) {{}}
  }}
  var annotations = loadAnnotations();
  var showMarker = true;

  var lightbox = document.getElementById('lightbox');
  var lbImg = document.getElementById('lightbox-img');
  var lbMarker = document.getElementById('lightbox-marker');
  var lbCaption = document.getElementById('lightbox-caption');
  var lbPrev = document.getElementById('lightbox-prev');
  var lbNext = document.getElementById('lightbox-next');
  var lbClose = document.getElementById('lightbox-close');
  var lbImages = [], lbIdx = 0, lbMarkerData = null;

  function openLightbox(images, startIdx, markerData) {{
    lbImages = images; lbIdx = startIdx; lbMarkerData = markerData || null;
    showLbFrame();
    lightbox.classList.add('open');
  }}
  function showLbFrame() {{
    lbImg.src = lbImages[lbIdx];
    lbCaption.textContent = (lbIdx + 1) + ' / ' + lbImages.length + '  —  ' + lbImages[lbIdx].split('/').pop();
    lbPrev.disabled = lbIdx === 0;
    lbNext.disabled = lbIdx === lbImages.length - 1;
    if (showMarker && lbMarkerData) {{
      var cx = lbMarkerData.x, cy = lbMarkerData.y, r = lbMarkerData.r;
      lbMarker.innerHTML =
        '<span class="marker-tick tl" style="left:' + (cx - r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick tr" style="left:' + (cx + r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick bl" style="left:' + (cx - r) + '%;top:' + (cy + r) + '%;"></span>' +
        '<span class="marker-tick br" style="left:' + (cx + r) + '%;top:' + (cy + r) + '%;"></span>';
    }} else {{
      lbMarker.innerHTML = '';
    }}
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

  function renderCard(ev) {{
    var ann = annotations[ev.entry_key] || {{}};
    var markerHtml = '';
    if (showMarker) {{
      var cx = ev.crosshair_x_pct, cy = ev.crosshair_y_pct, r = ev.marker_radius_pct;
      markerHtml =
        '<span class="marker-tick tl" style="left:' + (cx - r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick tr" style="left:' + (cx + r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick bl" style="left:' + (cx - r) + '%;top:' + (cy + r) + '%;"></span>' +
        '<span class="marker-tick br" style="left:' + (cx + r) + '%;top:' + (cy + r) + '%;"></span>';
    }}
    var filmstrip = ev.images.length > 0
      ? ev.images.map(function(src, i) {{
          return '<span class="crop-wrap" data-idx="' + i + '"><img class="crop-thumb" src="' + src + '" loading="lazy">' + markerHtml + '</span>';
        }}).join('')
      : '';
    var noCrops = ev.images.length === 0 ? '<p class="no-crops">No crops found on disk for this event.</p>' : '';
    var chips = extraCols.map(function(c) {{
      return ev.extra[c] ? '<span class="meta-chip">' + c + ': ' + ev.extra[c] + '</span>' : '';
    }}).join('');
    var isDone = ann.verdict ? ' done' : '';

    return '<div class="card' + isDone + '" data-row="' + ev.entry_key + '">' +
      '<div class="card-header">' +
        '<span class="frame-label">Track ' + ev.track_id + ' · Frame ' + ev.peak_frame +
          (ev.parent_id ? ' · parent ' + ev.parent_id : '') + '</span>' +
        chips +
      '</div>' +
      (noCrops || '<div class="filmstrip">' + filmstrip + '</div>') +
      '<div class="verdict-row">' +
        '<label>Verdict:</label>' +
        '<select class="verdict-select">' + verdictOptsHtml + '</select>' +
      '</div>' +
      '<div class="notes-area"><textarea placeholder="Notes...">' + (ann.notes || '') + '</textarea></div>' +
    '</div>';
  }}

  function renderGrid() {{
    var grid = document.getElementById('grid');
    grid.innerHTML = manifest.map(renderCard).join('');

    grid.querySelectorAll('.crop-wrap[data-idx]').forEach(function(wrap) {{
      wrap.addEventListener('click', function() {{
        var card = wrap.closest('.card');
        var rowId = card.getAttribute('data-row');
        var ev = manifest.find(function(e) {{ return e.entry_key === rowId; }});
        if (ev && ev.images.length) {{
          openLightbox(ev.images, parseInt(wrap.getAttribute('data-idx'), 10),
            {{x: ev.crosshair_x_pct, y: ev.crosshair_y_pct, r: ev.marker_radius_pct}});
        }}
      }});
    }});

    grid.querySelectorAll('.card').forEach(function(card) {{
      var rowId = card.getAttribute('data-row');
      var ann = annotations[rowId] || {{}};
      var sel = card.querySelector('.verdict-select');
      sel.value = ann.verdict || '';
      sel.addEventListener('change', function() {{
        if (!annotations[rowId]) annotations[rowId] = {{}};
        annotations[rowId].verdict = sel.value;
        saveAnnotations(annotations);
        card.classList.toggle('done', !!sel.value);
        updateStats();
      }});
      var ta = card.querySelector('textarea');
      ta.addEventListener('input', function() {{
        if (!annotations[rowId]) annotations[rowId] = {{}};
        annotations[rowId].notes = ta.value;
        saveAnnotations(annotations);
      }});
    }});

    updateStats();
  }}

  function updateStats() {{
    var done = Object.keys(annotations).filter(function(k) {{ return annotations[k] && annotations[k].verdict; }}).length;
    document.getElementById('stats').innerHTML = done + ' / ' + manifest.length + ' reviewed';
  }}

  document.getElementById('filter-show-marker').addEventListener('change', function() {{
    showMarker = this.checked; renderGrid();
    if (lightbox.classList.contains('open')) showLbFrame();
  }});

  document.getElementById('export-btn').addEventListener('click', function() {{
    var lines = ['track_id,parent_id,peak_frame,verdict,notes'];
    manifest.forEach(function(ev) {{
      var ann = annotations[ev.entry_key] || {{}};
      var notes = '"' + (ann.notes || '').replace(/"/g, '""') + '"';
      lines.push([ev.track_id, ev.parent_id, ev.peak_frame, ann.verdict || '', notes].join(','));
    }});
    var blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'verdicts_' + {json.dumps(title.replace(" ", "_"))} + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }});

  renderGrid();
}})();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("sample_csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--folder-pattern", choices=["parent", "track"], required=True,
                         help="'parent' for pre-existing frame_X_parent_Y crop folders (splits), "
                              "'track' for frame_X_track_Y folders (deaths / extend_event_timeline.py output)")
    parser.add_argument("--verdict-options", required=True, help="comma-separated verdict choices")
    parser.add_argument("--extra-cols", default="", help="comma-separated sample_csv columns to show as chips")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    extra_cols = [c for c in args.extra_cols.split(",") if c]
    manifest = _build_manifest(run_dir, Path(args.sample_csv), args.folder_pattern, extra_cols)

    n_with_crops = sum(1 for m in manifest if m["images"])
    subtitle = f"{run_dir.name} · {len(manifest)} events · {n_with_crops} with crops found"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render_html(manifest, args.title, subtitle, run_dir.name, extra_cols, args.verdict_options.split(",")),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(manifest)} events, {n_with_crops} with crops)")


if __name__ == "__main__":
    main()
