"""Sortable/filterable gallery for `death` events, ranked by shape-descriptor outlier-ness.

Motivated by the 2026-07-09 regionprops spike ([[project_cell_split_counter_interesting_events]]):
aggregate eccentricity/solidity stats don't discriminate real-vs-false_positive splits or death-row
duration, but manually inspecting extreme low-solidity/high-eccentricity death rows showed
genuinely distinct-looking fragmented/granular cells vs. the smooth healthy neighbors around them.
This is the browsing tool to scan those extremes directly, instead of one-off contact sheets.

Death events never get vision review, so unlike split events there are no pre-existing
review_crops/ folders -- crops are generated here directly from the run's frame PNGs, centered on
the tracked centroid, with a crosshair marker (the crop is centered exactly on the flagged cell,
but the surrounding field is often crowded with other cells, so the crosshair disambiguates which
one is the actual target).

Usage:
    python scripts/reports/death_shape_browser.py data/output/<run_dir>
    python scripts/reports/death_shape_browser.py data/output/<run_dir> --events-csv events_with_shape.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_CROP_RADIUS = 110   # px around centroid -- smaller than review.py's 192, death crops only need
                      # to show the one flagged cell + immediate neighbors, not two daughters.
_FRAMES_BEFORE = 6    # lead-up frames shown before the death frame (no "after" -- track ends there)
_FRAME_STRIDE = 3     # matches src/review.py's sampling convention


def _find_frame(frame_dir: Path, index: int) -> Path | None:
    matches = list(frame_dir.glob(f"frame_{index:05d}_*.png"))
    return matches[0] if matches else None


def _crop_with_marker(frame_cache: dict, frame_dir: Path, frame_idx: int, cx: float, cy: float):
    if frame_idx not in frame_cache:
        path = _find_frame(frame_dir, frame_idx)
        frame_cache[frame_idx] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
    img = frame_cache[frame_idx]
    if img is None:
        return None
    h, w = img.shape
    r = _CROP_RADIUS
    x0, x1 = int(cx - r), int(cx + r)
    y0, y1 = int(cy - r), int(cy + r)
    px0, py0 = max(0, -x0), max(0, -y0)
    x0c, x1c, y0c, y1c = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    out = np.zeros((r * 2, r * 2), dtype=np.uint8)
    out[py0:py0 + (y1c - y0c), px0:px0 + (x1c - x0c)] = img[y0c:y1c, x0c:x1c]
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    tile = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(tile, (r, r), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
    cv2.circle(tile, (r, r), 22, (0, 255, 255), 1)
    return tile


def _build_crops(frame_dir: Path, rows: list[dict], out_dir: Path) -> dict[str, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_cache: dict[int, np.ndarray | None] = {}
    images_by_row: dict[str, list[str]] = {}
    for r in rows:
        row_id = f"{r['track_id']}_{r['peak_frame']}"
        peak_frame = int(r["peak_frame"])
        cx, cy = float(r["centroid_x"]), float(r["centroid_y"])
        frame_indices = [peak_frame - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
        frame_indices = [i for i in frame_indices if i >= 0] + [peak_frame]
        images = []
        for idx in frame_indices:
            tile = _crop_with_marker(frame_cache, frame_dir, idx, cx, cy)
            if tile is None:
                continue
            fname = f"{row_id}_f{idx:05d}.png"
            cv2.imwrite(str(out_dir / fname), tile)
            images.append(fname)
        images_by_row[row_id] = images
        if len(frame_cache) > 400:  # cap memory -- oldest frames rarely revisited given time ordering
            frame_cache.clear()
    return images_by_row


def _build_manifest(run_dir: Path, events_csv: str, crops_subdir: str) -> list[dict]:
    rows = list(csv.DictReader(open(run_dir / events_csv, encoding="utf-8", errors="replace")))
    deaths = [r for r in rows if r.get("split_topology") == "death"]
    if deaths and not deaths[0].get("eccentricity"):
        # ValueError, not SystemExit -- this needs to be catchable by a normal
        # `except Exception` when generate() is called from src/pipeline.py, which
        # SystemExit (a BaseException) would silently bypass.
        raise ValueError(
            f"{events_csv} has no eccentricity/solidity data -- this run predates the "
            "2026-07-09 regionprops columns. Re-run classify/output, or point --events-csv "
            "at a backfilled CSV (e.g. events_with_shape.csv)."
        )

    frame_dir = run_dir / "frames"
    out_dir = run_dir / "reports" / crops_subdir
    images_by_row = _build_crops(frame_dir, deaths, out_dir)

    manifest = []
    for r in deaths:
        row_id = f"{r['track_id']}_{r['peak_frame']}"
        ecc = float(r["eccentricity"]) if r.get("eccentricity") else None
        sol = float(r["solidity"]) if r.get("solidity") else None
        manifest.append({
            "row_id": row_id,
            "track_id": r["track_id"],
            "parent_id": r.get("parent_id") or "",
            "peak_frame": int(r["peak_frame"]),
            "eccentricity": ecc,
            "solidity": sol,
            "cell_area_px": float(r["cell_area_px"]) if r.get("cell_area_px") else None,
            "persistence": float(r["ai_confidence"]) if r.get("ai_confidence") else None,
            "near_edge": r.get("near_edge") == "1",
            "images": [f"{crops_subdir}/{fn}" for fn in images_by_row.get(row_id, [])],
        })
    return manifest


_CSS = """
:root {
  --surface-1:#fcfcfb;--page:#f4f4f2;--text-primary:#0b0b0b;--text-secondary:#52514e;
  --text-muted:#898781;--border:rgba(11,11,11,0.10);--series-1:#2a78d6;
  --outlier:#b8860b;--outlier-wash:#fbf3e0;
}
@media (prefers-color-scheme:dark){
  :root{
    --surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#ffffff;--text-secondary:#c3c2b7;
    --text-muted:#898781;--border:rgba(255,255,255,0.10);
    --outlier:#fab219;--outlier-wash:#2a2311;
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
.stats{font-size:11.5px;color:var(--text-muted);line-height:1.6;}
.main{margin-left:220px;padding:20px 24px;}
.main-header{margin-bottom:16px;}
.main-header h1{font-size:18px;font-weight:600;margin:0 0 2px;}
.main-header .subtitle{font-size:13px;color:var(--text-secondary);margin:0;}
.grid{display:flex;flex-direction:column;gap:14px;}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:16px 18px;border-left:3px solid var(--outlier);}
.card-header{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap;}
.frame-label{font-size:13px;font-weight:600;}
.metric-badge{font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:4px;white-space:nowrap;background:var(--outlier-wash);color:var(--outlier);}
.near-edge-chip{font-size:11px;padding:2px 7px;border-radius:4px;background:var(--outlier-wash);color:var(--outlier);}
.filmstrip{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.crop-wrap{position:relative;display:inline-block;line-height:0;border-radius:4px;overflow:hidden;border:1px solid var(--border);cursor:zoom-in;flex-shrink:0;}
.crop-thumb{display:block;width:120px;height:120px;}
.meta-row{font-size:11.5px;color:var(--text-muted);margin-bottom:0;}
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
"""


def _render_html(manifest: list[dict], run_name: str, total_deaths: int) -> str:
    subtitle = f"{run_name} · {total_deaths} death events · sorted by shape outlier-ness"
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Death shape browser — {run_name}</title>
<style>{_CSS}</style>
</head>
<body>

<div class="sidebar">
  <h2>Sort / Filter</h2>
  <div class="run-name">{run_name}</div>

  <div class="filter-group">
    <label>Sort by</label>
    <select id="sort-key">
      <option value="solidity_asc">Lowest solidity first (jagged)</option>
      <option value="eccentricity_desc">Highest eccentricity first (elongated)</option>
      <option value="eccentricity_asc">Lowest eccentricity first (roundest)</option>
      <option value="area_desc">Largest area first</option>
      <option value="area_asc">Smallest area first</option>
      <option value="persistence_desc">Longest-lived first</option>
      <option value="persistence_asc">Shortest-lived first (likely tracker dropout)</option>
    </select>
  </div>

  <div class="filter-group">
    <label>Max solidity: <span id="sol-val">1.00</span></label>
    <input type="range" id="filter-sol-max" min="0.5" max="1" step="0.01" value="1">
  </div>

  <div class="filter-group">
    <label>Min eccentricity: <span id="ecc-val">0.00</span></label>
    <input type="range" id="filter-ecc-min" min="0" max="1" step="0.01" value="0">
  </div>

  <div class="filter-group">
    <label><input type="checkbox" id="filter-hide-near-edge"> Hide near-edge</label>
  </div>

  <div class="stats" id="stats"></div>
</div>

<div class="main">
  <div class="main-header">
    <h1>Death shape browser</h1>
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
    lbCaption.textContent = (lbIdx + 1) + ' / ' + lbImages.length + '  —  ' + lbImages[lbIdx].split('/').pop();
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

  var sortKey = 'solidity_asc';
  var maxSol = 1.0;
  var minEcc = 0.0;
  var hideNearEdge = false;

  var sortFns = {{
    solidity_asc: function(a,b){{ return (a.solidity ?? 1) - (b.solidity ?? 1); }},
    eccentricity_desc: function(a,b){{ return (b.eccentricity ?? 0) - (a.eccentricity ?? 0); }},
    eccentricity_asc: function(a,b){{ return (a.eccentricity ?? 1) - (b.eccentricity ?? 1); }},
    area_desc: function(a,b){{ return (b.cell_area_px ?? 0) - (a.cell_area_px ?? 0); }},
    area_asc: function(a,b){{ return (a.cell_area_px ?? 0) - (b.cell_area_px ?? 0); }},
    persistence_desc: function(a,b){{ return (b.persistence ?? 0) - (a.persistence ?? 0); }},
    persistence_asc: function(a,b){{ return (a.persistence ?? 0) - (b.persistence ?? 0); }}
  }};

  function passesFilter(ev) {{
    if (ev.solidity != null && ev.solidity > maxSol) return false;
    if (ev.eccentricity != null && ev.eccentricity < minEcc) return false;
    if (hideNearEdge && ev.near_edge) return false;
    return true;
  }}

  function renderCard(ev) {{
    var filmstrip = ev.images.length > 0
      ? ev.images.map(function(src, i) {{
          return '<span class="crop-wrap" data-idx="' + i + '">' +
            '<img class="crop-thumb" src="' + src + '" loading="lazy"></span>';
        }}).join('')
      : '<p>No crops.</p>';
    var ecc = ev.eccentricity != null ? ev.eccentricity.toFixed(3) : '?';
    var sol = ev.solidity != null ? ev.solidity.toFixed(3) : '?';
    var area = ev.cell_area_px != null ? Math.round(ev.cell_area_px) : '?';
    var pers = ev.persistence != null ? ev.persistence.toFixed(2) : '?';
    var nearChip = ev.near_edge ? '<span class="near-edge-chip">near edge</span>' : '';

    return '<div class="card" data-row="' + ev.row_id + '">' +
      '<div class="card-header">' +
        '<span class="frame-label">Track ' + ev.track_id + ' · Frame ' + ev.peak_frame + '</span>' +
        '<span class="metric-badge">ecc ' + ecc + '</span>' +
        '<span class="metric-badge">sol ' + sol + '</span>' +
        nearChip +
      '</div>' +
      '<div class="filmstrip">' + filmstrip + '</div>' +
      '<div class="meta-row">area ' + area + 'px&sup2; · persistence score ' + pers + ' · parent ' + (ev.parent_id || 'none') + '</div>' +
    '</div>';
  }}

  function renderGrid() {{
    var visible = manifest.filter(passesFilter).slice().sort(sortFns[sortKey]);
    var grid = document.getElementById('grid');
    grid.innerHTML = visible.map(renderCard).join('');

    grid.querySelectorAll('.crop-wrap[data-idx]').forEach(function(wrap) {{
      wrap.addEventListener('click', function() {{
        var card = wrap.closest('.card');
        var rowId = card.getAttribute('data-row');
        var ev = manifest.find(function(e) {{ return e.row_id === rowId; }});
        if (ev && ev.images.length) {{
          openLightbox(ev.images, parseInt(wrap.getAttribute('data-idx'), 10));
        }}
      }});
    }});

    document.getElementById('stats').innerHTML = visible.length + ' / ' + manifest.length + ' events shown';
  }}

  document.getElementById('sort-key').addEventListener('change', function() {{ sortKey = this.value; renderGrid(); }});
  document.getElementById('filter-sol-max').addEventListener('input', function() {{
    maxSol = parseFloat(this.value); document.getElementById('sol-val').textContent = maxSol.toFixed(2); renderGrid();
  }});
  document.getElementById('filter-ecc-min').addEventListener('input', function() {{
    minEcc = parseFloat(this.value); document.getElementById('ecc-val').textContent = minEcc.toFixed(2); renderGrid();
  }});
  document.getElementById('filter-hide-near-edge').addEventListener('change', function() {{
    hideNearEdge = this.checked; renderGrid();
  }});

  renderGrid();
}})();
</script>
</body></html>
"""


def generate(
    run_dir: Path,
    events_csv: str = "events.csv",
    crops_subdir: str = "death_crops",
    out: Path | None = None,
) -> Path | None:
    """Build and write the death shape browser HTML for a run. Returns the output path,
    or None if there was nothing to show (no events.csv, no death rows, or the CSV
    predates the regionprops columns).

    Callable directly (e.g. from src/pipeline.py to auto-generate at the end of a run)
    as well as via this script's CLI -- see main() below.
    """
    if not (run_dir / events_csv).exists():
        print(f"  [death_shape_browser] no {events_csv} found in {run_dir}, skipping")
        return None

    try:
        manifest = _build_manifest(run_dir, events_csv, crops_subdir)
    except ValueError as exc:
        print(f"  [death_shape_browser] {exc}")
        return None

    if not manifest:
        print(f"  [death_shape_browser] no death events found in {run_dir}, skipping")
        return None

    out_path = out if out else run_dir / "reports" / "death_shape_browser.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(manifest, run_dir.name, len(manifest)), encoding="utf-8")

    print(f"  [death_shape_browser] wrote {out_path}")
    print(f"    {len(manifest)} death events, crops in {run_dir / 'reports' / crops_subdir}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="output run directory containing events.csv and frames/")
    parser.add_argument("--events-csv", default="events.csv",
                        help="CSV filename to read within run_dir (default events.csv; use "
                             "events_with_shape.csv for runs backfilled by the regionprops spike)")
    parser.add_argument("--crops-subdir", default="death_crops",
                        help="subdirectory under <run_dir>/reports/ to write generated crop PNGs to")
    parser.add_argument("--out", default=None,
                        help="output .html path (default: <run_dir>/reports/death_shape_browser.html)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else None
    result = generate(run_dir, args.events_csv, args.crops_subdir, out)
    if result is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
