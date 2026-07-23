"""One-off gallery for the 2026-07-23 necrosis-candidate search (see
scripts/eval_harness/find_necrosis_candidates.py) -- combines candidates across all 5
Bewo wells into a single sorted-by-confidence HTML page, reusing batch_review_viewer.py's
manifest builder and renderer.

Not general-purpose: batch_review_viewer.py assumes single-well output (image paths are
"../review_crops/..." relative to a <run_dir>/reports/ page); this rewrites those paths
for a combined page living in data/output/ directly, one level above every well dir.

Usage:
    python scripts/reports/necrosis_candidates_gallery.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reports.batch_review_viewer import _build_manifest, _render_html

REPO = Path(__file__).resolve().parents[2]
CANDIDATES_CSV = Path("G:/Projects/cell-split-counter-shared-data/human_review/necrosis_candidates_2026-07-23.csv")
OUT_PATH = REPO / "data/output/necrosis_candidates_gallery.html"
WELLS = ["M2", "M3", "M4", "M5", "M6"]


def _load_centroids(well: str) -> dict[int, dict]:
    events_csv = REPO / f"data/output/202660629_Bewop920x_{well}/events.csv"
    rows = list(csv.DictReader(events_csv.open(encoding="utf-8")))
    return {int(r["track_id"]): r for r in rows}


def main() -> None:
    rows = list(csv.DictReader(CANDIDATES_CSV.open(encoding="utf-8")))
    candidates = [r for r in rows if r["recalibrated_dropout"] == "False"]
    candidates.sort(key=lambda r: float(r["confidence"] or 0), reverse=True)

    combined_manifest = []
    for well in WELLS:
        well_candidates = [r for r in candidates if r["well"] == well]
        if not well_candidates:
            continue
        centroids = _load_centroids(well)
        well_csv_rows = []
        for r in well_candidates:
            src = centroids.get(int(r["track_id"]), {})
            well_csv_rows.append({
                "track_id": r["track_id"], "parent_id": r["parent_id"], "peak_frame": r["peak_frame"],
                "centroid_x": src.get("centroid_x", ""), "centroid_y": src.get("centroid_y", ""),
                "cell_area_px": src.get("cell_area_px", ""),
                "neighbor_distance_px": src.get("neighbor_distance_px", ""),
                "neighbor_area_px": src.get("neighbor_area_px", ""),
                "confidence": r["confidence"], "description": r["description"],
            })
        tmp_csv = REPO / f"data/output/_tmp_necrosis_{well}.csv"
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(well_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(well_csv_rows)

        run_dir = REPO / f"data/output/202660629_Bewop920x_{well}"
        manifest = _build_manifest(run_dir, tmp_csv, "track", ["confidence", "description"])
        tmp_csv.unlink()

        for m in manifest:
            m["images"] = [img.replace("../review_crops/", f"202660629_Bewop920x_{well}/review_crops/") for img in m["images"]]
            m["well"] = well
        combined_manifest.extend(manifest)

    # re-sort combined manifest by confidence (build order was per-well, not global)
    combined_manifest.sort(key=lambda m: float(m["extra"].get("confidence") or 0), reverse=True)

    n_with_crops = sum(1 for m in combined_manifest if m["images"])
    subtitle = f"5 wells (M2-M6) · {len(combined_manifest)} candidates · {n_with_crops} with crops found · sorted by confidence"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        _render_html(
            combined_manifest, "Necrosis candidates (2026-07-23)", subtitle,
            "necrosis_candidates_2026-07-23", ["confidence", "description"],
            ["genuine necrosis", "actually alive/dividing", "unsure"],
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH} ({len(combined_manifest)} candidates, {n_with_crops} with crops)")


if __name__ == "__main__":
    main()
