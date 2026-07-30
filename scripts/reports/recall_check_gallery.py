"""Recall-check gallery: 50 randomly sampled events the recalibrated death-review
prompt called "still alive/dropout" (recalibrated_dropout=True), across all 5 wells.

The 2026-07-23 necrosis search only ever looked inside the "predicted death" bucket
(207 candidates) -- this checks the other side: does real necrosis show up among what
the prompt confidently waved through as alive? If so, the prompt's necrosis RECALL is
worse than the golden-set validation (83.7%) suggested, not just its precision (12.5%
on the candidate pool). Same viewer/verdict/export pattern as
necrosis_candidates_gallery.py.

Usage:
    python scripts/reports/recall_check_gallery.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reports.batch_review_viewer import _build_manifest, _render_html

REPO = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO / "data/output/recall_check_sample.csv"
OUT_PATH = REPO / "data/output/recall_check_gallery.html"
WELLS = ["M2", "M3", "M4", "M5", "M6"]


def _load_centroids(well: str) -> dict[int, dict]:
    events_csv = REPO / f"data/output/202660629_Bewop920x_{well}/events.csv"
    rows = list(csv.DictReader(events_csv.open(encoding="utf-8")))
    return {int(r["track_id"]): r for r in rows}


def main() -> None:
    rows = list(csv.DictReader(SAMPLE_CSV.open(encoding="utf-8")))

    combined_manifest = []
    for well in WELLS:
        well_rows = [r for r in rows if r["well"] == well]
        if not well_rows:
            continue
        centroids = _load_centroids(well)
        well_csv_rows = []
        for r in well_rows:
            src = centroids.get(int(r["track_id"]), {})
            well_csv_rows.append({
                "track_id": r["track_id"], "parent_id": r["parent_id"], "peak_frame": r["peak_frame"],
                "centroid_x": src.get("centroid_x", ""), "centroid_y": src.get("centroid_y", ""),
                "cell_area_px": src.get("cell_area_px", ""),
                "neighbor_distance_px": src.get("neighbor_distance_px", ""),
                "neighbor_area_px": src.get("neighbor_area_px", ""),
                "confidence": r["confidence"],
            })
        tmp_csv = REPO / f"data/output/_tmp_recall_{well}.csv"
        with tmp_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(well_csv_rows[0].keys()))
            w.writeheader()
            w.writerows(well_csv_rows)

        run_dir = REPO / f"data/output/202660629_Bewop920x_{well}"
        manifest = _build_manifest(run_dir, tmp_csv, "track", ["confidence"])
        tmp_csv.unlink()

        for m in manifest:
            m["images"] = [img.replace("../review_crops/", f"202660629_Bewop920x_{well}/review_crops/") for img in m["images"]]
            m["well"] = well
        combined_manifest.extend(manifest)

    n_with_crops = sum(1 for m in combined_manifest if m["images"])
    subtitle = f"5 wells (M2-M6) · {len(combined_manifest)} random 'predicted alive' events · {n_with_crops} with crops found · recall check"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        _render_html(
            combined_manifest, "Recall check: predicted-alive sample (2026-07-23)", subtitle,
            "recall_check_2026-07-23", ["confidence"],
            ["correctly alive/dividing", "actually necrosis (missed)", "unsure"],
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_PATH} ({len(combined_manifest)} events, {n_with_crops} with crops)")


if __name__ == "__main__":
    main()
