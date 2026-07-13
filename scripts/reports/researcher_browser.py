"""Open-book filmstrip browser for stem-cell researchers.

Shows **potentially interesting events** from a pipeline run -- splits (normal, failed,
multi-way), abnormal-geometry/anomaly-flagged divisions, and (2026-07-12) cell deaths --
sorted by biological interest, with AI verdict and notes visible from the start -- unlike
spot_check_review.py, which is a blind QC tool. Primary audience is a researcher (or an
AI assistant helping a researcher) exploring what happened in a video.

Reframed 2026-07-12 away from "confirmed splits" as the organizing concept: a spot-check
investigation ([[project_cell_split_counter_confirmed_high_reliability]]) found the
pipeline's own high-confidence tier agrees with a human reviewer 0-25% of the time, so
`ai_confidence` is a sort key, never a trust signal or an auto-skip filter. Recall over
precision -- the 2-second human glance is the real correctness backstop, this tool's job
is just to shrink the haystack without dropping needles she'd have caught given unlimited
attention (matches the researcher's actual triage workflow: high volume, throw away if not
interesting, risk of missing something before discarding and having to retry -- see
[[project_cell_split_counter_interesting_events]]).

Death handling follows the maintainer's 2026-07-12 guidance -- death is not a flat in/out
toggle: plain death is only "mildly interesting" (excluded from the default "Interesting
only" view but still visible with it unchecked); death traceable to an earlier
micronucleus-flagged division of the same track is treated as tier-1 interesting
(micronucleus is a known precursor to death); a death the vision model called a likely
tracking dropout during division is flagged separately as a probable missed division, not
a death, since review_deaths() already makes that call. Distinguishing "drifted out of
the z-axis" (not interesting) from "died in-plane" (interesting) is NOT implemented --
review_deaths()'s prompt has no such option yet -- so that distinction cannot be surfaced
here; treat plain "mildly interesting" deaths as an unfiltered mix of both until that's
built.

Position marker (2026-07-13, real-usage feedback): the saved review_crops/ PNGs are
deliberately unmarked (a researcher may want a clean copy for her own reports/slides), but
crowded frames turned out to be ambiguous for a human reviewer too, not just the AI --
"which cell is the candidate" wasn't always obvious. A faint corner-bracket overlay
(toggleable, on by default) is drawn only in this browser view via CSS, never baked into
the underlying image files, using the same neighbor-aware adaptive_radius() the AI's own
marker uses so it points at the same place the AI was told to look.

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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.reports._crop_shared import CROP_RADIUS as _CROP_RADIUS
from scripts.reports._crop_shared import centroid_in_crop_pct as _centroid_in_crop_pct
from scripts.reports._crop_shared import frame_idx_from_name as _frame_idx_from_name
from scripts.reports._crop_shared import sampled_only as _sampled_only
from src.review import adaptive_radius as _adaptive_radius

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


def _is_split_type_mismatch(row: dict) -> bool:
    """Tracker topology said normal_split (2 children) but the model visually saw 3+
    daughters -- see docs/output_schema.md's multi_way undercounting gotcha (2026-07-09)."""
    return row.get("split_type") == "multi_way" and row.get("split_topology") == "normal_split"


def _interest_score(row: dict) -> tuple[int, float]:
    """Return (tier_score, confidence) for sorting. Higher = more interesting.

    row may carry a synthetic "_micronucleus_history" key (see _build_manifest) --
    death-only, never present in the raw CSV.
    """
    conf_col = _conf_col(row)
    conf = float(row.get(conf_col) or 0)
    acd = (row.get("acd_division_type") or "").lower()
    near = row.get("near_edge") == "1"
    active_flags = [f for f in _FLAG_COLS if row.get(f) == "1"]
    # anaphase_bridge alone no longer counts as a corroborated anomaly (2026-07-13: a real
    # human-annotated sample found 0/4 agreement on solo bridge calls -- already documented
    # as the least reliable flag; see generate_package_readme.py's "least reliable
    # abnormality flag" note). Still shown as an informational chip on the card -- this only
    # stops it from single-handedly promoting an event into the top interest tier. Any OTHER
    # flag, or bridge alongside another flag, still counts fully.
    other_flag = any(f != "anaphase_bridge" for f in active_flags)
    is_failed = row.get("split_topology") == "failed_split"
    mismatch = _is_split_type_mismatch(row)
    has_anomaly_note = bool((row.get("anomaly_notes") or "").strip())
    # The freeform anomaly_notes field is usually the model explaining its OWN flag/failed
    # call, not an independent second observation -- so it only counts as its own
    # (unchanged, pre-existing) "generic note" tier-1 path when there's no structured flag
    # AND no failed-split call already explaining it; it must NOT be used to "corroborate"
    # a solo bridge flag or a failed-split call, or the demotions above would be silently
    # defeated (found this exact loophole while validating the fix against real annotations
    # -- the model always writes some explanatory text alongside a flag/failed call).
    generic_note_only = has_anomaly_note and not active_flags and not is_failed
    # is_failed_split alone no longer earns top-tier status either (2026-07-13: same sample,
    # 0/5 agreement that a "failed" split_type call was a real failed division) -- still
    # shown as an informational chip, still promoted if corroborated by an actual OTHER
    # anomaly flag (not by its own explanatory text -- see note above).
    is_death = row.get("split_topology") == "death"

    if is_death:
        # Deaths never carry ai_confidence in the "trustworthy real/FP" sense splits do --
        # classification_source == "rule" means classify_track_ends's persistence score
        # only, no vision review at all yet (see review_deaths()'s docstring). Score
        # per the maintainer's 2026-07-12 death/micronucleus/dropout guidance, not confidence.
        reviewed = row.get("classification_source", "rule") != "rule"
        dropout = row.get("likely_division_dropout")
        if row.get("_micronucleus_history"):
            score = 40      # death preceded by micronucleus -- genuinely interesting
        elif not reviewed:
            score = 3       # rule-only guess, ~most turn out to be tracking dropouts (see
                             # [[project_cell_split_counter_interesting_events]] M5 finding:
                             # 139/194 rule-only deaths were dropouts) -- not trustworthy
        elif dropout == "1":
            score = 15      # probable missed division, not a death -- a pipeline gap worth
                             # a look, but not new biology
        else:
            score = 8       # plain reviewed real death -- "mildly interesting" per the maintainer
        if near:
            score -= 3
        return score, conf

    if conf <= 0:
        score = 5          # false positive / unconfirmed
    elif (other_flag or generic_note_only) and conf >= 0.5:
        score = 40         # Tier 1: anomaly-flagged + confirmed
    elif mismatch or (is_failed and other_flag):
        score = 35         # Tier 1b: failed division (only when corroborated by a real
                            # OTHER anomaly flag), or tracker undercounted a multi-way split
                            # (2026-07-09 -- biologically/correctness interesting on its own,
                            # independent of confidence tier or ACD geometry)
    elif acd in ("tripolar", "multipolar"):
        score = 30         # Tier 2: abnormal geometry
    elif conf >= 0.5:
        score = 20         # Tier 3: normal confirmed
    else:
        score = 10         # Tier 4: low confidence
    if near:
        score -= 3         # deprioritize near-edge within tier
    return score, conf


_DISPLAY_MARKER_SCALE = 1.7  # browser overlay sits further from the cell than the AI's own
                              # marker (2026-07-13 feedback) -- the AI's radius has to stay
                              # tight enough to disambiguate crowded neighbors within a small
                              # token-budget image; a human just needs orientation, so this
                              # pushes the same neighbor-aware radius outward without
                              # changing what's actually sent to the AI.


def _marker_radius_pct(row: dict) -> float:
    """Position marker radius, as a % of the crop's width, for the faint browser-side
    overlay (2026-07-13, real-usage feedback -- crowded frames are ambiguous to a human
    reviewer too, not just the AI, but a marker baked into the saved PNG would ruin a
    clean copy for a researcher's own report/presentation). Starts from adaptive_radius()
    -- the exact same neighbor-aware formula src/review.py used to position the AI's own
    marker, so it stays proportionally far from a close neighbor rather than a fixed
    offset that could land on top of one -- then scaled further out for display only."""
    def _f(key: str) -> float | None:
        v = row.get(key)
        return float(v) if v not in (None, "") else None

    radius_px = _adaptive_radius(
        _f("neighbor_distance_px"), cell_area_px=_f("cell_area_px"), neighbor_area_px=_f("neighbor_area_px"),
    )
    return (radius_px * _DISPLAY_MARKER_SCALE / (2 * _CROP_RADIUS)) * 100


def _birth_micronucleus_by_track(rows: list[dict]) -> dict[str, bool]:
    """Map track_id -> True if that track's OWN birth event (the split row where it first
    appears as a new daughter track_id) had micronucleus flagged. Most tracks in a run have
    no birth-split row at all (alive since frame 0) and are simply absent from this dict --
    callers should treat a missing key as "no history available", not "no micronucleus"."""
    birth: dict[str, bool] = {}
    for r in rows:
        if r.get("split_topology") not in ("normal_split", "multi_way_split", "failed_split"):
            continue
        tid = r.get("track_id")
        if tid and tid not in birth:
            birth[tid] = r.get("micronucleus") == "1"
    return birth


def _build_manifest(
    run_dir: Path,
    min_conf: float,
    include_fps: bool,
    thumb_zoom: int,
) -> list[dict]:
    rows = list(csv.DictReader(open(run_dir / "events.csv", encoding="utf-8", errors="replace")))

    # Deduplicate to one row per unique split point; keep every sibling too (folder lookup
    # below needs to try each daughter's track_id, not just the first row's).
    by_split: dict[tuple, dict] = {}
    siblings_by_split: dict[tuple, list[dict]] = {}
    for r in rows:
        # failed_split included 2026-07-09 -- previously excluded entirely, meaning a real,
        # confirmed failed division was invisible in this tool despite being a distinct,
        # biologically interesting event type. Still excluded from the "confirmed splits"
        # count in main() below, since it's not a completed division.
        if r.get("split_topology") not in ("normal_split", "multi_way_split", "failed_split"):
            continue
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        by_split.setdefault(key, r)
        siblings_by_split.setdefault(key, []).append(r)

    deaths = [r for r in rows if r.get("split_topology") == "death"]
    micronucleus_history = _birth_micronucleus_by_track(rows)

    # Folders are keyed on track_id (changed 2026-07-12 -- see src/review.py's
    # _save_debug_crops docstring for why parent_id collided between splits and deaths).
    crops_dir = run_dir / "review_crops"
    folder_re = re.compile(r"^frame_(\d+)_track_(\d+)$")
    folder_by_track: dict[str, Path] = {}
    if crops_dir.exists():
        for d in crops_dir.iterdir():
            m = folder_re.match(d.name)
            if m:
                folder_by_track[m.group(2)] = d

    def _resolve_crops(candidate_track_ids: list[str], peak_frame: str, centroid_x: str, centroid_y: str) -> dict:
        empty = {"images": [], "dense_images": [], "has_dense": False, "crosshair_x_pct": 50.0, "crosshair_y_pct": 50.0}
        folder = None
        for tid in candidate_track_ids:
            folder = folder_by_track.get(tid)
            if folder is not None:
                break
        if folder is None:
            return empty
        all_imgs = sorted(folder.glob("*.png"))
        imgs = _sampled_only(all_imgs, int(peak_frame))
        if not imgs:
            return empty
        crosshair_x_pct, crosshair_y_pct = 50.0, 50.0
        try:
            crosshair_x_pct, crosshair_y_pct = _centroid_in_crop_pct(
                imgs[0],
                float(centroid_x or _CROP_RADIUS),
                float(centroid_y or _CROP_RADIUS),
            )
        except Exception:
            pass
        images = [f"../review_crops/{folder.name}/{p.name}" for p in imgs]
        # Every consecutive frame in the review window is saved to disk (_build_dense_debug_
        # window, src/review.py) even though the AI only sees the stride-sampled subset above --
        # exposing it here lets a researcher ask for more temporal context around a slow/subtle
        # event without any new crop generation (2026-07-12 real-usage feedback: repeated
        # requests for "more frames" that were already sitting on disk, just not surfaced).
        dense_names = {p.name for p in imgs}
        dense_images = [
            {"src": f"../review_crops/{folder.name}/{p.name}", "sampled": p.name in dense_names}
            for p in all_imgs if p.name != "verdict.txt" and _frame_idx_from_name(p.name) is not None
        ]
        has_dense = len(dense_images) > len(images)
        return {
            "images": images, "dense_images": dense_images, "has_dense": has_dense,
            "crosshair_x_pct": crosshair_x_pct, "crosshair_y_pct": crosshair_y_pct,
        }

    events = []
    for (parent_id, peak_frame), row in by_split.items():
        conf_col = _conf_col(row)
        notes_col = _notes_col(row)
        conf = float(row.get(conf_col) or 0)

        if conf < min_conf and not include_fps:
            continue

        candidate_ids = [s.get("track_id", "") for s in siblings_by_split.get((parent_id, peak_frame), [row])]
        crops = _resolve_crops(candidate_ids, peak_frame, row.get("centroid_x"), row.get("centroid_y"))

        flags = [_FLAG_LABELS[f] for f in _FLAG_COLS if row.get(f) == "1"]
        acd = row.get("acd_division_type") or ""
        score, _ = _interest_score(row)

        events.append({
            "event_kind": "split",
            "entry_key": f"split_{row.get('track_id', '')}_{peak_frame}",
            "parent_id": parent_id,
            "track_id": row.get("track_id", ""),
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
            "is_death": False,
            "death_reviewed": False,
            "likely_missed_division": False,
            "micronucleus_history": False,
            "images": crops["images"],
            "dense_images": crops["dense_images"],
            "has_dense": crops["has_dense"],
            "has_crops": len(crops["images"]) > 0,
            "crosshair_x_pct": round(crops["crosshair_x_pct"], 2),
            "crosshair_y_pct": round(crops["crosshair_y_pct"], 2),
            "marker_radius_pct": round(_marker_radius_pct(row), 2),
            "interest_score": score,
        })

    for row in deaths:
        track_id = row.get("track_id", "")
        peak_frame = row.get("peak_frame", "")
        reviewed = row.get("classification_source", "rule") != "rule"
        dropout = row.get("likely_division_dropout") == "1"
        has_micro_history = bool(micronucleus_history.get(track_id))

        # Score against a copy carrying the synthetic history flag -- keeps _interest_score
        # ignorant of how that flag gets computed (birth-split lookup lives here, not there).
        scored_row = dict(row)
        scored_row["_micronucleus_history"] = has_micro_history
        score, conf = _interest_score(scored_row)

        crops = _resolve_crops([track_id], peak_frame, row.get("centroid_x"), row.get("centroid_y"))

        events.append({
            "event_kind": "death",
            "entry_key": f"death_{track_id}_{peak_frame}",
            "parent_id": row.get("parent_id") or "",
            "track_id": track_id,
            "peak_frame": peak_frame,
            "confidence": conf,
            "raw_ai_confidence": row.get("raw_ai_confidence") or None,
            "acd_division_type": "",
            "flags": [],
            "near_edge": row.get("near_edge") == "1",
            "bleach_risk": row.get("bleach_risk") or None,
            "classification_source": row.get("classification_source") or "",
            "ai_notes": row.get("ai_notes") or "",
            "anomaly_notes": row.get("anomaly_notes") or "",
            "review_error": row.get("review_error") == "1",
            "split_topology": "death",
            "split_type": "",
            "is_failed_split": False,
            "split_type_mismatch": False,
            "is_death": True,
            "death_reviewed": reviewed,
            "likely_missed_division": reviewed and dropout,
            "micronucleus_history": has_micro_history,
            "images": crops["images"],
            "dense_images": crops["dense_images"],
            "has_dense": crops["has_dense"],
            "has_crops": len(crops["images"]) > 0,
            "crosshair_x_pct": round(crops["crosshair_x_pct"], 2),
            "crosshair_y_pct": round(crops["crosshair_y_pct"], 2),
            "marker_radius_pct": round(_marker_radius_pct(row), 2),
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
.death-badge{background:var(--critical-wash);color:var(--critical);}
.flag-chip{display:inline-block;font-size:11px;padding:2px 7px;border-radius:4px;background:var(--tier1-wash);color:var(--tier1);margin-right:4px;margin-bottom:4px;white-space:nowrap;}
.near-edge-chip{background:var(--warning-wash);color:var(--warning);}
.error-chip{background:var(--critical-wash);color:var(--critical);}
.filmstrip{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.crop-wrap{position:relative;display:inline-block;line-height:0;border-radius:4px;overflow:hidden;border:1px solid var(--border);cursor:zoom-in;flex-shrink:0;}
.crop-thumb{display:block;width:120px;height:120px;background-repeat:no-repeat;background-color:var(--border);}
.crop-thumb.loading{background-image:none!important;}
.crop-wrap.dense-skipped{opacity:0.5;}
.crop-wrap.dense-sampled{border-color:var(--good);border-width:2px;}
.dense-legend{font-size:11.5px;color:var(--text-muted);margin:0 0 8px;}
.marker-tick{position:absolute;width:10px;height:10px;border-color:rgba(230,170,60,0.75);border-style:solid;border-width:0;pointer-events:none;}
.marker-tick.tl{border-top-width:1.5px;border-left-width:1.5px;}
.marker-tick.tr{border-top-width:1.5px;border-right-width:1.5px;transform:translateX(-100%);}
.marker-tick.bl{border-bottom-width:1.5px;border-left-width:1.5px;transform:translateY(-100%);}
.marker-tick.br{border-bottom-width:1.5px;border-right-width:1.5px;transform:translate(-100%,-100%);}

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
.lightbox-dense-toggle{position:absolute;top:16px;left:20px;color:white;font-size:12.5px;font-weight:600;cursor:pointer;background:rgba(255,255,255,0.12);border:none;border-radius:6px;padding:7px 12px;z-index:21;}
.lightbox-dense-toggle:hover{background:rgba(255,255,255,0.22);}
.lightbox-dense-toggle.hidden{display:none;}
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
    total_confirmed_splits: int,
    total_deaths: int,
    thumb_zoom: int,
) -> str:
    storage_key = f"researcher_{run_name}"
    # "confirmed" here means the AI's own verdict, not a human-verified split -- see
    # [[project_cell_split_counter_confirmed_high_reliability]], the pipeline's high-
    # confidence tier was found to agree with a human reviewer 0-25% of the time.
    subtitle = (
        f"{run_name} · {total_confirmed_splits} AI-confirmed splits · "
        f"{total_deaths} death events · {len(manifest)} total events"
    )

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
    <label title="Anomaly-flagged/failed/mismatched split, abnormal geometry, or a death traceable to an earlier micronucleus-flagged division — same tiering that colors the card border. Plain reviewed deaths and low-confidence splits are 'mildly interesting' and stay hidden here on purpose."><input type="checkbox" id="filter-interesting-only" checked> Interesting only</label>
  </div>

  <div class="filter-group">
    <label>Cell death events</label>
    <label><input type="checkbox" id="filter-show-deaths" checked> Show death events</label>
    <label><input type="checkbox" id="filter-hide-unreviewed-deaths" checked> Hide not-yet-reviewed deaths</label>
  </div>

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
    <label title="Every consecutive frame in the review window is saved on disk even though the AI only sees every 3rd -- this shows all of them, dimming the ones the AI didn't see."><input type="checkbox" id="filter-dense-mode"> Show every frame (more context)</label>
    <label title="Faint corner ticks positioned away from the candidate cell -- an overlay drawn only in this view, never saved into the underlying PNG files, so the raw crop on disk stays clean for reports/presentations."><input type="checkbox" id="filter-show-marker" checked> Show position marker</label>
  </div>

  <button class="export-btn" id="export-btn">Export researcher_notes.csv</button>

  <div class="stats" id="stats"></div>
</div>

<div class="main">
  <div class="main-header">
    <h1>Researcher browser</h1>
    <p class="subtitle">{subtitle} · <span id="header-shown-count">{len(manifest)}</span> shown with current filters</p>
  </div>
  <div class="grid" id="grid"></div>
</div>

<div class="lightbox" id="lightbox">
  <button class="lightbox-close" id="lightbox-close">&times;</button>
  <button class="lightbox-dense-toggle" id="lightbox-dense-toggle"></button>
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
  var lbDenseToggle = document.getElementById('lightbox-dense-toggle');
  var lbImages = [], lbIdx = 0, lbEvent = null, lbDenseMode = false;

  function frameIdxFromSrc(src) {{
    var m = src.split('/').pop().match(/^\d+_(?:before|split|after)_(\d+)\.png$/);
    return m ? parseInt(m[1], 10) : null;
  }}
  function lbFrameList() {{
    if (lbDenseMode && lbEvent && lbEvent.has_dense) {{
      return lbEvent.dense_images.map(function(f) {{ return f.src; }});
    }}
    return lbEvent ? lbEvent.images : [];
  }}
  function openLightbox(ev, initialDense, startIdx) {{
    lbEvent = ev;
    lbDenseMode = !!(initialDense && ev.has_dense);
    lbImages = lbFrameList();
    lbIdx = Math.min(startIdx, lbImages.length - 1);
    lbDenseToggle.classList.toggle('hidden', !ev.has_dense);
    updateDenseToggleLabel();
    showLbFrame();
    lightbox.classList.add('open');
  }}
  function updateDenseToggleLabel() {{
    lbDenseToggle.textContent = lbDenseMode ? 'Show only AI-reviewed frames' : 'Show every frame';
  }}
  function showLbFrame() {{
    lbImg.src = lbImages[lbIdx];
    lbCaption.textContent = (lbIdx + 1) + ' / ' + lbImages.length + '  \u2014  ' + lbImages[lbIdx].split('/').pop();
    lbPrev.disabled = lbIdx === 0;
    lbNext.disabled = lbIdx === lbImages.length - 1;
  }}
  lbDenseToggle.addEventListener('click', function(e) {{
    e.stopPropagation();
    if (!lbEvent || !lbEvent.has_dense) return;
    var currentFrameIdx = frameIdxFromSrc(lbImages[lbIdx]);
    lbDenseMode = !lbDenseMode;
    lbImages = lbFrameList();
    if (currentFrameIdx !== null) {{
      var match = lbImages.findIndex(function(src) {{ return frameIdxFromSrc(src) === currentFrameIdx; }});
      lbIdx = match !== -1 ? match : 0;
    }} else {{
      lbIdx = 0;
    }}
    updateDenseToggleLabel();
    showLbFrame();
  }});
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
  var filterInterestingOnly = true;
  var filterShowDeaths = true;
  var filterHideUnreviewedDeaths = true;
  var denseMode = false;
  var showMarker = true;

  function passesFilter(ev) {{
    if (ev.event_kind === 'death') {{
      if (!filterShowDeaths) return false;
      if (filterHideUnreviewedDeaths && ev.classification_source === 'rule') return false;
    }}
    if (filterInterestingOnly && tierClass(ev) === '') return false;
    if (filterHideFps && ev.event_kind !== 'death' && ev.confidence <= 0) return false;
    if (filterHideNearEdge && ev.near_edge) return false;
    if (filterMinConf > 0 && ev.event_kind !== 'death' && ev.confidence < filterMinConf) return false;
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
      var a = annotations[ev.entry_key];
      if (!a || !a.notes) return false;
    }}
    if (filterFlaggedOnly) {{
      var a2 = annotations[ev.entry_key];
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
    if (ev.event_kind === 'death') {{
      if (ev.micronucleus_history) return 'tier1';
      return '';   // plain death (reviewed or not) and probable missed-division are both
                   // "mildly interesting" at most -- deliberately excluded from the
                   // default Interesting-only view, per the maintainer's 2026-07-12 guidance.
    }}
    // anaphase_bridge alone, and is_failed_split alone, no longer grant tier1 by themselves
    // (2026-07-13: 0/4 and 0/5 agreement respectively in a real human-annotated sample) --
    // must be corroborated by another flag or a split_type mismatch. anomaly_notes is
    // usually the model explaining its OWN flag/failed call, not independent evidence, so
    // it must NOT count as corroboration for either -- it only grants tier1 on its own
    // (unchanged, pre-existing "generic note" path) when there's no structured flag AND no
    // failed-split call already explaining it. Mirrors _interest_score's
    // other_flag/generic_note_only in the Python build step exactly.
    var otherFlag = ev.flags.some(function(f) {{ return f !== 'anaphase bridge'; }});
    var genericNoteOnly = !!ev.anomaly_notes && ev.flags.length === 0 && !ev.is_failed_split;
    if ((otherFlag || genericNoteOnly) && ev.confidence >= 0.5) return 'tier1';
    if (ev.split_type_mismatch) return 'tier1';
    if (ev.is_failed_split && otherFlag) return 'tier1';
    if (ev.acd_division_type === 'tripolar' || ev.acd_division_type === 'multipolar') return 'tier2';
    return '';
  }}

  function renderCard(ev) {{
    var ann = annotations[ev.entry_key] || {{}};
    var tier = tierClass(ev);
    var isDeath = ev.event_kind === 'death';
    var confBadge = isDeath ? '' : '<span class="conf-badge ' + confClass(ev.confidence) + '">' +
      (ev.confidence > 0 ? ev.confidence.toFixed(2) : 'FP') + '</span>';
    var acdBadge = ev.acd_division_type ? '<span class="acd-badge">' + ev.acd_division_type + '</span>' : '';
    var deathBadge = isDeath ? '<span class="acd-badge death-badge">death</span>' : '';
    var flagChips = ev.flags.map(function(f) {{ return '<span class="flag-chip">' + f + '</span>'; }}).join('');
    var nearChip = ev.near_edge ? '<span class="flag-chip near-edge-chip">near edge</span>' : '';
    var errChip = ev.review_error ? '<span class="flag-chip error-chip">review error</span>' : '';
    var failedChip = ev.is_failed_split ? '<span class="flag-chip near-edge-chip">failed division</span>' : '';
    var mismatchChip = ev.split_type_mismatch ? '<span class="flag-chip error-chip">split_type mismatch: model saw multi_way</span>' : '';
    var anomalyChip = ev.anomaly_notes ? '<span class="flag-chip">anomaly noted</span>' : '';
    var microHistChip = ev.micronucleus_history ? '<span class="flag-chip">micronucleus history</span>' : '';
    var missedDivChip = ev.likely_missed_division ? '<span class="flag-chip near-edge-chip">possible missed division, not a death</span>' : '';
    var unreviewedChip = (isDeath && !ev.death_reviewed) ? '<span class="flag-chip near-edge-chip">not yet vision-reviewed</span>' : '';

    var showDense = denseMode && ev.has_dense;
    var frames = showDense
      ? ev.dense_images
      : ev.images.map(function(src) {{ return {{src: src, sampled: true}}; }});

    var markerHtml = '';
    if (showMarker) {{
      var cx = ev.crosshair_x_pct, cy = ev.crosshair_y_pct, r = ev.marker_radius_pct;
      markerHtml =
        '<span class="marker-tick tl" style="left:' + (cx - r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick tr" style="left:' + (cx + r) + '%;top:' + (cy - r) + '%;"></span>' +
        '<span class="marker-tick bl" style="left:' + (cx - r) + '%;top:' + (cy + r) + '%;"></span>' +
        '<span class="marker-tick br" style="left:' + (cx + r) + '%;top:' + (cy + r) + '%;"></span>';
    }}

    var filmstrip = '';
    if (frames.length > 0) {{
      filmstrip = frames.map(function(f, i) {{
        var src = f.src;
        var label = src.split('/').pop().replace(/^\d+_/, '').replace(/_\d+\.png$/, '');
        var denseCls = showDense ? (f.sampled ? ' dense-sampled' : ' dense-skipped') : '';
        return '<span class="crop-wrap' + denseCls + '" data-idx="' + i + '" title="' + label + '">' +
          '<span class="crop-thumb loading" data-bg="' +
          'background-image:url(' + src + ');' +
          'background-position:' + ev.crosshair_x_pct + '% ' + ev.crosshair_y_pct + '%;' +
          'background-size:' + thumbZoom + '%;" style=""></span>' +
          markerHtml +
          '</span>';
      }}).join('');
    }} else {{
      filmstrip = '<p class="no-crops-note">No crop images available for this event.</p>';
    }}

    var denseLegend = showDense ? '<p class="dense-legend">green border = frame the AI actually reviewed; dimmed = extra context frame, not seen by the AI</p>' : '';

    var rawConf = ev.raw_ai_confidence ? ' (raw: ' + parseFloat(ev.raw_ai_confidence).toFixed(2) + ')' : '';
    var bleach = ev.bleach_risk ? ' · bleach risk: ' + parseFloat(ev.bleach_risk).toFixed(2) : '';
    var meta = 'frame ' + ev.peak_frame + ' · track ' + ev.track_id +
      ' · ' + (ev.classification_source || 'rule') + rawConf + bleach;

    var savedNote = ann.notes ? '<span style="color:var(--text-secondary);font-style:italic;">Saved: ' +
      ann.notes.substring(0, 80) + (ann.notes.length > 80 ? '…' : '') + '</span>' : '';

    return '<div class="card ' + tier + '" data-key="' + ev.entry_key + '">' +
      '<div class="card-header">' +
        '<span class="frame-label">Frame ' + ev.peak_frame + '</span>' +
        confBadge + acdBadge + deathBadge +
        '<span>' + flagChips + anomalyChip + microHistChip + missedDivChip + unreviewedChip + failedChip + mismatchChip + nearChip + errChip + '</span>' +
      '</div>' +
      '<div class="filmstrip">' + filmstrip + '</div>' +
      denseLegend +
      (ev.ai_notes ? '<div class="ai-notes"><b>AI verdict:</b> &ldquo;' + ev.ai_notes + '&rdquo;</div>' : '') +
      (ev.anomaly_notes ? '<div class="anomaly-notes"><b>&#9888; AI anomaly note:</b> ' + ev.anomaly_notes + '</div>' : '') +
      '<div class="meta-row">' + meta + '</div>' +
      '<div class="annotation-area">' +
        '<textarea placeholder="Researcher notes…" data-key="' + ev.entry_key + '">' +
          (ann.notes ? ann.notes.replace(/</g,'&lt;') : '') + '</textarea>' +
        '<label>' +
          '<input type="checkbox" class="followup-check" data-key="' + ev.entry_key + '"' +
          (ann.followup ? ' checked' : '') + '> Flag for follow-up' +
        '</label>' +
        '<div class="saved-note" id="saved-' + ev.entry_key + '">' + savedNote + '</div>' +
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
        var key = card.getAttribute('data-key');
        var ev = manifest.find(function(e) {{ return e.entry_key === key; }});
        if (!ev) return;
        if ((denseMode && ev.has_dense ? ev.dense_images.length : ev.images.length) > 0) {{
          openLightbox(ev, denseMode, parseInt(wrap.getAttribute('data-idx'), 10));
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
    grid.querySelectorAll('textarea[data-key]').forEach(function(ta) {{
      var key = ta.getAttribute('data-key');
      var timer = null;
      ta.addEventListener('input', function() {{
        clearTimeout(timer);
        timer = setTimeout(function() {{
          if (!annotations[key]) annotations[key] = {{}};
          annotations[key].notes = ta.value;
          saveAnnotations(annotations);
          var el = document.getElementById('saved-' + key);
          if (el) el.textContent = ta.value ? 'Saved.' : '';
        }}, 600);
      }});
    }});

    // follow-up checkboxes
    grid.querySelectorAll('.followup-check').forEach(function(cb) {{
      cb.addEventListener('change', function() {{
        var key = cb.getAttribute('data-key');
        if (!annotations[key]) annotations[key] = {{}};
        annotations[key].followup = cb.checked;
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
    var headerCount = document.getElementById('header-shown-count');
    if (headerCount) headerCount.textContent = visibleCount;
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
  document.getElementById('filter-interesting-only').addEventListener('change', function() {{
    filterInterestingOnly = this.checked; renderGrid();
  }});
  document.getElementById('filter-show-deaths').addEventListener('change', function() {{
    filterShowDeaths = this.checked; renderGrid();
  }});
  document.getElementById('filter-hide-unreviewed-deaths').addEventListener('change', function() {{
    filterHideUnreviewedDeaths = this.checked; renderGrid();
  }});
  document.getElementById('filter-dense-mode').addEventListener('change', function() {{
    denseMode = this.checked; renderGrid();
  }});
  document.getElementById('filter-show-marker').addEventListener('change', function() {{
    showMarker = this.checked; renderGrid();
  }});

  // --- Export ---
  document.getElementById('export-btn').addEventListener('click', function() {{
    var lines = ['track_id,parent_id,peak_frame,event_kind,researcher_notes,flagged_for_followup,ai_confidence,acd_division_type,anomaly_flags'];
    manifest.forEach(function(ev) {{
      var ann = annotations[ev.entry_key] || {{}};
      if (!ann.notes && !ann.followup) return;
      var notes = (ann.notes || '').replace(/"/g, '""');
      var flags = ev.flags.join('; ');
      lines.push([
        ev.track_id, ev.parent_id, ev.peak_frame, ev.event_kind,
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
    total_deaths = sum(1 for r in all_rows if r.get("split_topology") == "death")

    manifest = _build_manifest(run_dir, min_conf, include_fps, thumb_zoom)
    if not manifest:
        print(f"  [researcher_browser] no events matched the filter criteria in {run_dir}, skipping")
        return None

    out_path = out if out else run_dir / "reports" / "researcher_browser.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = _render_html(manifest, run_dir.name, confirmed, total_deaths, thumb_zoom)
    out_path.write_text(html, encoding="utf-8")

    anomaly_count = sum(1 for e in manifest if e["flags"] or e["anomaly_notes"])
    abnormal_geom = sum(1 for e in manifest if e["acd_division_type"] in ("tripolar", "multipolar"))
    failed_count = sum(1 for e in manifest if e["is_failed_split"])
    mismatch_count = sum(1 for e in manifest if e["split_type_mismatch"])
    death_reviewed = sum(1 for e in manifest if e["is_death"] and e["death_reviewed"])
    death_micro = sum(1 for e in manifest if e["micronucleus_history"])
    death_missed_div = sum(1 for e in manifest if e["likely_missed_division"])
    print(f"  [researcher_browser] wrote {out_path}")
    print(f"    {len(manifest)} events · {confirmed} confirmed splits · {total_deaths} deaths total")
    print(f"    splits: {anomaly_count} anomaly-flagged · {abnormal_geom} tripolar/multipolar · "
          f"{failed_count} failed_split · {mismatch_count} split_type mismatch")
    print(f"    deaths: {death_reviewed}/{total_deaths} vision-reviewed · "
          f"{death_micro} with micronucleus history (tier-1 interesting) · "
          f"{death_missed_div} flagged as probable missed division")
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
