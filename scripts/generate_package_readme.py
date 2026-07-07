"""Generate a self-contained README.md + index.csv inside an output package directory.

Why this exists: an output package (events.csv + review_crops/ + optionally frames/)
is often handed to someone (or to *their* Claude session) with no other repo context --
no docs/, no memory of this project's history. Without this file, the reader has to
reverse-engineer what "confidence", "classification_source", or a review_crops folder
name even mean. This script writes both files directly into the package so they travel
with it.

Usage:
  python scripts/generate_package_readme.py data/output/nd2_m3_cloud
  python scripts/generate_package_readme.py data/output/nd2_m3_cloud --video-label "Bewo Batch2 M3"
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

COLUMN_DOCS = [
    ("event_id", "Row counter within this CSV only. No meaning across files/runs."),
    ("source_video", "Filename of the input video/ND2 file this row came from."),
    ("frame_range", "Approximation only: [peak_frame - 10, peak_frame]. NOT a measured "
        "metaphase-anaphase window -- just a fixed lookback for display/crops."),
    ("peak_frame", "The frame the split was actually detected at (first frame with 2+ "
        "separate daughter masks). 0-indexed."),
    ("centroid_x / centroid_y", "Pixel coordinates (raw frame space, no transform needed) "
        "of the dividing cell at peak_frame. Matches review_crops crop centers. Placed next "
        "to peak_frame since locating an event needs both together."),
    ("near_edge", "'1'/'0' -- centroid within ~100px of any frame boundary. Partial visibility "
        "at the image boundary produces messier/more uncertain classifications. Flag, don't "
        "exclude: near-edge splits still belong in total confirmed-split counts, but exclude "
        "them when computing abnormality-rate percentages (micronucleus %, etc.), since those "
        "need the whole cell visible."),
    ("cell_area_px", "The parent cell's Cellpose mask area (pixel count) at the split frame. "
        "Always populated regardless of pixel-size availability."),
    ("cell_size_um2", "cell_area_px converted to real units via the acquisition's µm/pixel, "
        "when known. Blank if pixel size is unknown -- auto-detected from ND2 metadata (varies "
        "per acquisition, never a fixed constant), or set via --pixel-size-um for AVI sources. "
        "Check summary.json's pixel_size_um field for what value (if any) this run used."),
    ("split_topology", "General event-type column (name predates non-split events): "
        "'normal_split' (1->2 daughters), 'multi_way_split' (1->3+), 'death' (track stopped "
        "away from the frame boundary before the video ended), or 'roi_exit' (track stopped "
        "near the frame boundary -- likely walked out of frame rather than died). death/"
        "roi_exit rows are rule-only (claude_confidence=1.0, never sent for vision review) -- "
        "v1 can't distinguish a real death from the tracker just losing the cell, so treat "
        "both as 'track ended here,' not a confirmed biological death. Split counts are "
        "purely geometric (how many daughters), distinct from acd_division_type. Fixed "
        "2026-07-07: a single-child node (track-ID continuation, not a real division) used "
        "to be mislabeled multi_way_split. Packages generated before that fix may still have "
        "singleton multi_way_split rows -- check sibling count (see Known limitations) before "
        "trusting multi_way_split counts in older packages."),
    ("track_id", "Track ID assigned to THIS daughter cell going forward."),
    ("parent_id", "Track ID of the cell that split to produce this daughter. Rows sharing "
        "the same (parent_id, peak_frame) are daughters of the same split event -- "
        "a normal split produces 2 rows, a multi-way split produces 3+."),
    ("claude_confidence", "0.0-1.0. For classification_source=claude: this is Claude's real/split "
        "confidence IF the verdict was 'real' -- it is forced to 0.0 whenever Claude's "
        "verdict was false_positive, even if Claude's raw confidence in that FP call was "
        "high. So claude_confidence=0.0 does not mean 'unreviewed' or 'low-confidence guess' -- "
        "it means Claude actively rejected this candidate. For classification_source=rule: "
        "this is a tracker-persistence heuristic (daughter masks surviving N frames), not a "
        "Claude judgment -- see tracker_persistence_score below, which is the same kind of "
        "number under a clearer name."),
    ("tracker_persistence_score", "The persistence-based score computed BEFORE Claude ever "
        "looked at the candidate (daughter masks surviving N frames / max frames) -- a tracker "
        "stability signal, not a probability that the split is real. Kept even after Claude "
        "review so you can compare tracker behavior vs. Claude's verdict. On its own this has "
        "NOT been a reliable real-vs-noise discriminator in crowded fields (real divisions and "
        "false positives can score similarly) -- don't filter on this column, filter on "
        "claude_confidence."),
    ("classification_source", "'claude' once Claude vision review ran on this event, 'rule' "
        "if it was auto-confirmed by the tracker without a Claude call."),
    ("claude_notes", "Claude's free-text description from the review call, populated "
        "regardless of verdict (real or false_positive)."),
    ("bleach_risk", "peak_frame / total_frames. Higher = later in the timelapse = more "
        "accumulated photobleaching. Treat acd_division_type / abnormality calls with more "
        "skepticism as this approaches 1.0."),
    ("acd_division_type", "'bipolar' / 'tripolar' / 'multipolar' -- spindle geometry, only "
        "populated for confirmed real events (claude_confidence > 0)."),
    ("misaligned_chromosomes / lagging_chromosome / anaphase_bridge / micronucleus / binucleation",
        "'1'/'0' flags from Claude's abnormality read, only populated for confirmed real "
        "events. NOTE: anaphase_bridge in particular has a history of over-calling on "
        "subtle nuclear indentations rather than real bridges -- treat it with more "
        "skepticism than the other four flags. binucleation (added 2026-07-07) is one cell "
        "body containing two nuclei that don't progressively separate -- distinct from "
        "split_topology=failed (which re-fuses back into one nucleus)."),
    ("anomaly_notes", "Claude's free-text notes on anything unusual about the event "
        "(artifacts, nearby debris, atypical morphology) beyond the four flagged categories."),
]

README_INTRO = """# {label} -- cell division analysis output

This folder is the output of an automated pipeline that detects cell division
(ACD) events in microscopy video/ND2 timelapses: segment cells per frame
(Cellpose), track them across frames (Trackastra), flag candidate splits, then
send ambiguous candidates to Claude for a real/false-positive verdict plus
division-type and abnormality classification (misaligned chromosomes, lagging
chromosome, anaphase bridge, micronucleus, binucleation).

**If you are an AI assistant reading this on someone's behalf:** the intended
workflow is that a human (a research scientist) reviews these candidate events
by eye -- this pipeline is a triage/discovery tool, not an automated counter.
Because of that, **missing a real division is costly (a permanently lost
finding) but a false positive that already made it into this folder is cheap**
(a few seconds to dismiss). Weigh recall above precision when summarizing or
filtering this data -- don't discard low-confidence rows without flagging them
as "worth a human look," and don't treat a `false_positive` Claude verdict as
final/infallible, especially for `anaphase_bridge` calls (see column notes below).

## What's in this folder
"""

FOLDER_DOCS = {
    "events.csv": "One row per **daughter cell** produced by a detected split candidate "
        "(a normal split = 2 rows sharing the same parent_id + peak_frame, multi-way = 3+). "
        "This includes BOTH Claude-confirmed real events and rejected false positives -- "
        "filter to `claude_confidence > 0` to get only confirmed divisions. See 'How to "
        "filter this data' and the column reference below.",
    "events_formatted.xlsx": "Same data as events.csv, reformatted as a real Excel Table "
        "(sortable/filterable columns, frozen header), plus a second `confirmed_splits` tab: "
        "one row per unique split (deduplicated by parent_id+peak_frame, confidence > 0 only), "
        "with an added `is_fallback_review` flag for the rare rows where the review step's error "
        "fallback fired instead of a real Claude verdict (see Known limitations). Regenerate "
        "anytime from the CSV via scripts/csv_to_xlsx.py -- this file is a derived convenience "
        "copy, not a second source of truth.",
    "review_crops": "Per-candidate before/split/after frame crops (centered on the dividing "
        "cell, ~384px) plus a verdict.txt with Claude's raw verdict/confidence/notes for "
        "that specific review call. Folder name format: `frame_<peak_frame, 5 digits>_parent_"
        "<parent_id>`. IMPORTANT: this is not guaranteed to cover every row in events.csv "
        "-- crop coverage can be partial or come from a different run than the current CSV. "
        "Absence of a crop folder for a given event does NOT mean the event is less real or "
        "less important; use index.csv (generated alongside this README) to see exactly "
        "which confirmed events do/don't have a crop available. Also note: verdict.txt's "
        "`confidence` field is Claude's raw confidence in whatever verdict it gave (real OR "
        "false_positive) -- this differs from events.csv's `claude_confidence` column, which "
        "is always 0.0 for false_positive rows regardless of Claude's raw confidence.",
    "frames": "Raw extracted video frames (one PNG per frame, 0-indexed to match peak_frame "
        "and centroid coordinates directly, no transform needed).",
    "summary.json": "Aggregate counts only (total_events, event_counts by type), plus a "
        "vision_usage block (api_calls, input/output tokens, estimated_cost_usd) on runs "
        "generated after 2026-07-06. Not present in every package.",
}


def _find_pairs(rows: list[dict]) -> dict[tuple, dict]:
    """One representative row per unique (parent_id, peak_frame) split point."""
    pairs = {}
    for r in rows:
        key = (r.get("parent_id", ""), r.get("peak_frame", ""))
        if key not in pairs:
            pairs[key] = r
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path, help="output package directory to document")
    parser.add_argument("--video-label", default=None,
                         help="human-readable label for the README title (defaults to source_video from the CSV)")
    args = parser.parse_args()

    package_dir: Path = args.package_dir
    events_csv = package_dir / "events.csv"
    if not events_csv.exists():
        raise SystemExit(f"no events.csv found at {events_csv}")

    with open(events_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    pairs = _find_pairs(rows)
    source_video = rows[0].get("source_video", "") if rows else ""
    label = args.video_label or source_video or package_dir.name

    real_pairs = {k: r for k, r in pairs.items() if r.get("claude_confidence") and float(r["claude_confidence"]) > 0}

    # Sibling count per split point, across ALL rows (not just confirmed) -- a genuine
    # normal_split always has exactly 2, a genuine multi_way_split should have 3+. If a
    # confirmed multi_way_split has only 1, it's not actually a multi-daughter division --
    # it's a single-child node mislabeled by classify.py's `else EventType.MULTI_WAY_SPLIT`
    # branch (catches len(children) == 1, not just 3+). Found 2026-07-06 on the aggressive-
    # threshold + gap-bridging run: every multi_way_split row in that run was a singleton.
    sibling_counts = Counter((r.get("parent_id", ""), r.get("peak_frame", "")) for r in rows)
    suspect_multiway = {
        k: r for k, r in real_pairs.items()
        if r.get("split_topology") == "multi_way_split" and sibling_counts[k] == 1
    }
    acd_counts = Counter(r.get("acd_division_type") or "unclassified" for r in real_pairs.values())
    abn_labels = {
        "misaligned_chromosomes": "misaligned_chromosomes",
        "lagging_chromosome":     "lagging_chromosome",
        "anaphase_bridge":        "anaphase_bridge",
        "micronucleus":           "micronucleus",
        "binucleation":           "binucleation",
    }
    abn_counts = {label_: sum(1 for r in real_pairs.values() if r.get(col) == "1")
                  for col, label_ in abn_labels.items()}

    review_crops_dir = package_dir / "review_crops"
    crop_folders = set()
    if review_crops_dir.exists():
        crop_folders = {p.name for p in review_crops_dir.iterdir() if p.is_dir()}

    def _crop_name(r: dict) -> str:
        return f"frame_{int(r['peak_frame']):05d}_parent_{r['parent_id']}"

    present_dirs = [name for name in ["events.csv", "events_formatted.xlsx", "review_crops", "frames", "summary.json"]
                    if (package_dir / name).exists()]

    lines = [README_INTRO.format(label=label)]
    for name in present_dirs:
        doc = FOLDER_DOCS.get(name, "")
        lines.append(f"- **`{name}`** -- {doc}")
    lines.append("")

    lines.append("## How to filter this data\n")
    lines.append("**Filter on `claude_confidence > 0` to get confirmed real events -- don't "
                  "sweep a numeric threshold like >=0.5 or >=0.7.** There is no validated "
                  "cutoff stricter than 0>0 for this pipeline version: `tracker_persistence_score` "
                  "(the pre-Claude signal) has been shown NOT to separate real divisions from "
                  "noise reliably in crowded fields, and Claude's own confidence *value* (0.75 "
                  "vs 0.85 vs 0.92) varies run-to-run by several points since there's no fixed "
                  "temperature/seed -- it isn't precise enough to threshold on. If you want to "
                  "prioritize which confirmed events to look at first, sort by claude_confidence "
                  "descending or by bleach_risk, don't filter by a cutoff.")
    lines.append("")
    lines.append("**For simple column filtering/sorting (e.g. \"just show me confirmed events\"), "
                  "use `events_formatted.xlsx` in Excel** -- it already has AutoFilter dropdowns "
                  "on every column, no need to ask an AI assistant to regenerate a filtered copy "
                  "of the data for that.")
    lines.append("")
    lines.append("**If an AI assistant needs to do something `events_formatted.xlsx` can't "
                  "(cross-referencing columns, computing something derived), it should write the "
                  "result to a new, clearly-named file (e.g. `filtered_<criteria>.csv`) alongside "
                  "this one -- never overwrite `events.csv` in place.** `events.csv` is the ground "
                  "truth for this run; anything derived from it should be a separate file so it's "
                  "obvious which one is authoritative.")
    lines.append("")

    lines.append("## Summary stats (computed from events.csv at generation time)\n")
    lines.append(f"- {len(rows)} total daughter-cell rows, {len(pairs)} unique candidate split points")
    lines.append(f"- {len(real_pairs)} confirmed real events (claude_confidence > 0)")
    if suspect_multiway:
        n_clean = len(real_pairs) - len(suspect_multiway)
        lines.append(f"  - **{n_clean} are solid** (normal_split with 2 daughter rows, or multi_way_split with 3+)")
        lines.append(f"  - **{len(suspect_multiway)} are labeled multi_way_split but only have 1 recorded daughter row "
                      f"-- likely NOT a genuine 3+-daughter division, see Known limitations below. Don't report "
                      f"the headline confirmed count without this caveat.**")
    for acd_type, count in sorted(acd_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - {acd_type}: {count}")
    flagged_any = sum(1 for r in real_pairs.values() if any(r.get(c) == "1" for c in abn_labels))
    lines.append(f"- {flagged_any} confirmed events flagged with at least one abnormality")
    for label_, count in abn_counts.items():
        lines.append(f"  - {label_}: {count}")
    if review_crops_dir.exists():
        matched = sum(1 for r in real_pairs.values() if _crop_name(r) in crop_folders)
        lines.append(f"- review_crops covers {len(crop_folders)} folders total; "
                      f"{matched}/{len(real_pairs)} confirmed events have a matching crop folder "
                      f"(see index.csv for exactly which ones)")
    lines.append("")

    lines.append("## Column reference (events.csv)\n")
    lines.append("| Column | Meaning |")
    lines.append("|---|---|")
    for col, doc in COLUMN_DOCS:
        doc_escaped = doc.replace("|", "\\|")
        lines.append(f"| `{col}` | {doc_escaped} |")
    lines.append("")

    lines.append("## Known limitations\n")
    lines.append("- `frame_range` is a fixed lookback window, not a measured metaphase-anaphase window.")
    lines.append("- Slow/gradual divisions are systematically under-detected -- a division that takes many "
                  "frames to visibly separate can be missed entirely or mis-timed. If something looks like it "
                  "should have been caught and wasn't, that's a known failure mode, not necessarily a data error.")
    lines.append("- Events near the edge of the frame (partial cell visibility) tend to produce noisier "
                  "division-type/abnormality classifications -- treat those with extra skepticism.")
    lines.append("- `anaphase_bridge` is the least reliable abnormality flag; has a history of over-calling on "
                  "subtle nuclear indentations that aren't real bridges.")
    if suspect_multiway:
        lines.append(f"- **`split_topology == 'multi_way_split'` rows with only 1 sibling row are a known labeling "
                      f"bug, not a verified 3+-daughter division.** `classify_events` labels any split as "
                      f"multi_way_split whenever it doesn't have exactly 2 daughters -- that includes a node with "
                      f"only 1 recorded child, which isn't a division at all as far as the code's own bookkeeping "
                      f"goes. Best current guess (not fully confirmed): the gap-bridging fix can occasionally merge "
                      f"one real daughter into an unrelated track's \"continuation,\" leaving its true sibling "
                      f"looking like a lone child -- so these events may still be real, just mislabeled and missing "
                      f"a row, rather than fabricated. This run has {len(suspect_multiway)} such rows out of "
                      f"{sum(1 for r in real_pairs.values() if r.get('split_topology')=='multi_way_split')} total "
                      f"confirmed multi_way_split rows. Treat them as \"needs a human look,\" not as confirmed "
                      f"triple/quadruple divisions.")
    lines.append("")

    (package_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    index_path = package_dir / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parent_id", "peak_frame", "claude_confidence", "classification_source",
                          "acd_division_type", "abnormalities", "has_review_crop", "review_crop_path"])
        for (parent_id, peak_frame), r in sorted(pairs.items(), key=lambda kv: int(kv[0][1]) if kv[0][1] else 0):
            abn = ",".join(label_ for col, label_ in abn_labels.items() if r.get(col) == "1")
            crop_name = _crop_name(r) if peak_frame else ""
            has_crop = crop_name in crop_folders
            writer.writerow([parent_id, peak_frame, r.get("claude_confidence", ""), r.get("classification_source", ""),
                              r.get("acd_division_type", ""), abn,
                              "yes" if has_crop else "no",
                              f"review_crops/{crop_name}" if has_crop else ""])

    print(f"Wrote {package_dir / 'README.md'}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
