"""CLI entrypoint: python main.py <video_path> [frame_step]."""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Claude's returned descriptions can contain non-ASCII characters (e.g. "->" as an
# actual arrow glyph). Windows defaults redirected stdout to the system codepage
# (cp1252), which can't encode them -- crashing mid-review with all prior work lost
# since events.csv is only written after review_ambiguous returns. Force UTF-8 with
# replacement instead of erroring.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.ingest import IngestConfig
from src.pipeline import run

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--tracker", choices=["iou", "trackastra"], default="trackastra")
    parser.add_argument("--tracker-mode", choices=["greedy", "ilp"], default="greedy", help="Trackastra linking mode (default: greedy; ilp catches compact/adjacent divisions greedy collapses, see docs/investigation_notes.md)")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame index to process (0-indexed)")
    parser.add_argument("--end-frame", type=int, default=None, help="last frame index (exclusive); default = all")
    parser.add_argument("--no-debug-crops", dest="debug_crops", action="store_false", default=True, help="skip saving review crops to data/review_crops/ (saved by default)")
    parser.add_argument("--reuse-masks", action="store_true", help="skip Cellpose and load existing memmaps from <frame-dir>/_memmap/")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0, help="Cellpose sensitivity knob; lower = more permissive (default 0.0, library default)")
    parser.add_argument("--flow-threshold", type=float, default=0.4, help="Cellpose sensitivity knob; higher = more permissive (default 0.4, library default)")
    parser.add_argument("--segmentation-model", choices=["cyto3", "nuclei"], default="cyto3", help="Cellpose pretrained model. cyto3 (default, generalist) or nuclei (dedicated nuclear-stain model -- faster and ~23%% fewer masks/frame than cyto3 on the 2026-07-07 benchmark, not yet scored against ground truth)")
    parser.add_argument("--pixel-size-um", type=float, default=None, help="µm/pixel override for cell_size_um2 in events.csv. Auto-detected from ND2 metadata when the source is .nd2 (varies per acquisition -- do not assume one project's value applies to another file). Required for AVI sources if you want cell_size_um2 populated; otherwise it's left blank.")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for events.csv and summary.json (default: data/output/<video filename stem>)")
    parser.add_argument("--frame-dir", type=Path, default=None, help="directory for extracted frames (default: <output-dir>/frames, so a run's frames/crops/events.csv all live together; pass a shared path like data/frames to reuse a cache across runs)")
    parser.add_argument("--vision-backend", choices=["claude", "gpt"], default="gpt", help="vision review model: gpt (default, Azure OpenAI, draws down Azure credit -- see src/review_gpt.py; with --gpt-reasoning-effort medium and the default --min-gpt-confidence floor, precision is close to/slightly better than Claude on the 2026-07-06/07 spike) or claude (Anthropic Claude Haiku, higher precision but more expensive)")
    parser.add_argument("--gpt-reasoning-effort", choices=["low", "medium", "high"], default="medium", help="GPT backend only. 'high' is currently unusable -- burns its token budget on hidden reasoning and fails ~68%% of calls (2026-07-07 finding)")
    parser.add_argument("--min-gpt-confidence", type=float, default=0.65, help="GPT backend only: verdicts below this self-reported confidence are downgraded to false_positive. 0.0 disables filtering. Raised from 0.5 to 0.65 on 2026-07-18 -- a repeats-based confirmation sweep (the original 0.85->0.5 sweep was later found confounded by GPT-5-mini's run-to-run non-determinism) found 0.65 with higher mean precision (0.214 vs 0.160) AND higher mean recall (0.600 vs 0.500) than 0.5, on 2 repeats x 24 golden events each -- not a strict precision/recall tradeoff point, dominates 0.5 on both axes in this sample. 0.85 remains clearly worse on recall (stable 0.200 across repeats, no overlap with either other floor). See scripts/eval_harness/README.md.")
    parser.add_argument("--no-review-deaths", dest="review_death_events", action="store_false", default=True, help="skip vision review of DEATH events (reviewed by default since 2026-07-10 -- see review_deaths in src/review.py; DEATH previously got no vision coverage at all, unlike splits)")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("data/output") / args.video_path.stem
    frame_dir = args.frame_dir or (output_dir / "frames")

    config = IngestConfig(video_path=args.video_path, frame_step=args.frame_step, roi=None)
    run(
        config,
        frame_dir=frame_dir,
        output_dir=output_dir,
        tracker=args.tracker,
        tracker_mode=args.tracker_mode,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_debug_crops=args.debug_crops,
        reuse_masks=args.reuse_masks,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        segmentation_model=args.segmentation_model,
        pixel_size_um=args.pixel_size_um,
        vision_backend=args.vision_backend,
        gpt_reasoning_effort=args.gpt_reasoning_effort,
        min_gpt_confidence=args.min_gpt_confidence,
        review_death_events=args.review_death_events,
    )
