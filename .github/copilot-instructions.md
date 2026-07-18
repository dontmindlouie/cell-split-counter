Cell Split Counter — Copilot instructions

Quick commands
- Create & activate venv (Windows):
  python -m venv .venv
  .venv\Scripts\activate
- Install runtime deps:
  pip install -r requirements.txt
- Install dev deps (tests):
  pip install -r requirements-dev.txt
- Run full pipeline:
  python main.py --video data/raw/your_video.avi [--reuse-masks] [--classify-divisions] [--vision-backend claude|gpt]
- Run a single test (pytest):
  pytest tests/test_classify.py::test_some_behavior -q
  or by substring: pytest -k "substring" -q

Environment / API keys
- For Claude (default vision review): set ANTHROPIC_API_KEY in the environment or a .env file at project root.
- For GPT backend: set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.
- Use .env for local runs; main.py loads it via python-dotenv.

High-level architecture (big picture)
- Entry: main.py parses CLI and builds an IngestConfig.
- Ingest (src/ingest.py): extracts frames (ND2 via nd2 package or cv2 otherwise), optional ROI, and detects per-acquisition pixel size for cell_size_um2.
- Segment (src/segment.py): runs Cellpose (cyto3 or nuclei) per-frame. Two modes:
  - segment_all: returns per-frame CellMask objects (memory-friendly cropped masks).
  - segment_video_arrays: batch-eval to produce memmapped arrays (frames.dat, labels.dat) for Trackastra.
  Memmaps live under <frame-dir>/_memmap; use --reuse-masks to load them instead of re-running Cellpose.
- Track (src/track.py): two backends:
  - trackastra (link_frames_trackastra): preferred; uses Trackastra model, patches internal functions to avoid large RAM allocations and writes tracked masks to memmaps.
  - simple IoU linking (link_frames) for small runs.
  Outputs a list of TrackNode objects describing lineage (track_id, parent_id, frame, mask metadata).
- Classify (src/classify.py): rule-based logic to detect NORMAL_SPLIT, MULTI_WAY_SPLIT, FAILED_SPLIT, DEATH, and AMBIGUOUS using daughter persistence and cascade-noise suppression.
- Review (src/review.py / src/review_gpt.py): ambiguous split candidates are routed to vision review (Anthropic Claude by default, or Azure OpenAI GPT). review_ambiguous implements 3-tier routing: suppress (very low confidence), review (ambiguous), auto-confirm (high confidence). The review call returns a JSON verdict used to overwrite/annotate events.
- Output (src/output.py): write events.csv (20+ columns) and summary.json. scripts/package_events.py creates per-event folders with crops and an index README.

Key conventions & repo-specific patterns
- Memmap-first approach: large arrays (frames, labels, tracked_masks, frames_float32) are written to disk memmaps to avoid multi-GB RAM spikes. Never remove memmap code unless keeping equivalent memory-friendly behavior.
- Reuse segmentation via --reuse-masks: when running multiple passes or review-only work, reuse existing memmaps in <frame-dir>/_memmap to save time and GPU.
- Trackastra monkeypatches: trackastra internals are intentionally patched at runtime to avoid huge allocations (normalize, apply_solution_graph_to_masks, np.stack redirection). Copilot should not naively inline/remove these patches when refactoring.
- Review prompt & JSON schema: review.py contains the system prompt and expects a strict JSON response (verdict, confidence, split_type, acd_division_type, boolean flags). Keep parsing robust to code fences and minor model output noise.
- Confidence routing defaults: lower_threshold=0.05, upper_threshold=1.0. GPT backend has additional min_gpt_confidence filtering (default 0.85) and a gpt_reasoning_effort tune.
- NEAR_EDGE_MARGIN_PX = 100 flags near-edge events; pipeline drops DEATH events that are near-edge before output.
- Output columns & encoding: events.csv is UTF-8 (non-ASCII from model notes possible). Use write_events_csv to preserve ordering/format.

Tests & dev notes
- Tests run with pytest; requirements-dev.txt includes pytest.
- Single-test invocation examples shown above.

Files to inspect for deeper changes
- main.py, src/{ingest,segment,track,classify,review,review_gpt,output,config}.py
- scripts/ for packaging, reports, and helper utilities.
- data/ contains raw, frames, and output artifacts; review crops written to data/review_crops when enabled.

If updating code, prioritize preserving memmap patterns and review prompt schema. The README.md contains usage examples and benchmark notes — incorporate when producing CLI help or docs.

Created from README.md and sources under src/ (ingest, segment, track, classify, review, output, config). If you want targeted additions (CI commands, extra examples, or MCP server suggestions), say which area to expand.