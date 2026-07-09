"""Repeated-sampling validation of the v2 (thin, off-cell) marker design.

Why this exists: the 2026-07-05 investigation's "marker causes a regression on
subtle divisions" finding was only ever validated with single-shot A/B testing --
never with the repeated-sampling protocol that the SAME investigation later proved
necessary on these exact borderline cases (Claude's verdict is genuinely
non-deterministic run-to-run here). This gives the marker question the rigor it
never got, using a genuinely different marker design (thin corner brackets, well
clear of the cell, not the old bold solid ring) against the same three reference
cases used throughout that investigation.

Usage:
  python scripts/repeat_sample_marker_test.py --n 8
  python scripts/repeat_sample_marker_test.py --n 8 --cases parent_1246_neighbor_misattribution
  python scripts/repeat_sample_marker_test.py --n 8 --backend gpt   # validate on GPT-5-mini too --
                                                                     # every finding so far is Claude-only,
                                                                     # and master made GPT the pipeline default.

TODO / backlog (2026-07-08, not started): the `--n` samples per case/condition run
sequentially (see the plain `for i in range(args.n)` loop in main()), unlike
src/review.py's real review_ambiguous(), which fires its API calls concurrently via
ThreadPoolExecutor(max_workers=10) since these are network-bound, not CPU-bound. A
full sweep here (multiple cases x 3 modes x n=8) takes 20-45+ minutes serially for no
real reason -- parallelizing this loop the same way would make sweeps (e.g. trying
several radius-fraction/size-k values back to back) much cheaper to iterate on.
"""

import argparse
import base64
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# GPT-5-mini descriptions sometimes include characters (e.g. U+2011 non-breaking
# hyphen) that aren't representable in Windows' default cp1252 stdout encoding when
# redirected to a file, crashing print() mid-run. Force UTF-8 so a rare character
# in a model's own text never takes down an otherwise-successful multi-hour run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic
import cv2
from dotenv import load_dotenv

load_dotenv()

from src.review import _SYSTEM  # byte-identical prompt base, no drift from the real pipeline
from src.config import CLAUDE_MODEL, GPT_DEPLOYMENT

from marker_experiment import (  # noqa: E402
    REFERENCE_CASES,
    _crop,
    _draw_corner_ticks,
    _find_frame,
    _sequence_indices,
    adaptive_radius,
)

_MARKER_PROMPT_LINE = (
    "\n\nThe candidate cell is indicated by four corner brackets (thin, dim orange) "
    "in each image, positioned clear of the cell itself, not touching it. Evaluate "
    "only the cell centered within the brackets -- ignore division or anomaly "
    "activity on any other cell visible in the frame."
)

# 2026-07-08: modes are "unmarked" (no brackets), "marked" (fixed _TICK_RADIUS=55px,
# the original v2 design), and "adaptive" (radius shrunk to stay clear of the
# nearest known neighbor -- see adaptive_radius() in marker_experiment.py). Only
# cases with a "neighbor_distance_px" entry support "adaptive".
_MODES = ("unmarked", "marked", "adaptive")
_BACKENDS = ("claude", "gpt")


def _build_images(
    case: dict, mode: str, backend: str = "claude",
    radius_fraction: float = 0.5, radius_margin: float = 5.0, size_k: float = 1.3,
) -> list[dict]:
    cx, cy = case["centroid"]
    radius = None
    if mode == "adaptive":
        radius = adaptive_radius(
            case["neighbor_distance_px"], margin=radius_margin, fraction=radius_fraction,
            cell_area_px=case.get("cell_area_px"), size_k=size_k,
        )
    blocks = []
    missing = []
    for idx, label in _sequence_indices(case["local_frame"]):
        path = _find_frame(case["frames_dir"], idx)
        if path is None:
            missing.append(idx)
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        crop, x0, y0 = _crop(img, cx, cy)
        if mode in ("marked", "adaptive"):
            crop = _draw_corner_ticks(crop, cx - x0, cy - y0, radius=radius)
        ok, buf = cv2.imencode(".png", crop)
        data = base64.standard_b64encode(buf.tobytes()).decode()
        if backend == "gpt":
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        else:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    if not blocks:
        raise RuntimeError(
            f"0 images found for this case (missing indices: {missing}) -- refusing to call the API "
            f"with nothing to look at. Check frames_dir resolves correctly regardless of cwd."
        )
    if missing:
        print(f"  [warn] {len(missing)} frame(s) missing from sequence: {missing}", file=sys.stderr)
    return blocks


def _call_once(
    client, case: dict, mode: str, backend: str = "claude",
    radius_fraction: float = 0.5, radius_margin: float = 5.0, size_k: float = 1.3,
) -> dict:
    images = _build_images(
        case, mode, backend=backend, radius_fraction=radius_fraction,
        radius_margin=radius_margin, size_k=size_k,
    )
    system = _SYSTEM.format(before=8, after=8, stride=3)
    if mode in ("marked", "adaptive"):
        system += _MARKER_PROMPT_LINE
    text_note = {
        "type": "text",
        "text": f"Candidate split, {len(images)} frames shown across the review window.",
    }

    if backend == "gpt":
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT, max_completion_tokens=2000, reasoning_effort="medium",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": images + [text_note]},
            ],
        )
        text = response.choices[0].message.content.strip()
        return json.loads(text)

    response = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=512, system=system,
        messages=[{"role": "user", "content": images + [text_note]}],
    )
    text = response.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="repeated samples per case per condition")
    parser.add_argument("--cases", nargs="*", default=None, help="subset of REFERENCE_CASES keys to run (default: all)")
    parser.add_argument("--backend", choices=_BACKENDS, default="claude", help="vision backend to validate (default: claude)")
    parser.add_argument("--radius-fraction", type=float, default=0.5, help="fraction of neighbor_distance_px used for the adaptive radius (default: 0.5, Claude-tuned)")
    parser.add_argument("--radius-margin", type=float, default=5.0, help="px subtracted after applying radius-fraction (default: 5.0)")
    parser.add_argument("--size-k", type=float, default=1.3, help="radius floor = size_k * candidate cell's own Cellpose-derived radius, only for cases with cell_area_px (default: 1.3)")
    args = parser.parse_args()

    if args.backend == "gpt":
        from openai import AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise SystemExit("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set (.env)")
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set (.env)")
        client = anthropic.Anthropic(api_key=api_key)

    cases = {k: v for k, v in REFERENCE_CASES.items() if args.cases is None or k in args.cases}

    results = {}
    for name, case in cases.items():
        print(f"=== {name} (expected: {case['expected']}) ===")
        modes = ["unmarked", "marked"]
        if "neighbor_distance_px" in case:
            modes.append("adaptive")
        for condition in modes:
            verdicts = []
            for i in range(args.n):
                try:
                    r = _call_once(
                        client, case, condition, backend=args.backend,
                        radius_fraction=args.radius_fraction, radius_margin=args.radius_margin,
                        size_k=args.size_k,
                    )
                    verdicts.append({
                        "verdict": r.get("verdict"), "confidence": r.get("confidence"),
                        "description": r.get("description"),
                    })
                except Exception as exc:
                    verdicts.append({"verdict": f"ERROR:{type(exc).__name__}", "confidence": None, "description": None})
            counts = Counter(v["verdict"] for v in verdicts)
            results[(name, condition)] = verdicts
            print(f"  {condition:10s} n={args.n}: {dict(counts)}")
            for v in verdicts:
                print(f"    [{v['verdict']} {v['confidence']}] {v['description']}")
        print()

    suffix = (
        f"_f{args.radius_fraction}_m{args.radius_margin}_k{args.size_k}"
        if args.radius_fraction != 0.5 or args.radius_margin != 5.0 or args.size_k != 1.3 else ""
    )
    out_path = Path(__file__).resolve().parents[1] / f"repeat_sample_results_{args.backend}{suffix}.json"
    out_path.write_text(json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in results.items()}, indent=2
    ))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
