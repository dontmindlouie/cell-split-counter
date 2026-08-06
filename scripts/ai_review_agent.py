"""Multi-turn AOAI agent that drives the researcher-facing MCP tools directly
(in-process, no stdio hop) to confirm mitosis candidates the way a human
reviewer does: find_candidates -> get_track_profile -> follow_cells_over_time
(as many times as it wants) -> annotate().

This is NOT the single-shot review.py/review_gpt.py path (crop -> one JSON
verdict, 20% precision on gpt-5-mini per review_gpt.py's docstring). The
hypothesis here is that giving the model the same interactive tools a human
uses -- and letting it look again, widen the window, check a sibling track --
recovers the accuracy the 8/12-scoring human-in-the-loop MCP eval sessions got
that single-shot classification never reached.

PILOT USAGE (one well, one candidate, cheapest possible run):
    uv run python scripts/ai_review_agent.py --well M12_RUES2 --limit 1

Output:
    - A verdict, if the model calls annotate(), lands in the SAME
      <bundle>/<well>/annotations.csv a human review session writes to --
      annotator is tagged "aoai-<deployment>-agent-v1" so it's trivially
      filterable/excludable, never confused with a human row.
    - A run manifest (turns, tokens, whether a verdict was reached) is written
      to data/ai_review_runs/<well>/<track_id>.json for tuning -- this is the
      audit trail, not the finding.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AzureOpenAI

from cell_mcp_server.tools_candidates import find_candidates
from cell_mcp_server.tools_filmstrip import follow_cells_over_time
from cell_mcp_server.tools_browse import get_track_profile
from cell_mcp_server.tools_output import annotate

RUN_LOG_DIR = Path("data/ai_review_runs")

# A stronger deployment than GPT_DEPLOYMENT (config.py's "gpt-5-mini", picked
# for cost on the single-shot path). Multi-turn tool-calling reliability and
# judgment matter more here than per-call cost -- override via --deployment.
DEFAULT_AGENT_DEPLOYMENT = "gpt-5"

# Hard stop on turns. A turn is one assistant message; each one that calls
# follow_cells_over_time can carry up to 12 images (MAX_IMAGES in
# cell_mcp_server/server.py), so 8 turns is already a lot of context to hold a
# judgment across -- this is a circuit breaker against the model looping
# (re-requesting the same window, "let me check once more" indefinitely), not
# a budget a well-behaved review should ever hit.
MAX_TURNS = 8

# Second circuit breaker, orthogonal to turns: a single turn that requests an
# unusually wide follow_cells_over_time window can burn tens of thousands of
# tokens in images alone. Track cumulative usage and cut the run off even if
# it's still under MAX_TURNS.
MAX_TOKENS_PER_CANDIDATE = 60_000

AGENT_SYSTEM_PROMPT = """\
You are reviewing time-lapse microscopy of dividing cells, using the same \
tools a human researcher uses. Your job: for each candidate given to you, \
decide whether the cell actually divides (mitosis) and record your verdict.

Ground rules:
- Look before you decide. Call follow_cells_over_time on a candidate at least \
once before forming a verdict; get_track_profile first if you want to know \
whether it's worth spending images on.
- If the window doesn't show the outcome, widen after_min rather than \
guessing -- on some cell lines the mitotic figure appears well after the \
frame where the tracker link changes.
- Call annotate() exactly once per candidate you reach a verdict on, with \
outcome_class one of: "divides", "does_not_divide", "undetermined". Use \
"undetermined" honestly rather than forcing a guess -- a false "divides" is \
worse than an honest "undetermined".
- Put your reasoning in annotate()'s notes field in one or two sentences \
(what you saw, which frame convinced you).
- Do not call annotate() more than once for the same track_id.
- When you have called annotate() for every candidate you were given, stop \
calling tools and reply with a one-line summary.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_track_profile",
            "description": "Free (no images). Shape/size/brightness sparkline for one track, to decide if it's worth spending images on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "well": {"type": "string"},
                    "track_id": {"type": "integer"},
                },
                "required": ["well", "track_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_cells_over_time",
            "description": "The main look. Close-up filmstrip of one cell or a mother+daughters set over time. See system prompt for widening guidance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "well": {"type": "string"},
                    "track_id": {"type": "integer", "description": "Follow one cell. Mutually exclusive with track_ids."},
                    "track_ids": {"type": "array", "items": {"type": "integer"}, "description": "Follow a mother+daughters member set (pass [mother_id] to include recorded daughters). Mutually exclusive with track_id."},
                    "centre_frame": {"type": "integer", "description": "Centre the window here (use find_candidates' cond_f for a member set)."},
                    "before_min": {"type": "number"},
                    "after_min": {"type": "number", "description": "Widen this if the outcome isn't visible yet."},
                },
                "required": ["well"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate",
            "description": "Record your verdict for one cell. Never call twice for the same track_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "well": {"type": "string"},
                    "track_id": {"type": "integer"},
                    "outcome_class": {"type": "string", "enum": ["divides", "does_not_divide", "undetermined"]},
                    "condensation_frame": {"type": "integer"},
                    "metaphase_frame": {"type": "integer"},
                    "anaphase_frame": {"type": "integer"},
                    "daughter_ids": {"type": "array", "items": {"type": "integer"}},
                    "notes": {"type": "string"},
                },
                "required": ["well", "track_id", "outcome_class"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_track_profile": get_track_profile,
    "follow_cells_over_time": follow_cells_over_time,
    "annotate": annotate,
}


def _to_openai_content(result) -> list[dict]:
    """Convert the MCP TextContent/ImageContent list a tool returns into
    OpenAI chat content blocks. Mirrors review_gpt.py's _load_image_block."""
    if isinstance(result, str):
        return [{"type": "text", "text": result}]
    blocks = []
    for item in result:
        kind = getattr(item, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": item.text})
        elif kind == "image":
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
            })
        else:
            blocks.append({"type": "text", "text": str(item)})
    return blocks


def run_agent_on_well(
    client: AzureOpenAI,
    deployment: str,
    well: str,
    limit: int,
    sort_by: str,
) -> dict:
    seed = find_candidates(well, pool="division", sort_by=sort_by, limit=limit)

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Well: {well}\nCandidates (from find_candidates, pool=division, "
                f"sort_by={sort_by}):\n{seed}\n\nReview each one."
            ),
        },
    ]

    total_tokens = 0
    annotated_tracks: set[int] = set()
    turn = 0
    stop_reason = "completed"

    for turn in range(1, MAX_TURNS + 1):
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        total_tokens += resp.usage.total_tokens
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break  # model summarized without more tool use -- done or gave up

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            name = tc.function.name
            try:
                if name == "annotate":
                    if args.get("track_id") in annotated_tracks:
                        result = f"Already annotated track_id {args.get('track_id')} this run -- skipped."
                    else:
                        args.setdefault("annotator", f"aoai-{deployment}-agent-v1")
                        notes = args.get("notes", "") or ""
                        args["notes"] = (notes + f" [agent turn={turn}]").strip()
                        result = TOOL_DISPATCH[name](**args)
                        annotated_tracks.add(args.get("track_id"))
                else:
                    result = TOOL_DISPATCH[name](**args)
            except Exception as e:  # noqa: BLE001 -- feed the error back to the model, don't crash the run
                result = f"ERROR calling {name}: {e}"

            content = _to_openai_content(result) if name != "annotate" and not isinstance(result, str) else [{"type": "text", "text": str(result)}]
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

        if total_tokens > MAX_TOKENS_PER_CANDIDATE:
            stop_reason = "token_budget_exhausted"
            messages.append({
                "role": "user",
                "content": (
                    "Token budget for this run is exhausted. Give your best verdict "
                    "now for any candidate you haven't annotated yet (use "
                    "'undetermined' if you're not sure), then stop."
                ),
            })
            # one last forced turn, then hard stop regardless of tool_calls
            resp = client.chat.completions.create(model=deployment, messages=messages, tools=TOOLS, tool_choice="auto")
            total_tokens += resp.usage.total_tokens
            msg = resp.choices[0].message
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "annotate":
                        args = json.loads(tc.function.arguments or "{}")
                        args.setdefault("annotator", f"aoai-{deployment}-agent-v1")
                        args["notes"] = (args.get("notes", "") + " [forced by token budget]").strip()
                        TOOL_DISPATCH["annotate"](**args)
                        annotated_tracks.add(args.get("track_id"))
            break
    else:
        stop_reason = "max_turns_exhausted"

    manifest = {
        "well": well,
        "sort_by": sort_by,
        "requested": limit,
        "annotated_track_ids": sorted(annotated_tracks),
        "turns_used": turn,
        "total_tokens": total_tokens,
        "stop_reason": stop_reason,
        "deployment": deployment,
        "timestamp": time.time(),
    }
    RUN_LOG_DIR.joinpath(well).mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_LOG_DIR / well / f"run_{int(manifest['timestamp'])}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--well", required=True, help="well name from list_wells()")
    ap.add_argument("--limit", type=int, default=1, help="candidates to review this run")
    ap.add_argument("--sort-by", default="condensation", help="find_candidates sort_by; 'condensation' targets 'did this cell divide'")
    ap.add_argument("--deployment", default=DEFAULT_AGENT_DEPLOYMENT)
    args = ap.parse_args()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise EnvironmentError("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY not set")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2025-04-01-preview", max_retries=8, timeout=90.0)

    manifest = run_agent_on_well(client, args.deployment, args.well, args.limit, args.sort_by)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
