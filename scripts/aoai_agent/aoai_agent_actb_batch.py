"""Run the validated multi-turn gpt-5 verification agent (find_prophase_onset +
mandatory wide backward check) against a random 15-candidate sample from each ACTB
well's division pool (find_candidates(pool="division", sort_by="random", seed=42)),
30 candidates total. No ground-truth CSV exists for these wells, so this is an
exploratory scale test of the validated design, not a scored eval.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

sys.stdout.reconfigure(encoding="utf-8")

import cell_mcp_server as _cm
_cm.BUNDLE = Path("data/bundle")
from cell_mcp_server.tools_filmstrip import follow_cells_over_time, list_nearby_tracks, watch_location_over_time
from cell_mcp_server.tools_candidates import find_prophase_onset

load_dotenv(dotenv_path=".env")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview",
    max_retries=8,
    timeout=120.0,
)

# Rerun: only the 3 candidates that hit the API's 50-image cap in the first pass.
CANDIDATES = {
    "20251016_ACTB_M2_red": [
        (910, [916, 917], 262), (954, [966, 967], 278),
    ],
    "20251016_ACTB_M3": [
        (7275, [7304, 7305], 563),
    ],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "follow_cells_over_time",
        "description": (
            "Follow one cell, or a mother and her daughters, over time as close-up images. "
            "Give EITHER track_id (one cell) OR track_ids (a set -- crop centres on the mean "
            "position of whichever members are present each frame). Window defaults to a "
            "generous 120 min before / 180 min after the membership transition; pass "
            "start_frame/end_frame to override."),
        "parameters": {"type": "object", "properties": {
            "track_id": {"type": "integer"},
            "track_ids": {"type": "array", "items": {"type": "integer"}},
            "start_frame": {"type": "integer"}, "end_frame": {"type": "integer"},
            "centre_frame": {"type": "integer"},
            "before_min": {"type": "number"}, "after_min": {"type": "number"},
            "max_images": {"type": "integer"}, "crop_um": {"type": "number"},
            "marker": {"type": "boolean"},
        }},
    }},
    {"type": "function", "function": {
        "name": "list_nearby_tracks",
        "description": (
            "Every object segmented near a place and time -- free, text only, no images. "
            "Anchor on track_id (its last position/frame) or an explicit x/y/frame. Returns "
            "track_id | frames | n | area_um2 | dist_um | coexists_with -- two real "
            "daughters must COEXIST (overlapping frame spans)."),
        "parameters": {"type": "object", "properties": {
            "track_id": {"type": "integer"}, "x": {"type": "number"}, "y": {"type": "number"},
            "frame": {"type": "integer"}, "before_min": {"type": "number"},
            "after_min": {"type": "number"}, "radius_um": {"type": "number"},
            "new_only": {"type": "boolean"},
        }},
    }},
    {"type": "function", "function": {
        "name": "watch_location_over_time",
        "description": (
            "Watch a fixed PLACE over time (not a cell's own mask) -- useful when the mask "
            "breaks exactly when the event happens, and the ONLY way to look further back "
            "than any track's own start (a place has no lifetime limit, unlike a track id). "
            "Give x/y (full-frame pixels) or anchor_track_id."),
        "parameters": {"type": "object", "properties": {
            "start_frame": {"type": "integer"}, "end_frame": {"type": "integer"},
            "x": {"type": "number"}, "y": {"type": "number"},
            "anchor_track_id": {"type": "integer"}, "max_images": {"type": "integer"},
            "crop_um": {"type": "number"},
        }, "required": ["start_frame", "end_frame"]},
    }},
    {"type": "function", "function": {
        "name": "find_prophase_onset",
        "description": (
            "Cheap, text-only, no images: looks for where a mother's chromatin condensation "
            "actually BEGAN, earlier than her own track -- her first tracked frame is where "
            "the ID started, not necessarily where the biology did. Walks backward through "
            "short (<=10 min) id-hop gaps (segmentation dropout/re-acquisition) and scores "
            "each resolved frame on a brightness-rise/DNA-conservation signal. A RANKING "
            "SIGNAL, not a verdict -- confirm on the pixels via follow_cells_over_time. "
            "LIMITATION: only bridges SHORT tracking gaps. If it reports 'no predecessor "
            "resolves', that means the gap is too large for this mechanism -- it does NOT "
            "mean no earlier prophase exists. In that case you MUST fall back to "
            "watch_location_over_time, stepping backward well beyond any track's own start."),
        "parameters": {"type": "object", "properties": {
            "track_id": {"type": "integer", "description": "The mother track from the division candidate."},
        }, "required": ["track_id"]},
    }},
]

SYSTEM_PROMPT = (
    "You are an expert reviewer verifying candidate cell-division (mitosis) events from "
    "live-cell fluorescence microscopy timelapse, for the cell-split-counter pipeline. "
    "You have real tools -- follow_cells_over_time, list_nearby_tracks, "
    "watch_location_over_time, find_prophase_onset -- the same ones a human researcher uses "
    "in this MCP session. Investigate across as many turns as you need before deciding. Do "
    "not render a final verdict on the first turn unless already certain -- use at least "
    "one tool call first.\n\n"
    "MANDATORY BEFORE ANY false_positive VERDICT -- a real division's staging (prophase, "
    "metaphase) can occur HOURS before the frame where the tracker's link ends (the "
    "'peak_frame'/transition), especially when the mother track itself only spans a frame "
    "or two right at the transition. A short mother track is NOT evidence against division "
    "-- it may just mean the object wasn't segmented/tracked as its own id until close to "
    "the event. Before concluding false_positive, you MUST do BOTH of the following, not "
    "just one: "
    "(1) call find_prophase_onset on the mother track_id -- cheap, no images; "
    "(2) if it reports no result (or you remain unsure after confirming what it found), "
    "call watch_location_over_time anchored on the mother's own last known position, with "
    "start_frame set at least 50 frames before peak_frame -- look for chromatin "
    "condensation, a metaphase plate, or a cleavage furrow anywhere in that wider range "
    "before concluding nothing divided. Only render false_positive after you have done "
    "both and found no such evidence anywhere in the widened window.\n\n"
    "When done, respond with ONLY a JSON object (no tool call, no other text): "
    '{"verdict": "real" | "false_positive", "confidence": <float 0-1>, '
    '"split_type": "symmetric" | "asymmetric" | "multi_way" | "failed" | null, '
    '"notes": "<cite specific frames/tool calls that drove your verdict, including what the '
    'mandatory wide backward check found>"}'
)

TOOL_FNS = {
    "follow_cells_over_time": follow_cells_over_time,
    "list_nearby_tracks": list_nearby_tracks,
    "watch_location_over_time": watch_location_over_time,
    "find_prophase_onset": find_prophase_onset,
}


def _run_tool(name, args, well):
    args = dict(args)
    args["well"] = well
    fn = TOOL_FNS[name]
    result = fn(**args)
    if isinstance(result, str):
        return result, []
    text_parts, images = [], []
    for item in result:
        if getattr(item, "type", None) == "text":
            text_parts.append(item.text)
        elif getattr(item, "type", None) == "image":
            images.append(item.data)
    return "\n".join(text_parts), images


MAX_IMAGES_PER_REQUEST = 40  # Azure's hard cap is 50; leave headroom


def _trim_images(messages, keep_under=MAX_IMAGES_PER_REQUEST):
    """Collapse the OLDEST image-carrying user messages to a text placeholder once
    the running image count gets too close to the API's 50-image-per-request cap.

    Only touches the standalone `role=user` image messages this script appends
    after each tool call -- never the `role=tool` responses, which the API
    requires one-per-tool_call_id and which stay intact regardless. The model
    already reasoned about older images in its own prior-turn text (which stays
    in the conversation), so trimming the raw pixels for evidence it has already
    processed and moved past doesn't erase what it concluded -- it just frees
    budget for the newest tool calls, which are what actually drive the verdict.
    """
    image_msgs = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    total = sum(1 for m in image_msgs for part in m["content"] if part.get("type") == "image_url")
    if total <= keep_under:
        return
    for m in image_msgs:  # oldest first, since messages/image_msgs preserve order
        if total <= keep_under:
            break
        n_here = sum(1 for part in m["content"] if part.get("type") == "image_url")
        if n_here == 0:
            continue
        label = next((p["text"] for p in m["content"] if p.get("type") == "text"), "images")
        m["content"] = [{"type": "text",
                          "text": f"[{label} -- {n_here} image(s) omitted here to stay under "
                                  f"the API's image limit; see your own reasoning above for "
                                  f"what was concluded from them]"}]
        total -= n_here


def verify_one(well, parent_id, daughters, peak_frame, max_turns=9):
    user_prompt = (
        f"Candidate division to verify:\n"
        f"  well: {well}\n"
        f"  parent (mother) track_id: {parent_id}\n"
        f"  recorded daughter track_ids: {daughters}\n"
        f"  peak_frame: {peak_frame}\n\n"
        "This candidate is from a random, unscored sample of the well's division pool "
        "(find_candidates, no prior verdict) -- verify it from scratch using your tools."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]
    total_cost, PRICE_IN, PRICE_OUT = 0.0, 1.25, 10.00
    n_tool_calls = 0

    for turn in range(1, max_turns + 1):
        _trim_images(messages)
        resp = client.chat.completions.create(
            model="gpt-5", messages=messages, tools=TOOLS,
            max_completion_tokens=3000, reasoning_effort="low",
        )
        u = resp.usage
        total_cost += (u.prompt_tokens / 1e6) * PRICE_IN + (u.completion_tokens / 1e6) * PRICE_OUT
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            return {"turns": turn, "tool_calls": n_tool_calls, "cost": total_cost, "raw": msg.content}
        messages.append({"role": "assistant", "content": msg.content,
                          "tool_calls": [{"id": tc.id, "type": "function",
                                          "function": {"name": tc.function.name,
                                                       "arguments": tc.function.arguments}}
                                         for tc in tool_calls]})
        pending_images = []
        for tc in tool_calls:
            n_tool_calls += 1
            args = json.loads(tc.function.arguments or "{}")
            try:
                text, images = _run_tool(tc.function.name, args, well)
            except Exception as e:
                text, images = f"ERROR: {e}", []
            messages.append({"role": "tool", "tool_call_id": tc.id,
                              "content": (text or "(no text)") +
                              (f"\n\n[{len(images)} image(s) attached next message]" if images else "")})
            if images:
                pending_images.append((tc.function.name, args, images))
        if pending_images:
            content = []
            for name, args, images in pending_images:
                content.append({"type": "text", "text": f"Images from {name}({args}):"})
                for b64 in images:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            messages.append({"role": "user", "content": content})

    return {"turns": max_turns, "tool_calls": n_tool_calls, "cost": total_cost, "raw": None}


def main():
    results = []
    total_cost = 0.0
    t0 = time.time()
    all_events = [(well, *c) for well, cands in CANDIDATES.items() for c in cands]
    for i, (well, parent_id, daughters, peak_frame) in enumerate(all_events):
        print(f"\n{'='*10} [{i+1}/{len(all_events)}] {well} parent={parent_id} "
              f"daughters={daughters} frame={peak_frame} {'='*10}", flush=True)
        try:
            r = verify_one(well, parent_id, daughters, peak_frame)
        except Exception as e:
            print("  EXCEPTION:", e)
            r = {"turns": None, "tool_calls": None, "cost": 0.0, "raw": None}
        total_cost += r["cost"]

        verdict = split_type = confidence = None
        if r["raw"]:
            try:
                parsed = json.loads(r["raw"])
                verdict = parsed.get("verdict")
                split_type = parsed.get("split_type")
                confidence = parsed.get("confidence")
            except Exception:
                pass
        print(f"  -> turns={r['turns']} tool_calls={r['tool_calls']} cost=${r['cost']:.4f} "
              f"verdict={verdict} conf={confidence} split_type={split_type} "
              f"running_total=${total_cost:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

        results.append({
            "well": well, "parent_id": parent_id, "daughters": daughters,
            "peak_frame": peak_frame, "turns": r["turns"], "tool_calls": r["tool_calls"],
            "cost": r["cost"], "verdict": verdict, "confidence": confidence,
            "split_type": split_type, "raw": r["raw"],
        })

    out_df = pd.DataFrame(results)
    out_path = "aoai_actb_batch_results_rerun3.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n\n{'='*20} SUMMARY {'='*20}")
    print(out_df.groupby("well")["verdict"].value_counts(dropna=False))
    print(f"Total cost: ${total_cost:.3f}  |  n={len(out_df)}  |  "
          f"avg turns={out_df['turns'].mean():.1f}  |  saved to {out_path}")


if __name__ == "__main__":
    main()
