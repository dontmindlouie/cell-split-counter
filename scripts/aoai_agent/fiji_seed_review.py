"""Reverse-direction workflow: you scan the raw .nd2 in Fiji yourself, jot down
rough (well, frame, x, y) sightings as you go, paste them in below, and this
resolves each to a track and hands it to the SAME validated gpt-5 agent (wide
window + mandatory find_prophase_onset check) used in aoai_agent_actb_batch.py --
but asks it to REFINE (stage timing: condensation/metaphase/anaphase/exit frames)
and analyze, not just verdict a pre-existing candidate. This is the human->AI
direction; aoai_agent_actb_batch.py is AI-found candidates for human review.

Paste sightings into SIGHTINGS below, one per line:
    <well> <frame> <x> <y> [optional free-text note]
Blank lines and lines starting with # are ignored.

--- Getting sightings OUT of Fiji's AutoMeasure and INTO the line format above ---

1. Analyze > Set Measurements: check Area, Mean gray value, Min & max gray value,
   Centroid, Stack Position. **UNCHECK "Scaled units"** -- nd2s opened via Bio-Formats
   carry a pixel-size calibration (Fiji's title bar shows e.g. "885.50x885.50 microns
   (2048x2048)"), so with "Scaled units" ON, X/Y/Area come out in MICRONS, but
   tracks.csv (and _nearest_detection below) work in raw PIXELS. Leaving it checked
   silently feeds the wrong-unit coordinate in -- usually caught by the 30um snap
   check below (garbage distance), but not always. Unchecking it avoids the whole
   problem: X/Y then come out in the same pixel space as the bundle.
2. Toolbar: long-press the (multi-)point tool icon, pick the plain **Point** tool
   (not Multi-point -- only Point's options dialog has Auto-Measure). Double-click
   its icon, check "Auto-Measure".
3. Click each cell of interest across frames. The Results window fills in as you go
   (Window menu if it doesn't appear, or force one with Ctrl+M).
4. Copy with Ctrl+A, Ctrl+C. This copies DATA ROWS ONLY, no header -- with the
   config above the column order is fixed:
       n | Area | Mean | Min | Max | X | Y | Ch | Frame
   e.g. `1  0.000  50  50  50  741  1440  2  194` -> X=741, Y=1440, Frame=194.
   (Pasting a screenshot of the Results window instead works too, and is more
   robust against Set Measurements drifting out of this exact config, since the
   header text is then visible.)
5. Convert each row to a SIGHTINGS line:
     - well: not in the Results table at all -- read it off the Fiji window title
       bar / the .nd2 filename.
     - frame: Fiji's "Frame" column is 1-based; the pipeline's frame numbering
       (tracks.csv, this script) is 0-based -- **subtract 1**.
     - x, y: the X/Y columns directly (already pixels per step 1).
   So Results row `1  0.000  50  50  50  741  1440  2  194` in well
   `20251016_ACTB_M2` becomes: `20251016_ACTB_M2 193 741 1440`
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

sys.stdout.reconfigure(encoding="utf-8")

import cell_mcp_server as _cm
_cm.BUNDLE = Path("data/bundle")
from cell_mcp_server.tools_filmstrip import (
    follow_cells_over_time, list_nearby_tracks, watch_location_over_time, _nearest_detection,
)
from cell_mcp_server.tools_candidates import find_prophase_onset
from cell_mcp_server.tools_output import show_cells_in_browser

load_dotenv(dotenv_path=".env")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-04-01-preview",
    max_retries=8,
    timeout=120.0,
)

# ---- PASTE SIGHTINGS HERE ----
SIGHTINGS = """
# well frame x y [note]
20251016_ACTB_M3 221 1399 524   already-confirmed test case, real division
"""

_NEAREST_SNAP_UM = 30.0  # if the nearest tracked object is farther than this from
                          # the pasted point, the click probably missed -- flag it
                          # rather than silently investigating the wrong cell.

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
            "track_id": {"type": "integer", "description": "The mother track to investigate."},
        }, "required": ["track_id"]},
    }},
]

SYSTEM_PROMPT = (
    "You are an expert reviewer analyzing a cell that a human researcher, scanning the raw "
    "microscopy footage directly in Fiji, thought looked like it might be dividing (mitosis) "
    "or otherwise worth flagging. Their (well, frame, x, y) is only approximate -- it has "
    "already been snapped to the nearest tracked object at that frame, given to you as a "
    "track_id. Your job is to REFINE and ANALYZE, not just verdict: investigate with your "
    "real tools (follow_cells_over_time, list_nearby_tracks, watch_location_over_time, "
    "find_prophase_onset -- the same ones a human researcher uses in this MCP session) across "
    "as many turns as you need, then report what is actually happening and, if it is a real "
    "division, the specific frame numbers for each stage you can identify.\n\n"
    "Do not render a final verdict on the first turn unless already certain -- use at least "
    "one tool call first. A real division's staging (prophase, metaphase) can occur HOURS "
    "before the frame the human happened to click on, and the mother's own track may only "
    "span a frame or two right at the transition -- a short mother track is NOT evidence "
    "against division. Before concluding false_positive, you MUST do BOTH: "
    "(1) call find_prophase_onset on the resolved track_id -- cheap, no images; "
    "(2) if it reports no result (or you remain unsure after confirming what it found), call "
    "watch_location_over_time anchored on the track's own last known position, with "
    "start_frame set at least 50 frames before the flagged frame -- look for chromatin "
    "condensation, a metaphase plate, or a cleavage furrow anywhere in that wider range "
    "before concluding nothing divided.\n\n"
    "When done, respond with ONLY a JSON object (no tool call, no other text): "
    '{"verdict": "real" | "false_positive" | "unsure", "confidence": <float 0-1>, '
    '"split_type": "symmetric" | "asymmetric" | "multi_way" | "failed" | null, '
    '"condensation_frame": <int|null>, "metaphase_frame": <int|null>, '
    '"anaphase_frame": <int|null>, "exit_frame": <int|null>, '
    '"daughter_track_ids": [<int>, ...] | null, '
    '"notes": "<cite specific frames/tool calls that drove your conclusion, including what '
    'the mandatory wide backward check found>"}'
)

TOOL_FNS = {
    "follow_cells_over_time": follow_cells_over_time,
    "list_nearby_tracks": list_nearby_tracks,
    "watch_location_over_time": watch_location_over_time,
    "find_prophase_onset": find_prophase_onset,
}

MAX_IMAGES_PER_REQUEST = 40


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


def _trim_images(messages, keep_under=MAX_IMAGES_PER_REQUEST):
    image_msgs = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    total = sum(1 for m in image_msgs for part in m["content"] if part.get("type") == "image_url")
    if total <= keep_under:
        return
    for m in image_msgs:
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


def parse_sightings(text):
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=4)
        well, frame, x, y = parts[0], int(parts[1]), float(parts[2]), float(parts[3])
        note = parts[4] if len(parts) > 4 else ""
        out.append((well, frame, x, y, note))
    return out


def resolve_sighting(well, frame, x, y):
    hit = _nearest_detection(well, frame, x, y)
    if hit is None:
        return None, f"nothing tracked in {well} f{frame} at all -- empty frame in tracks.csv?"
    track_id, dist_um = hit
    if dist_um > _NEAREST_SNAP_UM:
        return None, (f"nearest tracked object ({track_id}) is {dist_um:.1f} um away -- "
                       f"farther than the {_NEAREST_SNAP_UM} um snap radius, likely a miss")
    return track_id, f"snapped to track {track_id}, {dist_um:.1f} um away"


def refine_one(well, track_id, flagged_frame, note, max_turns=9):
    user_prompt = (
        f"Human-flagged sighting to investigate:\n"
        f"  well: {well}\n"
        f"  resolved track_id: {track_id}\n"
        f"  flagged frame: {flagged_frame}\n"
        + (f"  researcher's note: {note}\n" if note else "")
        + "\nThis was NOT found by the pipeline's own candidate detector -- a human spotted it "
        "directly in the raw footage. Investigate from scratch."
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
    sightings = parse_sightings(SIGHTINGS)
    if not sightings:
        print("No sightings pasted into SIGHTINGS -- nothing to do.")
        return

    results = []
    events_by_well = {}
    total_cost = 0.0
    t0 = time.time()

    for i, (well, frame, x, y, note) in enumerate(sightings):
        print(f"\n{'='*10} [{i+1}/{len(sightings)}] {well} f{frame} @({x:.0f},{y:.0f}) {'='*10}", flush=True)
        track_id, resolve_note = resolve_sighting(well, frame, x, y)
        print(f"  resolve: {resolve_note}")
        if track_id is None:
            results.append({"well": well, "flagged_frame": frame, "x": x, "y": y, "note": note,
                             "track_id": None, "verdict": "UNRESOLVED", "raw": resolve_note})
            continue

        r = refine_one(well, track_id, frame, note)
        total_cost += r["cost"]
        parsed = {}
        if r["raw"]:
            try:
                parsed = json.loads(r["raw"])
            except Exception:
                pass
        verdict = parsed.get("verdict")
        print(f"  -> turns={r['turns']} tool_calls={r['tool_calls']} cost=${r['cost']:.4f} "
              f"verdict={verdict} conf={parsed.get('confidence')} "
              f"stages=c{parsed.get('condensation_frame')}/m{parsed.get('metaphase_frame')}/"
              f"a{parsed.get('anaphase_frame')}/e{parsed.get('exit_frame')} "
              f"running_total=${total_cost:.3f} elapsed={time.time()-t0:.0f}s", flush=True)

        results.append({"well": well, "flagged_frame": frame, "x": x, "y": y, "note": note,
                         "track_id": track_id, **parsed, "raw": r["raw"]})

        if track_id is not None:
            label = (f"[{(verdict or 'UNKNOWN').upper()} conf={parsed.get('confidence')}] "
                     f"human-flagged @({x:.0f},{y:.0f}) f{frame} -> track {track_id}"
                     + (f" -- {note}" if note else ""))
            events_by_well.setdefault(well, []).append({
                "track_id": track_id, "label": label,
                "centre_frame": parsed.get("metaphase_frame") or frame,
            })

    for well, events in events_by_well.items():
        path = show_cells_in_browser(well, events,
                                      note="Human-flagged sightings, AI-refined stage timing and verdict.")
        print(f"\n{well} report:\n{path}")

    print(f"\nTotal cost: ${total_cost:.3f}  |  n={len(sightings)}")


if __name__ == "__main__":
    main()
