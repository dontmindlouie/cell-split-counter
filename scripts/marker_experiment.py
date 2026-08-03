"""Spike: generate sample marked crops for visual review before spending any API budget.

Draws a NEW marker design -- thin corner-bracket ticks, well clear of the cell, dim
(not saturated) color -- on the same 384px crop src/review.py sends to Claude/GPT.
This is deliberately different from the closed 2026-07-05 investigation's marker: that
one was a bold, saturated, solid ring, and it caused a real regression on subtle
divisions EVEN AT 70px radius (confirmed clear of the object) -- so "further away"
alone was already tried and already failed. This design combines distance AND
thinness/low-saturation, which was proposed but never tried.

No API calls here -- just renders marked crops to samples/ for a human to look at
before any repeated-sampling validation is run.

TODO / backlog (2026-07-08, not started): try a crosshair-style AI-facing marker
(the same thin-reticle-with-a-center-gap design already used in the human-facing
spot-check tool's crosshair, see spot_check_review.py's `crosshairHtml`), positioned
further from the cell than the current corner brackets (`_TICK_RADIUS=55`). Not yet
attempted as a distinct variant -- current brackets are the only shape tested so far.
"""

import math
import sys
from pathlib import Path

import cv2

_CROP_RADIUS = 192  # matches src/review.py exactly
_TICK_RADIUS = 55     # px from centroid -- shrunk from 80 (v1): at 80px in a dense
                       # field the 4 brackets implied a ~160x160px box spanning
                       # several neighboring cells, not isolating one
_TICK_LEN = 14         # px, length of each corner-bracket arm (scaled down with radius)
_TICK_COLOR = (60, 170, 230)   # BGR, dim/muted orange-amber -- not saturated cyan
_TICK_THICKNESS = 2
_EDGE_MARGIN = 6       # px -- keep ticks from being clipped exactly at the canvas edge

# Absolute, resolved from this file's location -- not cwd. A relative path here
# previously broke silently: running this script (or importing it) from scripts/
# instead of the worktree root made every frame lookup return None, which sent
# ZERO images to the API with no error, and the model correctly-but-uselessly
# said false_positive every time because it had nothing to look at.
_REPO_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_CASES = {
    "parent_1246_neighbor_misattribution": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_m3/frames",
        "centroid": (905.0, 12.0),
        "local_frame": 43,  # raw 188
        "expected": "false_positive (tracked cell doesn't divide, neighbor does -- confirmed twice in Fiji)",
    },
    "parent_647_real_multipolar": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_m3/frames",
        "centroid": (594.0, 677.0),
        "local_frame": 29,  # raw 174
        "expected": "real, multipolar (confirmed, no contradiction anywhere)",
    },
    "tom20_gt_event6_subtle_regression_case": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_tom20gt6/frames",
        "centroid": (1231.0, 415.0),
        "local_frame": 44,  # raw 44, same numbering (no offset applied when copying)
        "expected": "real (confirmed via independent raw-mask tracing) -- this is the case the old bold marker broke",
    },
    "parent_2167_neighbor_misattribution_clean": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_bewo2167/frames",
        "centroid": (284.7369565217391, 282.004347826087),
        "local_frame": 619,  # same numbering as bewo_m2 output, no offset
        "expected": (
            "false_positive (marked/tracked cell doesn't split, a neighbor does -- "
            "spot-check note: 'great neighbor split where the marker cell doesn't "
            "split'). near_edge=0 in events.csv, unlike parent_1246 -- this is the "
            "clean (non-edge-clamped) neighbor-misattribution case parent_1246 couldn't "
            "be, so it isolates whether the marker helps with neighbor disambiguation "
            "specifically, independent of the edge-clamping confound."
        ),
        # 2026-07-08: unlike parent_2535/1609, this neighbor was never itself a
        # tracked event in events.csv (only visible by eye), so there's no clean
        # single "neighbor distance" -- connected-component analysis of the split
        # frame found bright blobs at 21px, 34px, 54px, etc. from the target, i.e.
        # this field is dense enough that MULTIPLE other nuclei sit within any
        # radius large enough to still clear the cell body itself (_TICK_RADIUS_MIN
        # is 15px). Using 21px (nearest blob) as the adaptive-radius test distance,
        # expecting this may be the case where radius-shrinking alone can't work.
        "neighbor_distance_px": 21.0,
        # cell_area_px pulled from bewo_m2_2026-07-07/events.csv (parent_id 2167) --
        # this cell's own Cellpose mask area, already computed by the real pipeline.
        "cell_area_px": 460.0,
    },
    # REJECTED 2026-07-08 (maintainer visual review) -- do not reintroduce without
    # actually overlaying both candidate + neighbor centroids on one frame first.
    # These two were found by an automated centroid-distance/peak-frame proximity
    # script over events.csv, then pattern-matched to Claude's text description
    # instead of being visually confirmed. parent_3004: "does not look like a
    # candidate" on direct review. parent_2560: "the target cell is just
    # disappearing and reappearing" -- a flicker/out-of-focus artifact, not a
    # genuine division. Left out of REFERENCE_CASES; see git history or the
    # project memory for the full (wrong) reasoning if this needs revisiting.
    "parent_2535_simultaneous_confirmed": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_bewo2535/frames",
        "centroid": (407.6134969325153, 386.6503067484663),
        "local_frame": 615,  # same numbering as bewo_m2 output, no offset
        "expected": (
            "real (maintainer's human spot-check verdict, 2026-07-08, via the "
            "--parent-ids filmstrip review of dual-marker-verified candidate pairs) -- "
            "note: 'both target and neighbor divide at the same time.' Raw verdict.txt "
            "already said real at confidence 0.67 with its own text flagging the exact "
            "confusion: 'The field is crowded with other mitotic events which "
            "complicates interpretation' -- but the pipeline's confidence floor "
            "downgraded it to false_positive in events.csv anyway. This is the second "
            "confirmed instance of the parent_2300 pattern (candidate and neighbor "
            "genuinely dividing concurrently), found via centroid-distance/peak-frame "
            "proximity search + dual-marker visual verification, then human-confirmed "
            "-- not just automated pattern-matching this time. Neighbor: parent_2533 "
            "(confidence 0.86, ~47px away, peak frame 612 -- only 3 frames off), which "
            "the maintainer separately called 'unsure... I think neighbor and target divides "
            "at the same time.'"
        ),
        "neighbor_distance_px": 47.0,
        "cell_area_px": 163.0,  # from bewo_m2_2026-07-07/events.csv, parent_id 2535
    },
    "parent_1609_simultaneous_confirmed": {
        "frames_dir": _REPO_ROOT / "data/smoke_test_bewo1609/frames",
        "centroid": (384.7393026941363, 278.18225039619654),
        "local_frame": 468,  # same numbering as bewo_m2 output, no offset
        "expected": (
            "real (maintainer's human spot-check verdict, 2026-07-08) -- note: 'target "
            "starts to divide, neighbor to the south right divides during this time "
            "but a little far from the target.' Same floor-downgrade bug as "
            "parent_2535: raw verdict.txt said real at 0.67, floored to false_positive "
            "in events.csv. Neighbor: parent_1801 (confidence 0.92, ~52px away, peak "
            "frame 466 -- 2 frames off), which the maintainer called 'unsure' on its own "
            "account ('target is either dividing really slowly or not dividing. "
            "neighbor is just starting to divide') -- so this pair has some additional "
            "ambiguity baked in beyond parent_2535, worth treating as the second-string "
            "case rather than the primary one."
        ),
        "neighbor_distance_px": 52.0,
        # from bewo_m2_2026-07-07/events.csv, parent_id 1609 -- notably ~3x the
        # mask area of parent_2535 (~40px vs ~14px implied diameter). 2026-07-08
        # hypothesis: this case's GPT regression at radius=21px (fraction=0.5) may
        # not be pure "backend fragmentation bias" -- 21px barely clears this cell's
        # own ~20px physical radius, so the box may be sized to the RESTING cell,
        # not to how far the daughters spread apart during an actual division,
        # visually clipping the split itself. See adaptive_radius()'s size-floor.
        "cell_area_px": 1262.0,
    },
}


def _crop(img, cx, cy):
    h, w = img.shape[:2]
    y0, y1 = max(0, int(cy - _CROP_RADIUS)), min(h, int(cy + _CROP_RADIUS))
    x0, x1 = max(0, int(cx - _CROP_RADIUS)), min(w, int(cx + _CROP_RADIUS))
    return img[y0:y1, x0:x1], x0, y0


_TICK_RADIUS_MIN = 15  # px -- floor for the adaptive radius; below this the brackets
                        # start overlapping the cell body itself (typical diameter
                        # ~15-30px), defeating "clear of the cell" entirely.


def adaptive_radius(
    neighbor_distance_px: float | None,
    margin: float = 5.0,
    fraction: float = 0.5,
    radius_min: int = _TICK_RADIUS_MIN,
    cell_area_px: float | None = None,
    size_k: float = 1.3,
) -> int:
    """Shrink the bracket radius so the marked box can't enclose a nearby neighbor,
    while not shrinking it below what the CANDIDATE cell's own size needs.

    2026-07-08 finding: the fixed 55px radius (v2) was shown -- via direct visual
    inspection of marked crops, confirmed by the maintainer -- to be the actual variable
    behind the marker helping vs. hurting. parent_1246/parent_2167 regressed because
    the neighbor sat INSIDE the same bracket box; parent_1609 worked because the
    neighbor sat outside it. This caps the radius at `fraction` of the distance to
    the nearest neighbor (minus a small margin) so the box structurally cannot
    contain both. Returns the unchanged default radius if no neighbor distance is
    known.

    2026-07-08 GPT-5-mini follow-up: n=8 validation showed the default fraction=0.5
    (tuned for/validated on Claude Haiku) regressed on parent_1609 -- GPT-5-mini read
    a genuine division as "fragmentation/apoptosis" under a too-tight crop. A
    backend-specific fraction=0.75 fixed it, but a cleaner explanation surfaced from
    `cell_area_px` (the candidate's own Cellpose mask area, already computed by the
    real pipeline, unlike neighbor_distance_px which sometimes doesn't exist):
    parent_1609's own cell is ~3x the mask area of parent_2535's (~40px vs ~14px
    implied diameter), and radius=21px (fraction=0.5) barely clears its ~20px own
    physical radius -- the box was sized to the RESTING cell, not to how far real
    daughters spread apart mid-division, so it may have visually clipped the split
    itself rather than merely excluding a neighbor. `size_k` sets the floor to
    `size_k * cell_own_radius_px` (derived from `cell_area_px` via area=pi*r^2) so a
    genuinely large candidate cell gets a floor that scales with its own geometry,
    on top of `radius_min` for cases with no size data. This composes with the
    neighbor-distance cap: whichever constraint is more restrictive still wins --
    a very close neighbor (parent_2167, 21px) still forces a small radius regardless
    of this floor, since `min(_TICK_RADIUS, ...)` is applied after.
    """
    if neighbor_distance_px is None:
        return _TICK_RADIUS
    floor = radius_min
    if cell_area_px is not None:
        cell_own_radius = math.sqrt(cell_area_px / math.pi)
        floor = max(floor, size_k * cell_own_radius)
    computed = neighbor_distance_px * fraction - margin
    return int(max(floor, min(_TICK_RADIUS, computed)))


def _draw_corner_ticks(crop, local_cx, local_cy, radius=None):
    """4 short L-shaped brackets at `radius` (default _TICK_RADIUS) from the point,
    pointing inward.

    Deliberately NOT a continuous ring (less total ink than the closed
    investigation's marker) and NOT touching the cell itself.

    Corner positions are clamped to the crop's actual bounds -- v1 didn't do this
    and silently dropped ticks off-canvas on edge-clamped crops (e.g. parent_1246,
    only 204px tall vs. the nominal 384px), leaving only 2 of 4 brackets visible
    with no indication anything was cut off.
    """
    out = crop.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    r = radius if radius is not None else _TICK_RADIUS
    l = _TICK_LEN
    m = _EDGE_MARGIN
    corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]  # dx, dy sign per corner
    for dx, dy in corners:
        cx = max(m, min(w - m, local_cx + dx * r))
        cy = max(m, min(h - m, local_cy + dy * r))
        # horizontal arm -- also clamp the far end so it can't run off-canvas
        hx = max(m, min(w - m, cx - dx * l))
        cv2.line(out, (int(cx), int(cy)), (int(hx), int(cy)), _TICK_COLOR, _TICK_THICKNESS, cv2.LINE_AA)
        # vertical arm
        vy = max(m, min(h - m, cy - dy * l))
        cv2.line(out, (int(cx), int(cy)), (int(cx), int(vy)), _TICK_COLOR, _TICK_THICKNESS, cv2.LINE_AA)
    return out


def _find_frame(frames_dir: Path, index: int) -> Path | None:
    matches = list(frames_dir.glob(f"frame_{index:05d}_*.png"))
    return matches[0] if matches else None


# Match src/review.py exactly -- same before/after span and stride, so the sequence
# generated here is what the real pipeline actually sends, not an approximation.
_FRAMES_BEFORE = 8
_FRAMES_AFTER = 8
_FRAME_STRIDE = 3


def _sequence_indices(center: int) -> list[tuple[int, str]]:
    before = [center - i * _FRAME_STRIDE for i in range(_FRAMES_BEFORE, 0, -1)]
    after = [center + i * _FRAME_STRIDE for i in range(1, _FRAMES_AFTER + 1)]
    labeled = [(i, "before") for i in before if i >= 0] + [(center, "split")] + [(i, "after") for i in after]
    return labeled


def main():
    out_dir = _REPO_ROOT / "samples"
    out_dir.mkdir(exist_ok=True)

    for name, case in REFERENCE_CASES.items():
        cx, cy = case["centroid"]
        neighbor_dist = case.get("neighbor_distance_px")
        radius = (
            adaptive_radius(neighbor_dist, cell_area_px=case.get("cell_area_px"))
            if neighbor_dist is not None else None
        )
        seq_dir = out_dir / f"{name}_sequence"
        seq_dir.mkdir(exist_ok=True)

        n_written = 0
        for pos, (idx, label) in enumerate(_sequence_indices(case["local_frame"])):
            path = _find_frame(case["frames_dir"], idx)
            if path is None:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            crop, x0, y0 = _crop(img, cx, cy)
            local_cx, local_cy = cx - x0, cy - y0
            default_radius = radius if radius is not None else _TICK_RADIUS
            marked = _draw_corner_ticks(crop, local_cx, local_cy, radius=default_radius)
            cv2.imwrite(str(seq_dir / f"{pos:02d}_{label}_{idx:05d}_marked.png"), marked)
            cv2.imwrite(str(seq_dir / f"{pos:02d}_{label}_{idx:05d}_unmarked.png"), crop)
            if radius is not None:
                fixed_marked = _draw_corner_ticks(crop, local_cx, local_cy)  # default _TICK_RADIUS, comparison only
                cv2.imwrite(str(seq_dir / f"{pos:02d}_{label}_{idx:05d}_fixed55.png"), fixed_marked)
            n_written += 1

        path = _find_frame(case["frames_dir"], case["local_frame"])
        if path is None:
            print(f"[SKIP] {name}: split frame {case['local_frame']} not found in {case['frames_dir']}")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        crop, x0, y0 = _crop(img, cx, cy)
        local_cx, local_cy = cx - x0, cy - y0

        unmarked_path = out_dir / f"{name}_unmarked.png"
        cv2.imwrite(str(unmarked_path), crop)

        # 2026-07-08: adaptive radius (shrunk to clear the nearest known neighbor) is
        # now the default "marked" output whenever a neighbor distance is known -- the
        # validated fixed 55px box regressed on parent_1246/parent_2167 precisely
        # because it could enclose a simultaneously-dividing neighbor. The old fixed
        # radius is still written out (suffix _fixed55) for side-by-side comparison,
        # but it is no longer what "_marked.png" means.
        default_radius = radius if radius is not None else _TICK_RADIUS
        marked = _draw_corner_ticks(crop, local_cx, local_cy, radius=default_radius)
        marked_path = out_dir / f"{name}_marked.png"
        cv2.imwrite(str(marked_path), marked)

        fixed_path = None
        if radius is not None:
            fixed_marked = _draw_corner_ticks(crop, local_cx, local_cy)  # default _TICK_RADIUS
            fixed_path = out_dir / f"{name}_fixed55.png"
            cv2.imwrite(str(fixed_path), fixed_marked)

        print(f"[OK] {name}")
        print(f"     source: {path.name}, centroid ({cx:.0f},{cy:.0f}) -> local ({local_cx:.0f},{local_cy:.0f})")
        print(f"     full sequence ({n_written} frames): {seq_dir}")
        print(f"     crop size: {crop.shape[1]}x{crop.shape[0]}")
        print(f"     expected: {case['expected']}")
        print(f"     -> {unmarked_path}")
        print(f"     -> {marked_path} (radius={default_radius}px{' adaptive' if radius is not None else ' fixed default'})")
        if fixed_path is not None:
            print(f"     -> {fixed_path} (fixed radius={_TICK_RADIUS}px, neighbor at {neighbor_dist:.0f}px, for comparison)")
        print()


if __name__ == "__main__":
    sys.exit(main())
