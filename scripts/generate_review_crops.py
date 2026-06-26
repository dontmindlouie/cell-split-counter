"""
Generate side-by-side before/after crops of unmatched detected split events
for manual review.

Usage: python scripts/generate_review_crops.py [n_samples] [tolerance]
  n_samples  how many splits to sample (default 24)
  tolerance  frame window for matching against GT (default 5)

Output: data/output/review_crops/  -- one PNG per split event
"""
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.segment import segment_frame
from src.track import link_frames

FRAME_DIR   = Path("data/frames")
EVENTS_CSV  = Path("data/output/events.csv")
GT_XLSX     = Path("data/ground_truth/ACD_analysis.xlsx")
GT_SHEET    = "iPSC_nTSC_Tom20_ACTB_ZO1"
GT_ROW_START, GT_ROW_END = 19, 51
OUT_DIR     = Path("data/output/review_crops")
CROP_PAD    = 180   # px around daughters centroid
BRIGHTNESS  = 4.0   # multiplier for dim fluorescence frames


def load_gt_peaks(tolerance: int) -> set[int]:
    wb = openpyxl.load_workbook(GT_XLSX, data_only=True)
    ws = wb[GT_SHEET]
    gt = []
    for row in ws.iter_rows(min_row=GT_ROW_START, max_row=GT_ROW_END, values_only=True):
        v = row[2] if len(row) > 2 else None
        if v is None:
            continue
        if isinstance(v, (int, float)):
            gt.append(int(v))
        elif isinstance(v, str):
            m = re.match(r"(\d+)\s*-\s*(\d+)", v.strip())
            if m:
                gt.append((int(m.group(1)) + int(m.group(2))) // 2)
    # expand to set of all frames that are "near" a GT event
    near = set()
    for g in gt:
        for d in range(g - tolerance, g + tolerance + 1):
            near.add(d)
    return near


def load_unmatched_split_frames(near_gt: set[int]) -> list[int]:
    """Return sorted list of unique peak_frames (1-indexed, GT convention) for
    splits that are NOT within tolerance of any ground-truth event."""
    with open(EVENTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    # convert pipeline 0-indexed peak_frame -> 1-indexed for GT comparison
    peak_frames = sorted({int(r["peak_frame"]) + 1 for r in rows})
    return [p for p in peak_frames if p not in near_gt]


def segment_pair(frame_idx: int):
    """Segment frame_idx-1 and frame_idx, return (masks_prev, masks_cur, nodes)."""
    prev_path = FRAME_DIR / f"frame_{frame_idx-1:05d}_raw{frame_idx-1:05d}.png"
    cur_path  = FRAME_DIR / f"frame_{frame_idx:05d}_raw{frame_idx:05d}.png"
    masks = {
        frame_idx - 1: segment_frame(prev_path, frame_idx - 1),
        frame_idx:     segment_frame(cur_path,  frame_idx),
    }
    nodes = link_frames(masks)
    return masks, nodes


def find_splits(nodes, frame_idx: int):
    """Return list of (parent_node, [daughter_nodes]) for splits at frame_idx."""
    # parent track_id -> list of daughter nodes at frame_idx
    daughters_by_parent: dict[int, list] = {}
    for n in nodes:
        if n.frame == frame_idx and n.parent_id is not None:
            daughters_by_parent.setdefault(n.parent_id, []).append(n)
    # only real splits (2+ daughters)
    splits = []
    parent_map = {n.track_id: n for n in nodes if n.frame == frame_idx - 1}
    for pid, daughters in daughters_by_parent.items():
        if len(daughters) >= 2 and pid in parent_map:
            splits.append((parent_map[pid], daughters))
    return splits


def draw_mask_contour(img: np.ndarray, cell_mask, color: tuple, dot_radius: int = 5) -> None:
    """Draw the Cellpose mask boundary as a thin outline + small centroid dot."""
    y0, y1, x0, x1 = cell_mask.bbox
    # reconstruct full-size boolean patch for this cell's region
    full_patch = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    full_patch[cell_mask.local_mask] = 255
    contours, _ = cv2.findContours(full_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # shift contour coords back to full-frame position
    shifted = [c + np.array([[[x0, y0]]]) for c in contours]
    cv2.drawContours(img, shifted, -1, color, 2)
    cx, cy = int(cell_mask.centroid[0]), int(cell_mask.centroid[1])
    cv2.circle(img, (cx, cy), dot_radius, color, -1)


def make_crop(img_prev, img_cur, parent_node, daughter_nodes, frame_idx: int, label: str) -> np.ndarray:
    # find center from daughters
    cxs = [int(d.mask.centroid[0]) for d in daughter_nodes]
    cys = [int(d.mask.centroid[1]) for d in daughter_nodes]
    cx, cy = int(np.mean(cxs)), int(np.mean(cys))

    def crop_brightened(img, cx, cy):
        h, w = img.shape[:2]
        x0 = max(0, cx - CROP_PAD); x1 = min(w, cx + CROP_PAD)
        y0 = max(0, cy - CROP_PAD); y1 = min(h, cy + CROP_PAD)
        patch = img[y0:y1, x0:x1].copy()
        patch = np.clip(patch.astype(np.float32) * BRIGHTNESS, 0, 255).astype(np.uint8)
        return patch, x0, y0

    # annotate prev frame with parent contour
    prev_ann = img_prev.copy()
    draw_mask_contour(prev_ann, parent_node.mask, (0, 255, 255))

    # annotate cur frame with daughter contours
    DCOLORS = [(0, 200, 255), (255, 140, 0), (0, 255, 100), (200, 0, 255)]
    cur_ann = img_cur.copy()
    for i, d in enumerate(daughter_nodes):
        draw_mask_contour(cur_ann, d.mask, DCOLORS[i % len(DCOLORS)])

    prev_crop, _, _ = crop_brightened(prev_ann, cx, cy)
    cur_crop,  _, _ = crop_brightened(cur_ann,  cx, cy)

    # pad both to same size
    ph, pw = prev_crop.shape[:2]
    ch, cw = cur_crop.shape[:2]
    th, tw = max(ph, ch), max(pw, cw)
    def pad(p, th, tw):
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        out[:p.shape[0], :p.shape[1]] = p
        return out
    prev_crop = pad(prev_crop, th, tw)
    cur_crop  = pad(cur_crop,  th, tw)

    # add text banners
    banner_h = 40
    def banner(img, text):
        bar = np.zeros((banner_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.vstack([bar, img])

    prev_crop = banner(prev_crop, f"frame {frame_idx-1}  (parent, cyan)")
    cur_crop  = banner(cur_crop,  f"frame {frame_idx}  (daughters)")

    side_by_side = np.hstack([prev_crop, cur_crop])

    # top label bar
    top = np.zeros((50, side_by_side.shape[1], 3), dtype=np.uint8)
    cv2.putText(top, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 255, 100), 2)
    return np.vstack([top, side_by_side])


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    tolerance = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth (tolerance ±{tolerance})...")
    near_gt = load_gt_peaks(tolerance)

    print("Finding unmatched split frames...")
    unmatched_frames = load_unmatched_split_frames(near_gt)
    print(f"  {len(unmatched_frames)} unique peak frames not near GT")

    # sample evenly across the video
    step = max(1, len(unmatched_frames) // n_samples)
    sample = unmatched_frames[::step][:n_samples]
    print(f"  sampling {len(sample)} frames: {sample[:8]}...")

    saved = 0
    for fi in sample:
        if fi == 0:
            continue  # no prev frame
        print(f"  frame {fi}...", end=" ", flush=True)
        masks, nodes = segment_pair(fi)
        splits = find_splits(nodes, fi)

        if not splits:
            print("no splits found in local tracking")
            continue

        # load raw images for annotation
        prev_img = cv2.imread(str(FRAME_DIR / f"frame_{fi-1:05d}_raw{fi-1:05d}.png"))
        cur_img  = cv2.imread(str(FRAME_DIR / f"frame_{fi:05d}_raw{fi:05d}.png"))
        if prev_img is None or cur_img is None:
            print("image load failed")
            continue

        for si, (parent, daughters) in enumerate(splits):
            label = f"peak_frame={fi}  split_{si+1}of{len(splits)}"
            out = make_crop(prev_img, cur_img, parent, daughters, fi, label)
            out_path = OUT_DIR / f"frame{fi:04d}_split{si+1:02d}.png"
            cv2.imwrite(str(out_path), out)
            saved += 1
            print(f"saved {out_path.name}", end="  ")

        print()

    print(f"\nDone. {saved} review images in {OUT_DIR}/")


if __name__ == "__main__":
    main()
