"""Per-frame cell position lookup from the cached Trackastra tracked-mask memmap,
without re-running segmentation/tracking. Built 2026-07-18 as the foundation for
lineage-tracking crops -- a researcher's Claude following a track's full lifetime
(potentially 50-200+ frames between divisions), not just the AI review's fixed
+/-24 frame window. See [[project_cell_split_counter_marker_tracking_backlog]] and
[[project_cell_split_counter_researcher_browser_extend_timeline]].

events.csv only carries a single (cx, cy) snapshot per event, taken at that event's
own peak_frame -- nothing for the frames in between two divisions. Reconstructing a
real per-frame trajectory means going back to frames/_memmap/tracked_masks.dat
(written by src.track's Trackastra call): Trackastra's RAW per-frame CTC label maps.

Critical subtlety: a raw label in tracked_masks.dat is NOT always the same value as
events.csv's track_id. src.track._bridge_track_gaps() remaps raw labels into
canonical lineages after tracking (merging a track across a brief Cellpose
detection gap -- see that function's docstring), and events.csv's track_id reflects
THAT remapping, not the raw pixel values. The remapping itself was never persisted
to disk -- only used in-memory to build TrackNode objects before being collapsed
into events.csv. Rather than reimplementing that gap-bridging heuristic a second
time (a duplication pattern this project has been burned by before -- see
scripts/reports/_crop_shared.py's docstring on review_gpt.py silently drifting from
review.py), this module imports _bridge_track_gaps() itself and feeds it a
begin/end-frame table built from a single scan of tracked_masks.dat.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.track import _bridge_track_gaps

_DTYPE = np.uint16  # must match src/segment.py's label_maps.dat / tracked_masks.dat dtype


def _frame_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "frames").glob("frame_*.png"))


def _open_tracked_masks(run_dir: Path) -> np.memmap:
    frame_paths = _frame_paths(run_dir)
    if not frame_paths:
        raise FileNotFoundError(f"no frame_*.png files under {run_dir / 'frames'}")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    T, H, W = len(frame_paths), first.shape[0], first.shape[1]

    path = run_dir / "frames" / "_memmap" / "tracked_masks.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- this run's cached Trackastra output isn't on disk, "
            "can't reconstruct per-frame positions without a pipeline rerun"
        )
    expected = T * H * W * np.dtype(_DTYPE).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"tracked_masks.dat size mismatch -- expected {expected / 1e9:.2f}GB for "
            f"{T} frames at {H}x{W}, found {actual / 1e9:.2f}GB. Stale/wrong run_dir?"
        )
    return np.memmap(path, dtype=_DTYPE, mode="r", shape=(T, H, W))


def _tracking_index_cache_path(run_dir: Path) -> Path:
    return run_dir / "frames" / "_memmap" / "canonical_labels.json"


def _build_tracking_index(
    tracked_masks: np.memmap, run_dir: Path,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """(canonical_of, begin_of, end_of) -- canonical_of maps raw Trackastra CTC
    label -> canonical (post-gap-bridge) label matching events.csv's track_id;
    begin_of/end_of are each RAW label's own first/last-seen frame (pre-bridge --
    a canonical/track_id's true lifetime bounds are the min begin_of/max end_of
    across every raw label that bridges into it, see track_lifetime()).

    Cached to disk since building it requires one full pass over every frame
    (np.unique per frame) -- cheap relative to a pipeline rerun, but not free on a
    long video. Cache keyed on tracked_masks.dat's file size so a stale cache from
    a different run/regeneration doesn't silently get reused.
    """
    import pandas as pd

    cache_path = _tracking_index_cache_path(run_dir)
    masks_path = run_dir / "frames" / "_memmap" / "tracked_masks.dat"
    masks_size = masks_path.stat().st_size
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("_tracked_masks_size") == masks_size and "canonical_of" in cached:
            return (
                {int(k): v for k, v in cached["canonical_of"].items()},
                {int(k): v for k, v in cached["begin_of"].items()},
                {int(k): v for k, v in cached["end_of"].items()},
            )

    T = tracked_masks.shape[0]
    begin_of: dict[int, int] = {}
    end_of: dict[int, int] = {}
    for t in range(T):
        for lbl in np.unique(tracked_masks[t]):
            lbl = int(lbl)
            if lbl == 0:
                continue
            if lbl not in begin_of:
                begin_of[lbl] = t
            end_of[lbl] = t

    df_ctc = pd.DataFrame({
        "label": list(begin_of.keys()),
        "begin": [begin_of[l] for l in begin_of],
        "end": [end_of[l] for l in begin_of],
    })
    canonical_of = _bridge_track_gaps(df_ctc, tracked_masks, "label", "begin", "end")

    cache_path.write_text(
        json.dumps({
            "_tracked_masks_size": masks_size,
            "canonical_of": {str(k): v for k, v in canonical_of.items()},
            "begin_of": {str(k): v for k, v in begin_of.items()},
            "end_of": {str(k): v for k, v in end_of.items()},
        }),
        encoding="utf-8",
    )
    return canonical_of, begin_of, end_of


def track_lifetime(run_dir: Path, track_id: int) -> tuple[int, int] | None:
    """(first_frame, last_frame) this track_id actually appears in tracked_masks.dat,
    across every raw label bridged into it -- None if track_id never appears at all
    (e.g. a typo'd id, or one that classify.py dropped before writing events.csv).
    """
    tracked_masks = _open_tracked_masks(run_dir)
    canonical_of, begin_of, end_of = _build_tracking_index(tracked_masks, run_dir)
    raw_labels = [raw for raw, canon in canonical_of.items() if canon == track_id]
    if not raw_labels:
        return None
    return min(begin_of[r] for r in raw_labels), max(end_of[r] for r in raw_labels)


def per_frame_centroids(
    run_dir: Path, track_id: int, frame_lo: int | None = None, frame_hi: int | None = None,
) -> dict[int, tuple[float, float]]:
    """Real per-frame (cx, cy) centroid for `track_id`. Defaults to the track's own
    observed lifetime (track_lifetime()) rather than the whole video -- correct and
    far cheaper for a short-lived track in a long video. A frame with no data (the
    cell genuinely wasn't detected that frame -- see _bridge_track_gaps's docstring
    on brief Cellpose detection gaps that DIDN'T get bridged, e.g. 2+ plausible
    successors) is simply absent from the returned dict, never interpolated or guessed.
    """
    tracked_masks = _open_tracked_masks(run_dir)
    canonical_of, begin_of, end_of = _build_tracking_index(tracked_masks, run_dir)
    raw_labels = {raw for raw, canon in canonical_of.items() if canon == track_id}
    if not raw_labels:
        return {}

    T = tracked_masks.shape[0]
    if frame_lo is None or frame_hi is None:
        lifetime_lo, lifetime_hi = min(begin_of[r] for r in raw_labels), max(end_of[r] for r in raw_labels)
    lo = lifetime_lo if frame_lo is None else max(0, frame_lo)
    hi = lifetime_hi if frame_hi is None else min(T - 1, frame_hi)

    out: dict[int, tuple[float, float]] = {}
    for t in range(lo, hi + 1):
        frame_arr = np.asarray(tracked_masks[t])
        for raw in raw_labels:
            ys, xs = np.nonzero(frame_arr == raw)
            if len(xs) > 0:
                out[t] = (float(xs.mean()), float(ys.mean()))
                break
    return out
