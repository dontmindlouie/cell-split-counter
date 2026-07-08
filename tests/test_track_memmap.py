"""Tests for memmap helpers and Trackastra monkeypatch infrastructure in src/track.py.

Coverage goals
--------------
1. _normalize_on_memmap  — in-place op, output range, subsample guard
2. _make_float32_memmap  — disk-backed (np.memmap), correct shape/dtype/values
3. _bridge_track_gaps    — single-successor gap merged; real split (2 successors) left alone;
                           too-far successor ignored; track at video end never bridged
4. _label_to_cellmask    — bbox, centroid, area, local_mask
5. Trackastra patch targets exist and are patchable/restorable (catches version-bump renames
   that would silently break the RAM-reduction patches in link_frames_trackastra)
6. _stack_to_memmap interceptor logic — two interception cases produce correct outputs
"""

import numpy as np
import pandas as pd
import pytest

from src.track import (
    _bridge_track_gaps,
    _label_to_cellmask,
    _make_float32_memmap,
    _normalize_on_memmap,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _uint8_memmap(path, shape, fill=128):
    """Create a uint8 memmap filled with *fill* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    mm[:] = fill
    mm.flush()
    return mm


def _tracked_video(T, H, W, placements):
    """Return a (T, H, W) uint16 label array from a list of (frame, label, y0, x0, h, w)."""
    video = np.zeros((T, H, W), dtype=np.uint16)
    for f, label, y0, x0, h, w in placements:
        video[f, y0 : y0 + h, x0 : x0 + w] = label
    return video


def _ctc_df(rows):
    """Build a minimal CTC-style DataFrame with columns (label, t_start, t_end, parent)."""
    return pd.DataFrame(rows, columns=["label", "t_start", "t_end", "parent"])


# ── _normalize_on_memmap ──────────────────────────────────────────────────────

class TestNormalizeOnMemmap:
    def test_returns_same_object(self):
        """Operation must be in-place; no copy should be allocated."""
        arr = (np.random.rand(5, 20, 20) * 200 + 50).astype(np.float32)
        result = _normalize_on_memmap(arr, subsample=None)
        assert result is arr

    def test_output_spans_unit_range(self):
        """After p1/p99.8 normalization most values land in [0, 1]."""
        rng = np.random.default_rng(0)
        arr = (rng.random((10, 32, 32)) * 200 + 50).astype(np.float32)
        _normalize_on_memmap(arr, subsample=None)
        assert float(arr.min()) < 0.05
        assert float(arr.max()) > 0.95

    def test_subsample_skipped_on_small_array(self):
        """Subsampling guard: if dims ≤ 64*subsample the full array is used (no IndexError)."""
        arr = np.ones((4, 16, 16), dtype=np.float32) * 100
        # 16 < 64*4 = 256 → subsample path is bypassed; result should still be normalized
        _normalize_on_memmap(arr, subsample=4)
        # Constant array: (val - mi) / (ma - mi + eps) → 0; just confirm no exception raised
        assert arr.shape == (4, 16, 16)

    def test_subsample_used_on_large_array(self):
        """Subsampling path is exercised when both spatial dims exceed 64*subsample."""
        rng = np.random.default_rng(1)
        # 300 > 64*4 = 256 → subsample triggered
        arr = (rng.random((3, 300, 300)) * 200 + 50).astype(np.float32)
        result = _normalize_on_memmap(arr, subsample=4)
        assert result is arr  # still in-place


# ── _make_float32_memmap ─────────────────────────────────────────────────────

class TestMakeFloat32Memmap:
    def test_result_is_disk_backed_memmap(self, tmp_path):
        """Output must be a np.memmap, not a plain ndarray (RAM spike guard)."""
        frames = _uint8_memmap(tmp_path / "_memmap" / "frames.dat", shape=(3, 8, 8))
        result = _make_float32_memmap(frames)
        assert isinstance(result, np.memmap)

    def test_dtype_is_float32(self, tmp_path):
        frames = _uint8_memmap(tmp_path / "_memmap" / "frames.dat", shape=(4, 8, 8))
        result = _make_float32_memmap(frames)
        assert result.dtype == np.float32

    def test_shape_matches_input(self, tmp_path):
        shape = (5, 12, 16)
        frames = _uint8_memmap(tmp_path / "_memmap" / "frames.dat", shape=shape)
        result = _make_float32_memmap(frames)
        assert result.shape == shape

    def test_pixel_values_preserved(self, tmp_path):
        """Each uint8 pixel value must equal its float32 counterpart."""
        frames = _uint8_memmap(tmp_path / "_memmap" / "frames.dat", shape=(2, 8, 8), fill=200)
        result = _make_float32_memmap(frames)
        assert np.allclose(result, 200.0)

    def test_file_written_to_memmap_dir(self, tmp_path):
        """frames_float32.dat must be written next to frames.dat (same _memmap dir)."""
        frames = _uint8_memmap(tmp_path / "_memmap" / "frames.dat", shape=(2, 8, 8))
        _make_float32_memmap(frames)
        assert (tmp_path / "_memmap" / "frames_float32.dat").exists()


# ── _bridge_track_gaps ────────────────────────────────────────────────────────

class TestBridgeTrackGaps:
    def test_single_nearby_successor_is_merged(self):
        """One nearby successor within gap window → continuous track (same canonical id)."""
        T, H, W = 3, 50, 50
        video = _tracked_video(T, H, W, [
            (0, 1, 10, 10, 8, 8),
            (1, 1, 10, 10, 8, 8),
            (2, 2, 10, 10, 8, 8),  # track 2 resumes at same location
        ])
        df = _ctc_df([(1, 0, 1, 0), (2, 2, 2, 0)])
        canonical = _bridge_track_gaps(
            df, video, "label", "t_start", "t_end", max_gap_frames=1, max_gap_dist=40.0
        )
        assert canonical[1] == canonical[2]

    def test_two_nearby_successors_not_merged(self):
        """Two nearby successors (real split) → tracks remain distinct canonical ids."""
        T, H, W = 3, 80, 80
        video = _tracked_video(T, H, W, [
            (0, 1, 30, 30, 10, 10),
            (1, 1, 30, 30, 10, 10),
            (2, 2, 20, 20,  8,  8),  # daughter 1
            (2, 3, 40, 40,  8,  8),  # daughter 2
        ])
        df = _ctc_df([(1, 0, 1, 0), (2, 2, 2, 0), (3, 2, 2, 0)])
        canonical = _bridge_track_gaps(
            df, video, "label", "t_start", "t_end", max_gap_frames=1, max_gap_dist=60.0
        )
        assert canonical[1] != canonical[2]
        assert canonical[1] != canonical[3]

    def test_successor_beyond_distance_threshold_not_merged(self):
        """Successor outside max_gap_dist → tracks remain distinct."""
        T, H, W = 3, 200, 200
        video = _tracked_video(T, H, W, [
            (0, 1, 10, 10, 8, 8),
            (1, 1, 10, 10, 8, 8),
            (2, 2, 160, 160, 8, 8),  # far away
        ])
        df = _ctc_df([(1, 0, 1, 0), (2, 2, 2, 0)])
        canonical = _bridge_track_gaps(
            df, video, "label", "t_start", "t_end", max_gap_frames=1, max_gap_dist=40.0
        )
        assert canonical[1] != canonical[2]

    def test_track_running_to_video_end_not_bridged(self):
        """Tracks alive at the final frame are not gap-candidates and stay distinct."""
        T, H, W = 3, 50, 50
        video = _tracked_video(T, H, W, [
            (0, 1, 10, 10, 8, 8),
            (1, 1, 10, 10, 8, 8),
            (2, 1, 10, 10, 8, 8),  # track 1 lives to final frame (t_end == T-1)
            (2, 2, 12, 12, 8, 8),  # track 2 also at final frame
        ])
        df = _ctc_df([(1, 0, 2, 0), (2, 2, 2, 0)])  # t_end=2 == T-1 for track 1
        canonical = _bridge_track_gaps(
            df, video, "label", "t_start", "t_end", max_gap_frames=1, max_gap_dist=40.0
        )
        assert canonical[1] != canonical[2]

    def test_every_label_has_canonical_entry(self):
        """Return dict contains all label ids, not just bridged ones."""
        T, H, W = 2, 30, 30
        video = _tracked_video(T, H, W, [(0, 1, 5, 5, 6, 6), (1, 2, 5, 5, 6, 6)])
        df = _ctc_df([(1, 0, 0, 0), (2, 1, 1, 0)])
        canonical = _bridge_track_gaps(
            df, video, "label", "t_start", "t_end", max_gap_frames=1, max_gap_dist=40.0
        )
        assert 1 in canonical and 2 in canonical


# ── _label_to_cellmask ────────────────────────────────────────────────────────

class TestLabelToCellmask:
    def _label_map(self):
        arr = np.zeros((20, 20), dtype=np.uint16)
        arr[4:9, 6:12] = 7  # label 7: 5 rows × 6 cols = 30 px
        return arr

    def test_bbox_matches_nonzero_extent(self):
        cell = _label_to_cellmask(self._label_map(), label_id=7, frame=3)
        assert cell.bbox == (4, 9, 6, 12)  # (y0, y1, x0, x1)

    def test_area_equals_pixel_count(self):
        cell = _label_to_cellmask(self._label_map(), label_id=7, frame=3)
        assert cell.area == 30.0

    def test_centroid_at_rectangle_centre(self):
        cell = _label_to_cellmask(self._label_map(), label_id=7, frame=3)
        # centre of cols 6-11 = 8.5; centre of rows 4-8 = 6.0
        cx, cy = cell.centroid
        assert abs(cx - 8.5) < 0.1
        assert abs(cy - 6.0) < 0.1

    def test_local_mask_matches_label_pixels(self):
        label_map = self._label_map()
        cell = _label_to_cellmask(label_map, label_id=7, frame=3)
        y0, y1, x0, x1 = cell.bbox
        expected = (label_map[y0:y1, x0:x1] == 7)
        assert np.array_equal(cell.local_mask, expected)

    def test_frame_stored_on_mask(self):
        cell = _label_to_cellmask(self._label_map(), label_id=7, frame=5)
        assert cell.frame == 5
        assert cell.mask_id == 7


# ── Trackastra patch-target sanity ────────────────────────────────────────────

class TestTrackastraPatchTargets:
    """Verify that the module attributes patched in link_frames_trackastra exist and
    survive a patch-restore cycle.  These tests fail loudly if a trackastra upgrade
    renames the targets, which would silently break the RAM-reduction patches."""

    def test_model_api_normalize_is_patchable(self):
        import trackastra.model.model_api as _model_api

        original = _model_api.normalize
        sentinel = lambda x, **kw: x
        _model_api.normalize = sentinel
        try:
            assert _model_api.normalize is sentinel
        finally:
            _model_api.normalize = original
        assert _model_api.normalize is original

    def test_model_api_apply_solution_graph_to_masks_is_patchable(self):
        import trackastra.model.model_api as _model_api

        original = _model_api.apply_solution_graph_to_masks
        sentinel = lambda g, m, **kw: None
        _model_api.apply_solution_graph_to_masks = sentinel
        try:
            assert _model_api.apply_solution_graph_to_masks is sentinel
        finally:
            _model_api.apply_solution_graph_to_masks = original
        assert _model_api.apply_solution_graph_to_masks is original

    def test_tracking_utils_np_stack_is_patchable(self):
        import trackastra.tracking.utils as _ttu

        original = _ttu.np.stack
        sentinel = lambda *a, **k: None
        _ttu.np.stack = sentinel
        try:
            assert _ttu.np.stack is sentinel
        finally:
            _ttu.np.stack = original
        assert _ttu.np.stack is original


# ── _stack_to_memmap interceptor logic ───────────────────────────────────────

class TestStackToMemmapLogic:
    """Unit-test the two-case interception logic used in link_frames_trackastra
    to redirect large np.stack calls to a disk memmap, without running Trackastra."""

    @staticmethod
    def _make_interceptor(T, H, W, dtype, tracked_masks_path):
        orig_stack = np.stack

        def stack_to_memmap(arrays, *args, **kwargs):
            # Case 1: list of T (H,W) arrays of the right dtype → allocate memmap
            if (
                isinstance(arrays, (list, tuple))
                and len(arrays) == T
                and np.asarray(arrays[0]).shape == (H, W)
                and np.asarray(arrays[0]).dtype == dtype
            ):
                return np.memmap(tracked_masks_path, dtype=dtype, mode="w+", shape=(T, H, W))
            # Case 2: already-filled (T,H,W) memmap passed back → return as-is
            if (
                isinstance(arrays, np.ndarray)
                and arrays.shape == (T, H, W)
                and arrays.dtype == dtype
            ):
                return arrays
            return orig_stack(arrays, *args, **kwargs)

        return stack_to_memmap

    def test_list_of_arrays_creates_disk_memmap(self, tmp_path):
        T, H, W, dtype = 3, 10, 10, np.uint16
        path = tmp_path / "tracked_masks.dat"
        fn = self._make_interceptor(T, H, W, dtype, path)
        result = fn([np.zeros((H, W), dtype=dtype) for _ in range(T)])
        assert isinstance(result, np.memmap)
        assert result.shape == (T, H, W)
        assert result.dtype == dtype

    def test_already_filled_memmap_returned_unchanged(self, tmp_path):
        T, H, W, dtype = 3, 10, 10, np.uint16
        path = tmp_path / "tracked_masks.dat"
        fn = self._make_interceptor(T, H, W, dtype, path)
        existing = np.memmap(path, dtype=dtype, mode="w+", shape=(T, H, W))
        result = fn(existing)
        assert result is existing

    def test_unrelated_stack_delegates_to_numpy(self, tmp_path):
        T, H, W, dtype = 3, 10, 10, np.uint16
        path = tmp_path / "tracked_masks.dat"
        fn = self._make_interceptor(T, H, W, dtype, path)
        # A normal list of float32 arrays should pass through to np.stack unchanged
        arrays = [np.ones((5,), dtype=np.float32) for _ in range(4)]
        result = fn(arrays)
        expected = np.stack(arrays)
        assert np.array_equal(result, expected)
