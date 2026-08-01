"""Caches must notice a rebuild.

These were lru_cache'd on the well name alone, so nothing could invalidate them. A
long-running server that had called list_wells() once held every manifest for the life
of the process: on 2026-07-31 a session read BeWo M3's pre-rebuild track count more than
an hour after the rebuild, and concluded the well still needed rebuilding.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp  # noqa: E402


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    w = tmp_path / "W1"
    (w / "frames").mkdir(parents=True)
    (w / "manifest.json").write_text(json.dumps({"n_tracks": 100, "n_frames": 2}))
    pd.DataFrame([{"track_id": 1, "frame": 0}]).to_csv(w / "tracks.csv", index=False)
    monkeypatch.setattr(cell_mcp, "BUNDLE", tmp_path)
    cell_mcp._manifest.cache_clear()
    cell_mcp._tracks.cache_clear()
    return w


def test_a_rebuilt_manifest_is_re_read_not_served_from_cache(bundle):
    assert cell_mcp._manifest("W1")["n_tracks"] == 100
    (bundle / "manifest.json").write_text(json.dumps({"n_tracks": 111, "n_frames": 2}))
    import os
    st = (bundle / "manifest.json").stat()
    os.utime(bundle / "manifest.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert cell_mcp._manifest("W1")["n_tracks"] == 111, "stale read: the 7/31 bug"


def test_an_unchanged_file_is_still_served_from_cache(bundle):
    first = cell_mcp._tracks("W1")
    assert cell_mcp._tracks("W1") is first, "must not re-parse a 500k-row csv per call"


def test_the_cache_stays_bounded(bundle, monkeypatch):
    """21 wells x a 100k-500k row table would not fit in memory unbounded."""
    for i in range(2, 15):
        d = bundle.parent / f"W{i}"
        d.mkdir()
        pd.DataFrame([{"track_id": i, "frame": 0}]).to_csv(d / "tracks.csv", index=False)
    for i in range(1, 15):
        cell_mcp._tracks(f"W{i}")
    assert cell_mcp._tracks.cache_size() == 8, "14 wells read, 8 retained"
