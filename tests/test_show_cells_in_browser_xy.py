"""Tests for show_cells_in_browser's x/y event shape.

Added 2026-08-15 (researcher feedback): the tool previously accepted only
track_id/track_ids, forcing every "just show me this raw clicked point" request
through a track-snapping detour -- which is exactly how a neighbour mix-up
happened (a calm, correctly-snapped-but-irrelevant track got rendered instead
of the real event). x/y renders the same fixed-point crop
watch_location_over_time already does, as an alternative event shape.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    tracks = pd.DataFrame([
        {"track_id": 1, "frame": 0, "cx": 100.0, "cy": 100.0,
         "area_px": 400.0, "n_masks_in_frame": 1, "intensity_mean": 100.0},
        {"track_id": 1, "frame": 1, "cx": 100.0, "cy": 100.0,
         "area_px": 400.0, "n_masks_in_frame": 1, "intensity_mean": 100.0},
    ])
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: tracks)
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": 0.5, "n_frames": 10, "width_px": 512, "height_px": 512,
        "frame_timestamps_ms": [f * 300_000 for f in range(10)],
    })
    monkeypatch.setattr(cell_mcp_server, "_frame_png",
                        lambda well, f: np.full((512, 512), 40, dtype=np.uint8))
    monkeypatch.setattr(cell_mcp_server, "_hours", lambda well, f: f * 0.1)
    monkeypatch.setattr(cell_mcp_server, "_elapsed_str", lambda well, f: f"{f * 5}m")
    return "fake"


def test_xy_event_renders_a_page(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = cell_mcp_server.show_cells_in_browser(
        fake, events=[{"x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}])
    path = out.splitlines()[0]
    html = Path(path).read_text(encoding="utf-8")
    assert "fixed at (100, 100)" in html
    assert "(100, 100)" in html  # default label falls back to the coordinate


def test_xy_event_requires_both_coordinates(fake):
    with pytest.raises(ValueError, match="only one of x/y"):
        cell_mcp_server.show_cells_in_browser(
            fake, events=[{"x": 100.0, "start_frame": 0, "end_frame": 1}])


def test_xy_event_requires_a_frame_window(fake):
    with pytest.raises(ValueError, match="needs start_frame and end_frame"):
        cell_mcp_server.show_cells_in_browser(
            fake, events=[{"x": 100.0, "y": 100.0}])


def test_missing_shape_still_rejected(fake):
    with pytest.raises(ValueError, match="needs 'track_id', 'track_ids', or 'x'/'y'"):
        cell_mcp_server.show_cells_in_browser(fake, events=[{"label": "nothing"}])


def test_custom_label_used_for_xy_event(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = cell_mcp_server.show_cells_in_browser(
        fake, events=[{"x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1,
                       "label": "researcher click #4"}])
    path = out.splitlines()[0]
    html = Path(path).read_text(encoding="utf-8")
    assert "researcher click #4" in html
