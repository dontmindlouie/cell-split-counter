"""Tests for show_cells_in_browser's x/y (kind="point") event shape.

Added 2026-08-15 (researcher feedback): the tool previously accepted only
track_id/track_ids, forcing every "just show me this raw clicked point" request
through a track-snapping detour -- which is exactly how a neighbour mix-up
happened (a calm, correctly-snapped-but-irrelevant track got rendered instead
of the real event). x/y renders the same fixed-point crop
watch_location_over_time already does, as an alternative event shape.

Updated 2026-08-16: every event now declares its shape with a required `kind`
key ("track"/"family"/"point") instead of it being inferred from whichever
other keys happen to be present, so a wrong or missing id key fails with
"kind='point' but no 'x'/'y' (or 'waypoints') given" instead of a generic
"needs track_id/track_ids/x/y" that doesn't say which one was intended.
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


def test_point_event_renders_a_page(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = cell_mcp_server.show_cells_in_browser(
        fake, events=[{"kind": "point", "x": 100.0, "y": 100.0,
                       "start_frame": 0, "end_frame": 1}])
    path = out.splitlines()[0]
    html = Path(path).read_text(encoding="utf-8")
    assert "fixed at (100, 100)" in html
    assert "(100, 100)" in html  # default label falls back to the coordinate


def test_point_event_requires_both_coordinates(fake):
    with pytest.raises(ValueError, match="no 'x'/'y' \\(or 'waypoints'\\) given"):
        cell_mcp_server.show_cells_in_browser(
            fake, events=[{"kind": "point", "x": 100.0, "start_frame": 0, "end_frame": 1}])


def test_point_event_waypoints_renders_a_page(fake, tmp_path, monkeypatch):
    # 2026-08-16 field feedback: a drifting untracked object (debris/fragment)
    # needs the crop centre to move too, not stay fixed on the click that
    # started the search -- see render.py's _fixed_point_frames docstring.
    monkeypatch.chdir(tmp_path)
    out = cell_mcp_server.show_cells_in_browser(
        fake, events=[{"kind": "point", "waypoints": [(0, 100.0, 100.0), (1, 150.0, 150.0)],
                       "start_frame": 0, "end_frame": 1}])
    path = out.splitlines()[0]
    html = Path(path).read_text(encoding="utf-8")
    assert "interpolated across 2 waypoints" in html
    assert "drift (2 waypoints)" in html  # default label falls back to a waypoint count


def test_point_event_requires_a_frame_window(fake):
    with pytest.raises(ValueError, match="needs start_frame and end_frame"):
        cell_mcp_server.show_cells_in_browser(
            fake, events=[{"kind": "point", "x": 100.0, "y": 100.0}])


def test_missing_kind_is_rejected(fake):
    with pytest.raises(ValueError, match="needs a 'kind' of 'track', 'family', or 'point'"):
        cell_mcp_server.show_cells_in_browser(fake, events=[{"label": "nothing"}])


def test_unknown_kind_is_rejected(fake):
    with pytest.raises(ValueError, match="needs a 'kind' of 'track', 'family', or 'point'"):
        cell_mcp_server.show_cells_in_browser(
            fake, events=[{"kind": "coordinate", "x": 100.0, "y": 100.0,
                           "start_frame": 0, "end_frame": 1}])


def test_kind_track_without_track_id_is_rejected(fake):
    with pytest.raises(ValueError, match="kind='track' but no 'track_id' given"):
        cell_mcp_server.show_cells_in_browser(fake, events=[{"kind": "track"}])


def test_kind_family_without_track_ids_is_rejected(fake):
    with pytest.raises(ValueError, match="kind='family' but no 'track_ids' given"):
        cell_mcp_server.show_cells_in_browser(fake, events=[{"kind": "family"}])


def test_custom_label_used_for_point_event(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = cell_mcp_server.show_cells_in_browser(
        fake, events=[{"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0,
                       "end_frame": 1, "label": "researcher click #4"}])
    path = out.splitlines()[0]
    html = Path(path).read_text(encoding="utf-8")
    assert "researcher click #4" in html


def test_many_events_split_into_linked_pages(fake, tmp_path, monkeypatch):
    # 2026-10-17 (researcher, ZO1_STB_P2): 60 events on one page came out at 2.3 GB.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cell_mcp_server._tools_output_mod, "_EVENTS_PER_PAGE", 2)
    ev = {"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}
    out = cell_mcp_server.show_cells_in_browser(fake, events=[ev] * 5)
    paths = [Path(l) for l in out.splitlines() if l.endswith(".html") and not l.startswith("file:")]
    assert [p.name.rsplit("_", 1)[1] for p in paths] == [
        "part1of3.html", "part2of3.html", "part3of3.html"]
    middle = paths[1].read_text(encoding="utf-8")
    assert f'href="{paths[0].name}"' in middle and f'href="{paths[2].name}"' in middle
    assert "events 3&ndash;4 of 5" in middle
    assert "<h2>3. " in middle                       # headings keep the global numbering
    assert "openLightbox(0,0)" in middle             # but lightbox indices are per page
    assert "data:image/jpeg;base64," in middle and "image/png" not in middle
    assert paths[2].read_text(encoding="utf-8").count("<section") == 1


def test_bad_event_late_in_list_fails_before_any_page_is_written(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cell_mcp_server._tools_output_mod, "_EVENTS_PER_PAGE", 2)
    ok = {"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}
    with pytest.raises(ValueError, match=r"events\[4\]"):
        cell_mcp_server.show_cells_in_browser(fake, events=[ok] * 4 + [{"kind": "track"}])
    assert not list(tmp_path.rglob("*.html"))


def test_point_marker_frames_the_target_without_covering_it(fake):
    """marker_um: corner brackets around the point (2026-10-28: crowded blind cards
    asked "which mitosis is the target?"), off unless asked."""
    from cell_mcp_server import render
    kw = dict(max_images=None, crop_um=60.0, color=True, scale_bar=False, crosshair=False)
    _, plain = render._fixed_point_frames(fake, 0, 1, 100.0, 100.0, None, **kw)
    header, marked = render._fixed_point_frames(fake, 0, 1, 100.0, 100.0, None,
                                                **kw, marker_um=12.0)
    assert "corner brackets" in header
    diff = np.any(plain[0] != marked[0], axis=-1)
    ys, xs = np.nonzero(diff)
    h, w = diff.shape
    assert len(ys) > 0
    # Drawn around the centre, at ~6 um (12 px at 0.5 um/px, before upscale) out,
    # and nothing on the point itself.
    assert not diff[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5].any()
    assert abs(xs.mean() - w / 2) < 3 and abs(ys.mean() - h / 2) < 3


def test_point_marker_is_off_by_default_on_the_page(fake, tmp_path, monkeypatch):
    seen = []
    real = cell_mcp_server._fixed_point_frames
    monkeypatch.setattr(cell_mcp_server, "_fixed_point_frames",
                        lambda *a, **k: (seen.append(k.get("marker_um")), real(*a, **k))[1])
    monkeypatch.chdir(tmp_path)
    ev = {"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}
    cell_mcp_server.show_cells_in_browser(fake, events=[ev])
    cell_mcp_server.show_cells_in_browser(fake, events=[dict(ev, marker=True, marker_um=10)])
    assert seen[0] is None and seen[-1] == 10.0


def test_point_crosshair_is_opt_in_on_the_page(fake, tmp_path, monkeypatch):
    seen = []
    real = cell_mcp_server._fixed_point_frames
    monkeypatch.setattr(cell_mcp_server, "_fixed_point_frames",
                        lambda *a, **k: (seen.append(k.get("crosshair")), real(*a, **k))[1])
    monkeypatch.chdir(tmp_path)
    ev = {"kind": "point", "x": 100.0, "y": 100.0, "start_frame": 0, "end_frame": 1}
    cell_mcp_server.show_cells_in_browser(fake, events=[ev])
    cell_mcp_server.show_cells_in_browser(fake, events=[dict(ev, crosshair=True)])
    assert seen[0] is False and seen[-1] is True
