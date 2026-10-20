"""Tests for measure_points (2026-10-20).

Validated on real data first: all 954 of the 2026-10-19 TSC batch2 / Bewo M5 Fiji
clicks, fed in as raw Fiji columns, gave the same 854 inside-a-mask as the earlier
hand scoring, recovered Bewo M5's off-calibration Fiji X scale (0.566454 vs the
file's 0.567490), and flagged the same frame-shifted clicks (M4 #56, M6 #16/#18).
These tests pin the pieces that run on every call without needing a bundle or .nd2.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402

UM_PX = 0.5


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Two frames, 64x64, 0.5 um/px. Cell A: raw_label 7 -> track 70, a 10x10 square at
    x 10-19, y 10-19. Cell B: raw_label 8 -> track 80, x 40-49, y 40-49, present in
    frame 0 only. One channel, AF488, with nucleus/ring means."""
    lab0 = np.zeros((64, 64), np.uint16)
    lab0[10:20, 10:20] = 7
    lab0[40:50, 40:50] = 8
    lab1 = np.zeros((64, 64), np.uint16)
    lab1[10:20, 10:20] = 7
    labs = {0: lab0, 1: lab1}
    rows = [
        dict(track_id=70, frame=0, raw_label=7, cx=14.5, cy=14.5, area_um2=25.0, intensity_mean=100.0,
             intensity_integrated=10000.0, n_masks_in_frame=1, AF488_nuc_mean=50.0, AF488_ring_mean=20.0),
        dict(track_id=80, frame=0, raw_label=8, cx=44.5, cy=44.5, area_um2=25.0, intensity_mean=200.0,
             intensity_integrated=20000.0, n_masks_in_frame=1, AF488_nuc_mean=150.0, AF488_ring_mean=40.0),
        dict(track_id=70, frame=1, raw_label=7, cx=14.5, cy=14.5, area_um2=25.0, intensity_mean=100.0,
             intensity_integrated=10000.0, n_masks_in_frame=2, AF488_nuc_mean=60.0, AF488_ring_mean=20.0),
    ]
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda well: pd.DataFrame(rows))
    monkeypatch.setattr(cell_mcp_server, "_label_img", lambda well, f: labs.get(f))
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "pixel_size_um": UM_PX, "n_frames": 2, "frame_timestamps_ms": [0, 180_000],
        "track_multiplicity": {"suspect_tracks": [80]},
        "channels": [{"name": "AF488", "column": "AF488"}],
    })
    monkeypatch.setattr(cell_mcp_server, "BUNDLE", tmp_path / "bundle")
    return "fake"


def _rows(out):
    lines = out.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("label,"))
    return list(csv.DictReader(lines[i:]))


def test_inside_snapped_and_none(fake):
    out = cell_mcp_server.measure_points(fake, points=[
        {"frame": 0, "x": 15, "y": 15, "label": "in"},
        {"frame": 0, "x": 22, "y": 15, "label": "snap"},   # 3 px = 1.5 um off the edge
        {"frame": 0, "x": 30, "y": 30, "label": "none"},
    ], units="px")
    r = {x["label"]: x for x in _rows(out)}
    assert r["in"]["match"] == "inside" and r["in"]["track_id"] == "70"
    assert r["snap"]["match"].startswith("snapped 1.5 um") and r["snap"]["track_id"] == "70"
    assert r["none"]["match"] == "none"
    assert "no cell at: none" in out


def test_values_diameter_channels_and_warnings(fake):
    out = cell_mcp_server.measure_points(fake, points=[
        {"frame": 0, "x": 45, "y": 45}, {"frame": 1, "x": 15, "y": 15}], units="px")
    a, b = _rows(out)
    assert float(a["diameter_um"]) == pytest.approx(2 * np.sqrt(25 / np.pi), abs=0.01)
    assert a["AF488_nuc_mean"] == "150.0"
    # frame 0 field median of AF488_nuc_mean over tracks 70, 80 = 100
    assert float(a["AF488_nuc_mean_vs_field"]) == pytest.approx(1.5)
    assert "merged-cells track" in a["warn"]
    assert "2 shapes share this id" in b["warn"]
    assert float(b["hours"]) == pytest.approx(0.05)


def test_frame_tolerance_finds_a_flickered_out_mask(fake):
    pts = [{"frame": 1, "x": 45, "y": 45}]  # cell B is only in frame 0
    assert _rows(cell_mcp_server.measure_points(fake, points=pts, units="px"))[0]["match"] == "none"
    r = _rows(cell_mcp_server.measure_points(fake, points=pts, units="px", frame_tolerance=1))[0]
    assert r["match"] == "inside (frame -1)" and r["track_id"] == "80"


def test_fiji_columns_are_one_based_and_whole_pixels_read_as_px(fake):
    out = cell_mcp_server.measure_points(fake, points=[
        {"Slice": 1, "X": 15, "Y": 15}, {"Slice": 2, "X": 16, "Y": 12}, {"Slice": 1, "X": 45, "Y": 41}])
    assert "1-based, Fiji" in out and "units: pixels" in out
    assert [r["frame"] for r in _rows(out)] == ["0", "1", "0"]


def test_micron_fit_recovers_an_off_calibration_x_scale():
    """Fiji's X calibration 0.2% off the file's (as on Bewo M5): only the fitted scale
    puts every click back on its whole pixel."""
    rng = np.random.default_rng(0)
    px = rng.integers(0, 1024, size=(40, 2))
    sx, sy = 0.566455, 0.56749
    pts = [{"x_in": x * sx, "y_in": y * sy} for x, y in px]
    note = cell_mcp_server._resolve_units(pts, 0.56749, "auto")
    assert "fitted to whole pixels" in note and "differs slightly" in note
    assert all(abs(p["x_px"] - x) < 0.05 and abs(p["y_px"] - y) < 0.05 for p, (x, y) in zip(pts, px))


def test_non_click_microns_fall_back_to_file_scale():
    pts = [{"x_in": v, "y_in": v} for v in (10.13, 55.71, 201.37, 333.33)]
    note = cell_mcp_server._resolve_units(pts, 0.5, "auto")
    assert "file's" in note and pts[0]["x_px"] == pytest.approx(20.26)


def test_duplicates_flagged(fake):
    out = cell_mcp_server.measure_points(fake, points=[
        {"frame": 0, "x": 12, "y": 12, "label": "a"}, {"frame": 1, "x": 18, "y": 18, "label": "b"},
        {"frame": 0, "x": 45, "y": 45, "label": "c"}], units="px")
    assert "possible duplicate clicks: a/b" in out
    assert "c/" not in out and "/c" not in out


def test_csv_in_writes_beside_input_and_refuses_the_bundle(fake, tmp_path):
    src = tmp_path / "clicks.csv"
    pd.DataFrame({"click": [1, 2], "frame0": [0, 0], "x_px": [15, 45], "y_px": [15, 45]}).to_csv(src, index=False)
    out = cell_mcp_server.measure_points(fake, csv_path=str(src))
    written = tmp_path / "clicks_measured.csv"
    assert written.is_file() and str(written) in out
    assert list(pd.read_csv(written).track_id) == [70, 80]
    (tmp_path / "bundle" / "fake").mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the bundle"):
        cell_mcp_server.measure_points(fake, csv_path=str(src), out_csv=str(tmp_path / "bundle" / "fake" / "x.csv"))


def test_exactly_one_input(fake):
    with pytest.raises(ValueError, match="exactly one"):
        cell_mcp_server.measure_points(fake)
