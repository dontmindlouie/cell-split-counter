"""Tests for per-channel bundles (2026-10-18).

Multi-channel wells now measure every channel per nucleus (inside the mask and in a
ring just outside it) and store each channel as its own grey stack, so the tools and
a researcher's own scripts can reach a second marker without reopening the ND2. The
ZO1_STB experiment (ZO1 + H2B) was the first user: its scripts were reading 69 GB off
an external drive to get ZO1.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402
from scripts.build_bundle import channel_column, measure_channels  # noqa: E402

META = [{"index": 0, "column": "AF488"}, {"index": 1, "column": "AF594"}]


def test_channel_column_is_a_safe_identifier():
    assert channel_column("AF488") == "AF488"
    assert channel_column("Cy5 (far red)") == "Cy5_far_red"
    assert channel_column("???") == "ch"


def test_nucleus_and_ring_means_are_exact():
    lab = np.zeros((40, 40), np.uint16)
    lab[15:25, 15:25] = 1
    ch = np.zeros((2, 40, 40), np.uint16)
    ch[0][lab == 1] = 100                    # inside the nucleus
    ring = np.zeros_like(lab, bool)
    ring[12:28, 12:28] = True
    ring[lab > 0] = False
    ch[0][ring] = 500                        # a junction-like rim, 3 px wide
    ch[0][(ch[0] == 0)] = 9                  # far background: must not leak into the ring
    ch[1][lab == 1] = 2000
    m = measure_channels(lab, ch, ring_px=3, channel_meta=META)[1]
    assert m["AF488_nuc_mean"] == 100
    assert m["AF594_nuc_mean"] == 2000
    # expand_labels uses Euclidean distance, so the 3 px ring rounds its corners:
    # every ring pixel is still a rim pixel, none is background.
    assert m["AF488_ring_mean"] == 500


def test_neighbouring_rings_split_the_gap_and_never_enter_another_nucleus():
    lab = np.zeros((20, 40), np.uint16)
    lab[5:15, 5:15] = 1
    lab[5:15, 19:29] = 2                     # 4 px gap between the two nuclei
    ch = np.zeros((2, 20, 40), np.uint16)
    ch[0][lab == 1] = 1000                   # bright nucleus 1 must not appear in 2's ring
    ch[0][lab == 2] = 50
    ch[0][:, 15:17] = 10                     # gap half nearer nucleus 1
    ch[0][:, 17:19] = 30                     # gap half nearer nucleus 2
    m = measure_channels(lab, ch, ring_px=6, channel_meta=META)
    from skimage.segmentation import expand_labels
    ring = expand_labels(lab, 6)
    ring[lab > 0] = 0
    assert not ((ring == 1) & (ring == 2)).any()
    for k in (1, 2):
        assert m[k][f"AF488_ring_mean"] < 50      # neither ring picked up nucleus 1's 1000
    assert m[1]["AF488_nuc_mean"] == 1000


def test_fully_enclosed_nucleus_gets_no_ring_value_instead_of_zero():
    lab = np.ones((30, 30), np.uint16)       # nucleus 1 fills everything but a hole
    lab[10:20, 10:20] = 2                    # nucleus 2 enclosed by nucleus 1
    ch = np.full((2, 30, 30), 7, np.uint16)
    m = measure_channels(lab, ch, ring_px=3, channel_meta=META)
    assert "AF488_nuc_mean" in m[2]
    assert "AF488_ring_mean" not in m[2]


@pytest.fixture
def channel_bundle(tmp_path, monkeypatch):
    well = "W"
    chans = [
        {"index": 0, "name": "AF488", "column": "AF488", "dir": "channels/AF488",
         "color_rgb": [54, 255, 0], "segmentation": False},
        {"index": 1, "name": "AF594", "column": "AF594", "dir": "channels/AF594",
         "color_rgb": [255, 0, 0], "segmentation": True},
    ]
    root = tmp_path / well
    (root / "frames").mkdir(parents=True)
    g = np.zeros((8, 8), np.uint8); g[:, :4] = 200
    r = np.zeros((8, 8), np.uint8); r[:4, :] = 100
    for c, img in (("AF488", g), ("AF594", r)):
        (root / "channels" / c).mkdir(parents=True)
        cv2.imwrite(str(root / "channels" / c / "frame_00000.png"), img)
    cv2.imwrite(str(root / "frames" / "frame_00000.png"), r)
    tracks = pd.DataFrame([
        {"track_id": 1, "frame": f, "cx": 4.0, "cy": 4.0, "AF488_nuc_mean": 100.0 + f,
         "AF488_ring_mean": 400.0 - 10 * f, "AF594_nuc_mean": 2000.0, "AF594_ring_mean": 50.0}
        for f in range(10)
    ] + [
        {"track_id": 2, "frame": f, "cx": 9.0, "cy": 9.0, "AF488_nuc_mean": 100.0,
         "AF488_ring_mean": 200.0, "AF594_nuc_mean": 1000.0, "AF594_ring_mean": None}
        for f in range(10)
    ])
    monkeypatch.setattr(cell_mcp_server, "BUNDLE", tmp_path)
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {
        "n_frames": 10, "channels": chans, "ring_um": 3.0, "track_multiplicity": {}})
    monkeypatch.setattr(cell_mcp_server, "_tracks", lambda w: tracks)
    monkeypatch.setattr(cell_mcp_server, "_hours", lambda w, f: f * 0.05)
    return well, g, r


def test_frame_is_composed_from_channels_when_no_stored_composite(channel_bundle):
    well, g, r = channel_bundle
    img = cell_mcp_server._frame_png(well, 0)
    assert img.shape == (8, 8, 3)
    # Same recipe as src/ingest.py's frames_display: tint by LUT, per-plane max (BGR).
    expect = np.maximum(
        np.stack([g * 0.0, g * 1.0, g * (54 / 255)], -1),
        np.stack([r * 0.0, r * 0.0, r * 1.0], -1)).astype(np.uint8)
    assert np.array_equal(img, expect)


def test_stored_composite_still_wins_for_older_bundles(channel_bundle, tmp_path):
    well, _, _ = channel_bundle
    (tmp_path / well / "frames_display").mkdir()
    cv2.imwrite(str(tmp_path / well / "frames_display" / "frame_00000.png"),
                np.full((8, 8, 3), 77, np.uint8))
    assert (cell_mcp_server._frame_png(well, 0) == 77).all()


def test_list_channels_names_roles_and_columns(channel_bundle):
    out = cell_mcp_server.list_channels(channel_bundle[0])
    assert "AF594 | #ff0000 | SEGMENTED" in out
    assert "AF488_ring_mean" in out
    assert "Ask the researcher" in out


def test_channel_profile_reports_raw_and_field_relative_values(channel_bundle):
    out = cell_mcp_server.get_channel_profile(channel_bundle[0], [1, 2], "af488", region="ring")
    assert "AF488 ring brightness" in out
    assert "min 310, median 355, max 400" in out          # track 1 raw ring values
    assert "track 2:" in out and "min 200" in out


def test_channel_profile_handles_a_track_with_no_ring_pixels(channel_bundle):
    out = cell_mcp_server.get_channel_profile(channel_bundle[0], [2], "AF594", region="ring")
    assert "no ring pixels" in out


def test_unknown_or_ambiguous_channel_is_rejected(channel_bundle):
    with pytest.raises(ValueError, match="not found"):
        cell_mcp_server.get_channel_profile(channel_bundle[0], [1], "Cy5")
    with pytest.raises(ValueError, match="ambiguous"):
        cell_mcp_server.get_channel_profile(channel_bundle[0], [1], "AF")


def test_old_bundle_without_channels_says_rebuild(monkeypatch):
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda w: {"n_frames": 1})
    assert "no per-channel measurements" in cell_mcp_server.list_channels("old")
    with pytest.raises(ValueError, match="rebuilding"):
        cell_mcp_server.get_channel_profile("old", [1], "AF488")
