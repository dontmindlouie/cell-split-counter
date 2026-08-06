"""Tests for cell_mcp's edge-distance readout.

Derived at read time from cx/cy rather than stored as a bundle column, so these pin
the geometry and the "don't persist it" decision: a stored column would only describe
bundles built after it was added, which is the trap `solidity` fell into.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cell_mcp_server  # noqa: E402


@pytest.fixture
def fake_well(monkeypatch):
    """A 1024x1024 frame at 0.5 um/px, so 1 px == 0.5 um and the numbers are obvious."""
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "width_px": 1024, "height_px": 1024, "pixel_size_um": 0.5,
    })
    return "fake"


def test_edge_um_uses_the_nearest_of_all_four_sides(fake_well):
    # Dead centre: 512 px to every side -> 256 um.
    assert cell_mcp_server._edge_um(fake_well, 512, 512) == pytest.approx(256.0)
    # Near the left edge: 10 px -> 5 um, and left is nearer than top.
    assert cell_mcp_server._edge_um(fake_well, 10, 400) == pytest.approx(5.0)
    # Near the bottom: 1024-1020 = 4 px -> 2 um.
    assert cell_mcp_server._edge_um(fake_well, 500, 1020) == pytest.approx(2.0)
    # Near the right edge -- the M14 case, a track at cx=1010 of 1024.
    assert cell_mcp_server._edge_um(fake_well, 1010, 500) == pytest.approx(7.0)


def test_edge_um_is_nan_when_the_manifest_lacks_dimensions(monkeypatch):
    """Older bundles carry no width_px/height_px. That must degrade to 'unknown'
    rather than to a confidently wrong distance."""
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {"pixel_size_um": 0.5})
    v = cell_mcp_server._edge_um("fake", 512, 512)
    assert v != v, "expected NaN, not a fabricated number"


def test_edge_um_scales_with_pixel_size(monkeypatch):
    """The answer is in microns, so it must track calibration -- the whole reason
    the bundle hard-fails rather than defaulting a pixel size."""
    monkeypatch.setattr(cell_mcp_server, "_manifest", lambda well: {
        "width_px": 100, "height_px": 100, "pixel_size_um": 2.0,
    })
    assert cell_mcp_server._edge_um("fake", 10, 50) == pytest.approx(20.0)
