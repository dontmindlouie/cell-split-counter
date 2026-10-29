"""The scale bar is the calibration check a researcher trusts over the numbers, so its
length must be what its label says. 2026-10-28: a stray "/ 4" (since a95294d, 8/6) drew
the "10 um" bar 2.5 um long, and dropped it from the wider lightbox crops altogether."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cell_mcp_server import render  # noqa: E402


def _bar_px(img):
    # the bar is the widest solid white run in the bottom rows
    rows = (img[-20:, :, 0] == 255)
    best = 0
    for r in rows:
        run = 0
        for v in r:
            run = run + 1 if v else 0
            best = max(best, run)
    return best


def test_bar_is_as_long_as_its_label():
    img = np.zeros((360, 360, 3), np.uint8)
    out = render._scale_bar(img, um_per_px=0.25, target_um=10.0)
    assert abs(_bar_px(out) - 40) <= 1


def test_wide_lightbox_crop_still_gets_a_bar():
    img = np.zeros((360, 360, 3), np.uint8)  # 200 um across 360 px
    out = render._scale_bar(img, um_per_px=200 / 360, target_um=10.0)
    assert abs(_bar_px(out) - 18) <= 1
