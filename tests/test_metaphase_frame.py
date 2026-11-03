import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "metaphase_frame", Path(__file__).resolve().parents[1] / "scripts" / "metaphase_frame.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def _obj(brightness, x=100.0, y=100.0):
    """One plate object per frame at (x, y), plus a brighter neighbour far away."""
    rows = [{"frame": f, "x": x, "y": y, "mean": v} for f, v in brightness.items()]
    rows += [{"frame": f, "x": x + 200, "y": y, "mean": 9999.0} for f in brightness]
    return pd.DataFrame(rows)


def test_last_bright_frame_before_segregation():
    cand = pd.DataFrame([{"first": 10, "last": 13, "x": 100.0, "y": 100.0, "minutes": 100.0}])
    fate = pd.DataFrame([{"cand": 0, "mitotic_duration_min": 50.0}])     # split at frame 15
    # peak at 12; 13 still >= 85% of it; 14 has dropped (anaphase blur); 15 is the split
    obj = _obj({9: 100, 10: 150, 11: 180, 12: 200, 13: 175, 14: 120, 15: 90})
    out = mf.estimate(cand, fate, obj, frame_min=10.0)
    assert out.loc[0, "metaphase_frame"] == 13
    assert out.loc[0, "split_frame"] == 15
    assert out.loc[0, "basis"] == "before segregation"


def test_no_segregation_uses_scan_window():
    cand = pd.DataFrame([{"first": 10, "last": 12, "x": 100.0, "y": 100.0, "minutes": 100.0}])
    fate = pd.DataFrame([{"cand": 0, "mitotic_duration_min": np.nan}])
    obj = _obj({10: 200, 11: 100, 12: 100, 13: 300})                    # 13 is past `last`
    out = mf.estimate(cand, fate, obj, frame_min=10.0)
    assert out.loc[0, "metaphase_frame"] == 10
    assert np.isnan(out.loc[0, "split_frame"])
    assert out.loc[0, "basis"] == "no segregation seen"
