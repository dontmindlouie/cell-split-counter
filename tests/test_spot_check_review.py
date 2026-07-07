"""Tests for the spot-check review tool's sampling/scoring logic (scripts/reports/spot_check_review.py).

Focused on the exact class of bug found live on 2026-07-08: pipeline_verdict must
reflect what events.csv actually shipped, not verdict.txt's raw pre-floor call --
these differ specifically on gpt_floor_downgrade events, and a live human session
had its scoring silently inverted for that bucket before this was caught.
"""

import struct

from scripts.reports.spot_check_review import (
    _bucket_for,
    _centroid_in_crop_pct,
    _effective_verdict,
    _png_size,
)


def _fake_png(tmp_path, width, height, name="fake.png"):
    # _png_size only reads the IHDR chunk (bytes 16-24) -- doesn't need to be a
    # fully valid/decodable PNG for this test's purposes.
    header = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    path = tmp_path / name
    path.write_bytes(header + b"\x00" * 8)
    return path


def _row(confidence="0.0", near_edge="0", **flags):
    base = {
        "claude_confidence": confidence,
        "near_edge": near_edge,
        "misaligned_chromosomes": "0", "lagging_chromosome": "0",
        "anaphase_bridge": "0", "micronucleus": "0", "binucleation": "0",
    }
    base.update(flags)
    return base


# --- _effective_verdict: the exact bug this session found -----------------------

def test_effective_verdict_no_verdict_is_always_real():
    assert _effective_verdict(None, 0.0) == "real"
    assert _effective_verdict(None, 0.9) == "real"


def test_effective_verdict_gpt_floor_downgrade_is_false_positive():
    # This is the bucket definition itself: raw verdict says real, CSV confidence 0.
    verdict = {"verdict": "real", "confidence": "0.72"}
    assert _effective_verdict(verdict, 0.0) == "false_positive"


def test_effective_verdict_confirmed_real_stays_real():
    verdict = {"verdict": "real", "confidence": "0.9"}
    assert _effective_verdict(verdict, 0.9) == "real"


def test_effective_verdict_rejected_stays_false_positive():
    verdict = {"verdict": "false_positive", "confidence": "0.0"}
    assert _effective_verdict(verdict, 0.0) == "false_positive"


# --- _bucket_for: priority order matters -----------------------------------------

def test_bucket_no_verdict():
    assert _bucket_for(_row(), None) == "no_verdict"


def test_bucket_gpt_floor_downgrade_takes_priority_over_near_edge():
    row = _row(confidence="0.0", near_edge="1")
    verdict = {"verdict": "real", "confidence": "0.7"}
    assert _bucket_for(row, verdict) == "gpt_floor_downgrade"


def test_bucket_anomaly_flagged_requires_confirmed():
    verdict = {"verdict": "real", "confidence": "0.86"}
    confirmed_with_anomaly = _row(confidence="0.86", micronucleus="1")
    assert _bucket_for(confirmed_with_anomaly, verdict) == "anomaly_flagged"
    # Anomaly flag set but confidence below 0.5 -- not "confirmed", falls through.
    unconfirmed_with_anomaly = _row(confidence="0.3", micronucleus="1")
    assert _bucket_for(unconfirmed_with_anomaly, verdict) != "anomaly_flagged"


def test_bucket_near_edge():
    verdict = {"verdict": "real", "confidence": "0.6"}
    assert _bucket_for(_row(confidence="0.6", near_edge="1"), verdict) == "near_edge"


def test_bucket_confirmed_high():
    verdict = {"verdict": "real", "confidence": "0.9"}
    assert _bucket_for(_row(confidence="0.9"), verdict) == "confirmed_high"


def test_bucket_false_positive_default():
    verdict = {"verdict": "false_positive", "confidence": "0.0"}
    assert _bucket_for(_row(confidence="0.0"), verdict) == "false_positive"


# --- _centroid_in_crop_pct: crosshair placement, incl. edge-clamped crops -------

def test_centroid_dead_center_when_not_edge_clamped(tmp_path):
    # Cellpose centroid far from any frame edge -- crop is the full 384x384, so the
    # centroid sits at exactly R=192 from each edge -> dead center (50%, 50%).
    path = _fake_png(tmp_path, 384, 384)
    x_pct, y_pct = _centroid_in_crop_pct(path, cx=480.0, cy=392.0)
    assert x_pct == 50.0
    assert y_pct == 50.0


def test_centroid_shifts_toward_clamped_edge(tmp_path):
    # centroid_x=28 is less than R=192, so the crop was clamped on the left (x0=0)
    # and the true saved width is cx + R = 220 -- centroid sits at 28/220, not 50%.
    path = _fake_png(tmp_path, 220, 384)
    x_pct, y_pct = _centroid_in_crop_pct(path, cx=28.0, cy=392.0)
    assert abs(x_pct - (28 / 220 * 100)) < 0.01
    assert y_pct == 50.0


def test_png_size_reads_ihdr(tmp_path):
    path = _fake_png(tmp_path, 100, 200)
    assert _png_size(path) == (100, 200)
