"""Tests for review_ambiguous three-tier routing (no real API calls)."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.classify import EventType, LineageEvent
from src.review import review_ambiguous


def make_event(track_id, frame=10, confidence=0.5, source="rule", parent_id=1):
    return LineageEvent(
        track_id=track_id,
        parent_id=parent_id,
        frame=frame,
        event_type=EventType.NORMAL_SPLIT,
        classification_source=source,
        confidence=confidence,
        centroid=(100.0, 100.0),
    )


# ── suppression tier ──────────────────────────────────────────────────────────

def test_events_below_lower_threshold_are_dropped(tmp_path):
    events = [make_event(1, confidence=0.0), make_event(2, confidence=0.04)]
    result = review_ambiguous(events, tmp_path, lower_threshold=0.05, upper_threshold=1.0)
    assert result == []


def test_events_at_lower_threshold_are_not_suppressed(tmp_path):
    event = make_event(1, confidence=0.05)
    with patch("src.review.anthropic.Anthropic") as mock_cls, \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"verdict":"false_positive","confidence":0.0,"split_type":null,"description":"noise"}')]
        )
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)
    # event was routed to Claude (not suppressed), came back as FP with confidence 0.0
    assert len(result) == 1
    assert result[0].classification_source == "claude"


# ── auto-confirm tier ─────────────────────────────────────────────────────────

def test_events_at_upper_threshold_pass_through_unchanged(tmp_path):
    event = make_event(1, confidence=1.0)
    result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)
    assert result == [event]


def test_auto_confirm_does_not_call_api(tmp_path):
    events = [make_event(i, confidence=1.0) for i in range(5)]
    with patch("src.review.anthropic.Anthropic") as mock_cls:
        review_ambiguous(events, tmp_path, lower_threshold=0.05, upper_threshold=1.0)
    mock_cls.assert_not_called()


# ── claude review tier ────────────────────────────────────────────────────────

def test_daughters_sharing_split_point_share_one_api_call(tmp_path):
    # Two daughters from the same parent/frame → one Claude call, not two.
    d1 = make_event(track_id=2, frame=10, confidence=0.5, parent_id=1)
    d2 = make_event(track_id=3, frame=10, confidence=0.5, parent_id=1)

    with patch("src.review._review_split") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = ("real", 0.9, "clean division")
        review_ambiguous([d1, d2], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert mock_review.call_count == 1


def test_claude_real_verdict_updates_source_and_confidence(tmp_path):
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_split") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = ("real", 0.85, "symmetric: clear division")
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert len(result) == 1
    assert result[0].classification_source == "claude"
    assert result[0].confidence == 0.85
    assert result[0].claude_notes == "symmetric: clear division"


def test_claude_false_positive_zeroes_confidence_but_keeps_notes(tmp_path):
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_split") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = ("false_positive", 0.1, "z-plane focus drift")
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert result[0].confidence == 0.0
    assert result[0].claude_notes == "z-plane focus drift"


def test_max_reviews_cap_passes_excess_events_unchanged(tmp_path):
    # 3 unique split points, cap=2 → third passes through unchanged.
    events = [
        make_event(track_id=2, frame=10, parent_id=1, confidence=0.5),
        make_event(track_id=3, frame=20, parent_id=4, confidence=0.5),
        make_event(track_id=5, frame=30, parent_id=6, confidence=0.5),
    ]
    with patch("src.review._review_split") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = ("real", 0.9, "")
        result = review_ambiguous(events, tmp_path, lower_threshold=0.05, upper_threshold=1.0, max_reviews=2)

    assert mock_review.call_count == 2
    # Third event passes through with original rule source
    capped = next(e for e in result if e.track_id == 5)
    assert capped.classification_source == "rule"


def test_missing_api_key_raises(tmp_path):
    event = make_event(1, confidence=0.5)
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)
