"""Tests for review_ambiguous three-tier routing (no real API calls)."""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.classify import EventType, LineageEvent
from src.review import review_ambiguous, review_deaths


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
            content=[MagicMock(text='{"verdict":"false_positive","confidence":0.0,"split_type":null,"description":"noise",'
                                     '"acd_division_type":null,"misaligned_chromosomes":null,"lagging_chromosome":null,'
                                     '"anaphase_bridge":null,"micronucleus":null,"anomaly_notes":null}')]
        )
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)
    # event was routed to Claude (not suppressed), came back as FP with confidence 0.0
    assert len(result) == 1
    assert result[0].classification_source == "claude-haiku-4-5"


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

    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.9, "split_type": None,
                                     "description": "clean division", "acd_division_type": "bipolar",
                                     "misaligned_chromosomes": False, "lagging_chromosome": False,
                                     "anaphase_bridge": False, "micronucleus": False, "anomaly_notes": None}
        review_ambiguous([d1, d2], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert mock_review.call_count == 1


def test_claude_real_verdict_updates_source_confidence_and_classification(tmp_path):
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.85, "split_type": "symmetric",
                                     "description": "clear division", "acd_division_type": "bipolar",
                                     "misaligned_chromosomes": True, "lagging_chromosome": False,
                                     "anaphase_bridge": False, "micronucleus": False, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert len(result) == 1
    assert result[0].classification_source == "claude-haiku-4-5"
    assert result[0].confidence == 0.85
    assert result[0].ai_notes == "symmetric: clear division"
    assert result[0].acd_division_type == "bipolar"
    assert result[0].misaligned_chromosomes is True
    assert result[0].split_type == "symmetric"
    assert result[0].event_type == EventType.NORMAL_SPLIT


def test_claude_false_positive_zeroes_confidence_keeps_notes_no_classification(tmp_path):
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "false_positive", "confidence": 0.1, "split_type": None,
                                     "description": "z-plane focus drift", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert result[0].confidence == 0.0
    assert result[0].ai_notes == "z-plane focus drift"
    assert result[0].acd_division_type is None
    assert result[0].split_type is None


# ── failed-split reclassification (un-shelved 2026-07-09) ────────────────────

def test_confirmed_failed_split_reclassifies_event_type(tmp_path):
    event = make_event(1, confidence=0.5)
    assert event.event_type == EventType.NORMAL_SPLIT
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.8, "split_type": "failed",
                                     "description": "daughters re-fused", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert result[0].event_type == EventType.FAILED_SPLIT
    assert result[0].split_type == "failed"
    # still a real, confirmed event -- not zeroed out like a false positive
    assert result[0].confidence == 0.8


def test_false_positive_failed_split_type_is_impossible_stays_unclassified(tmp_path):
    # split_type is only meaningful when verdict == real; a false_positive result should
    # never carry split_type through, regardless of what the (malformed) response contains.
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "false_positive", "confidence": 0.1, "split_type": "failed",
                                     "description": "noise", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert result[0].event_type == EventType.NORMAL_SPLIT
    assert result[0].split_type is None


def test_gpt_floor_skipped_for_failed_split(tmp_path):
    # Real-API validation (2026-07-09) found genuine failed-division confidence clusters
    # at 0.72-0.82, below the 0.85 GPT floor -- the floor must not suppress this category,
    # or the un-shelved FAILED_SPLIT reclassification never gets a chance to run.
    event = make_event(1, confidence=0.5)
    with patch("src.review_gpt.review_and_classify_gpt") as mock_review, \
         patch("openai.AzureOpenAI"), \
         patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "test", "AZURE_OPENAI_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.75, "split_type": "failed",
                                     "description": "re-fuses", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0,
                                   backend="gpt", min_gpt_confidence=0.85)

    assert result[0].event_type == EventType.FAILED_SPLIT
    assert result[0].split_type == "failed"
    assert result[0].confidence == 0.75  # NOT zeroed by the floor


def test_gpt_floor_still_applies_to_non_failed_splits(tmp_path):
    # Same below-floor confidence, but a normal split_type -- floor behavior must be
    # unchanged here; only split_type=="failed" is exempted.
    event = make_event(1, confidence=0.5)
    with patch("src.review_gpt.review_and_classify_gpt") as mock_review, \
         patch("openai.AzureOpenAI"), \
         patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "test", "AZURE_OPENAI_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.75, "split_type": "symmetric",
                                     "description": "borderline division", "acd_division_type": "bipolar",
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0,
                                   backend="gpt", min_gpt_confidence=0.85)

    assert result[0].event_type == EventType.NORMAL_SPLIT
    assert result[0].confidence == 0.0
    assert result[0].split_type is None


def test_multi_way_mismatch_is_flagged_not_silently_reclassified(tmp_path):
    # Tracker topology only found 2 children (NORMAL_SPLIT), but the model visually saw 3+
    # daughters. We don't silently override tracker topology -- split_type carries the
    # mismatch so a downstream reader can catch it (see docs/output_schema.md gotcha).
    event = make_event(1, confidence=0.5)
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.8, "split_type": "multi_way",
                                     "description": "three daughters visible", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
        result = review_ambiguous([event], tmp_path, lower_threshold=0.05, upper_threshold=1.0)

    assert result[0].event_type == EventType.NORMAL_SPLIT
    assert result[0].split_type == "multi_way"


def test_max_reviews_cap_passes_excess_events_unchanged(tmp_path):
    # 3 unique split points, cap=2 → third passes through unchanged.
    events = [
        make_event(track_id=2, frame=10, parent_id=1, confidence=0.5),
        make_event(track_id=3, frame=20, parent_id=4, confidence=0.5),
        make_event(track_id=5, frame=30, parent_id=6, confidence=0.5),
    ]
    with patch("src.review._review_and_classify") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {"verdict": "real", "confidence": 0.9, "split_type": None,
                                     "description": "", "acd_division_type": None,
                                     "misaligned_chromosomes": None, "lagging_chromosome": None,
                                     "anaphase_bridge": None, "micronucleus": None, "anomaly_notes": None}
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


# ── backend content-drift guard ───────────────────────────────────────────────
# 2026-07-10: review_and_classify_gpt used to hand-duplicate _review_and_classify's
# content-building loop. The frame-order/offset labeling fix (backlog #25) silently
# never reached the GPT backend -- the one M5 actually uses -- for a full day because
# nothing exercised review_and_classify_gpt's own prompt-building logic. Both backends
# now call the same _build_review_content() helper; this test asserts their outputs
# stay structurally identical (same text labels, same image count/order) even though
# the image *blocks* differ (Claude's {"type":"image", ...} vs GPT's
# {"type":"image_url", ...}), so a future change to one can't silently diverge again.

def _make_frame(frame_dir: Path, idx: int) -> None:
    import cv2
    import numpy as np
    img = np.full((20, 20), idx % 256, dtype=np.uint8)
    cv2.imwrite(str(frame_dir / f"frame_{idx:05d}_x.png"), img)


def test_claude_and_gpt_backends_send_structurally_identical_content(tmp_path):
    """Calls the real _review_and_classify and review_and_classify_gpt (client mocked,
    no network) and captures the actual `content` list each one sent, rather than
    calling _build_review_content directly -- a test that bypasses the backend
    functions themselves would NOT have caught the actual 2026-07-10 bug, since that
    bug was review_and_classify_gpt having its own duplicated loop instead of calling
    the shared helper at all."""
    from src.review import _FRAMES_AFTER, _FRAMES_BEFORE, _FRAME_STRIDE, _review_and_classify
    from src.review_gpt import review_and_classify_gpt

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    event = make_event(1, frame=100)
    span = _FRAMES_BEFORE * _FRAME_STRIDE
    for idx in range(event.frame - span, event.frame + span + 1):
        _make_frame(frame_dir, idx)

    fake_json = ('{"verdict":"real","confidence":0.9,"split_type":"symmetric","description":"d",'
                 '"acd_division_type":"bipolar","misaligned_chromosomes":false,"lagging_chromosome":false,'
                 '"anaphase_bridge":false,"micronucleus":false,"binucleation":false,"anomaly_notes":null}')

    claude_client = MagicMock()
    claude_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=fake_json)],
        usage=MagicMock(input_tokens=1, output_tokens=1),
    )
    _review_and_classify(claude_client, event, frame_dir, "claude-model")
    claude_content = claude_client.messages.create.call_args.kwargs["messages"][0]["content"]

    gpt_client = MagicMock()
    gpt_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=fake_json))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
    )
    review_and_classify_gpt(gpt_client, event, frame_dir, "gpt-model")
    gpt_content = gpt_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]

    assert len(claude_content) == len(gpt_content) == (_FRAMES_BEFORE + 1 + _FRAMES_AFTER) * 2 + 1
    for claude_block, gpt_block in zip(claude_content, gpt_content):
        if claude_block["type"] == "text":
            assert gpt_block["type"] == "text"
            assert claude_block["text"] == gpt_block["text"]
        else:
            assert claude_block["type"] == "image"
            assert gpt_block["type"] == "image_url"

    claude_texts = [b["text"] for b in claude_content if b["type"] == "text"]
    assert "chronological order" in claude_texts[0]
    assert "chronological order" in claude_texts[-1]


# ── review_deaths (backlog #23/#27, 2026-07-10) ───────────────────────────────────

def make_death_event(track_id, frame=100, confidence=0.8, parent_id=None):
    return LineageEvent(
        track_id=track_id,
        parent_id=parent_id,
        frame=frame,
        event_type=EventType.DEATH,
        classification_source="rule",
        confidence=confidence,
        centroid=(100.0, 100.0),
    )


def test_review_deaths_empty_list_returns_empty(tmp_path):
    assert review_deaths([], tmp_path, backend="claude") == []


def test_review_deaths_invalid_backend_raises(tmp_path):
    with pytest.raises(ValueError, match="backend must be"):
        review_deaths([make_death_event(1)], tmp_path, backend="bogus")


def test_review_deaths_missing_api_key_raises(tmp_path):
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            review_deaths([make_death_event(1)], tmp_path, backend="claude")


def test_review_deaths_updates_fields_from_verdict(tmp_path):
    event = make_death_event(1, confidence=0.6)
    with patch("src.review._review_death_event") as mock_review, \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        mock_review.return_value = {
            "likely_division_dropout": True, "confidence": 0.75,
            "description": "chromatin condensing, likely prophase", "anomaly_notes": None,
        }
        result = review_deaths([event], tmp_path, backend="claude")

    assert len(result) == 1
    reviewed = result[0]
    assert reviewed.likely_division_dropout is True
    assert reviewed.confidence == 0.75
    assert reviewed.ai_notes == "chromatin condensing, likely prophase"
    # event_type/split_topology must NOT change -- review_deaths flags for a human,
    # it can't reclassify a death into a split without the (lost) daughter track data.
    assert reviewed.event_type == EventType.DEATH


def test_review_deaths_review_error_is_marked_and_type_unchanged(tmp_path):
    event = make_death_event(1)
    with patch("src.review._review_death_event", side_effect=RuntimeError("boom")), \
         patch("src.review.anthropic.Anthropic"), \
         patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        result = review_deaths([event], tmp_path, backend="claude")

    assert result[0].review_error is True
    assert result[0].event_type == EventType.DEATH
    assert result[0].likely_division_dropout is None


def test_claude_and_gpt_death_review_backends_send_structurally_identical_content(tmp_path):
    """Same drift guard as the split-review test above, applied proactively to the
    new death-review path so this can't repeat the 2026-07-10 backend-drift bug."""
    from src.review import _FRAMES_AFTER, _FRAMES_BEFORE, _FRAME_STRIDE, _review_death_event
    from src.review_gpt import review_death_gpt

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    event = make_death_event(1, frame=100)
    span = _FRAMES_BEFORE * _FRAME_STRIDE
    for idx in range(event.frame - span, event.frame + span + 1):
        _make_frame(frame_dir, idx)

    fake_json = '{"likely_division_dropout":false,"confidence":0.9,"description":"d","anomaly_notes":null}'

    claude_client = MagicMock()
    claude_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=fake_json)],
        usage=MagicMock(input_tokens=1, output_tokens=1),
    )
    _review_death_event(claude_client, event, frame_dir, "claude-model")
    claude_content = claude_client.messages.create.call_args.kwargs["messages"][0]["content"]

    gpt_client = MagicMock()
    gpt_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=fake_json))],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
    )
    review_death_gpt(gpt_client, event, frame_dir, "gpt-model")
    gpt_content = gpt_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]

    assert len(claude_content) == len(gpt_content) == (_FRAMES_BEFORE + 1 + _FRAMES_AFTER) * 2 + 1
    for claude_block, gpt_block in zip(claude_content, gpt_content):
        if claude_block["type"] == "text":
            assert gpt_block["type"] == "text"
            assert claude_block["text"] == gpt_block["text"]
        else:
            assert claude_block["type"] == "image"
            assert gpt_block["type"] == "image_url"

    claude_texts = [b["text"] for b in claude_content if b["type"] == "text"]
    assert "track-end" in claude_texts[0]
    assert "track-end" in claude_texts[-1]
