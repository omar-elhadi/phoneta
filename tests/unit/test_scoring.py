"""Tests for scoring thresholds and color assignment."""

from __future__ import annotations

import pytest

from phoneta.core.alignment.sequence import (
    DELETION,
    INSERTION,
    MATCH,
    SUBSTITUTION,
    AlignedPair,
)
from phoneta.core.metrics.scoring import (
    CONFIDENCE_THRESHOLD,
    GREEN,
    RED,
    RED_THRESHOLD,
    YELLOW,
    PhonemeFeedback,
    WordScore,
    score_word,
)


def _pair(ref, user, kind=MATCH) -> AlignedPair:
    return AlignedPair(ref=ref, user=user, kind=kind)


class TestPhonemeFeedback:
    @pytest.mark.parametrize(
        "kind,confidence,expected_flagged",
        [
            (MATCH, 1.0, False),
            (MATCH, 0.9, False),
            (MATCH, CONFIDENCE_THRESHOLD, False),  # boundary — not flagged (< threshold, not ≤)
            (MATCH, CONFIDENCE_THRESHOLD - 0.01, True),
            (MATCH, 0.3, True),
            (SUBSTITUTION, 1.0, True),  # kind error always flagged
            (DELETION, 0.9, True),
            (INSERTION, 0.9, True),
        ],
    )
    def test_flagged(self, kind, confidence, expected_flagged) -> None:
        pair = _pair("t", "t" if kind == MATCH else "z", kind)
        fb = PhonemeFeedback.from_pair(pair, confidence)
        assert fb.flagged == expected_flagged


class TestScoreWord:
    def test_perfect_match(self) -> None:
        pairs = [_pair("t", "t"), _pair("ə", "ə"), _pair("m", "m")]
        ws = score_word("tem", pairs)
        assert ws.accuracy == 1.0
        assert ws.color == GREEN
        assert ws.has_error is False

    def test_one_substitution_red(self) -> None:
        pairs = [
            _pair("k", "k"),
            _pair("æ", "ʌ", SUBSTITUTION),
            _pair("t", "t"),
        ]
        ws = score_word("cat", pairs)
        assert ws.accuracy == pytest.approx(2 / 3)
        assert ws.color == RED
        assert ws.has_error is True

    def test_yellow_via_prosody_issue(self) -> None:
        """Phonemes all correct + prosody variance → yellow (spec's minor variance)."""
        pairs = [_pair("t", "t"), _pair("ɛ", "ɛ"), _pair("s", "s"), _pair("t", "t")]
        ws = score_word("test", pairs, prosody_issue=True)
        assert ws.accuracy == 1.0
        assert ws.color == YELLOW
        assert ws.has_error is False

    def test_no_prosody_issue_stays_green(self) -> None:
        pairs = [_pair("t", "t"), _pair("ɛ", "ɛ"), _pair("s", "s"), _pair("t", "t")]
        ws = score_word("test", pairs)
        assert ws.color == GREEN

    def test_prosody_issue_does_not_override_phoneme_error(self) -> None:
        """A substitution is red even when prosody also flags variance."""
        pairs = [_pair("t", "t"), _pair("ɛ", "ə", SUBSTITUTION), _pair("t", "t")]
        ws = score_word("tet", pairs, prosody_issue=True)
        assert ws.color == RED

    def test_red_via_accuracy(self) -> None:
        """Accuracy below RED_THRESHOLD → red even with 1.0 confidences."""
        pairs = [_pair("a", "a")] + [_pair("x", "y", SUBSTITUTION) for _ in range(3)]
        ws = score_word("test", pairs, [1.0] * 4)
        assert ws.accuracy < RED_THRESHOLD
        assert ws.color == RED

    def test_empty_pairs(self) -> None:
        ws = score_word("oov", [])
        assert ws.accuracy == 1.0
        assert ws.color == GREEN
        assert ws.feedback == ()
        assert ws.has_error is False

    def test_confidence_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            score_word("x", [_pair("a", "a")], [1.0, 1.0])

    def test_confidences_default_to_one(self) -> None:
        pairs = [_pair("t", "t"), _pair("ə", "ə")]
        ws = score_word("te", pairs)
        for fb in ws.feedback:
            assert fb.confidence == 1.0

    def test_wordscore_has_error_empty(self) -> None:
        assert WordScore(word="x", accuracy=1.0, color=GREEN, feedback=()).has_error is False

    def test_each_feedback_has_correct_ref_user(self) -> None:
        pairs = [
            _pair("t", "t"),
            _pair("ə", None, DELETION),
            _pair(None, "s", INSERTION),
        ]
        ws = score_word("tes", pairs)
        assert ws.feedback[0].ref == "t"
        assert ws.feedback[0].user == "t"
        assert ws.feedback[1].ref == "ə"
        assert ws.feedback[1].user is None
        assert ws.feedback[2].ref is None
        assert ws.feedback[2].user == "s"