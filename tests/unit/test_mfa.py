"""Tests for forced alignment — both MFA and word-level fallback."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from phoneta.core.alignment.mfa import (
    FALLBACK_CONFIDENCE,
    AlignmentResult,
    ForcedAligner,
    PhonemeSegment,
)


class TestPhonemeSegment:
    def test_immutable(self) -> None:
        seg = PhonemeSegment(phoneme="t", start_s=0.1, end_s=0.2, confidence=0.95)
        assert seg.phoneme == "t"
        assert seg.confidence == 0.95


class TestForcedAlignerFallback:
    def test_fallback_distributes_words_evenly(self) -> None:
        aligner = ForcedAligner(lang="en")
        result = aligner._fallback("hello world", duration_s=2.0)

        assert result.method == "fallback"
        assert len(result.segments) == 2
        hello, world = result.segments
        assert hello.phoneme == "hello"
        assert hello.start_s == 0.0
        assert hello.end_s == 1.0
        assert hello.confidence == FALLBACK_CONFIDENCE
        assert world.phoneme == "world"
        assert world.start_s == 1.0
        assert world.end_s == 2.0

    def test_fallback_empty_text(self) -> None:
        result = ForcedAligner()._fallback("", duration_s=3.0)
        assert result.segments == ()
        assert result.method == "fallback"

    def test_fallback_zero_duration(self) -> None:
        result = ForcedAligner()._fallback("hello", duration_s=0.0)
        assert result.segments == ()
        assert result.method == "fallback"

    def test_fallback_single_word(self) -> None:
        result = ForcedAligner()._fallback("bonjour", duration_s=1.5)
        assert len(result.segments) == 1
        assert result.segments[0].phoneme == "bonjour"
        assert result.segments[0].start_s == 0.0
        assert result.segments[0].end_s == 1.5

    def test_fallback_with_hyphenated_word(self) -> None:
        result = ForcedAligner()._fallback("well-being test", duration_s=2.0)
        assert len(result.segments) == 2
        assert result.segments[0].phoneme == "well-being"
        assert result.segments[1].phoneme == "test"

    def test_alignment_result_dataclass(self) -> None:
        seg = PhonemeSegment(phoneme="t", start_s=0.0, end_s=0.1, confidence=0.9)
        ar = AlignmentResult(segments=(seg,), method="mfa")
        assert len(ar.segments) == 1
        assert ar.method == "mfa"


class TestForcedAlignerMfaPath:
    def test_mfa_success(self) -> None:
        """When MFA alignment works, its segments are returned directly."""
        segs = [
            PhonemeSegment(phoneme="t", start_s=0.0, end_s=0.2, confidence=0.98),
            PhonemeSegment(phoneme="ə", start_s=0.2, end_s=0.4, confidence=0.91),
        ]
        aligner = ForcedAligner(lang="en")
        with patch.object(aligner, "_run_mfa_alignment", return_value=segs):
            result = aligner.align("audio.wav", "the", duration_s=0.5)

        assert result.method == "mfa"
        assert result.segments == tuple(segs)

    def test_mfa_failure_falls_back(self) -> None:
        """Any MFA failure degrades to the word-level fallback, never raises."""
        aligner = ForcedAligner(lang="en")
        with patch.object(aligner, "_run_mfa_alignment", side_effect=RuntimeError("boom")):
            result = aligner.align("audio.wav", "hello world", duration_s=2.0)

        assert result.method == "fallback"
        assert len(result.segments) == 2

    def test_import_error_falls_back(self) -> None:
        """Missing MFA install routes to fallback."""
        aligner = ForcedAligner(lang="en")
        with patch("builtins.__import__", side_effect=ImportError("no mfa")):
            result = aligner.align("audio.wav", "hello", duration_s=1.0)
        assert result.method == "fallback"
        assert len(result.segments) == 1

    @pytest.mark.parametrize(
        "exc", [ImportError("x"), FileNotFoundError("x"), RuntimeError("x"), OSError("x")]
    )
    def test_all_expected_exceptions_fall_back(self, exc) -> None:
        aligner = ForcedAligner(lang="en")
        with patch.object(aligner, "_align_with_mfa", side_effect=exc):
            result = aligner.align("audio.wav", "bonjour", duration_s=1.0)
        assert result.method == "fallback"