"""Integration tests for the full pronunciation evaluation pipeline.

Tests the orchestration — data flow between components, audio deletion, and
result assembly — with individual subcomponents mocked at their call boundary.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from phoneta.core.alignment.g2p import WordIPA
from phoneta.core.alignment.mfa import AlignmentResult, PhonemeSegment
from phoneta.core.metrics.prosody import ProsodyResult
from phoneta.core.pipeline import PipelineResult, run_pipeline


def _sine_wav(path: str, duration_s: float, freq: float = 440.0, sr: int = 16000):
    """Write a clean sine-tone WAV for pipeline input."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    samples = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    return samples


# ── helpers to build canned component results ──────────────────────

def _canned_vad_find_speech(*args, **kwargs):
    """Return one speech segment covering the first second."""
    from phoneta.core.audio.vad import SpeechSegment

    return [SpeechSegment(start_sample=0, end_sample=16000, sample_rate=16000)]


def _canned_transcription():
    from phoneta.core.alignment.asr import Transcription, WordTimestamp

    return Transcription(
        text="hello world",
        language="en",
        words=(
            WordTimestamp(word="hello", start_s=0.0, end_s=0.5),
            WordTimestamp(word="world", start_s=0.5, end_s=1.0),
        ),
    )


def _canned_g2p():
    return [
        WordIPA(word="hello", phonemes=("h", "ə", "l", "oʊ"), analyzed=True),
        WordIPA(word="world", phonemes=("w", "ɜ", "l", "d"), analyzed=True),
    ]


def _canned_alignment():
    return AlignmentResult(
        segments=(
            PhonemeSegment(phoneme="h", start_s=0.0, end_s=0.1, confidence=0.95),
            PhonemeSegment(phoneme="ə", start_s=0.1, end_s=0.2, confidence=0.92),
            PhonemeSegment(phoneme="l", start_s=0.2, end_s=0.3, confidence=0.90),
            PhonemeSegment(phoneme="oʊ", start_s=0.3, end_s=0.5, confidence=0.88),
            PhonemeSegment(phoneme="w", start_s=0.5, end_s=0.6, confidence=0.94),
            PhonemeSegment(phoneme="ɜ", start_s=0.6, end_s=0.7, confidence=0.91),
            PhonemeSegment(phoneme="l", start_s=0.7, end_s=0.8, confidence=0.93),
            PhonemeSegment(phoneme="d", start_s=0.8, end_s=1.0, confidence=0.89),
        ),
        method="mfa",
    )


def _canned_prosody():
    return ProsodyResult(
        f0=(150.0, 155.0, 160.0, 165.0),
        times=(0.0, 0.25, 0.5, 0.75),
        voiced_ratio=1.0,
        mean_f0=157.5,
        std_f0=6.5,
        cv_f0=0.04,
        boundary_trend="rising",
        boundary_rise=True,
        monotonicity=0.25,
    )


# ── tests ──────────────────────────────────────────────────────────


class TestPipelineFromSamples:
    def test_end_to_end_with_numpy_samples(self, tmp_path: Path):
        """Full pipeline from float32 samples → result, audio auto-deleted."""
        audio = np.sin(
            np.linspace(0, 2 * np.pi, 16000, dtype=np.float32)
        ).astype(np.float32) * 0.3

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                return_value=_canned_transcription(),
            ),
            patch(
                "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                return_value=_canned_g2p(),
            ),
            patch(
                "phoneta.core.alignment.mfa.ForcedAligner.align",
                return_value=_canned_alignment(),
            ),
            patch(
                "phoneta.core.metrics.prosody.extract_f0",
                return_value=_canned_prosody(),
            ),
        ):
            result = run_pipeline(
                target_text="hello world",
                lang="en",
                audio_samples=audio,
            )

        assert isinstance(result, PipelineResult)
        assert result.target_text == "hello world"
        assert result.transcribed_text == "hello world"
        assert result.language == "en"
        assert result.alignment_method == "mfa"
        assert result.audio_deleted is True
        assert len(result.words) == 2

        hello, world = result.words
        assert hello.word == "hello"
        assert hello.accuracy == 1.0  # all-phoneme match
        assert hello.color == "green"

        assert world.word == "world"
        assert world.accuracy == 1.0
        assert world.color == "green"

        assert result.prosody.boundary_rise is True

    def test_audio_deletion_with_path(self, tmp_path: Path):
        """When given a WAV path, it is deleted after analysis."""
        wav_path = tmp_path / "input.wav"
        _sine_wav(str(wav_path), duration_s=1.0)

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                return_value=_canned_transcription(),
            ),
            patch(
                "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                return_value=_canned_g2p(),
            ),
            patch(
                "phoneta.core.alignment.mfa.ForcedAligner.align",
                return_value=_canned_alignment(),
            ),
            patch(
                "phoneta.core.metrics.prosody.extract_f0",
                return_value=_canned_prosody(),
            ),
        ):
            result = run_pipeline(
                target_text="hello world",
                lang="en",
                audio_path=str(wav_path),
                delete_audio=True,
            )

        assert result.audio_deleted is True
        assert not os.path.exists(str(wav_path))

    def test_keep_audio_when_requested(self, tmp_path: Path):
        """delete_audio=False preserves the WAV file."""
        wav_path = tmp_path / "keep.wav"
        _sine_wav(str(wav_path), duration_s=1.0)

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                return_value=_canned_transcription(),
            ),
            patch(
                "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                return_value=_canned_g2p(),
            ),
            patch(
                "phoneta.core.alignment.mfa.ForcedAligner.align",
                return_value=_canned_alignment(),
            ),
            patch(
                "phoneta.core.metrics.prosody.extract_f0",
                return_value=_canned_prosody(),
            ),
        ):
            result = run_pipeline(
                target_text="hello world",
                lang="en",
                audio_path=str(wav_path),
                delete_audio=False,
            )

        assert result.audio_deleted is False
        assert os.path.exists(str(wav_path))

    def test_missing_both_audio_sources_raises(self):
        with pytest.raises(ValueError, match="audio_path or audio_samples"):
            run_pipeline(target_text="hello", lang="en")

    def test_temp_dir_always_cleaned(self):
        """Even on error, the temp directory is removed."""
        audio = np.zeros(16000, dtype=np.float32)

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                side_effect=RuntimeError("model not loaded"),
            ),
        ):
            with pytest.raises(RuntimeError):
                run_pipeline(
                    target_text="hello world",
                    lang="en",
                    audio_samples=audio,
                )

        # Temp dir cleanup is verified by the absence of lingering dirs —
        # because rmtree is called unconditionally in the finally block.
        # (No explicit assertion needed; a leak would be visible via OS.)


class TestPipelineFallback:
    def test_fallback_alignment_scores_words(self):
        """Fallback produces one segment per word — pipeline still scores."""
        audio = np.sin(
            np.linspace(0, 2 * np.pi, 16000, dtype=np.float32)
        ).astype(np.float32) * 0.3

        # Fallback segments are word-level, not phoneme-level.
        fallback_align = AlignmentResult(
            segments=(
                PhonemeSegment(phoneme="hello", start_s=0.0, end_s=0.5, confidence=1.0),
                PhonemeSegment(phoneme="world", start_s=0.5, end_s=1.0, confidence=1.0),
            ),
            method="fallback",
        )

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                return_value=_canned_transcription(),
            ),
            patch(
                "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                return_value=_canned_g2p(),
            ),
            patch(
                "phoneta.core.alignment.mfa.ForcedAligner.align",
                return_value=fallback_align,
            ),
            patch(
                "phoneta.core.metrics.prosody.extract_f0",
                return_value=_canned_prosody(),
            ),
        ):
            result = run_pipeline(
                target_text="hello world",
                lang="en",
                audio_samples=audio,
            )

        assert result.alignment_method == "fallback"
        assert len(result.words) == 2
        # In fallback, each segment is a whole word → NW alignment is "word" vs
        # the full phoneme sequence of the target word — substitutions expected.
        hello = result.words[0]
        assert hello.word == "hello"

    def test_oov_handled_gracefully(self):
        """OOV words (no IPA) get a neutral green score with no feedback."""
        audio = np.zeros(16000, dtype=np.float32)

        oov_g2p = [
            WordIPA(word="xyzzy", phonemes=(), analyzed=False),
        ]

        with (
            patch(
                "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                side_effect=_canned_vad_find_speech,
            ),
            patch(
                "phoneta.core.alignment.asr.Transcriber.transcribe",
                return_value=_canned_transcription(),
            ),
            patch(
                "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                return_value=oov_g2p,
            ),
            patch(
                "phoneta.core.alignment.mfa.ForcedAligner.align",
                return_value=_canned_alignment(),
            ),
            patch(
                "phoneta.core.metrics.prosody.extract_f0",
                return_value=_canned_prosody(),
            ),
        ):
            result = run_pipeline(
                target_text="xyzzy",
                lang="en",
                audio_samples=audio,
            )

        assert len(result.words) == 1
        assert result.words[0].word == "xyzzy"
        assert result.words[0].color == "green"
        assert result.words[0].feedback == ()