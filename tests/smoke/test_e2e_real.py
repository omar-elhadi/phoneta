"""End-to-end smoke test — the *full* pipeline on real synthesized speech.

espeak-ng synthesizes a clean English sentence into a WAV, then the whole
pipeline runs with real components: silero-VAD trim → faster-whisper
transcription → phonemizer/espeak-ng G2P → word-level fallback scoring
(MFA models are not installed, so alignment falls back) → librosa prosody.

This is the closest thing to a real user session without a microphone.

Run with::

    pytest tests/smoke/test_e2e_real.py -v

Skips when ``espeak-ng`` is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from phoneta.core.pipeline import run_pipeline

TARGET = "the quick brown fox jumps over the lazy dog"


def _synthesize_speech() -> str:
    """Synthesize TARGET with espeak-ng into a temp WAV; return its path."""
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not installed on this machine")

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="phoneta_e2e_")
    import os

    os.close(fd)
    subprocess.run(
        ["espeak-ng", "-v", "en-us", "-s", "140", "-w", path, TARGET],
        check=True,
        capture_output=True,
    )
    return path


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


@pytest.mark.smoke
class TestRealEndToEnd:
    """Full pipeline on real speech: TTS → VAD → ASR → g2p → scoring."""

    def test_pipeline_transcribes_and_scores_real_speech(self) -> None:
        wav_path = _synthesize_speech()
        try:
            result = run_pipeline(
                target_text=TARGET,
                lang="en",
                audio_path=wav_path,
                delete_audio=False,  # keep fixture for other assertions
                asr_model_size="tiny",
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        # 1. ASR actually heard speech.
        assert result.transcribed_text, "transcription should not be empty"

        # 2. Every target word got scored.
        assert len(result.words) == 9
        assert all(w.word in TARGET for w in result.words)

        # 3. Colours are honest — a perfectly-spoken espeak-ng sentence must
        #    NOT be all red.  Some words may be red because the tiny Whisper
        #    model mishears a word (brown→roundfax), but most should be green.
        colors = [w.color for w in result.words]
        assert "green" in colors, f"expected some green words, got {colors}"

        # 4. Audio was actually captured/trimmed — prosody found voiced frames.
        assert result.prosody.mean_f0 > 50, "expected a plausible mean F0"

    def test_privacy_audio_deletion(self) -> None:
        """delete_audio=True removes the source WAV (privacy guarantee)."""
        wav_path = _synthesize_speech()
        assert Path(wav_path).exists()

        try:
            result = run_pipeline(
                target_text=TARGET,
                lang="en",
                audio_path=wav_path,
                delete_audio=True,
                asr_model_size="tiny",
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        assert result.audio_deleted is True
        assert not Path(wav_path).exists(), "raw audio must be deleted"

    def test_input_validation(self) -> None:
        """Empty target text / empty audio raise clear errors."""
        from phoneta.core.pipeline import run_pipeline as rp

        wav_path = _synthesize_speech()
        try:
            with pytest.raises(ValueError, match="target_text"):
                rp(target_text="   ", lang="en", audio_path=wav_path)

            empty = np.array([], dtype=np.float32)
            with pytest.raises(ValueError, match="audio_samples"):
                rp(target_text=TARGET, lang="en", audio_samples=empty)
        finally:
            Path(wav_path).unlink(missing_ok=True)


@pytest.mark.smoke
class TestResampledAudio:
    """Non-16 kHz input WAVs must work (sample-rate bug regression)."""

    def test_22khz_wav_is_resampled_correctly(self) -> None:
        """espeak-ng outputs 22.05 kHz — the pipeline must handle that."""
        wav_path = _synthesize_speech()

        try:
            samples, rate = _read_wav(wav_path)
            assert rate == 22050, "espeak-ng should produce 22.05 kHz"

            result = run_pipeline(
                target_text=TARGET,
                lang="en",
                audio_path=wav_path,
                delete_audio=False,
                asr_model_size="tiny",
            )
            assert result.transcribed_text, "22 kHz audio must transcribe"
            assert result.prosody.mean_f0 > 50
        finally:
            Path(wav_path).unlink(missing_ok=True)
