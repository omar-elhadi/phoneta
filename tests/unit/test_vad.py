"""Unit tests for voice-activity detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from phoneta.core.audio.vad import SpeechSegment, VoiceActivityDetector, trim_silence


def _sine(duration_s: float, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Generate a float32 sine tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.3


def _silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _mock_silero(prob_values: np.ndarray) -> MagicMock:
    """Return a mock silero-vad model that produces the given per-frame probs.

    Real silero returns a torch Tensor; we mimic its ``.numpy()`` interface.
    """
    mock_model = MagicMock()
    output = MagicMock()
    output.numpy.return_value = prob_values
    mock_model.return_value = output

    def _mock_timestamps(audio, model, threshold, sampling_rate):
        result: list[dict] = []
        in_speech = False
        start = 0
        step = 512  # rough
        for i, p in enumerate(prob_values):
            if p >= threshold and not in_speech:
                in_speech = True
                start = i * step
            elif p < threshold and in_speech:
                in_speech = False
                result.append({"start": start, "end": i * step})
        if in_speech:
            result.append({"start": start, "end": len(prob_values) * step})
        return result

    utils = (_mock_timestamps,)
    return mock_model, utils


class TestSpeechSegment:
    def test_duration(self) -> None:
        seg = SpeechSegment(start_sample=0, end_sample=16000, sample_rate=16000)
        assert seg.duration_s == 1.0


class TestVoiceActivityDetector:
    def test_is_speech_on_tone(self) -> None:
        audio = _sine(0.5)
        probs = np.full(16, 0.9, dtype=np.float32)  # all speech
        mock_model, mock_utils = _mock_silero(probs)

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            vad = VoiceActivityDetector(threshold=0.5)
            assert vad.is_speech(audio) is True

    def test_is_speech_on_silence(self) -> None:
        audio = _silence(0.5)
        probs = np.full(16, 0.1, dtype=np.float32)
        mock_model, mock_utils = _mock_silero(probs)

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            vad = VoiceActivityDetector(threshold=0.5)
            assert vad.is_speech(audio) is False

    def test_find_speech_segments(self) -> None:
        # 16 frames: 8 speech then 4 silence then 4 speech
        probs = np.array(
            [0.9] * 8 + [0.1] * 4 + [0.9] * 4, dtype=np.float32
        )
        mock_model, mock_utils = _mock_silero(probs)

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            vad = VoiceActivityDetector(threshold=0.5)
            segments = vad.find_speech(_sine(0.5))

            assert len(segments) == 2
            assert segments[0].start_sample < segments[0].end_sample
            assert segments[1].start_sample > segments[0].end_sample  # gap


class TestTrimSilence:
    def test_trim(self) -> None:
        # 0.5 s silence + 0.5 s tone + 0.5 s silence
        audio = np.concatenate([_silence(0.5), _sine(0.5), _silence(0.5)])
        # silero's mock timestamps map frame i -> sample i*512; derive probs
        # from the actual audio so frame boundaries match the tone location.
        frames = len(audio) // 512
        probs = np.array(
            [
                0.9
                if np.max(np.abs(audio[i * 512 : (i + 1) * 512])) > 0.01
                else 0.0
                for i in range(frames)
            ],
            dtype=np.float32,
        )

        mock_model, mock_utils = _mock_silero(probs)
        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            trimmed = trim_silence(audio)
            assert len(trimmed) < len(audio)  # shorter
            # trimming removes silence only — every non-silent sample survives
            assert np.count_nonzero(np.abs(trimmed) > 0.01) == np.count_nonzero(
                np.abs(audio) > 0.01
            )
            assert np.max(np.abs(trimmed)) > 0.0

    def test_all_silence_returns_original(self) -> None:
        audio = _silence(0.5)
        probs = np.full(10, 0.0, dtype=np.float32)
        mock_model, mock_utils = _mock_silero(probs)

        with patch("torch.hub.load", return_value=(mock_model, mock_utils)):
            trimmed = trim_silence(audio)
            assert len(trimmed) == len(audio)