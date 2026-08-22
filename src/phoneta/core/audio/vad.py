"""Voice-activity detection using the local silero-vad model.

All inference is CPU-only; no network calls are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


SILERO_SR = 16000  # silero-vad operates at 16 kHz
_WINDOW_MS = 30  # silero expects 30 ms windows


@dataclass(frozen=True)
class SpeechSegment:
    """A contiguous speech region."""

    start_sample: int
    end_sample: int
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate


class VoiceActivityDetector:
    """Silero-VAD wrapper — detects speech probability and trims silence.

    The underlying model is loaded once on first use and reused for the
    lifetime of the process.

    Usage::

        vad = VoiceActivityDetector()
        segments = vad.find_speech(audio_samples)  # list of SpeechSegment
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Parameters
        ----------
        threshold:
            Speech probability threshold (0.0 – 1.0).  Frames with probability
            ≥ threshold are considered speech.
        """
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _model(self):
        """Lazy-load the silero-vad model."""
        if not hasattr(self, "_vad_model"):
            import torch

            _model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model = _model
            self._get_speech_timestamps = _utils[0]
        return self._vad_model

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Return ``True`` when at least one frame exceeds the speech threshold.

        Parameters
        ----------
        audio:
            float32 mono array.
        sample_rate:
            Sample rate of *audio*.  Resampled to 16 kHz if needed.
        """
        prob = self.speech_probability(audio, sample_rate)
        return float(np.max(prob)) >= self.threshold

    def speech_probability(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return per-frame speech probabilities (0–1) for *audio*.

        The returned array has one value per 30 ms step.
        """
        if sample_rate != SILERO_SR:
            audio = _resample(audio, sample_rate, SILERO_SR)

        import torch

        model = self._model()
        tensor = torch.from_numpy(audio).float()
        with torch.no_grad():
            prob = model(tensor, SILERO_SR).numpy()
        return prob

    def find_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeechSegment]:
        """Return contiguous speech segments above the threshold.

        Parameters
        ----------
        audio:
            float32 mono.
        sample_rate:
            Sample rate of *audio*.
        """
        if sample_rate != SILERO_SR:
            audio = _resample(audio, sample_rate, SILERO_SR)

        model = self._model()  # also populates self._get_speech_timestamps
        timestamps = self._get_speech_timestamps(
            audio,
            model,
            threshold=self.threshold,
            sampling_rate=SILERO_SR,
        )
        segments: list[SpeechSegment] = []
        for ts in timestamps:
            segments.append(
                SpeechSegment(
                    start_sample=int(ts["start"]),
                    end_sample=int(ts["end"]),
                    sample_rate=SILERO_SR,
                )
            )
        return segments


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    pad_samples: int = 800,  # 50 ms at 16 kHz
) -> np.ndarray:
    """Strip leading and trailing silence from *audio*.

    Returns the trimmed audio (same sample rate), or the original if no speech
    is detected.
    """
    vad = VoiceActivityDetector(threshold=threshold)
    segments = vad.find_speech(audio, sample_rate)

    if not segments:
        return audio  # nothing to trim

    start = max(0, segments[0].start_sample - pad_samples)
    end = min(len(audio), segments[-1].end_sample + pad_samples)
    return audio[start:end]


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample float32 mono audio using linear interpolation (zero-dep fallback).

    For production the caller should install ``librosa`` or ``soxr``, but this
    lightweight path keeps the VAD module importable without those deps.
    """
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    new_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)