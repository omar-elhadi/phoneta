"""Record microphone audio at 16 kHz mono WAV with a live RMS level meter."""

from __future__ import annotations

import io
import wave
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecordResult:
    """The result of a completed recording."""

    samples: np.ndarray  # float32 mono, shape (n_samples,)
    sample_rate: int  # 16000
    duration_s: float
    peak_rms: float  # highest per-frame RMS during recording

    def to_wav_bytes(self) -> bytes:
        """Encode the recording as 16-bit PCM WAV in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            pcm = (self.samples * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


def _rms(frame: np.ndarray) -> float:
    """Root-mean-square of a mono float32 frame."""
    sq = np.mean(frame.astype(np.float64) ** 2)
    return float(np.sqrt(sq)) if sq > 0 else 0.0


class AudioRecorder:
    """Capture system microphone audio at 16 kHz mono.

    Provides an optional live-level-meter callback that fires at a configurable
    interval so the UI can paint a volume bar.

    Usage::

        rec = AudioRecorder(duration_s=5.0)
        result = rec.record()
        wav_bytes = result.to_wav_bytes()

    Parameters
    ----------
    duration_s:
        Maximum recording length in seconds.
    level_interval_s:
        How often (in seconds) the ``on_level`` callback fires during
        recording.  Larger values reduce CPU overhead.
    on_level:
        Optional callback receiving ``(rms: float)`` on every level interval.
        Called from the audio callback thread — the caller is responsible for
        thread-safety.
    """

    TARGET_SR = 16000
    _CHANNELS = 1

    def __init__(
        self,
        duration_s: float = 10.0,
        level_interval_s: float = 0.05,
        on_level: Callable[[float], None] | None = None,
    ) -> None:
        self.duration_s = float(duration_s)
        self.level_interval_s = max(float(level_interval_s), 0.01)
        self.on_level = on_level

    def record(self) -> RecordResult:
        """Block until recording completes; return the captured audio."""
        import sounddevice as sd  # late import — only needed when recording

        frames: list[np.ndarray] = []
        peak_rms = 0.0
        samples_since_last_callback = 0
        callback_interval_samples = int(self.TARGET_SR * self.level_interval_s)

        def _callback(indata: np.ndarray, _frames, _time, _status) -> None:
            nonlocal peak_rms, samples_since_last_callback
            mono = indata[:, 0].copy()
            frames.append(mono)
            current_rms = _rms(mono)
            if current_rms > peak_rms:
                peak_rms = current_rms

            samples_since_last_callback += len(mono)
            if (
                self.on_level is not None
                and samples_since_last_callback >= callback_interval_samples
            ):
                samples_since_last_callback = 0
                self.on_level(current_rms)

        samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
        if samplerate != self.TARGET_SR:
            # resampling is cheap enough via sounddevice's built-in converter
            sd.default.samplerate = self.TARGET_SR

        sd.rec(
            int(self.duration_s * self.TARGET_SR),
            samplerate=self.TARGET_SR,
            channels=self._CHANNELS,
            dtype="float32",
            callback=_callback,
        )
        sd.wait()

        samples = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
        duration = len(samples) / self.TARGET_SR
        self._last_result = RecordResult(
            samples=samples,
            sample_rate=self.TARGET_SR,
            duration_s=duration,
            peak_rms=peak_rms,
        )
        return self._last_result