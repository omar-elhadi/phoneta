"""Unit tests for the audio recorder."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from phoneta.core.audio.recorder import AudioRecorder, RecordResult, _rms


class TestRMS:
    def test_silent_frame(self) -> None:
        assert _rms(np.zeros(100, dtype=np.float32)) == 0.0

    def test_unity_sine(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi, 1600, dtype=np.float32))
        rms = _rms(samples)
        assert 0.6 < rms < 0.8  # sin RMS ≈ 0.707


class TestRecordResult:
    def test_to_wav_bytes(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi, 16000, dtype=np.float32)).astype(np.float32)
        result = RecordResult(samples=samples, sample_rate=16000, duration_s=1.0, peak_rms=0.5)
        wav = result.to_wav_bytes()
        assert len(wav) > 44  # WAV header
        assert wav[:4] == b"RIFF"


class TestAudioRecorder:
    def test_defaults(self) -> None:
        rec = AudioRecorder()
        assert rec.duration_s == 10.0
        assert rec.level_interval_s == 0.05
        assert rec.on_level is None

    def test_recording_pipeline(self) -> None:
        """Simulate a full recording via mocked sounddevice."""
        fake_samples = np.random.randn(16000).astype(np.float32)  # 1 s

        with (
            patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}),
            patch("sounddevice.default") as mock_default,
            patch("sounddevice.rec") as mock_rec,
            patch("sounddevice.wait"),
        ):
            mock_default.samplerate = 16000

            def capture_side_effect(frames, **kwargs):
                cb = kwargs.get("callback")
                if cb:
                    stereo = np.column_stack([fake_samples[:frames], fake_samples[:frames] * 0])
                    cb(stereo, frames, None, None)

            mock_rec.side_effect = capture_side_effect

            rec = AudioRecorder(duration_s=1.0)
            result = rec.record()

            assert isinstance(result, RecordResult)
            assert result.sample_rate == 16000
            assert result.duration_s == pytest.approx(1.0, abs=0.1)
            assert result.peak_rms > 0

    def test_level_meter_callback(self) -> None:
        """on_level fires periodically during recording."""
        callback_values: list[float] = []
        fake_samples = np.random.randn(96000).astype(np.float32)  # 6 s

        with (
            patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}),
            patch("sounddevice.default"),
            patch("sounddevice.rec") as mock_rec,
            patch("sounddevice.wait"),
        ):
            def capture_side_effect(frames, **kwargs):
                cb = kwargs.get("callback")
                if cb:
                    # Deliver in small blocks like real sounddevice does
                    for start in range(0, frames, 1024):
                        block = fake_samples[start : start + 1024]
                        stereo = np.column_stack([block, block * 0])
                        cb(stereo, len(block), None, None)

            mock_rec.side_effect = capture_side_effect

            AudioRecorder(
                duration_s=6.0, level_interval_s=0.1,
                on_level=callback_values.append,
            ).record()

            # 6 s / 0.1 = ~60 callbacks (allow tolerance)
            assert 30 < len(callback_values) < 120
            assert all(isinstance(v, float) for v in callback_values)

    def test_no_callback_does_not_crash(self) -> None:
        with (
            patch("sounddevice.query_devices", return_value={"default_samplerate": 16000}),
            patch("sounddevice.default"),
            patch("sounddevice.rec"),
            patch("sounddevice.wait"),
        ):
            AudioRecorder(duration_s=0.1, on_level=None).record()