"""Tests for prosody analysis — pure F0 analysis + mocked librosa extractor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phoneta.core.metrics.prosody import (
    ProsodyResult,
    analyze_f0,
    boundary_trend,
    extract_f0,
    monotonicity,
)


def _f0_from_hz(values: list[float], n_frames: int = 20) -> np.ndarray:
    """Build an F0 array where *values* are evenly distributed and unvoiced
    frames (NaN) are sprinkled in between the last voiced frame and the end."""
    out = np.full(n_frames, np.nan, dtype=np.float64)
    step = max(1, n_frames // len(values))
    for i, v in enumerate(values):
        out[i * step] = float(v)
    return out


def _all_voiced(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


class TestBoundaryTrend:
    def test_rising(self) -> None:
        f0 = _f0_from_hz([100, 101, 110, 120])
        assert boundary_trend(f0) == "rising"

    def test_falling(self) -> None:
        f0 = _f0_from_hz([150, 140, 125, 110])
        assert boundary_trend(f0) == "falling"

    def test_flat_simple(self) -> None:
        f0 = _f0_from_hz([120, 121, 119, 121])
        assert boundary_trend(f0) == "flat"

    def test_too_few_voiced(self) -> None:
        f0 = _f0_from_hz([120], n_frames=5)
        assert boundary_trend(f0) == "flat"

    def test_all_nan(self) -> None:
        f0 = np.full(10, np.nan, dtype=np.float64)
        assert boundary_trend(f0) == "flat"

    def test_zero_start(self) -> None:
        f0 = np.array([0.0, 100.0, 200.0], dtype=np.float64)
        assert boundary_trend(f0) == "flat"


class TestMonotonicity:
    def test_constant_tone(self) -> None:
        f0 = _all_voiced([120.0, 120.0, 120.0, 120.0])
        assert monotonicity(f0) == pytest.approx(1.0)

    def test_varying(self) -> None:
        f0 = _all_voiced([100.0, 140.0, 80.0, 200.0])
        assert monotonicity(f0) < 0.5

    def test_single_frame(self) -> None:
        assert monotonicity(_all_voiced([120.0])) == 1.0

    def test_empty_voiced(self) -> None:
        assert monotonicity(np.full(10, np.nan)) == 1.0


class TestAnalyzeF0:
    def test_all_voiced_rising(self) -> None:
        f0 = _all_voiced([150.0, 155.0, 160.0, 165.0, 170.0, 180.0])
        result = analyze_f0(f0)
        assert result.voiced_ratio == 1.0
        assert result.mean_f0 == pytest.approx(163.3333, abs=1e-3)
        assert result.cv_f0 > 0.0
        assert result.boundary_trend == "rising"
        assert result.boundary_rise is True
        assert 0.0 <= result.monotonicity <= 1.0

    def test_all_unvoiced(self) -> None:
        f0 = np.full(8, np.nan, dtype=np.float64)
        result = analyze_f0(f0)
        assert result.voiced_ratio == 0.0
        assert result.mean_f0 == 0.0
        assert result.std_f0 == 0.0
        assert result.boundary_trend == "flat"
        assert result.boundary_rise is False
        assert result.monotonicity == 1.0

    def test_mixed(self) -> None:
        f0 = _f0_from_hz([120, 125, 130], n_frames=12)
        result = analyze_f0(f0)
        # 3 voiced / 12 frames
        assert 0.0 < result.voiced_ratio < 1.0

    def test_f0_times_stored_as_tuples(self) -> None:
        f0 = _all_voiced([100.0, 110.0, 120.0])
        times = np.array([0.0, 0.01, 0.02])
        result = analyze_f0(f0, times)
        assert isinstance(result.f0, tuple)
        assert isinstance(result.times, tuple)
        assert len(result.f0) == 3
        assert result.times[0] == 0.0

    def test_times_default(self) -> None:
        f0 = _all_voiced([100.0])
        result = analyze_f0(f0)
        assert result.times[0] == 0.0


class TestExtractF0:
    def test_mocked_librosa(self) -> None:
        f0 = np.array([100.0, np.nan, 110.0, 120.0], dtype=np.float64)
        times = np.array([0.0, 0.032, 0.064, 0.096], dtype=np.float64)

        mock_pyin = MagicMock(return_value=(f0, None, None))
        mock_times_like = MagicMock(return_value=times)

        with (
            patch("librosa.pyin", new=mock_pyin),
            patch("librosa.times_like", new=mock_times_like),
        ):
            result = extract_f0(np.ones(1600, dtype=np.float32), 16000)

        mock_pyin.assert_called_once()
        mock_times_like.assert_called_once()
        assert result.voiced_ratio == 0.75

    def test_pyin_args_forwarded(self) -> None:
        mock_pyin = MagicMock(return_value=(np.array([100.0]), None, None))
        mock_times_like = MagicMock(return_value=np.array([0.0]))

        with (
            patch("librosa.pyin", new=mock_pyin),
            patch("librosa.times_like", new=mock_times_like),
        ):
            extract_f0(
                np.ones(4000, dtype=np.float32),
                sample_rate=8000,
                fmin=80.0,
                fmax=1000.0,
                frame_length=1024,
                hop_length=256,
            )

        call_kwargs = mock_pyin.call_args.kwargs
        assert call_kwargs["fmin"] == 80.0
        assert call_kwargs["fmax"] == 1000.0
        assert call_kwargs["frame_length"] == 1024
        assert call_kwargs["sr"] == 8000