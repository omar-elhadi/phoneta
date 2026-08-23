"""Pitch (F0) contour analysis for prosody feedback.

Analysis is **user-only** — there is no synthetic reference curve (per the
product decision); the goal is to describe *how* the user spoke: whether the
voice rises at the end of the utterance (e.g. French rising intonation),
how dynamic the delivery is, and how monotone it sounds.

``librosa`` is a heavy dependency and is imported lazily inside
:func:`extract_f0`; the pure analysis functions (:func:`analyze_f0` and
friends) operate on plain ``(f0, times)`` arrays so they are fully
unit-testable without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Relative thresholds for classifying the end-of-utterance pitch trend.
_TREND_RATIO = 1.10  # last-quartile median ≥ 1.10× first-quartile => rising
_TREND_RATIO_INV = 0.90  # ≤ 0.90× => falling

# Relative per-frame change below which a voice is considered "monotone".
_MONOTONE_REL_CHANGE = 0.02

FMIN_DEFAULT = 65.0  # Hz (C2)
FMAX_DEFAULT = 2093.0  # Hz (C7)


@dataclass(frozen=True)
class ProsodyResult:
    """Summary of the user's pitch contour.

    ``f0`` / ``times`` keep the full per-frame curve (NaN marks unvoiced
    frames) so the UI can draw it; the remaining fields are statistics over
    voiced frames only.
    """

    f0: tuple[float, ...]
    times: tuple[float, ...]
    voiced_ratio: float  # fraction of frames that are voiced (0..1)
    mean_f0: float  # Hz
    std_f0: float  # Hz
    cv_f0: float  # coefficient of variation (std/mean) — stress-variance proxy
    boundary_trend: str  # "rising" | "falling" | "flat"
    boundary_rise: bool  # True when the utterance ends on a rising pitch
    monotonicity: float  # 0..1 — 1.0 means fully monotone


def _voiced(f0: np.ndarray) -> np.ndarray:
    return f0[~np.isnan(f0)]  # type: ignore[no-any-return]  # old numpy stubs lack indexing types


def boundary_trend(f0: np.ndarray) -> str:
    """Classify the pitch trend between the start and end of the utterance.

    Compares the median F0 of the first vs last quarter of voiced frames.
    """
    v = _voiced(f0)
    if len(v) < 4:
        return "flat"
    quarter = len(v) // 4
    start = float(np.median(v[:quarter]))
    end = float(np.median(v[-quarter:]))
    if start <= 0:
        return "flat"
    ratio = end / start
    if ratio >= _TREND_RATIO:
        return "rising"
    if ratio <= _TREND_RATIO_INV:
        return "falling"
    return "flat"


def monotonicity(f0: np.ndarray) -> float:
    """Fraction of adjacent voiced frames whose pitch barely changes.

    1.0 means the voice never varied (fully monotone); lower values mean a
    more dynamic contour.
    """
    v = _voiced(f0)
    if len(v) < 2:
        return 1.0
    rel_change = np.abs(np.diff(v)) / np.maximum(v[:-1], 1e-9)
    return float(np.mean(rel_change < _MONOTONE_REL_CHANGE))


def analyze_f0(f0: np.ndarray, times: np.ndarray | None = None) -> ProsodyResult:
    """Summarise an F0 curve (with NaN for unvoiced frames) into a result.

    Pure function — no librosa needed.  ``times`` defaults to frame indices.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    if times is None:
        times = np.arange(len(f0), dtype=np.float64)
    else:
        times = np.asarray(times, dtype=np.float64)

    v = _voiced(f0)
    n = len(f0)
    if len(v) == 0:
        return ProsodyResult(
            f0=tuple(f0.tolist()),
            times=tuple(times.tolist()),
            voiced_ratio=0.0,
            mean_f0=0.0,
            std_f0=0.0,
            cv_f0=0.0,
            boundary_trend="flat",
            boundary_rise=False,
            monotonicity=1.0,
        )

    mean = float(np.mean(v))
    std = float(np.std(v))
    cv = std / mean if mean > 0 else 0.0
    trend = boundary_trend(f0)
    return ProsodyResult(
        f0=tuple(f0.tolist()),
        times=tuple(times.tolist()),
        voiced_ratio=float(len(v) / n) if n else 0.0,
        mean_f0=mean,
        std_f0=std,
        cv_f0=cv,
        boundary_trend=trend,
        boundary_rise=trend == "rising",
        monotonicity=monotonicity(f0),
    )


def extract_f0(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = FMIN_DEFAULT,
    fmax: float = FMAX_DEFAULT,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> ProsodyResult:
    """Extract and summarise the F0 contour of *audio* (float32 mono).

    Uses ``librosa.pyin`` (loaded lazily).  Returns a :class:`ProsodyResult`
    with NaN marking unvoiced frames.
    """
    import librosa

    f0, _voiced_flag, _probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    return analyze_f0(f0, times)
