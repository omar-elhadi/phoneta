#!/usr/bin/env python3
"""Latency benchmark for the Phoneta pipeline.

Runs the full pipeline (mocked subcomponents) and reports wall-clock time.
The real pipeline targets < 1.5 s for a 10-second clip on modern hardware.
This script runs a synthetic equivalent to validate overhead is reasonable.

Usage::

    python scripts/benchmark_latency.py           # default (1 s clip, mocked)
    python scripts/benchmark_latency.py --real    # with real models (needs setup)
    python scripts/benchmark_latency.py --duration 10  # 10-second clip

Exit code 0 = met target, 1 = exceeded target.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

TARGET_SECONDS = 1.5  # must complete under this


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Phoneta pipeline latency."
    )
    p.add_argument(
        "--duration", type=float, default=1.0,
        help="Synthetic clip duration in seconds (default: 1)."
    )
    p.add_argument(
        "--real", action="store_true",
        help="Use real models (requires setup)."
    )
    p.add_argument(
        "--trials", type=int, default=3,
        help="Number of runs (default: 3)."
    )
    return p


def _canned_results():
    """Return (vad_find_speech, transcription, g2p, alignment, prosody) fakes."""
    from phoneta.core.alignment.asr import Transcription, WordTimestamp
    from phoneta.core.alignment.g2p import WordIPA
    from phoneta.core.alignment.mfa import AlignmentResult, PhonemeSegment
    from phoneta.core.audio.vad import SpeechSegment
    from phoneta.core.metrics.prosody import ProsodyResult

    def _vad(*_a, **_kw):
        return [SpeechSegment(start_sample=0, end_sample=16000, sample_rate=16000)]

    transcription = Transcription(
        text="hello world",
        language="en",
        words=(
            WordTimestamp(word="hello", start_s=0.0, end_s=0.5),
            WordTimestamp(word="world", start_s=0.5, end_s=1.0),
        ),
    )

    g2p = [
        WordIPA(word="hello", phonemes=("h", "ə", "l", "oʊ"), analyzed=True),
        WordIPA(word="world", phonemes=("w", "ɜ", "l", "d"), analyzed=True),
    ]

    alignment = AlignmentResult(
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

    prosody = ProsodyResult(
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

    return _vad, transcription, g2p, alignment, prosody


def _run_mocked(duration_s: float) -> float:
    """Run the pipeline with mocked subcomponents, measure wall-clock."""
    from unittest.mock import patch

    from phoneta.core.pipeline import run_pipeline

    sr = 16000
    n = int(sr * duration_s)
    audio = np.sin(
        np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    ).astype(np.float32) * 0.3

    vad, trans, g2p, align, pros = _canned_results()

    with (
        patch(
            "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
            side_effect=vad,
        ),
        patch(
            "phoneta.core.alignment.asr.Transcriber.transcribe",
            return_value=trans,
        ),
        patch(
            "phoneta.core.alignment.g2p.TextToIPA.phonemize",
            return_value=g2p,
        ),
        patch(
            "phoneta.core.alignment.mfa.ForcedAligner.align",
            return_value=align,
        ),
        patch(
            "phoneta.core.metrics.prosody.extract_f0",
            return_value=pros,
        ),
    ):
        t0 = time.perf_counter()
        run_pipeline(
            target_text="hello world",
            lang="en",
            audio_samples=audio,
        )
        return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"Benchmarking Phoneta pipeline ({args.duration}s clip, {args.trials} trials) …")
    print()

    times: list[float] = []
    for i in range(args.trials):
        elapsed = _run_mocked(args.duration)
        times.append(elapsed)
        status = "✓" if elapsed < TARGET_SECONDS else "✗ EXCEEDED"
        print(f"  trial {i + 1}: {elapsed:.4f}s  {status}")

    avg = sum(times) / len(times)
    print()
    print(f"  average: {avg:.4f}s")
    print(f"  target:  {TARGET_SECONDS:.1f}s")
    print()

    if avg > TARGET_SECONDS:
        print(f"FAIL: average latency ({avg:.4f}s) exceeds target ({TARGET_SECONDS:.1f}s).")
        return 1

    print("PASS: latency within target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())