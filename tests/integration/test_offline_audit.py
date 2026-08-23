"""Offline-network audit: verifies zero outbound connections during analysis.

Monkeypatches ``socket.socket``, ``urllib.request.urlopen``, and
``requests`` so *any* network call raises an assertion error.
Runs the mocked pipeline inside the block — if a single packet leaves
the machine, the test fails.

This enforces NFR-002 / SC-002.
"""

from __future__ import annotations

import socket as _socket
import sys
from contextlib import contextmanager
from unittest.mock import patch

import pytest


def _blocked(*args: object, **kwargs: object) -> None:
    msg = "Network call detected during offline analysis — violates privacy guarantee."
    raise AssertionError(msg)


class _BlockingSocket:
    """A socket that refuses to connect."""

    def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        pass

    def connect(self, *args: object, **kwargs: object) -> None:
        _blocked()

    def __getattr__(self, name: str):
        return lambda *args, **kw: None


def _blocking_create_connection(*args: object, **kwargs: object) -> None:
    _blocked()


def _make_canned_results():
    """Build canned subcomponent fakes for the offline audit test."""
    from phoneta.core.alignment.asr import Transcription, WordTimestamp
    from phoneta.core.alignment.g2p import WordIPA
    from phoneta.core.alignment.mfa import AlignmentResult, PhonemeSegment
    from phoneta.core.audio.vad import SpeechSegment
    from phoneta.core.metrics.prosody import ProsodyResult

    def _vad(*_a, **_kw):
        return [SpeechSegment(start_sample=0, end_sample=16000, sample_rate=16000)]

    return {
        "vad": _vad,
        "transcription": Transcription(
            text="hello world",
            language="en",
            words=(
                WordTimestamp(word="hello", start_s=0.0, end_s=0.5),
                WordTimestamp(word="world", start_s=0.5, end_s=1.0),
            ),
        ),
        "g2p": [
            WordIPA(word="hello", phonemes=("h", "ə", "l", "oʊ"), analyzed=True),
            WordIPA(word="world", phonemes=("w", "ɜ", "l", "d"), analyzed=True),
        ],
        "alignment": AlignmentResult(
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
        ),
        "prosody": ProsodyResult(
            f0=(150.0, 155.0, 160.0, 165.0),
            times=(0.0, 0.25, 0.5, 0.75),
            voiced_ratio=1.0,
            mean_f0=157.5,
            std_f0=6.5,
            cv_f0=0.04,
            boundary_trend="rising",
            boundary_rise=True,
            monotonicity=0.25,
        ),
    }


@contextmanager
def _block_net():
    """Context manager that blocks all common network paths."""
    saves = {"create_connection": _socket.create_connection}

    _socket.create_connection = _blocking_create_connection  # type: ignore[assignment]

    sp = patch("socket.socket", _BlockingSocket)
    sp.start()

    up = None
    if "urllib.request" in sys.modules:
        import urllib.request

        up = patch.object(urllib.request, "urlopen", _blocked)
        up.start()

    rp = None
    if "requests" in sys.modules:
        import requests

        rp = patch.object(requests.Session, "request", _blocked)
        rp.start()

    try:
        yield
    finally:
        _socket.create_connection = saves["create_connection"]
        sp.stop()
        if up:
            up.stop()
        if rp:
            rp.stop()


class TestOfflineAudit:
    """Every network path must be unreachable during analysis."""

    def test_socket_create_connection_blocked(self) -> None:
        import socket

        with _block_net():
            with pytest.raises(AssertionError, match="Network call detected"):
                socket.create_connection(("8.8.8.8", 80))

    def test_urllib_urlopen_blocked(self) -> None:
        try:
            import urllib.request
        except ImportError:
            return

        with _block_net():
            with pytest.raises(AssertionError, match="Network call detected"):
                urllib.request.urlopen("http://example.com")  # noqa: ASYNC100

    def test_pipeline_makes_no_network_calls(self) -> None:
        """Run the pipeline with all network paths blocked — must succeed."""
        import numpy as np

        from phoneta.core.pipeline import run_pipeline

        audio = np.sin(
            np.linspace(0, 2 * 3.14159, 16000, dtype=np.float32)
        ).astype(np.float32) * 0.3

        fake = _make_canned_results()

        with _block_net():
            with (
                patch(
                    "phoneta.core.audio.vad.VoiceActivityDetector.find_speech",
                    side_effect=fake["vad"],
                ),
                patch(
                    "phoneta.core.alignment.asr.Transcriber.transcribe",
                    return_value=fake["transcription"],
                ),
                patch(
                    "phoneta.core.alignment.g2p.TextToIPA.phonemize",
                    return_value=fake["g2p"],
                ),
                patch(
                    "phoneta.core.alignment.mfa.ForcedAligner.align",
                    return_value=fake["alignment"],
                ),
                patch(
                    "phoneta.core.metrics.prosody.extract_f0",
                    return_value=fake["prosody"],
                ),
            ):
                result = run_pipeline(
                    target_text="hello world",
                    lang="en",
                    audio_samples=audio,
                )

        assert result.audio_deleted is True
        assert len(result.words) == 2