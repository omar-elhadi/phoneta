"""Smoke tests that exercise the *real* ML components with a fixture WAV.

These tests validate that each heavy dependency (silero-vad, faster-whisper,
librosa pyin, phonemizer) actually loads and runs on this machine.

Run with::

    pytest tests/smoke/ -v

On first run, faster-whisper downloads the ``tiny`` model (~39 MB).
Subsequent runs are fully offline.

The ``tests/smoke/conftest.py`` removes the root conftest's fake modules
before the session starts and restores them after.

Fixtures: ``tests/fixtures/sweep_3s_16k.wav`` — a 3 s 16 kHz sine sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sweep_3s_16k.wav"


def _read_fixture() -> np.ndarray:
    """Read the sweep fixture as float32 numpy array."""
    import wave

    with wave.open(str(FIXTURE), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ═══════════════════════════════════════════════════════════════════════
#  silero-vad  (via torch.hub)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestSileroVADReal:
    """Exercise the real silero-vad model (loaded via torch.hub)."""

    def test_model_loads_with_torch_hub(self) -> None:
        """torch.hub.load('snakers4/silero-vad') returns a working model."""
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        assert model is not None
        assert callable(utils[0])  # get_speech_timestamps

    def test_our_vad_wrapper_loads_real_model(self) -> None:
        """VoiceActivityDetector._model() loads the real silero-vad."""
        from phoneta.core.audio.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        model = vad._model()  # force-loads torch.hub model
        assert model is not None

    def test_speech_probability_on_silence(self) -> None:
        """Detector returns near-zero probability for silence (512-sample chunk).

        The raw silero-vad model expects exactly 512 samples at 16 kHz;
        our wrapper's ``find_speech`` handles chunking automatically, but
        ``speech_probability`` passes audio directly to the model.
        """
        from phoneta.core.audio.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        silence = np.zeros(512, dtype=np.float32)
        prob = vad.speech_probability(silence)
        assert isinstance(prob, np.ndarray)
        assert float(prob.max()) < 0.5

    def test_find_speech_on_fixture(self) -> None:
        """find_speech runs on the sweep fixture without crashing."""
        from phoneta.core.audio.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        samples = _read_fixture()
        segments = vad.find_speech(samples, sample_rate=16000)
        assert isinstance(segments, list)


# ═══════════════════════════════════════════════════════════════════════
#  faster-whisper
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestFasterWhisperReal:
    """Exercise faster-whisper with the ``tiny`` model (~39 MB download)."""

    def test_loads_and_runs_inference(self) -> None:
        """Model loads and transcribe returns segments + info."""
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(FIXTURE), language="en")

        texts = list(segments)
        assert isinstance(texts, list)
        assert info.language is not None

    def test_our_transcriber_wrapper(self) -> None:
        """Transcriber._load() loads the real model; transcribe returns text."""
        from phoneta.core.alignment.asr import Transcriber

        transcriber = Transcriber(model_size="tiny")
        _model = transcriber._load()
        assert _model is not None

        result = transcriber.transcribe(str(FIXTURE))
        assert result.text is not None
        assert result.language is not None
        assert isinstance(result.words, tuple)


# ═══════════════════════════════════════════════════════════════════════
#  librosa  pyin  (F0 extraction)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestLibrosaPyinReal:
    """Exercise librosa.pyin F0 extraction on fixture audio."""

    def test_extracts_f0_from_sweep(self) -> None:
        """pyin runs on the sweep and returns plausible F0 values."""
        import librosa

        samples = _read_fixture()

        f0, voiced_flag, voiced_prob = librosa.pyin(
            samples,
            fmin=100.0,
            fmax=800.0,
            sr=16000,
            frame_length=2048,
        )

        assert f0 is not None
        assert voiced_flag is not None
        assert voiced_prob is not None
        assert int(voiced_flag.sum()) > 0, "sine sweep should have voiced frames"

    def test_our_prosody_wrapper(self) -> None:
        """extract_f0 returns a ProsodyResult with plausible values."""
        from phoneta.core.metrics.prosody import extract_f0

        samples = _read_fixture()
        result = extract_f0(samples, sample_rate=16000)

        assert result.f0 is not None
        assert len(result.f0) == len(result.times)
        assert result.mean_f0 > 0
        assert result.boundary_trend in ("rising", "falling", "flat")


# ═══════════════════════════════════════════════════════════════════════
#  phonemizer  (needs espeak-ng system package)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestPhonemizerReal:
    """Exercise phonemizer + espeak-ng if the system package is installed."""

    def test_espeak_backend_available(self) -> None:
        """EspeakBackend constructs — skipped if espeak-ng not installed."""
        from phonemizer.backend import EspeakBackend

        try:
            backend = EspeakBackend("en-us")
            assert backend is not None
        except RuntimeError as exc:
            if "espeak" in str(exc).lower():
                pytest.skip("espeak-ng not installed on this machine")
            raise

    def test_english_ipa_runs(self) -> None:
        """English text → IPA works."""
        from phonemizer.backend import EspeakBackend

        try:
            backend = EspeakBackend("en-us")
        except RuntimeError as exc:
            if "espeak" in str(exc).lower():
                pytest.skip("espeak-ng not installed")
            raise

        result = backend.phonemize(["hello world"])
        assert len(result) == 1
        assert len(result[0]) > 0  # non-empty IPA

    def test_french_ipa_runs(self) -> None:
        """French text → IPA works."""
        from phonemizer.backend import EspeakBackend

        try:
            backend = EspeakBackend("fr-fr")
        except RuntimeError as exc:
            if "espeak" in str(exc).lower():
                pytest.skip("espeak-ng not installed")
            raise

        result = backend.phonemize(["bonjour le monde"])
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_our_g2p_wrapper(self) -> None:
        """TextToIPA.phonemize returns WordIPA objects."""
        from phoneta.core.alignment.g2p import TextToIPA

        g2p = TextToIPA()
        try:
            result = g2p.phonemize("hello world", lang="en")
        except RuntimeError as exc:
            if "espeak" in str(exc).lower():
                pytest.skip("espeak-ng not installed")
            raise

        assert len(result) == 2
        assert result[0].word == "hello"
        assert result[0].analyzed is True
        assert len(result[0].phonemes) > 0
        assert result[1].word == "world"
        assert result[1].analyzed is True


# ═══════════════════════════════════════════════════════════════════════
#  sounddevice
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.smoke
class TestSoundDeviceReal:
    """Exercise sounddevice — query devices, check no crash."""

    def test_query_devices_works(self) -> None:
        """sounddevice.query_devices() returns device list."""
        import sounddevice as sd

        devices = sd.query_devices()
        assert devices is not None

    def test_our_recorder_constructs(self) -> None:
        """AudioRecorder can be instantiated."""
        from phoneta.core.audio.recorder import AudioRecorder

        rec = AudioRecorder(duration_s=1.0)
        assert rec.duration_s == 1.0
        assert rec.TARGET_SR == 16000