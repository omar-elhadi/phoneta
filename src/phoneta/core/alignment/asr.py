"""Speech-to-text via faster-whisper (local ``base`` model by default).

The model is loaded once on first use and reused.  ``faster_whisper`` is a
heavy dependency and is only imported when a transcription is actually
requested, so the rest of the app (and the test suite) can run without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WordTimestamp:
    """A transcribed word with its location in the audio."""

    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Transcription:
    """Result of transcribing one audio file."""

    text: str
    language: Optional[str]
    words: tuple[WordTimestamp, ...]


class Transcriber:
    """Transcribe local audio files with faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        model_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_dir = Path(model_dir) if model_dir else None
        self._model = None  # lazy

    def _load(self):
        """Load the Whisper model once (idempotent)."""
        if self._model is None:
            from faster_whisper import WhisperModel

            kwargs = {"device": self.device, "compute_type": self.compute_type}
            if self.model_dir is not None:
                kwargs["download_root"] = str(self.model_dir)
            self._model = WhisperModel(self.model_size, **kwargs)
        return self._model

    def transcribe(self, audio_path: str | Path) -> Transcription:
        """Transcribe *audio_path* and return text plus word timestamps.

        Word timestamps require ``word_timestamps=True``; if the installed
        faster-whisper build does not support them, we retry without and
        return an empty word list rather than failing the pipeline.
        """
        model = self._load()
        path = str(audio_path)

        try:
            segments, info = model.transcribe(path, word_timestamps=True)
            words = tuple(
                WordTimestamp(word=w.word, start_s=float(w.start), end_s=float(w.end))
                for segment in segments
                for w in (segment.words or [])
            )
        except TypeError:
            segments, info = model.transcribe(path)
            words = ()

        text = " ".join(seg.text.strip() for seg in segments).strip()
        language = getattr(info, "language", None)
        return Transcription(text=text, language=language, words=words)
