"""Forced alignment via Montreal Forced Aligner (MFA).

MFA requires pretrained acoustic models and pronunciation dictionaries per
language, which are downloaded during the model-setup phase (Phase 7).  Until
they exist — or if alignment fails at runtime — :meth:`ForcedAligner.align`
degrades gracefully to a word-level fallback that distributes the reference
words uniformly across the audio duration, so the pipeline never hard-fails on
missing models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)

# Confidence placeholder used by the word-level fallback: no phoneme-level
# signal is available, so nothing is flagged purely on confidence.
FALLBACK_CONFIDENCE = 1.0


@dataclass(frozen=True)
class PhonemeSegment:
    """A single phoneme with its location and alignment confidence."""

    phoneme: str
    start_s: float
    end_s: float
    confidence: float


@dataclass(frozen=True)
class AlignmentResult:
    """Output of forced alignment (or its fallback)."""

    segments: tuple[PhonemeSegment, ...]
    method: str  # "mfa" | "fallback"


class ForcedAligner:
    """Align recorded audio to the reference text at phoneme granularity."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def align(
        self,
        audio_path: str | Path,
        ref_text: str,
        duration_s: float,
    ) -> AlignmentResult:
        """Return phoneme segments for *audio_path* against *ref_text*.

        Attempts a real MFA alignment first; any failure (missing models,
        missing binary, runtime error) falls back to uniform word-level timing.
        """
        try:
            return self._align_with_mfa(audio_path, ref_text)
        except (ImportError, FileNotFoundError, RuntimeError, OSError):
            return self._fallback(ref_text, duration_s)

    def _align_with_mfa(self, audio_path: str | Path, ref_text: str) -> AlignmentResult:
        """Run MFA; raises if the model stack is unavailable or fails."""
        import montreal_forced_aligner  # noqa: F401  (import check only)

        segments = self._run_mfa_alignment(str(audio_path), ref_text, self.lang)
        return AlignmentResult(segments=tuple(segments), method="mfa")

    def _run_mfa_alignment(
        self, audio_path: str, ref_text: str, lang: str
    ) -> list[PhonemeSegment]:
        """Thin seam around the actual MFA call — overridden/patched in tests.

        Implemented in the model-setup phase once acoustic models + dictionary
        paths are configurable; raising here routes to the fallback.
        """
        raise RuntimeError("MFA alignment not configured yet")

    def _fallback(self, ref_text: str, duration_s: float) -> AlignmentResult:
        """Distribute reference words uniformly across the audio.

        Each word becomes a single segment spanning an equal slice of the
        audio.  This gives the UI word-level colouring even before phoneme
        models are installed.
        """
        words = _WORD_RE.findall(ref_text)
        if not words or duration_s <= 0:
            return AlignmentResult(segments=(), method="fallback")

        step = duration_s / len(words)
        segments: list[PhonemeSegment] = []
        for i, word in enumerate(words):
            segments.append(
                PhonemeSegment(
                    phoneme=word,
                    start_s=round(i * step, 4),
                    end_s=round((i + 1) * step, 4),
                    confidence=FALLBACK_CONFIDENCE,
                )
            )
        return AlignmentResult(segments=tuple(segments), method="fallback")
