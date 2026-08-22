"""Text-to-IPA conversion via phonemizer + espeak-ng (English & French).

``phonemizer`` and the espeak-ng backend are heavy system dependencies and are
imported lazily — importing this module never touches them.  Words the backend
cannot phonemize (OOV) are returned with ``analyzed=False`` and an empty
phoneme list so the pipeline can mark them "not analyzed".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# espeak language codes used by phonemizer's backend.
_ESPEAK_LANGS = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
}

# Words: letters, digits and intra-word apostrophes/hyphens (keeps
# "don't", "l'homme", "well-being" together as single tokens).
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)


@dataclass(frozen=True)
class WordIPA:
    """IPA conversion result for a single word of the target text."""

    word: str
    phonemes: tuple[str, ...]  # empty tuple => not analyzed / OOV
    analyzed: bool

    @property
    def ipa_string(self) -> str:
        return " ".join(self.phonemes)


class TextToIPA:
    """Convert target text into a sequence of IPA phonemes, word by word."""

    def __init__(self, backend: str = "espeak") -> None:
        self.backend = backend

    def _backend_for(self, lang: str):
        """Lazily build (and cache) a phonemizer backend for *lang*."""
        cache_key = f"{self.backend}:{lang}"
        cached = getattr(self, "_backends", None)
        if cached is None:
            cached = self._backends = {}
        if cache_key not in cached:
            espeak_lang = _ESPEAK_LANGS.get(lang, lang)
            from phonemizer.backend import EspeakBackend

            cached[cache_key] = EspeakBackend(
                language=espeak_lang,
                preserve_punctuation=False,
                with_stress=False,
            )
        return cached[cache_key]

    def phonemize(self, text: str, lang: str = "en") -> list[WordIPA]:
        """Convert *text* into a per-word IPA breakdown for language *lang*.

        Returns
        -------
        list[WordIPA]
            One entry per word, in order.  Words the backend could not
            phonemize have ``analyzed=False``.
        """
        backend = self._backend_for(lang)
        words = _WORD_RE.findall(text)

        results: list[WordIPA] = []
        for word in words:
            raw = backend.phonemize([word])[0].strip()
            phonemes = tuple(p for p in raw.split(" ") if p)
            results.append(
                WordIPA(
                    word=word,
                    phonemes=phonemes,
                    analyzed=bool(phonemes),
                )
            )
        return results

    def phonemize_flat(self, text: str, lang: str = "en") -> tuple[str, ...]:
        """Return just the flattened phoneme tokens (analyzed words only).

        Convenience for callers that only need the reference phoneme sequence.
        """
        tokens: list[str] = []
        for wp in self.phonemize(text, lang):
            tokens.extend(wp.phonemes)
        return tuple(tokens)


def supported_languages() -> tuple[str, ...]:
    """Languages the converter understands without extra config."""
    return tuple(_ESPEAK_LANGS)
