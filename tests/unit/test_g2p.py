"""Tests for text-to-IPA conversion (mocked phonemizer backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from phoneta.core.alignment.g2p import TextToIPA, WordIPA


class TestWordIPA:
    def test_ipa_string(self) -> None:
        wp = WordIPA(word="hello", phonemes=("h", "ə", "l", "oʊ"), analyzed=True)
        assert wp.ipa_string == "h ə l oʊ"


class TestTextToIPA:
    @staticmethod
    def _mock_backend(phonemize_result: list[str]):
        """Return a class whose instances have ``.phonemize()``.

        Each call to ``.phonemize()`` consumes the next pre-baked result
        so the mock correctly maps word → IPA when looping over multiple
        words in the same text.
        """
        it = iter(phonemize_result)

        class FakeBackend:
            def __init__(self, **kwargs):
                pass

            def phonemize(self, words):
                return [next(it)]

        return FakeBackend

    def test_phonemize_english(self) -> None:
        fake = self._mock_backend(["t ɛ s t", "m iː"])
        with patch("phonemizer.backend.EspeakBackend", new=fake):
            converter = TextToIPA()
            result = converter.phonemize("test me", lang="en")
            assert len(result) == 2
            assert result[0].word == "test"
            assert result[0].phonemes == ("t", "ɛ", "s", "t")
            assert result[0].analyzed is True
            assert result[1].word == "me"
            assert result[1].phonemes == ("m", "iː")
            assert result[1].analyzed is True

    def test_phonemize_french(self) -> None:
        fake = self._mock_backend(["b ɔ̃ ʒ u ʁ"])
        with patch("phonemizer.backend.EspeakBackend", new=fake):
            converter = TextToIPA()
            result = converter.phonemize("bonjour", lang="fr")
            assert result[0].word == "bonjour"
            assert result[0].phonemes == ("b", "ɔ̃", "ʒ", "u", "ʁ")
            assert result[0].analyzed is True

    def test_oov_word(self) -> None:
        """Empty phonemizer output → WordIPA.analyzed == False."""
        fake = self._mock_backend([""])
        with patch("phonemizer.backend.EspeakBackend", new=fake):
            converter = TextToIPA()
            result = converter.phonemize("xyzzy", lang="en")
            assert result[0].word == "xyzzy"
            assert result[0].phonemes == ()
            assert result[0].analyzed is False

    def test_backend_caching(self) -> None:
        """Same (backend, lang) pair reuses the backend instance."""
        backend = MagicMock()
        backend_class = MagicMock(return_value=backend)
        with patch("phonemizer.backend.EspeakBackend", new=backend_class):
            converter = TextToIPA()
            converter._backend_for("en")
            converter._backend_for("en")
            assert backend_class.call_count == 1  # cached

    def test_phonemize_flat(self) -> None:
        fake = self._mock_backend(["h ə l oʊ"])
        with patch("phonemizer.backend.EspeakBackend", new=fake):
            converter = TextToIPA()
            tokens = converter.phonemize_flat("hello", lang="en")
            assert tokens == ("h", "ə", "l", "oʊ")

    def test_punctuation_preserved_as_words(self) -> None:
        """Words only — punctuation and whitespace are skipped."""
        fake = self._mock_backend(["h aɪ", "ð ɛ ɹ"])
        with patch("phonemizer.backend.EspeakBackend", new=fake):
            converter = TextToIPA()
            result = converter.phonemize("Hi, there!", lang="en")
            assert len(result) == 2  # "Hi" and "there"