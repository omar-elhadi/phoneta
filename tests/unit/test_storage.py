"""Tests for local SQLite practice-history storage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from phoneta.core.metrics.scoring import GREEN, RED, PhonemeFeedback, WordScore
from phoneta.storage.db import PracticeStore


def _green_word(word: str) -> WordScore:
    return WordScore(
        word=word,
        accuracy=1.0,
        color=GREEN,
        feedback=(),
    )


def _red_word(word: str) -> WordScore:
    return WordScore(
        word=word,
        accuracy=0.5,
        color=RED,
        feedback=(
            PhonemeFeedback(
                ref="t", user="d", kind="substitution", confidence=0.9, flagged=True
            ),
        ),
    )


@pytest.fixture
def store() -> PracticeStore:
    """Create a store backed by a temp file (automatically cleaned up)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield PracticeStore(path)
    try:
        os.unlink(path)
    except OSError:
        pass


class TestPracticeStore:
    def test_empty_store(self, store: PracticeStore) -> None:
        assert store.session_count() == 0
        assert store.list_sessions() == []

    def test_save_and_list(self, store: PracticeStore) -> None:
        words = (_green_word("hello"), _red_word("world"))
        sid = store.save_session(
            target_text="hello world",
            lang="en",
            words=words,
            alignment_method="fallback",
        )
        assert sid > 0
        assert store.session_count() == 1

        sessions = store.list_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s.target_text == "hello world"
        assert s.lang == "en"
        assert s.alignment == "fallback"
        assert len(s.words) == 2

        hello, world = s.words
        assert hello.word == "hello"
        assert hello.color == GREEN
        assert world.word == "world"
        assert world.color == RED
        assert len(world.feedback) == 1
        assert world.feedback[0].kind == "substitution"

    def test_get_session_by_id(self, store: PracticeStore) -> None:
        words = (_green_word("bonjour"),)
        sid = store.save_session("bonjour", "fr", words)
        s = store.get_session(sid)
        assert s is not None
        assert s.target_text == "bonjour"
        assert s.words[0].word == "bonjour"

    def test_get_nonexistent(self, store: PracticeStore) -> None:
        assert store.get_session(999) is None

    def test_multiple_sessions_ordered(self, store: PracticeStore) -> None:
        for text in ("first", "second", "third"):
            store.save_session(text, "en", (_green_word(text),))
        sessions = store.list_sessions()
        assert [s.target_text for s in sessions] == ["third", "second", "first"]

    def test_clear(self, store: PracticeStore) -> None:
        store.save_session("hello", "en", (_green_word("hello"),))
        assert store.session_count() == 1
        removed = store.clear()
        assert removed > 0
        assert store.session_count() == 0
        assert store.list_sessions() == []

    def test_no_audio_stored(self, store: PracticeStore) -> None:
        """Verify the db file contains no binary audio markers."""
        store.save_session("test", "en", (_green_word("test"),))
        raw = Path(store.db_path).read_bytes()
        assert b"RIFF" not in raw  # WAV header
        assert b"WAVEfmt" not in raw

    def test_clear_empty(self, store: PracticeStore) -> None:
        removed = store.clear()
        assert removed == 0