"""Tests for the local practice-history view."""

from __future__ import annotations

from phoneta.core.metrics.scoring import GREEN, RED, WordScore
from phoneta.storage.db import PracticeStore, SessionRow
from phoneta.ui.history_view import format_session_date, session_summary


def _word(word: str, accuracy: float, color: str) -> WordScore:
    return WordScore(word=word, accuracy=accuracy, color=color, feedback=())


def test_format_session_date_is_stable() -> None:
    assert format_session_date(0).endswith(("00:00", "01:00", "02:00"))


def test_session_summary_contains_key_details() -> None:
    session = SessionRow(
        id=1,
        created_at=0,
        target_text="hello world",
        lang="en",
        alignment="fallback",
        words=(_word("hello", 1.0, GREEN), _word("world", 0.5, RED)),
    )
    summary = session_summary(session)
    assert "hello world" in summary
    assert "75%" in summary
    assert "en" in summary


def test_history_view_empty_state() -> None:
    from phoneta.ui.history_view import HistoryView

    view = HistoryView(PracticeStore("/tmp/phoneta-history-test.db"))
    view.refresh()
    assert "No practice sessions" in view.list.item(0).text()


def test_history_view_populates_rows(tmp_path) -> None:
    from phoneta.ui.history_view import HistoryView

    store = PracticeStore(tmp_path / "history.db")
    store.save_session("hello", "en", (_word("hello", 1.0, GREEN),))
    view = HistoryView(store)
    view.refresh()
    assert view.list.count() == 1
    assert "hello" in view.list.item(0).text()
