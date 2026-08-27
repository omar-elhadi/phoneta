"""Tests for result-view design logic (pure helpers + faked-Qt wiring)."""

from __future__ import annotations

from phoneta.core.metrics.scoring import GREEN, RED, PhonemeFeedback, WordScore
from phoneta.ui import theme
from phoneta.ui.result_view import (
    _EMPTY_HINT,
    ResultView,
    banner_color,
    grid_columns,
    summary_message,
)


def _ws(
    word: str,
    accuracy: float,
    color: str,
    *,
    flagged: bool = False,
) -> WordScore:
    feedback = (
        PhonemeFeedback(ref="a", user="a", kind="match", confidence=1.0, flagged=False),
    )
    if flagged:
        feedback = (
            PhonemeFeedback(
                ref="a", user="b", kind="substitution", confidence=0.4, flagged=True
            ),
        )
    return WordScore(
        word=word, accuracy=accuracy, color=color, feedback=feedback
    )


class TestGridColumns:
    def test_zero_or_negative_gives_one(self) -> None:
        assert grid_columns(0) == 1
        assert grid_columns(-3) == 1

    def test_small_sentences(self) -> None:
        assert grid_columns(1) == 1
        assert grid_columns(4) == 4

    def test_long_sentences_cap_at_six(self) -> None:
        assert grid_columns(9) == 6
        assert grid_columns(30) == 6


class TestBannerColor:
    def test_all_good_is_green(self) -> None:
        assert banner_color(0.95, 0) == theme.COLORS["green"]

    def test_zero_errors_always_yellow_or_better(self) -> None:
        assert banner_color(0.5, 0) == theme.COLORS["yellow"]

    def test_some_errors_is_yellow_or_red(self) -> None:
        assert banner_color(0.7, 2) in (theme.COLORS["yellow"], theme.COLORS["red"])

    def test_many_errors_is_red(self) -> None:
        assert banner_color(0.3, 5) == theme.COLORS["red"]


class TestSummaryMessage:
    def test_empty_shows_hint(self) -> None:
        assert summary_message(()) == _EMPTY_HINT

    def test_perfect_session(self) -> None:
        words = (_ws("hi", 1.0, GREEN), _ws("there", 0.9, GREEN))
        msg = summary_message(words)
        assert "90%" in msg or "95%" in msg
        assert "0 to review" in msg

    def test_counts_flagged_words(self) -> None:
        words = (
            _ws("good", 0.9, GREEN),
            _ws("bad", 0.3, RED, flagged=True),
        )
        msg = summary_message(words)
        assert "1 to review" in msg


class TestResultViewWiring:
    """Widget-level wiring using the conftest Qt fakes."""

    def test_initial_empty_state(self) -> None:
        view = ResultView()
        assert view.lbl_summary.text() == _EMPTY_HINT
        assert view._words == ()

    def test_set_results_clears_old_cards(self) -> None:
        view = ResultView()
        view.set_results((_ws("one", 1.0, GREEN),))
        assert len(view._cards) == 1
        view.set_results((_ws("a", 1.0, GREEN), _ws("b", 1.0, GREEN)))
        assert len(view._cards) == 2

    def test_banner_updates_with_results(self) -> None:
        view = ResultView()
        view.set_results((_ws("word", 1.0, GREEN),))
        assert "0 to review" in view.lbl_summary.text()

    def test_click_handler_emits(self) -> None:
        view = ResultView()
        ws = _ws("tap", 1.0, GREEN)
        seen: list[tuple[int, WordScore]] = []
        view.word_clicked.connect(lambda i, w: seen.append((i, w)))
        view._make_click_handler(3, ws)(None)
        assert seen == [(3, ws)]
