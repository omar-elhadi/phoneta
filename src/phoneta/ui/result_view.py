"""Result view — colour-coded word cards with an overall score banner.

Each word is a clickable card coloured by its
:class:`~phoneta.core.metrics.scoring.WordScore`.  Cards wrap across rows so
long sentences stay readable.  Layout maths lives in pure helpers for easy
unit testing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from phoneta.core.metrics.scoring import GREEN, RED, YELLOW, WordScore
from phoneta.ui.theme import COLORS

_COLOUR_MAP = {
    GREEN: (COLORS["green"], COLORS["green-bg"], COLORS["green-border"]),
    YELLOW: (COLORS["yellow"], COLORS["yellow-bg"], COLORS["yellow-border"]),
    RED: (COLORS["red"], COLORS["red-bg"], COLORS["red-border"]),
}

_EMPTY_HINT = "Record a phrase to see colour-coded feedback here."


def grid_columns(n_words: int) -> int:
    """How many word cards fit per row (wraps long sentences)."""
    return max(1, min(6, n_words))


def banner_color(avg_accuracy: float, n_errors: int) -> str:
    """Banner colour from the session's average accuracy and error count."""
    if n_errors == 0 and avg_accuracy >= 0.85:
        return COLORS["green"]
    if n_errors == 0 or avg_accuracy >= 0.6:
        return COLORS["yellow"]
    return COLORS["red"]


def summary_message(words: tuple[WordScore, ...]) -> str:
    """One-line encouragement + statistics for the session."""
    if not words:
        return _EMPTY_HINT
    avg = sum(w.accuracy for w in words) / len(words)
    n_err = sum(1 for w in words if w.has_error)
    pct = f"{avg:.0%}"
    if n_err == 0 and avg >= 0.85:
        mood = "Excellent pronunciation!"
    elif n_err == 0:
        mood = "Great job — every word recognised."
    elif avg >= 0.6:
        mood = "Good — a few words need attention (tap one for details)."
    else:
        mood = "Keep practising — tap a red word to see what to fix."
    return f"{pct} overall · {len(words)} words · {n_err} to review — {mood}"


def _word_card(word_score: WordScore) -> QFrame:
    """Build a single word card with the right colour scheme."""
    fg, bg, border = _COLOUR_MAP.get(word_score.color, _COLOUR_MAP[GREEN])

    card = QFrame()
    card.setFrameShape(QFrame.Shape.StyledPanel)
    card.setStyleSheet(
        f"""
        QFrame {{
            background: {bg};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        """
    )
    card.setCursor(Qt.CursorShape.PointingHandCursor)
    card.setToolTip(
        f"{word_score.word}: {word_score.accuracy:.0%} accuracy"
        + (" — click for phoneme details" if word_score.has_error else "")
    )

    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 6, 10, 6)
    layout.setSpacing(2)

    label = QLabel(word_score.word)
    label.setStyleSheet(
        f"color: {fg}; font-size: 18px; font-weight: bold;"
        " border: none; background: transparent;"
    )
    layout.addWidget(label)

    pct = QLabel(f"{word_score.accuracy:.0%}")
    pct.setStyleSheet(
        f"color: {fg}; font-size: 11px; border: none; background: transparent;"
    )
    layout.addWidget(pct)

    return card


class ResultView(QWidget):
    """Wrapped grid of colour-coded word cards plus a score banner."""

    word_clicked = Signal(int, WordScore)  # index, score

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._words: tuple[WordScore, ...] = ()
        self._cards: list[QFrame] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── score banner ─────────────────────────────────────────
        self.lbl_summary = QLabel(_EMPTY_HINT)
        self.lbl_summary.setObjectName("score_banner")
        self.lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_summary.setWordWrap(True)
        layout.addWidget(self.lbl_summary)

        # ── scrollable word grid ─────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(8)
        self._scroll.setWidget(self.container)

        layout.addWidget(self._scroll)

    def set_results(self, words: tuple[WordScore, ...]) -> None:
        """Replace the displayed word cards with *words*."""
        self._words = words

        for card in self._cards:
            card.deleteLater()
        self._cards.clear()

        cols = grid_columns(len(words))
        for i, ws in enumerate(words):
            card = _word_card(ws)
            card.mouseReleaseEvent = self._make_click_handler(i, ws)  # type: ignore[method-assign]
            self._cards.append(card)
            row, col = divmod(i, cols)
            self.grid.addWidget(card, row, col)

        self._update_banner(words)

    def _update_banner(self, words: tuple[WordScore, ...]) -> None:
        self.lbl_summary.setText(summary_message(words))
        if not words:
            self.lbl_summary.setStyleSheet("color: #6b7280; font-size: 14px;")
            return
        avg = sum(w.accuracy for w in words) / len(words)
        n_err = sum(1 for w in words if w.has_error)
        self.lbl_summary.setStyleSheet(
            f"color: {banner_color(avg, n_err)}; font-size: 15px; font-weight: 600;"
        )

    def _make_click_handler(self, idx: int, ws: WordScore):
        """Return a callable that emits ``word_clicked``."""

        def handler(_event) -> None:
            self.word_clicked.emit(idx, ws)

        return handler
