"""Result view — colour-coded target-text rendering.

Each word is shown as a clickable coloured label derived from its
:class:`~phoneta.core.metrics.scoring.WordScore`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from phoneta.core.metrics.scoring import GREEN, RED, YELLOW, WordScore

_COLOUR_MAP = {
    GREEN: ("#2e7d32", "#e8f5e9", "#a5d6a7"),
    YELLOW: ("#f9a825", "#fff8e1", "#ffe082"),
    RED: ("#c62828", "#ffebee", "#ef9a9a"),
}


def _word_card(word_score: WordScore) -> QFrame:
    """Build a single word card with the right colour scheme."""
    fg, bg, border = _COLOUR_MAP.get(
        word_score.color, _COLOUR_MAP[GREEN]
    )

    card = QFrame()
    card.setFrameShape(QFrame.Shape.StyledPanel)
    card.setStyleSheet(
        f"""
        QFrame {{
            background: {bg};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 4px 10px;
        }}
        """
    )
    card.setCursor(Qt.CursorShape.PointingHandCursor)
    card.setToolTip(
        f"{word_score.word}: {word_score.accuracy:.0%} accuracy"
        + (" \u26a0" if word_score.has_error else "")
    )

    layout = QVBoxLayout(card)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(2)

    label = QLabel(word_score.word)
    label.setStyleSheet(
        f"color: {fg}; font-size: 18px; font-weight: bold; border: none; background: transparent;"
    )
    layout.addWidget(label)

    pct = QLabel(f"{word_score.accuracy:.0%}")
    pct.setStyleSheet(
        f"color: {fg}; font-size: 11px; border: none; background: transparent;"
    )
    layout.addWidget(pct)

    return card


class ResultView(QWidget):
    """Scrollable row of colour-coded word cards."""

    word_clicked = Signal(int, WordScore)  # index, score

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._words: tuple[WordScore, ...] = ()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.container = QWidget()
        self.flow = QHBoxLayout(self.container)
        self.flow.setSpacing(8)
        self.flow.addStretch()
        self.scroll.setWidget(self.container)

        layout.addWidget(self.scroll)

        # Summary line
        self.lbl_summary = QLabel("")
        self.lbl_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_summary)

    def set_results(self, words: tuple[WordScore, ...]) -> None:
        """Replace the displayed word cards with *words*."""
        self._words = words

        # Clear existing cards
        while self.flow.count() > 1:  # stretch is last
            item = self.flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        for i, ws in enumerate(words):
            card = _word_card(ws)
            card.mousePressEvent = self._make_click_handler(i, ws)
            self.flow.insertWidget(self.flow.count() - 1, card)

        if words:
            n_err = sum(1 for w in words if w.has_error)
            avg = sum(w.accuracy for w in words) / len(words)
            self.lbl_summary.setText(
                f"{len(words)} words · {avg:.0%} accuracy · "
                f"{n_err} word{'s' if n_err != 1 else ''} need attention"
            )
        else:
            self.lbl_summary.setText("")

    def _make_click_handler(self, idx: int, ws: WordScore):
        """Return a callable that emits ``word_clicked``."""

        def handler(_event) -> None:
            self.word_clicked.emit(idx, ws)

        return handler