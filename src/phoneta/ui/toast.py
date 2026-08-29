"""Non-blocking user notifications for long-running operations."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QWidget

from phoneta.ui.theme import COLORS


def recording_message(started: bool) -> str:
    """Return the status text for recording start/finish transitions."""
    if started:
        return "Recording started — speak now."
    return "Recording finished — analysing your pronunciation …"


def analysis_message(done: bool, word_count: int = 0) -> str:
    """Return the status text for analysis start/finish transitions."""
    if not done:
        return "Analysing pronunciation … this may take a moment."
    return f"Analysis complete — {word_count} word{'s' if word_count != 1 else ''} scored."


class Toast(QLabel):
    """A temporary, non-modal message displayed inside the parent window."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.setObjectName("toast")
        self.setWordWrap(True)
        self.setStyleSheet(
            f"background: {COLORS['surface']}; color: {COLORS['text']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 8px; "
            "padding: 10px 14px; font-weight: 600;"
        )
        self.hide()

    def show_message(self, message: str, duration_ms: int = 3500) -> None:
        """Show *message* without interrupting the current interaction."""
        self.setText(message)
        self.adjustSize()
        self.show()
        self.raise_()
        self._timer.start(duration_ms)
