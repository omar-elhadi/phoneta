"""Practice history view.

History contains scores and target text only; raw recordings are never stored.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from phoneta.storage.db import PracticeStore, SessionRow


def format_session_date(timestamp: float) -> str:
    """Format a stored Unix timestamp for a compact history row."""
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M")


def session_summary(session: SessionRow) -> str:
    """Build the user-facing summary for one stored session."""
    if session.words:
        accuracy = sum(word.accuracy for word in session.words) / len(session.words)
        reviewed = sum(word.has_error for word in session.words)
    else:
        accuracy = 0.0
        reviewed = 0
    return (
        f"{format_session_date(session.created_at)} · {session.lang} · "
        f"{accuracy:.0%} · {reviewed} to review\n{session.target_text}"
    )


class HistoryView(QWidget):
    """Compact list of recent local practice sessions."""

    def __init__(self, store: PracticeStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel("Recent practice")
        self.lbl_title.setObjectName("subheading")
        layout.addWidget(self.lbl_title)
        self.list = QListWidget()
        self.list.setToolTip("Your recent scores, stored only on this computer")
        layout.addWidget(self.list)

    def refresh(self) -> None:
        """Reload recent sessions from SQLite."""
        self.list.clear()
        sessions = self._store.list_sessions(limit=20)
        if not sessions:
            self.list.addItem("No practice sessions yet — your results will appear here.")
            return
        for session in sessions:
            self.list.addItem(QListWidgetItem(session_summary(session)))
