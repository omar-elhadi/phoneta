"""Privacy badge widget — the \"100% Offline Mode\" indicator.

A small, always-visible label that reassures the user no data leaves the
machine.  It also serves as the visual anchor for the app's core selling
point: *zero telemetry, fully local*.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class PrivacyBadge(QWidget):
    """Green badge reading \"🔒 100% Offline Mode\"."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("\U0001f512 100% Offline Mode")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("privacy_badge")
        label.setStyleSheet(
            """
            QLabel#privacy_badge {
                color: #2e7d32;
                background: #e8f5e9;
                border: 1px solid #a5d6a7;
                border-radius: 6px;
                padding: 4px 16px;
                font-weight: bold;
            }
            """
        )
        layout.addWidget(label)