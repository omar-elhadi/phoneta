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
        # styled centrally via QLabel#privacy_badge in ui/theme.py
        label.setObjectName("privacy_badge")
        layout.addWidget(label)