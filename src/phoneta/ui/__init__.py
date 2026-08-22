"""Phoneta user interface (PySide6)."""

from .inspector import InspectorDialog
from .main_window import MainWindow
from .privacy_badge import PrivacyBadge
from .recorder_view import RecorderView
from .result_view import ResultView

__all__ = [
    "InspectorDialog",
    "MainWindow",
    "PrivacyBadge",
    "RecorderView",
    "ResultView",
]