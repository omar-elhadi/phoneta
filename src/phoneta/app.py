"""PySide6 application bootstrap."""

from __future__ import annotations

import sys


def _version() -> str:
    """Resolve the package version without importing heavy deps."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("phoneta")
    except PackageNotFoundError:
        return "0.0.0"


def run() -> int:
    """Create and run the Qt application."""
    from PySide6.QtWidgets import QApplication

    from phoneta.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Phoneta")
    app.setOrganizationName("Phoneta")
    app.setApplicationVersion(_version())
    from phoneta.ui.theme import build_stylesheet

    app.setStyleSheet(build_stylesheet())

    from phoneta.ui.icon import app_icon

    icon = app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    # ── first-run model setup ──────────────────────────────────
    from phoneta.models.registry import all_present
    from phoneta.ui.setup_screen import SetupScreen

    if not all_present():
        setup = SetupScreen()
        if setup.exec() != SetupScreen.DialogCode.Accepted:
            # User skipped — open main window anyway (limited functionality)
            pass

    # ── main window ────────────────────────────────────────────
    window = MainWindow()
    window.show()
    result: int = app.exec()
    return result
