"""PySide6 application bootstrap."""

from __future__ import annotations

import sys


def run() -> int:
    """Create and run the Qt application."""
    from PySide6.QtWidgets import QApplication

    from phoneta.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Phoneta")
    app.setOrganizationName("Phoneta")
    app.setApplicationVersion("0.1.0")

    # Dark-friendly base style
    app.setStyleSheet(
        """
        QMainWindow { background: #fafafa; }
        QLabel { color: #212121; }
        QLineEdit, QComboBox { padding: 4px; }
        QPushButton {
            background: #1565c0; color: white; border: none;
            border-radius: 6px; padding: 6px 20px; font-weight: bold;
        }
        QPushButton:hover { background: #1976d2; }
        QPushButton:disabled { background: #90caf9; }
        """
    )

    window = MainWindow()
    window.show()
    result: int = app.exec()
    return result
