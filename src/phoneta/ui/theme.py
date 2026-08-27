"""Phoneta design system — colour tokens and the app-wide stylesheet.

Keeping this in one place gives the whole app a consistent look and lets the
stylesheet be unit-tested without a display server.
"""

from __future__ import annotations

# ── palette ────────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    # base surfaces
    "bg": "#f5f7fa",
    "surface": "#ffffff",
    "border": "#d7dce3",
    # text
    "text": "#1f2430",
    "text-muted": "#6b7280",
    # brand / primary
    "primary": "#1565c0",
    "primary-hover": "#1976d2",
    "primary-disabled": "#a9c7e8",
    # feedback (matches scoring colours)
    "green": "#2e7d32",
    "green-bg": "#e8f5e9",
    "green-border": "#a5d6a7",
    "yellow": "#f9a825",
    "yellow-bg": "#fff8e1",
    "yellow-border": "#ffe082",
    "red": "#c62828",
    "red-bg": "#ffebee",
    "red-border": "#ef9a9a",
    # level meter
    "meter-low": "#43a047",
    "meter-mid": "#f9a825",
    "meter-high": "#e53935",
}

_FONT_STACK = (
    "-apple-system, 'Segoe UI', Roboto, 'Ubuntu', 'Cantarell', 'Noto Sans', sans-serif"
)


def build_stylesheet() -> str:
    """Return the application-wide Qt stylesheet (QSS)."""
    c = COLORS
    return f"""
QMainWindow, QDialog {{
    background: {c["bg"]};
}}
QWidget {{
    font-family: {_FONT_STACK};
    font-size: 13px;
    color: {c["text"]};
}}
QLabel {{
    background: transparent;
}}
QLabel#heading {{
    font-size: 20px;
    font-weight: 700;
}}
QLabel#muted {{
    color: {c["text-muted"]};
}}
QLineEdit, QComboBox {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 6px 10px;
}}
QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {c["primary"]};
}}
QPushButton {{
    background: {c["primary"]};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 22px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {c["primary-hover"]}; }}
QPushButton:disabled {{ background: {c["primary-disabled"]}; }}
QPushButton#record {{
    background: {c["red"]};
    font-size: 15px;
    padding: 10px 28px;
}}
QPushButton#record:hover {{ background: #d32f2f; }}
QPushButton#record:checked {{ background: {c["primary"]}; }}
QPushButton#record:disabled {{ background: #ef9a9a; }}
QPushButton#ghost {{
    background: transparent;
    color: {c["primary"]};
    border: 1px solid {c["primary"]};
}}
QPushButton#ghost:hover {{ background: {c["green-bg"]}; }}
QProgressBar {{
    background: {c["border"]};
    border: none;
    border-radius: 4px;
}}
QProgressBar::chunk {{ border-radius: 4px; }}
QStatusBar {{
    background: {c["surface"]};
    color: {c["text-muted"]};
    border-top: 1px solid {c["border"]};
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QTableWidget {{
    background: {c["surface"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    gridline-color: {c["border"]};
}}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {c["border"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QLabel#privacy_badge {{
    color: {c["green"]};
    background: {c["green-bg"]};
    border: 1px solid {c["green-border"]};
    border-radius: 6px;
    padding: 4px 16px;
    font-weight: bold;
}}
"""


def meter_color(rms: float) -> str:
    """Colour for the live level meter given an RMS in ``[0, 1]``."""
    if rms < 0.08:
        return COLORS["meter-low"]
    if rms < 0.30:
        return COLORS["meter-mid"]
    return COLORS["meter-high"]
