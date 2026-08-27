"""Tests for the centralised design system (pure Python, no Qt)."""

from __future__ import annotations

import re

from phoneta.ui import theme

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class TestColorTokens:
    def test_all_tokens_are_hex(self) -> None:
        for name, value in theme.COLORS.items():
            assert _HEX.match(value), f"{name} = {value!r} is not a hex colour"

    def test_required_tokens_present(self) -> None:
        for key in ("bg", "primary", "green", "yellow", "red", "surface", "border"):
            assert key in theme.COLORS

    def test_feedback_colors_distinct(self) -> None:
        assert len({theme.COLORS["green"], theme.COLORS["yellow"], theme.COLORS["red"]}) == 3


class TestBuildStylesheet:
    def test_returns_nonempty_string(self) -> None:
        qss = theme.build_stylesheet()
        assert isinstance(qss, str)
        assert len(qss) > 200

    def test_styles_core_widgets(self) -> None:
        qss = theme.build_stylesheet()
        for selector in ("QMainWindow", "QPushButton", "QLineEdit", "QProgressBar", "QStatusBar"):
            assert selector in qss

    def test_styles_privacy_badge_by_object_name(self) -> None:
        assert "QLabel#privacy_badge" in theme.build_stylesheet()

    def test_record_button_variant(self) -> None:
        assert "QPushButton#record" in theme.build_stylesheet()

    def test_no_unclosed_braces(self) -> None:
        qss = theme.build_stylesheet()
        assert qss.count("{") == qss.count("}")


class TestMeterColor:
    def test_silence_is_green(self) -> None:
        assert theme.meter_color(0.0) == theme.COLORS["meter-low"]

    def test_normal_speech_is_amber(self) -> None:
        assert theme.meter_color(0.15) == theme.COLORS["meter-mid"]

    def test_loud_is_red(self) -> None:
        assert theme.meter_color(0.5) == theme.COLORS["meter-high"]

    def test_thresholds_monotonic(self) -> None:
        """Color escalates as loudness rises; no regressions at boundaries."""
        low = theme.meter_color(0.05)
        mid = theme.meter_color(0.2)
        high = theme.meter_color(0.4)
        assert low != mid != high
