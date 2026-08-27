"""Tests for the bundled app icon."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from phoneta import __version__
from phoneta.ui.icon import icon_path


class TestVersion:
    def test_version_bumped(self) -> None:
        assert __version__ == "0.2.0"


class TestIconAsset:
    def test_icon_path_resolves(self) -> None:
        path = icon_path()
        assert path is not None
        assert Path(path).exists()

    def test_icon_is_valid_xml(self) -> None:
        path = icon_path()
        assert path is not None
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")

    def test_icon_has_viewbox(self) -> None:
        path = icon_path()
        assert path is not None
        root = ET.parse(path).getroot()
        assert root.get("viewBox") == "0 0 128 128"


class TestIconLoader:
    def test_app_icon_never_raises(self) -> None:
        """Loader returns an icon or None — never raises, even without Qt GUI."""
        from phoneta.ui.icon import app_icon

        app_icon()  # the point is: no exception
