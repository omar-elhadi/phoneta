"""Application icon loading.

The icon ships inside the package as an SVG so it works identically in dev
runs and PyInstaller bundles.  Loading degrades gracefully: if Qt's SVG
plugin or the file is unavailable, callers get an empty icon instead of a
crash.
"""

from __future__ import annotations

from importlib.resources import files

_ICON_RESOURCE = "assets/phoneta.svg"


def icon_path() -> str | None:
    """Filesystem path of the bundled icon, or None if unavailable."""
    try:
        return str(files("phoneta").joinpath(_ICON_RESOURCE))
    except Exception:
        return None


def app_icon():
    """QIcon for the app/window icon; empty icon when Qt or file is absent."""
    try:
        from PySide6.QtGui import QIcon
    except Exception:
        return None

    path = icon_path()
    if path is None:
        return QIcon()
    return QIcon(path)
