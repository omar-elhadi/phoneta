"""PySide6 application bootstrap.

Kept separate from ``__main__`` so the CLI and GUI entry points can diverge
cleanly while the engine is still under construction.
"""

from __future__ import annotations


def run() -> int:
    """Create and run the Qt application (implemented in a later phase)."""
    raise NotImplementedError("The PySide6 UI is implemented in a later phase.")
