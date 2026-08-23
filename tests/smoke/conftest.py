"""Smoke conftest — use real modules instead of root conftest's fakes.

When you run ``pytest tests/smoke/``, this conftest removes the fake torch,
sounddevice, librosa, etc. modules so the real ones are imported.

**Run smoke tests separately from unit tests** — they are excluded from the
default collection via ``norecursedirs`` in pyproject.toml.
"""

from __future__ import annotations

import sys

_FAKE_PREFIXES = (
    "torch",
    "torchaudio",
    "sounddevice",
    "phonemizer",
    "faster_whisper",
    "montreal_forced_aligner",
    "librosa",
)

_saved: dict[str, object] = {}


def pytest_sessionstart(session) -> None:
    """Remove conftest fakes so real modules are importable."""
    for key in list(sys.modules.keys()):
        if key.startswith(_FAKE_PREFIXES):
            _saved[key] = sys.modules[key]
            del sys.modules[key]


def pytest_sessionfinish(session, exitstatus) -> None:
    """Restore the fakes so subsequent test runs in-process are clean."""
    for key, mod in _saved.items():
        sys.modules[key] = mod
    _saved.clear()