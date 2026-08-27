# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Phoneta — standalone Linux (later Windows) desktop app.

Usage::

    pyinstaller phoneta.spec

The built app goes to ``dist/Phoneta/``.  Models are NOT bundled — users run
through the first-run download flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules  # type: ignore[import-untyped]

# ── paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".").resolve()
SRC = PROJECT_ROOT / "src"

# ── hidden imports for lazy-loaded backends ────────────────────────
HIDDEN_IMPORTS = [
    # audio
    "sounddevice",
    "numpy",
    # ASR
    "faster_whisper",
    "ctranslate2",
    # g2p
    "phonemizer",
    "phonemizer.backend",
    # alignment
    "montreal_forced_aligner",
    "montreal_forced_aligner.models",
    # pitch
    "librosa",
    # VAD
    "silero_vad",
    "torch",
    # UI
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # storage (stdlib, but explicit)
    "sqlite3",
]

# ── binary / data includes ─────────────────────────────────────────
BINARIES: list[tuple[str, str]] = []

# Collect phonemizer backend data files
try:
    import phonemizer

    PHONEMIZER_DATA = collect_data_files("phonemizer")
except Exception:
    PHONEMIZER_DATA = []

# Collect librosa data files
try:
    import librosa

    LIBROSA_DATA = collect_data_files("librosa")
except Exception:
    LIBROSA_DATA = []

# App icon asset (bundled inside the phoneta package)
ICON_DATA = [(str(SRC / "phoneta" / "assets" / "phoneta.svg"), "phoneta/assets")]

# ── spec ───────────────────────────────────────────────────────────
a = Analysis(
    [str(SRC / "phoneta" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=BINARIES,
    datas=PHONEMIZER_DATA + LIBROSA_DATA + ICON_DATA,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Phoneta",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)