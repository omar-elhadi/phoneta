# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Phoneta — standalone Windows desktop app.

Usage::

    pyinstaller phoneta-windows.spec

The built app goes to ``dist/Phoneta/``.  Models are NOT bundled — users run
through the first-run download flow.

Windows-specific notes:
    - espeak-ng and PortAudio DLLs must be on the build machine (CI installs them).
    - ``console=False`` produces a pure GUI app (no terminal window).
"""

from __future__ import annotations

import os
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
    # storage
    "sqlite3",
    # Windows extras (DLL loading)
    "_sounddevice_data",
    "sounddevice_data",
]

# ── binary / data includes ─────────────────────────────────────────
BINARIES: list[tuple[str, str]] = []

# Try to locate espeak-ng binaries on the build machine
ESPEAK_PATHS = [
    r"C:\Program Files\eSpeak NG",
    r"C:\Program Files (x86)\eSpeak NG",
    os.environ.get("ESPEAK_PATH", ""),
]
for esp in ESPEAK_PATHS:
    if esp and Path(esp).is_dir():
        for dll in Path(esp).glob("*.dll"):
            BINARIES.append((str(dll), "."))
        for exe in Path(esp).glob("*.exe"):
            BINARIES.append((str(exe), "."))

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

# ── spec ───────────────────────────────────────────────────────────
a = Analysis(
    [str(SRC / "phoneta" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=BINARIES,
    datas=PHONEMIZER_DATA + LIBROSA_DATA,
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
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)