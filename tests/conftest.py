"""Pytest fixtures.

Injects lightweight fake ``sounddevice`` / ``torch`` / ``phonemizer`` /
``faster_whisper`` / ``montreal_forced_aligner`` modules into ``sys.modules``
so unit tests can run without the heavy runtime dependencies installed.  Real
packages, when present, are left untouched.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _NoGrad:
    """Fake ``torch.no_grad`` context manager."""

    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _from_numpy(array):
    """Fake ``torch.from_numpy`` — returns a tensor whose ``.float()`` is itself."""
    tensor = MagicMock()
    tensor.float.return_value = tensor
    return tensor


def _install_fakes() -> None:
    if "torch" not in sys.modules:
        fake_hub = _module("torch.hub", load=MagicMock())
        fake_torch = _module("torch", hub=fake_hub, no_grad=_NoGrad, from_numpy=_from_numpy)
        sys.modules["torch"] = fake_torch
        sys.modules["torch.hub"] = fake_hub

    if "sounddevice" not in sys.modules:
        fake_sd = _module(
            "sounddevice",
            query_devices=MagicMock(return_value={"default_samplerate": 16000}),
            rec=MagicMock(),
            wait=MagicMock(),
            default=MagicMock(),
        )
        sys.modules["sounddevice"] = fake_sd

    if "phonemizer" not in sys.modules:
        fake_ph = _module("phonemizer")
        fake_backend = _module("phonemizer.backend", EspeakBackend=MagicMock())
        fake_ph.backend = fake_backend
        sys.modules["phonemizer"] = fake_ph
        sys.modules["phonemizer.backend"] = fake_backend

    if "faster_whisper" not in sys.modules:
        fake_fw = _module("faster_whisper", WhisperModel=MagicMock())
        sys.modules["faster_whisper"] = fake_fw

    if "montreal_forced_aligner" not in sys.modules:
        sys.modules["montreal_forced_aligner"] = _module("montreal_forced_aligner")

    if "librosa" not in sys.modules:
        sys.modules["librosa"] = _module("librosa", pyin=MagicMock(), times_like=MagicMock())

    if "PySide6" not in sys.modules:
        _conv = _module(
            "PySide6.QtCore",
            Signal=type("Signal", (), {}),
            QThread=type("QThread", (), {}),
            Qt=MagicMock(),
        )
        _wid = _module(
            "PySide6.QtWidgets",
            QApplication=MagicMock(),
            QComboBox=MagicMock(),
            QDialog=MagicMock(),
            QFrame=MagicMock(),
            QGroupBox=MagicMock(),
            QHBoxLayout=MagicMock(),
            QHeaderView=MagicMock(),
            QLabel=MagicMock(),
            QLineEdit=MagicMock(),
            QMainWindow=MagicMock(),
            QMessageBox=MagicMock(),
            QProgressBar=MagicMock(),
            QPushButton=MagicMock(),
            QScrollArea=MagicMock(),
            QSizePolicy=MagicMock(),
            QTableWidget=MagicMock(),
            QTableWidgetItem=MagicMock(),
            QVBoxLayout=MagicMock(),
            QWidget=MagicMock(),
        )
        _pyside = _module("PySide6", QtCore=_conv, QtWidgets=_wid)
        sys.modules["PySide6"] = _pyside
        sys.modules["PySide6.QtCore"] = _conv
        sys.modules["PySide6.QtWidgets"] = _wid


_install_fakes()
