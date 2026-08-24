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
            sleep=MagicMock(),
            default=MagicMock(),
            InputStream=MagicMock(),
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
        class _FakeSignal:
            def __init__(self, *types: object) -> None:
                pass

            def connect(self, *args: object) -> None:
                pass

        class _FakeQThread:
            """A QThread stub whose __init__ accepts optional parent."""

            def __init__(self, parent: object = None) -> None:
                pass

            def start(self) -> None:
                pass

        _conv = _module(
            "PySide6.QtCore",
            Signal=_FakeSignal,
            QThread=_FakeQThread,
            Qt=MagicMock(),
            QObject=MagicMock(),
        )

        class _FakeQWidget:
            """A QWidget stub whose __init__ accepts optional parent."""

            def __init__(self, parent: object = None) -> None:
                pass

            def setContentsMargins(self, *args: object) -> None:
                pass

            def setLayout(self, *args: object) -> None:
                pass

            def setMinimumHeight(self, h: int) -> None:
                pass

            def setFixedSize(self, *args: object) -> None:
                pass

            def setStyleSheet(self, s: str) -> None:
                pass

            def setToolTip(self, s: str) -> None:
                pass

            def update(self) -> None:
                pass

            def findChild(self, *args: object, **kwargs: object) -> object:
                return None

            def sizeHint(self) -> object:
                from PySide6.QtCore import QSize  # noqa: F811
                return QSize()

            def isVisible(self) -> bool:
                return True

            def setVisible(self, v: bool) -> None:
                pass

            def setEnabled(self, v: bool) -> None:
                pass

            def setText(self, s: str) -> None:
                pass

        class _FakeQLabel(_FakeQWidget):
            def setWordWrap(self, w: bool) -> None:
                pass

            def text(self) -> str:
                return ""

        class _FakeQDialog(_FakeQWidget):
            """QDialog stub."""

            class DialogCode:
                Accepted = 1
                Rejected = 0

            def setModal(self, m: bool) -> None:
                pass

            def setWindowTitle(self, t: str) -> None:
                pass

            def resize(self, w: int, h: int) -> None:
                pass

            def exec(self) -> int:
                return self.DialogCode.Accepted

            def accept(self) -> None:
                pass

            def reject(self) -> None:
                pass

            def show(self) -> None:
                pass

        class _FakeQMainWindow(_FakeQWidget):
            def resize(self, w: int, h: int) -> None:
                pass

            def setCentralWidget(self, w: object) -> None:
                pass

            def setWindowTitle(self, t: str) -> None:
                pass

            def statusBar(self) -> object:
                return MagicMock()

        class _FakeQPushButton(_FakeQWidget):
            def click(self) -> None:
                pass

            clicked = _FakeSignal()

        class _FakeQProgressBar(_FakeQWidget):
            def setRange(self, lo: int, hi: int) -> None:
                pass

            def setValue(self, v: int) -> None:
                pass

        class _FakeQComboBox(_FakeQWidget):
            def addItems(self, items: list[str]) -> None:
                pass

            def currentText(self) -> str:
                return "English"

        _wid = _module(
            "PySide6.QtWidgets",
            QApplication=MagicMock(),
            QComboBox=_FakeQComboBox,
            QDialog=_FakeQDialog,
            QFrame=_FakeQWidget,
            QGroupBox=_FakeQWidget,
            QHBoxLayout=MagicMock(),
            QHeaderView=MagicMock(),
            QLabel=_FakeQLabel,
            QLineEdit=_FakeQWidget,
            QMainWindow=_FakeQMainWindow,
            QMessageBox=MagicMock(),
            QProgressBar=_FakeQProgressBar,
            QPushButton=_FakeQPushButton,
            QScrollArea=_FakeQWidget,
            QSizePolicy=MagicMock(),
            QTableWidget=_FakeQWidget,
            QTableWidgetItem=MagicMock(),
            QVBoxLayout=MagicMock(),
            QWidget=_FakeQWidget,
        )
        _pyside = _module("PySide6", QtCore=_conv, QtWidgets=_wid)
        sys.modules["PySide6"] = _pyside
        sys.modules["PySide6.QtCore"] = _conv
        sys.modules["PySide6.QtWidgets"] = _wid


_install_fakes()
