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
            """Class-level signal declaration with per-instance binding."""

            def __init__(self, *types: object) -> None:
                self._name = ""

            def __set_name__(self, owner: type, name: str) -> None:
                self._name = "_signal_" + name

            def __get__(self, obj: object, objtype: type | None = None) -> object:
                if obj is None:
                    return self
                bound = getattr(obj, self._name, None)
                if bound is None:
                    bound = _BoundSignal()
                    setattr(obj, self._name, bound)
                return bound

            def connect(self, *args: object) -> None:
                pass

            def emit(self, *args: object) -> None:
                pass

        class _BoundSignal:
            """Instance-bound signal with working connect/emit."""

            def __init__(self) -> None:
                self._slots: list[object] = []

            def connect(self, slot: object) -> None:
                self._slots.append(slot)

            def emit(self, *args: object) -> None:
                for slot in list(self._slots):
                    slot(*args)

        class _FakeQThread:
            """A QThread stub whose __init__ accepts optional parent."""

            def __init__(self, parent: object = None) -> None:
                pass

            def start(self) -> None:
                pass

        class _FakeQSettings:
            """QSettings stub with process-wide persistence like the real thing."""

            _store: dict[str, object] = {}

            def __init__(self, *args: object) -> None:
                pass

            def value(self, key: str, default: object = None, type: object = None) -> object:
                return self._store.get(key, default)

            def setValue(self, key: str, value: object) -> None:
                self._store[key] = value

        class _FakeQTimer:
            timeout = _FakeSignal()

            def __init__(self, parent: object = None) -> None:
                pass

            def setSingleShot(self, value: bool) -> None:
                pass

            def start(self, milliseconds: int) -> None:
                pass

        _conv = _module(
            "PySide6.QtCore",
            Signal=_FakeSignal,
            QTimer=_FakeQTimer,
            QThread=_FakeQThread,
            Qt=MagicMock(),
            QObject=MagicMock(),
            QSettings=_FakeQSettings,
        )

        class _FakeQWidget:
            """A QWidget stub whose __init__ accepts optional parent."""

            class Shape:
                StyledPanel = 6

            returnPressed = _FakeSignal()

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

            def show(self) -> None:
                self._visible = True

            def hide(self) -> None:
                self._visible = False

            def raise_(self) -> None:
                pass

            def adjustSize(self) -> None:
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
                self._enabled = v

            def isEnabled(self) -> bool:
                return getattr(self, "_enabled", True)

            def setObjectName(self, name: str) -> None:
                self._object_name = name

            def setFrameShape(self, shape: object) -> None:
                pass

            def setCursor(self, cursor: object) -> None:
                pass

            def deleteLater(self) -> None:
                pass

            def objectName(self) -> str:
                return getattr(self, "_object_name", "")

            def setText(self, s: str) -> None:
                self._text = s

            def setPlaceholderText(self, text: str) -> None:
                pass

            def setCurrentText(self, text: str) -> None:
                self._text = text

            def text(self) -> str:
                return getattr(self, "_text", "")

        class _FakeQLabel(_FakeQWidget):
            def __init__(self, text: object = "", parent: object = None) -> None:
                super().__init__(parent)
                if isinstance(text, str):
                    self.setText(text)

            def setWordWrap(self, w: bool) -> None:
                pass

            def setAlignment(self, alignment: object) -> None:
                pass

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

        class _FakeQStatusBar:
            def __init__(self) -> None:
                self._message = ""

            def showMessage(self, msg: str, timeout: int = 0) -> None:
                self._message = msg

            def currentMessage(self) -> str:
                return self._message

        class _FakeQMainWindow(_FakeQWidget):
            def __init__(self, parent: object = None) -> None:
                super().__init__(parent)
                self._status_bar = _FakeQStatusBar()

            def resize(self, w: int, h: int) -> None:
                pass

            def setCentralWidget(self, w: object) -> None:
                pass

            def setWindowTitle(self, t: str) -> None:
                pass

            def statusBar(self) -> _FakeQStatusBar:
                return self._status_bar

        class _FakeQPushButton(_FakeQWidget):
            def click(self) -> None:
                pass

            clicked = _FakeSignal()

        class _FakeQProgressBar(_FakeQWidget):
            def setRange(self, lo: int, hi: int) -> None:
                pass

            def setValue(self, v: int) -> None:
                pass

            def setTextVisible(self, visible: bool) -> None:
                pass

            def setMaximumHeight(self, h: int) -> None:
                pass

        class _FakeQComboBox(_FakeQWidget):
            def addItems(self, items: list[str]) -> None:
                self._items = list(items)
                if items:
                    self._text = items[0]

            def currentText(self) -> str:
                return getattr(self, "_text", "")

        class _FakeQListWidgetItem:
            def __init__(self, text: str) -> None:
                self._text = text

            def text(self) -> str:
                return self._text

        class _FakeQListWidget(_FakeQWidget):
            def __init__(self, parent: object = None) -> None:
                super().__init__(parent)
                self._items: list[_FakeQListWidgetItem] = []

            def addItem(self, item: object) -> None:
                if isinstance(item, str):
                    item = _FakeQListWidgetItem(item)
                self._items.append(item)

            def clear(self) -> None:
                self._items.clear()

            def count(self) -> int:
                return len(self._items)

            def item(self, index: int) -> _FakeQListWidgetItem:
                return self._items[index]

        class _FakeQScrollArea(_FakeQWidget):
            def setWidgetResizable(self, resizable: bool) -> None:
                pass

            def setWidget(self, widget: object) -> None:
                pass

            def setHorizontalScrollBarPolicy(self, policy: object) -> None:
                pass

        _wid = _module(
            "PySide6.QtWidgets",
            QApplication=MagicMock(),
            QComboBox=_FakeQComboBox,
            QDialog=_FakeQDialog,
            QFrame=_FakeQWidget,
            QGridLayout=MagicMock(),
            QGroupBox=_FakeQWidget,
            QHBoxLayout=MagicMock(),
            QHeaderView=MagicMock(),
            QLabel=_FakeQLabel,
            QLineEdit=_FakeQWidget,
            QListWidget=_FakeQListWidget,
            QListWidgetItem=_FakeQListWidgetItem,
            QMainWindow=_FakeQMainWindow,
            QMessageBox=MagicMock(),
            QProgressBar=_FakeQProgressBar,
            QPushButton=_FakeQPushButton,
            QScrollArea=_FakeQScrollArea,
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
