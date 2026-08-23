"""First-run setup dialog — download models with progress indicators."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phoneta.models.registry import ModelEntry, missing_models


class _DownloadWorker(QThread):
    """Run model downloads on a background thread, emitting per-model progress."""

    progress = Signal(str, int)  # model name, step: 0=start, 1=done, -1=failed
    all_done = Signal(bool)      # True = all succeeded

    def __init__(self, models: list[ModelEntry], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._models = models

    def run(self) -> None:
        all_ok = True
        for m in self._models:
            self.progress.emit(m.name, 0)
            try:
                if not _download_one(m):
                    self.progress.emit(m.name, -1)
                    all_ok = False
                else:
                    self.progress.emit(m.name, 1)
            except Exception:
                self.progress.emit(m.name, -1)
                all_ok = False
        self.all_done.emit(all_ok)


def _download_one(entry: ModelEntry) -> bool:
    """Download a single model entry.  Returns True on success."""
    name = entry.name

    if name == "whisper-base":
        from faster_whisper import WhisperModel
        WhisperModel("base", device="cpu", compute_type="int8")
        return True

    if name == "silero-vad":
        from silero_vad import load_silero_vad
        load_silero_vad(onnx=True)
        return True

    if name == "mfa-en":
        from montreal_forced_aligner.models import AcousticModel  # type: ignore[import-untyped]
        AcousticModel.download("english_mfa")
        return True

    if name == "mfa-fr":
        from montreal_forced_aligner.models import AcousticModel  # type: ignore[import-untyped]
        AcousticModel.download("french_mfa")
        return True

    if name == "espeak-ng":
        # espeak-ng must be installed by the user — warn and skip
        return False

    return False


class SetupScreen(QDialog):
    """First-run model download screen.

    Shows a list of required models with per-item progress indicators
    and a "Download All" button that fetches everything in the background.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phoneta — First Run Setup")
        self.resize(500, 380)
        self.setModal(True)

        self._bars: dict[str, QProgressBar] = {}
        self._labels: dict[str, QLabel] = {}
        self._worker: _DownloadWorker | None = None
        self._all_ok = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        header = QLabel(
            "<h2>Welcome to Phoneta</h2>"
            "<p>Phoneta is a <b>100% offline</b> pronunciation coach. "
            "Before your first session, a few models must be downloaded once. "
            "After this setup, Phoneta never connects to the internet.</p>"
        )
        header.setWordWrap(True)
        root.addWidget(header)

        models = missing_models()
        if not models:
            root.addWidget(QLabel("✓ All models are already present."))
            self._all_ok = True
            btn = QPushButton("Start Using Phoneta")
            btn.clicked.connect(self.accept)
            root.addWidget(btn)
            return

        root.addWidget(QLabel(f"<b>{len(models)} model(s) to download:</b>"))

        for m in models:
            row = QVBoxLayout()
            label = QLabel(f"  {m.description}")
            row.addWidget(label)
            bar = QProgressBar()
            bar.setRange(0, 0)  # indeterminate while downloading
            bar.setVisible(False)
            row.addWidget(bar)
            root.addLayout(row)
            self._labels[m.name] = label
            self._bars[m.name] = bar

        self._status = QLabel("")
        root.addWidget(self._status)

        self._btn_download = QPushButton("Download All Models")
        self._btn_download.clicked.connect(self._start_downloads)
        self._btn_download.setMinimumHeight(40)
        root.addWidget(self._btn_download)

        self._btn_skip = QPushButton("Skip Setup (offline-only)")
        self._btn_skip.setToolTip(
            "You can run the downloader later with:\n"
            "  python scripts/download_models.py"
        )
        self._btn_skip.clicked.connect(self.reject)
        root.addWidget(self._btn_skip)

    def _start_downloads(self) -> None:
        self._btn_download.setEnabled(False)
        self._btn_skip.setEnabled(False)
        self._status.setText("Downloading …")
        for bar in self._bars.values():
            bar.setVisible(True)

        models = missing_models()
        self._worker = _DownloadWorker(models, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.all_done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, name: str, step: int) -> None:
        bar = self._bars.get(name)
        if bar is None:
            return
        if step == 0:
            bar.setRange(0, 0)  # indeterminate
        elif step == 1:
            bar.setRange(0, 1)
            bar.setValue(1)
            label = self._labels.get(name)
            if label:
                label.setText(label.text() + "  ✓")
        else:
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setStyleSheet("QProgressBar::chunk { background: #c62828; }")
            label = self._labels.get(name)
            if label:
                label.setText(label.text() + "  ✗")

    def _on_done(self, all_ok: bool) -> None:
        self._all_ok = all_ok
        self._btn_skip.setEnabled(True)
        self._btn_skip.setText("Close")
        if all_ok:
            self._status.setText("All models ready — offline pronunciation coach is set up!")
            self.accept()
        else:
            self._status.setText(
                "Some downloads failed. Check your network connection and try again, "
                "or skip and use the command-line downloader later."
            )
            self._btn_download.setEnabled(True)
            self._btn_download.setText("Retry")

    @property
    def all_ready(self) -> bool:
        return self._all_ok