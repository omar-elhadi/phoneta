"""Main application window — top-level layout wiring all views.

Runs the pronunciation pipeline on a background thread so the UI never blocks.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phoneta.core.pipeline import PipelineResult, run_pipeline
from phoneta.ui.inspector import InspectorDialog
from phoneta.ui.privacy_badge import PrivacyBadge
from phoneta.ui.recorder_view import RecorderView
from phoneta.ui.result_view import ResultView

LANGS = {"English": "en", "French (français)": "fr"}


class _PipelineWorker(QThread):
    """Run the pipeline off the main (GUI) thread."""

    done = Signal(object)  # PipelineResult | Exception

    def __init__(
        self,
        target_text: str,
        lang: str,
        audio_samples,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = target_text
        self._lang = lang
        self._audio = audio_samples

    def run(self) -> None:
        try:
            result = run_pipeline(
                target_text=self._text,
                lang=self._lang,
                audio_samples=self._audio,
            )
            self.done.emit(result)
        except Exception as exc:
            self.done.emit(exc)


class MainWindow(QMainWindow):
    """Phoneta main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Phoneta — Offline Pronunciation Coach")
        self.resize(720, 560)

        self._last_prosody = None
        self._last_words = ()
        self._pipeline_worker: _PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── privacy badge (top-right anchored via row) ──────────
        top_row = QHBoxLayout()
        top_row.addStretch()
        top_row.addWidget(PrivacyBadge())
        root.addLayout(top_row)

        # ── target text input ───────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Target text:"))
        self.txt_target = QLineEdit()
        self.txt_target.setPlaceholderText("Type the phrase you want to practise …")
        self.txt_target.returnPressed.connect(self._start_pipeline)
        row.addWidget(self.txt_target)
        root.addLayout(row)

        # ── language selector ───────────────────────────────────
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(list(LANGS))
        lang_row.addWidget(self.cmb_lang)
        lang_row.addStretch()

        self.btn_analyse = QPushButton("\U0001f50d Analyse")
        self.btn_analyse.clicked.connect(self._start_pipeline)
        self.btn_analyse.setMinimumHeight(40)
        lang_row.addWidget(self.btn_analyse)
        root.addLayout(lang_row)

        # ── recorder view ───────────────────────────────────────
        self.recorder = RecorderView()
        self.recorder.recording_done.connect(self._on_recording)
        root.addWidget(self.recorder)

        # ── results ─────────────────────────────────────────────
        self.results = ResultView()
        self.results.word_clicked.connect(self._on_word_clicked)
        root.addWidget(self.results)

        # ── status bar ──────────────────────────────────────────
        self.statusBar().showMessage("Ready — offline mode")

    # ── slots ───────────────────────────────────────────────────────

    def _on_recording(self, result) -> None:
        """Recording finished → auto-start pipeline."""
        from phoneta.core.audio.recorder import RecordResult

        if not isinstance(result, RecordResult):
            return

        target = self.txt_target.text().strip()
        if not target:
            self.statusBar().showMessage("Enter target text first.")
            return

        lang = LANGS[self.cmb_lang.currentText()]
        self.statusBar().showMessage("Analysing pronunciation …")
        self.btn_analyse.setEnabled(False)

        self._pipeline_worker = _PipelineWorker(
            target_text=target,
            lang=lang,
            audio_samples=result.samples,
        )
        self._pipeline_worker.done.connect(self._on_pipeline_done)
        self._pipeline_worker.start()

    def _start_pipeline(self) -> None:
        """Manual analyse button — no recording, use last recording?"""
        self.statusBar().showMessage(
            "Press Record first, then speak — analysis starts automatically."
        )

    def _on_pipeline_done(self, result: object) -> None:
        self.btn_analyse.setEnabled(True)
        if isinstance(result, Exception):
            QMessageBox.warning(self, "Error", f"Analysis failed:\n{result}")
            self.statusBar().showMessage("Analysis failed.")
            return

        self._last_words = result.words
        self._last_prosody = result.prosody
        self.results.set_results(result.words)
        self.statusBar().showMessage(
            f"Done — {len(result.words)} words scored "
            f"· {result.alignment_method} · audio deleted: {result.audio_deleted}"
        )

    def _on_word_clicked(self, idx: int, ws) -> None:
        if self._last_prosody is None:
            return
        dlg = InspectorDialog(ws, self._last_prosody, parent=self)
        dlg.exec()