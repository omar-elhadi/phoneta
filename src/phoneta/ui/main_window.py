"""Main application window — top-level layout wiring all views.

Runs the pronunciation pipeline on a background thread so the UI never
blocks.  Target text and language persist between runs via QSettings.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSettings, QThread, Signal
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

from phoneta.core.metrics.prosody import ProsodyResult
from phoneta.core.metrics.scoring import WordScore
from phoneta.core.pipeline import PipelineResult, run_pipeline
from phoneta.ui.inspector import InspectorDialog
from phoneta.ui.privacy_badge import PrivacyBadge
from phoneta.ui.recorder_view import RecorderView
from phoneta.ui.result_view import ResultView

LANGS = {"English": "en", "French (français)": "fr"}


def error_hint(exc: Exception) -> str:
    """Map an exception onto an actionable, user-friendly message."""
    msg = str(exc).lower()
    if "microphone" in msg or "portaudio" in msg or "audio" in msg:
        hint = (
            "Check that a microphone is connected and not muted, "
            "then try recording again."
        )
    elif "model" in msg or "whisper" in msg or "download" in msg:
        hint = (
            "A speech model may be missing. Run it from the setup screen "
            "or with: python scripts/download_models.py"
        )
    elif "espeak" in msg or "phonemize" in msg:
        hint = (
            "Phoneme analysis needs espeak-ng. Install it with: "
            "sudo apt install espeak-ng (word scores still work without it)."
        )
    elif "empty" in msg:
        hint = "Type a target phrase and record yourself speaking it."
    else:
        hint = "Please try again. If it keeps happening, restart the app."
    return f"{type(exc).__name__}: {exc}\n\n{hint}"


class _PipelineWorker(QThread):
    """Run the pipeline off the main (GUI) thread."""

    done = Signal(object)

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
        self.resize(760, 600)

        self._last_prosody: ProsodyResult | None = None
        self._last_words: tuple[WordScore, ...] = ()
        self._last_audio: np.ndarray | None = None
        self._pipeline_worker: _PipelineWorker | None = None
        self._settings = QSettings("Phoneta", "Phoneta")
        self._build()
        self._restore_settings()

    # ── UI construction ─────────────────────────────────────────────

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ── top row: title + about + privacy badge ────────────────
        top_row = QHBoxLayout()
        title = QLabel("Phoneta")
        title.setObjectName("heading")
        top_row.addWidget(title)
        top_row.addStretch()

        self.btn_about = QPushButton("\u2139 About")
        self.btn_about.setObjectName("ghost")
        self.btn_about.setToolTip("About Phoneta")
        self.btn_about.clicked.connect(self._show_about)
        top_row.addWidget(self.btn_about)

        top_row.addWidget(PrivacyBadge())
        root.addLayout(top_row)

        # ── target text input ─────────────────────────────────────
        row = QHBoxLayout()
        label = QLabel("Phrase to practise:")
        label.setToolTip("The sentence you will read aloud")
        row.addWidget(label)
        self.txt_target = QLineEdit()
        self.txt_target.setPlaceholderText("Type the phrase you want to practise …")
        self.txt_target.setToolTip("Type a sentence, then record yourself saying it")
        self.txt_target.returnPressed.connect(self._reanalyse)
        row.addWidget(self.txt_target)
        root.addLayout(row)

        # ── language selector + analyse ───────────────────────────
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItems(list(LANGS))
        self.cmb_lang.setToolTip("Language of the target phrase")
        lang_row.addWidget(self.cmb_lang)
        lang_row.addStretch()

        self.btn_analyse = QPushButton("\U0001f50d Re-analyse")
        self.btn_analyse.setToolTip("Re-run analysis on your latest recording")
        self.btn_analyse.clicked.connect(self._reanalyse)
        self.btn_analyse.setMinimumHeight(40)
        self.btn_analyse.setEnabled(False)
        lang_row.addWidget(self.btn_analyse)
        root.addLayout(lang_row)

        # ── recorder view ─────────────────────────────────────────
        self.recorder = RecorderView()
        self.recorder.recording_done.connect(self._on_recording)
        root.addWidget(self.recorder)

        # ── results ───────────────────────────────────────────────
        self.results = ResultView()
        self.results.word_clicked.connect(self._on_word_clicked)
        root.addWidget(self.results)

        # ── status bar ────────────────────────────────────────────
        self.statusBar().showMessage("Ready — 100% offline, nothing leaves this machine")

    # ── settings persistence ─────────────────────────────────────────

    def _restore_settings(self) -> None:
        text = self._settings.value("target_text", "")
        if isinstance(text, str) and text:
            self.txt_target.setText(text)
        lang = self._settings.value("language", "")
        if isinstance(lang, str) and lang in LANGS:
            self.cmb_lang.setCurrentText(lang)

    def _save_settings(self) -> None:
        self._settings.setValue("target_text", self.txt_target.text().strip())
        self._settings.setValue("language", self.cmb_lang.currentText())

    # ── slots ────────────────────────────────────────────────────────

    def _on_recording(self, result) -> None:
        """Recording finished → store audio, auto-start pipeline."""
        from phoneta.core.audio.recorder import RecordResult

        if not isinstance(result, RecordResult):
            return
        self._last_audio = result.samples
        self.btn_analyse.setEnabled(True)
        self._reanalyse()

    def _reanalyse(self) -> None:
        """Analyse (or re-analyse) the latest recording."""
        target = self.txt_target.text().strip()
        if not target:
            self.statusBar().showMessage("Type a phrase to practise first.")
            return
        if self._last_audio is None:
            self.statusBar().showMessage(
                "Press Record, speak, then analysis starts automatically."
            )
            return

        lang = LANGS[self.cmb_lang.currentText()]
        self.statusBar().showMessage("Analysing pronunciation …")
        self.btn_analyse.setEnabled(False)

        self._pipeline_worker = _PipelineWorker(
            target_text=target,
            lang=lang,
            audio_samples=self._last_audio,
        )
        self._pipeline_worker.done.connect(self._on_pipeline_done)
        self._pipeline_worker.start()

    def _on_pipeline_done(self, result: object) -> None:
        self.btn_analyse.setEnabled(self._last_audio is not None)
        if isinstance(result, Exception):
            QMessageBox.warning(self, "Analysis failed", error_hint(result))
            self.statusBar().showMessage("Analysis failed — see the message for help.")
            return

        assert isinstance(result, PipelineResult)
        self._last_words = result.words
        self._last_prosody = result.prosody
        self.results.set_results(result.words)
        self._save_settings()
        self.statusBar().showMessage(
            f"Done — {len(result.words)} words scored · "
            f"{result.alignment_method} · audio deleted: {result.audio_deleted}"
        )

    def _on_word_clicked(self, idx: int, ws) -> None:
        if self._last_prosody is None:
            return
        dlg = InspectorDialog(ws, self._last_prosody, parent=self)
        dlg.exec()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Phoneta",
            "<b>Phoneta</b> — offline pronunciation coach.<br><br>"
            "Records your voice, aligns it against the target phrase and "
            "highlights every word: <span style='color:#2e7d32'>green</span> "
            "= good, <span style='color:#f9a825'>yellow</span> = close, "
            "<span style='color:#c62828'>red</span> = needs work. "
            "Click any word for phoneme-level detail.<br><br>"
            "100% offline — your audio never leaves this machine and is "
            "deleted immediately after analysis.",
        )
