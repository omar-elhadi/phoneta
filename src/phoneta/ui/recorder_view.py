"""Recorder view — microphone controls and live level meter.

Runs the :class:`~phoneta.core.audio.recorder.AudioRecorder` on a background
thread so the UI stays responsive.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from phoneta.core.audio.recorder import AudioRecorder, RecordResult


class _RecordWorker(QThread):
    """Runs ``AudioRecorder.record()`` off the main thread."""

    finished = Signal(object)  # RecordResult | Exception
    level_update = Signal(float)

    def __init__(
        self,
        duration_s: float,
        level_interval_s: float = 0.05,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._duration = duration_s
        self._interval = level_interval_s

    def run(self) -> None:
        try:
            recorder = AudioRecorder(
                duration_s=self._duration,
                level_interval_s=self._interval,
                on_level=self._emit_level,
            )
            result = recorder.record()
            self.finished.emit(result)
        except Exception as exc:
            self.finished.emit(exc)

    def _emit_level(self, rms: float) -> None:
        self.level_update.emit(rms)


class RecorderView(QWidget):
    """Mic input area: record/stop, level meter, duration label."""

    recording_done = Signal(object)  # RecordResult

    MAX_DURATION_S = 10.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: Optional[_RecordWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── top row: record button + duration ──────────────────────
        top = QHBoxLayout()
        self.btn_record = QPushButton("\U0001f3a4 Record")
        self.btn_record.clicked.connect(self._start_recording)
        self.btn_record.setMinimumHeight(44)
        top.addWidget(self.btn_record)

        self.btn_stop = QPushButton("\u23f9 Stop")
        self.btn_stop.clicked.connect(self._stop_recording)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(44)
        top.addWidget(self.btn_stop)

        self.lbl_duration = QLabel("0.0 s")
        self.lbl_duration.setStyleSheet("font-size: 14px;")
        top.addWidget(self.lbl_duration)
        top.addStretch()
        layout.addLayout(top)

        # ── level meter ───────────────────────────────────────────
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(False)
        self.level_meter.setMaximumHeight(16)
        layout.addWidget(self.level_meter)

        # ── status ─────────────────────────────────────────────────
        self.lbl_status = QLabel("Ready — select target text and press Record")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    # ── slots ──────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Recording … speak now.")

        self._worker = _RecordWorker(
            duration_s=self.MAX_DURATION_S,
            level_interval_s=0.05,
        )
        self._worker.level_update.connect(self._on_level)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _stop_recording(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(1000)
        self._reset_ui()
        self.lbl_status.setText("Recording stopped.")

    def _on_level(self, rms: float) -> None:
        # Normalise RMS to 0-100 (clamp reasonable speech range 0..0.5)
        value = min(int(rms * 200), 100)
        self.level_meter.setValue(value)
        # Update duration from worker elapsed if possible
        # (simplistic — just pulse the meter, worker handles actual duration)

    def _on_done(self, result: object) -> None:
        self._reset_ui()
        if isinstance(result, Exception):
            self.lbl_status.setText(f"Recording error: {result}")
            return
        self.lbl_status.setText(
            f"Recorded {result.duration_s:.1f} s — analysing …"
        )
        self.recording_done.emit(result)

    def _reset_ui(self) -> None:
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.level_meter.setValue(0)
        self._worker = None