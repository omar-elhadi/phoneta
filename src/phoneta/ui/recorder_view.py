"""Recorder view — microphone controls and live level meter.

Runs the :class:`~phoneta.core.audio.recorder.AudioRecorder` on a background
thread so the UI stays responsive.  Design logic lives in pure helpers so it
is unit-testable without a display server.
"""

from __future__ import annotations

import time

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
from phoneta.ui.theme import meter_color


def format_elapsed(elapsed: float, total: float) -> str:
    """Human countdown label, e.g. ``0:04 / 0:10``."""
    e = max(0, int(elapsed))
    t = max(0, int(total))
    return f"{e // 60}:{e % 60:02d} / {t // 60}:{t % 60:02d}"


def meter_value(rms: float) -> int:
    """Map RMS (speech range ≈ 0..0.5) onto a 0-100 progress value."""
    return min(int(max(rms, 0.0) * 200), 100)


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
    """Mic input area: one toggle button, countdown, colour-coded meter."""

    recording_done = Signal(object)  # RecordResult

    MAX_DURATION_S = 10.0
    _HINT = "Type a phrase above, then press Record and speak."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: _RecordWorker | None = None
        self._t0: float = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── top row: single toggle button + countdown ──────────────
        top = QHBoxLayout()
        self.btn_record = QPushButton("\U0001f3a4  Record")
        self.btn_record.setObjectName("record")
        self.btn_record.setToolTip("Click to start recording — click again to stop early")
        self.btn_record.clicked.connect(self._toggle_recording)
        self.btn_record.setMinimumHeight(48)
        top.addWidget(self.btn_record)

        self.lbl_duration = QLabel(format_elapsed(0.0, self.MAX_DURATION_S))
        self.lbl_duration.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.lbl_duration.setToolTip("Elapsed / maximum recording length")
        top.addWidget(self.lbl_duration)
        top.addStretch()
        layout.addLayout(top)

        # ── level meter ─────────────────────────────────────────────
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(False)
        self.level_meter.setMaximumHeight(14)
        layout.addWidget(self.level_meter)

        # ── status ──────────────────────────────────────────────────
        self.lbl_status = QLabel(self._HINT)
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    # ── slots ───────────────────────────────────────────────────────────

    def is_recording(self) -> bool:
        return self._worker is not None

    def _toggle_recording(self) -> None:
        if self.is_recording():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        self._t0 = time.monotonic()
        self.btn_record.setText("\u23f9  Stop & analyse")
        self.lbl_status.setText("Recording … speak now.")
        self._set_meter(0.0)

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
        self.lbl_status.setText("Recording stopped — nothing analysed.")

    def _on_level(self, rms: float) -> None:
        self._set_meter(rms)
        elapsed = time.monotonic() - self._t0
        self.lbl_duration.setText(format_elapsed(elapsed, self.MAX_DURATION_S))

    def _set_meter(self, rms: float) -> None:
        self.level_meter.setValue(meter_value(rms))
        self.level_meter.setStyleSheet(
            f"QProgressBar::chunk {{ background: {meter_color(rms)}; }}"
        )

    def _on_done(self, result: object) -> None:
        self._reset_ui()
        if isinstance(result, Exception):
            self.lbl_status.setText(
                "Recording error: no microphone found or access denied. "
                "Check that a mic is connected and try again."
            )
            return
        assert isinstance(result, RecordResult)
        if result.duration_s < 0.3:
            self.lbl_status.setText("Too short — hold Record a bit longer and speak.")
            return
        self.lbl_status.setText(f"Recorded {result.duration_s:.1f} s — analysing …")
        self.recording_done.emit(result)

    def _reset_ui(self) -> None:
        self.btn_record.setText("\U0001f3a4  Record")
        self.level_meter.setValue(0)
        self.lbl_duration.setText(format_elapsed(0.0, self.MAX_DURATION_S))
        self._worker = None
