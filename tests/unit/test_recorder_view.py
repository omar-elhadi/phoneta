"""Tests for recorder-view design logic and state machine (faked Qt)."""

from __future__ import annotations

from phoneta.ui.recorder_view import RecorderView, format_elapsed, meter_value


class TestFormatElapsed:
    def test_start(self) -> None:
        assert format_elapsed(0.0, 10.0) == "0:00 / 0:10"

    def test_midway(self) -> None:
        assert format_elapsed(4.7, 10.0) == "0:04 / 0:10"

    def test_minutes(self) -> None:
        assert format_elapsed(75.0, 600.0) == "1:15 / 10:00"

    def test_negative_clamped(self) -> None:
        assert format_elapsed(-1.0, 10.0) == "0:00 / 0:10"


class TestMeterValue:
    def test_silence(self) -> None:
        assert meter_value(0.0) == 0

    def test_typical_speech(self) -> None:
        assert 0 < meter_value(0.2) <= 100

    def test_clipping_clamped(self) -> None:
        assert meter_value(5.0) == 100

    def test_negative_clamped(self) -> None:
        assert meter_value(-0.5) == 0


class TestRecorderViewState:
    """State transitions using the conftest Qt fakes."""

    def _view(self) -> RecorderView:
        return RecorderView()

    def test_initial_state_not_recording(self) -> None:
        view = self._view()
        assert view.is_recording() is False
        assert view.lbl_status.text() == RecorderView._HINT

    def test_stop_when_idle_is_safe(self) -> None:
        view = self._view()
        view._stop_recording()  # must not raise
        assert view.is_recording() is False

    def test_reset_restores_button_and_labels(self) -> None:
        view = self._view()
        view.btn_record.setText("x")
        view.lbl_duration.setText("x")
        view._reset_ui()
        assert view.btn_record.text().startswith("\U0001f3a4")
        assert view.lbl_duration.text() == "0:00 / 0:10"

    def test_done_with_error_shows_friendly_hint(self) -> None:
        view = self._view()
        view._on_done(RuntimeError("boom"))
        assert "microphone" in view.lbl_status.text().lower()

    def test_done_with_short_clip_rejected(self) -> None:
        import numpy as np

        from phoneta.core.audio.recorder import RecordResult

        view = self._view()
        emitted: list[object] = []
        view.recording_done.connect(emitted.append)
        result = RecordResult(
            samples=np.zeros(100, dtype="float32"),
            sample_rate=16000,
            duration_s=0.1,
            peak_rms=0.1,
        )
        view._on_done(result)
        assert emitted == []
        assert "short" in view.lbl_status.text().lower()

    def test_done_with_valid_clip_emits(self) -> None:
        import numpy as np

        from phoneta.core.audio.recorder import RecordResult

        view = self._view()
        emitted: list[object] = []
        view.recording_done.connect(emitted.append)
        result = RecordResult(
            samples=np.zeros(16000, dtype="float32"),
            sample_rate=16000,
            duration_s=1.0,
            peak_rms=0.1,
        )
        view._on_done(result)
        assert emitted == [result]
