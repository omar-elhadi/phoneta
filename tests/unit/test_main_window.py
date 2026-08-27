"""Tests for main-window logic: error hints, settings, re-analyse gating."""

from __future__ import annotations

from PySide6.QtCore import QSettings  # faked by conftest in unit runs

from phoneta.ui.main_window import MainWindow, error_hint


class TestErrorHint:
    def test_mic_error_gets_mic_guidance(self) -> None:
        hint = error_hint(RuntimeError("PortAudio: no microphone available"))
        assert "microphone" in hint.lower()

    def test_model_error_gets_download_guidance(self) -> None:
        hint = error_hint(RuntimeError("whisper model missing"))
        assert "download_models" in hint

    def test_espeak_error_gets_install_guidance(self) -> None:
        hint = error_hint(RuntimeError("espeak-ng not found"))
        assert "espeak-ng" in hint

    def test_unknown_error_gets_generic_guidance(self) -> None:
        hint = error_hint(RuntimeError("quantum fluctuation"))
        assert "try again" in hint.lower()

    def test_includes_exception_type(self) -> None:
        hint = error_hint(ValueError("bad input"))
        assert hint.startswith("ValueError: bad input")


class TestSettingsPersistence:
    def setup_method(self) -> None:
        QSettings._store.clear()  # type: ignore[attr-defined]

    def test_save_stores_text_and_language(self) -> None:
        win = MainWindow()
        win.txt_target.setText("bonjour le monde")
        win._save_settings()
        assert QSettings._store["target_text"] == "bonjour le monde"  # type: ignore[attr-defined]
        assert QSettings._store["language"] == "English"  # type: ignore[attr-defined]

    def test_restore_roundtrip(self) -> None:
        QSettings._store["target_text"] = "the quick brown fox"  # type: ignore[attr-defined]
        QSettings._store["language"] = "French (français)"  # type: ignore[attr-defined]
        win = MainWindow()
        assert win.txt_target.text() == "the quick brown fox"
        assert win.cmb_lang.currentText() == "French (français)"

    def test_restore_ignores_unknown_language(self) -> None:
        QSettings._store["language"] = "Klingon"  # type: ignore[attr-defined]
        win = MainWindow()
        assert win.cmb_lang.currentText() == "English"  # first item

    def test_save_strips_whitespace(self) -> None:
        win = MainWindow()
        win.txt_target.setText("  padded  ")
        win._save_settings()
        assert QSettings._store["target_text"] == "padded"  # type: ignore[attr-defined]


class TestReanalyseGating:
    def setup_method(self) -> None:
        QSettings._store.clear()  # type: ignore[attr-defined]

    def test_reanalyse_without_text_shows_hint(self) -> None:
        win = MainWindow()
        win._reanalyse()
        assert "phrase" in win.statusBar().currentMessage().lower()

    def test_reanalyse_without_recording_is_noop(self) -> None:
        win = MainWindow()
        win.txt_target.setText("hello world")
        win._reanalyse()  # no recording yet — must not raise or start a worker
        assert win._pipeline_worker is None

    def test_analyse_disabled_until_first_recording(self) -> None:
        win = MainWindow()
        assert win.btn_analyse.isEnabled() is False


class TestRecordingFlow:
    def setup_method(self) -> None:
        QSettings._store.clear()  # type: ignore[attr-defined]

    def test_non_recordresult_ignored(self) -> None:
        win = MainWindow()
        win._on_recording("not a record result")
        assert win._last_audio is None

    def test_recording_enables_analyse(self) -> None:
        import numpy as np

        from phoneta.core.audio.recorder import RecordResult

        win = MainWindow()
        result = RecordResult(
            samples=np.zeros(16000, dtype="float32"),
            sample_rate=16000,
            duration_s=1.0,
            peak_rms=0.1,
        )
        win._on_recording(result)
        assert win._last_audio is not None
        assert win.btn_analyse.isEnabled() is True
