"""Tests for the first-run setup screen (mocked heavy deps)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from phoneta.models.registry import ModelEntry
from phoneta.ui.setup_screen import _DownloadWorker


class TestDownloadWorker:
    """Background download thread signal sequencing."""

    def test_constructor_stores_models(self) -> None:
        models = [
            ModelEntry(name="whisper-base", description="whisper", present=lambda: False),
        ]
        worker = _DownloadWorker(models)
        assert worker._models == models

    def test_progress_signal_wiring(self) -> None:
        """Signal.connect should be callable (non-crashing wiring test)."""
        models = [
            ModelEntry(name="whisper-base", description="whisper", present=lambda: False),
        ]
        worker = _DownloadWorker(models)

        captured: list[tuple[str, int]] = []
        worker.progress.connect(lambda n, s: captured.append((n, s)))

        captured_done: list[bool] = []
        worker.all_done.connect(lambda ok: captured_done.append(ok))

        # Don't start the thread — just verify signal wiring doesn't crash
        assert len(captured) == 0
        assert len(captured_done) == 0


class TestSetupScreenWithMockedModels:
    """SetupScreen behaviour when model presence is mocked."""

    @patch("phoneta.ui.setup_screen.missing_models", return_value=[])
    def test_all_present_sets_ready_flag(self, _mock_mm: MagicMock) -> None:
        screen = _construct_screen_safely()
        # When missing_models returns [], _all_ok = True and "All models are already present"
        assert screen._all_ok is True

    @patch(
        "phoneta.ui.setup_screen.missing_models",
        return_value=[
            ModelEntry(name="whisper-base", description="whisper desc", present=lambda: False),
            ModelEntry(name="silero-vad", description="silero desc", present=lambda: False),
        ],
    )
    def test_missing_creates_progress_bars(self, _mock_mm: MagicMock) -> None:
        screen = _construct_screen_safely()
        assert screen._all_ok is False
        assert len(screen._bars) == 2
        assert "whisper-base" in screen._bars
        assert "silero-vad" in screen._bars


def _construct_screen_safely():
    """Construct a SetupScreen without triggering real imports.

    We manually call SetupScreen.__init__ after patching missing_models.
    The constructor calls _build() which accesses the QDialog parent ctor.
    """
    from phoneta.ui.setup_screen import SetupScreen

    # Create with no parent; all QWidget constructors are mocked
    return SetupScreen()