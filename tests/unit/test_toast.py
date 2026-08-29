"""Tests for non-blocking UX messages."""

from phoneta.ui.toast import analysis_message, recording_message


def test_recording_messages_are_clear() -> None:
    assert "started" in recording_message(True)
    assert "analysing" in recording_message(False)


def test_analysis_messages_are_clear() -> None:
    assert "moment" in analysis_message(False)
    assert analysis_message(True, 1) == "Analysis complete — 1 word scored."
    assert analysis_message(True, 2) == "Analysis complete — 2 words scored."
