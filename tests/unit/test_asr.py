"""Tests for the faster-whisper ASR wrapper (mocked model)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from phoneta.core.alignment.asr import Transcriber, Transcription


class TestTranscriber:
    def test_transcribe_with_timestamps(self) -> None:
        """End-to-end with word timestamps enabled."""
        mock_seg = MagicMock()
        mock_seg.text = "hello world"
        mock_seg.words = [
            MagicMock(word="hello", start=0.0, end=0.5),
            MagicMock(word="world", start=0.6, end=1.0),
        ]
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            transcriber = Transcriber()
            result = transcriber.transcribe("fake.wav")

        assert isinstance(result, Transcription)
        assert result.text == "hello world"
        assert result.language == "en"
        assert len(result.words) == 2
        assert result.words[0].word == "hello"
        assert result.words[0].start_s == 0.0
        assert result.words[0].end_s == 0.5

    def test_transcribe_no_word_timestamps(self) -> None:
        """Falls back when word_timestamps is not supported."""
        mock_seg = MagicMock()
        mock_seg.text = "fallback"
        mock_seg.words = None
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model = MagicMock()
        # side_effect list: first call (word_timestamps) raises, second succeeds
        mock_model.transcribe.side_effect = [
            TypeError("not supported"),
            ([mock_seg], mock_info),
        ]

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            result = Transcriber().transcribe("fake.wav")

        assert result.words == ()

    def test_model_reuse(self) -> None:
        """WhisperModel is created only once."""
        mock = MagicMock()
        with patch("faster_whisper.WhisperModel", return_value=mock):
            transcriber = Transcriber()
            transcriber._load()
            transcriber._load()
            from faster_whisper import WhisperModel

            assert WhisperModel.call_count == 1

    def test_model_dir_forwarded(self) -> None:
        with patch("faster_whisper.WhisperModel") as mock_cls:
            Transcriber(model_dir="/tmp/models")._load()
            mock_cls.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="int8",
                download_root="/tmp/models",
            )