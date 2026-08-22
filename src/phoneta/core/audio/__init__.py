"""Audio capture and voice-activity detection."""

from phoneta.core.audio.recorder import AudioRecorder
from phoneta.core.audio.vad import VoiceActivityDetector, trim_silence

__all__ = ["AudioRecorder", "VoiceActivityDetector", "trim_silence"]