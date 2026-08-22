"""End-to-end pronunciation evaluation pipeline.

Orchestrates the full flow:

    record / audio file
    → VAD silence trimming
    → ASR transcription (faster-whisper)
    → reference-text to IPA (phonemizer / espeak-ng)
    → forced alignment (MFA or word-level fallback)
    → Needleman-Wunsch phoneme alignment per word
    → per-word scoring (green / yellow / red)
    → prosody (F0) analysis
    → auto-delete raw WAV (privacy guarantee)

Every heavy dependency is imported lazily inside the relevant submodule,
not here, so the pipeline module itself is always importable.
"""

from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from phoneta.core.alignment.g2p import TextToIPA, WordIPA
from phoneta.core.alignment.mfa import ForcedAligner, PhonemeSegment
from phoneta.core.alignment.sequence import align as nw_align
from phoneta.core.audio.vad import VoiceActivityDetector
from phoneta.core.metrics.prosody import ProsodyResult
from phoneta.core.metrics.scoring import WordScore, score_word


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of one pronunciation evaluation."""

    target_text: str
    transcribed_text: str
    language: str
    words: tuple[WordScore, ...]
    prosody: ProsodyResult
    audio_deleted: bool
    alignment_method: str = ""  # "mfa" | "fallback"


def _write_wav(samples: np.ndarray, sample_rate: int, path: str) -> float:
    """Write float32 mono samples to a 16-bit PCM WAV file.

    Returns the audio duration in seconds.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return len(samples) / sample_rate


def _segment_word_index_map(
    segments: tuple[PhonemeSegment, ...], word_ipas: list[WordIPA]
) -> list[list[PhonemeSegment]]:
    """Group phoneme (or word-level fallback) segments by target word.

    For real MFA output: phonemes are in reference-text order, so the per-word
    phoneme counts from g2p tell us how to split the segment list.
    For fallback: each segment IS a word — map 1:1.
    """
    # Detect fallback: each segment's ``phoneme`` is a whole word (has ≥2 chars
    # and no IPA diacritics).  Real phonemes are short IPA tokens.
    if all(len(s.phoneme) >= 2 and s.phoneme.isalpha() for s in segments):
        return [[s] for s in segments]

    # Real MFA: split by word-level phoneme counts.
    grouped: list[list[PhonemeSegment]] = []
    cursor = 0
    for wip in word_ipas:
        count = len(wip.phonemes) if wip.analyzed else 1  # OOV → 1 placeholder
        grouped.append(list(segments[cursor : cursor + count]))
        cursor += count
    return grouped


def _segments_to_user_tokens(segs: list[PhonemeSegment]) -> list[str]:
    """Extract phoneme tokens from a group of segments."""
    return [s.phoneme for s in segs]


def _score_words(
    word_ipas: list[WordIPA],
    segment_groups: list[list[PhonemeSegment]],
    prosody: ProsodyResult,
) -> list[WordScore]:
    """Align reference IPA to user segments and score each word."""
    scores: list[WordScore] = []
    for wip, segs in zip(word_ipas, segment_groups):
        if not wip.analyzed or not wip.phonemes:
            # OOV — mark green (neutral) with no feedback
            scores.append(
                WordScore(word=wip.word, accuracy=1.0, color="green", feedback=())
            )
            continue

        ref_tokens = list(wip.phonemes)
        user_tokens = _segments_to_user_tokens(segs)
        confidences = [s.confidence for s in segs]

        # Ensure confidences match the alignment columns — pad/trim to user_tokens
        if len(confidences) != len(user_tokens):
            confidences = [1.0] * len(user_tokens)

        alignment = nw_align(ref_tokens, user_tokens)
        # Map confidences to alignment columns (not raw segments) by distributing
        # the user-side confidences to the alignment positions that consume user.
        aligned_confs: list[float] = []
        user_idx = 0
        for pair in alignment:
            if pair.user is not None:
                aligned_confs.append(
                    confidences[user_idx] if user_idx < len(confidences) else 1.0
                )
                user_idx += 1
            else:
                aligned_confs.append(1.0)  # deletion — no user confidence

        # Prosody issue: mark yellow when boundary is flat/missing rise
        # and the word is at the end of the utterance.
        is_last = wip is word_ipas[-1]
        prosody_issue = is_last and not prosody.boundary_rise

        scores.append(
            score_word(
                wip.word,
                alignment,
                aligned_confs if aligned_confs else None,
                prosody_issue=prosody_issue,
            )
        )
    return scores


def run_pipeline(
    target_text: str,
    lang: str,
    audio_path: Optional[str | Path] = None,
    audio_samples: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    delete_audio: bool = True,
    vad_threshold: float = 0.5,
    asr_model_size: str = "base",
) -> PipelineResult:
    """Evaluate pronunciation of *target_text* against recorded audio.

    You must provide one of *audio_path* or *audio_samples*.

    Parameters
    ----------
    target_text:
        The text the user was supposed to say.
    lang:
        Language code (``"en"``, ``"fr"``).
    audio_path:
        Path to a 16 kHz mono WAV file.
    audio_samples:
        float32 mono numpy array at *sample_rate*.
    sample_rate:
        Sample rate of *audio_samples* (ignored when *audio_path* is used).
    delete_audio:
        If ``True``, the WAV file at *audio_path* (or the temp file created
        from *audio_samples*) is deleted after analysis — the privacy
        guarantee.
    vad_threshold:
        Speech-probability threshold for silence trimming.
    asr_model_size:
        faster-whisper model size (``"tiny"``, ``"base"``, ``"small"``, …).
    """
    temp_dir = tempfile.mkdtemp(prefix="phoneta_")
    own_path = False

    try:
        # ---- audio ingestion -------------------------------------------------
        if audio_samples is not None:
            wav_path = Path(temp_dir) / "recording.wav"
            duration_s = _write_wav(audio_samples, sample_rate, str(wav_path))
            own_path = True
        elif audio_path is not None:
            wav_path = Path(audio_path)
            # Determine duration from WAV header when possible.
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    duration_s = wf.getnframes() / wf.getframerate()
            except (wave.Error, EOFError, FileNotFoundError):
                duration_s = 1.0
        else:
            raise ValueError("One of audio_path or audio_samples is required")

        # ---- silence trim ----------------------------------------------------
        vad = VoiceActivityDetector(threshold=vad_threshold)
        trimmed_samples: np.ndarray
        if audio_samples is not None:
            trimmed_samples = _trim(vad, audio_samples, sample_rate)
        else:
            # Read WAV into samples for VAD; keep originals for prosody.
            with wave.open(str(wav_path), "rb") as wf:
                raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            orig_audio = raw.astype(np.float32) / 32768.0
            trimmed_samples = _trim(vad, orig_audio, sample_rate)

        trimmed_wav = Path(temp_dir) / "trimmed.wav"
        trimmed_duration = _write_wav(trimmed_samples, sample_rate, str(trimmed_wav))

        # ---- ASR -------------------------------------------------------------
        from phoneta.core.alignment.asr import Transcriber

        transcriber = Transcriber(model_size=asr_model_size)
        transcription = transcriber.transcribe(str(trimmed_wav))

        # ---- g2p -------------------------------------------------------------
        g2p = TextToIPA()
        word_ipas = g2p.phonemize(target_text, lang)

        # ---- forced alignment ------------------------------------------------
        aligner = ForcedAligner(lang=lang)
        alignment_result = aligner.align(
            str(trimmed_wav), target_text, trimmed_duration
        )

        # ---- per-word scoring ------------------------------------------------
        segment_groups = _segment_word_index_map(
            alignment_result.segments, word_ipas
        )

        # ---- prosody ---------------------------------------------------------
        from phoneta.core.metrics.prosody import extract_f0

        # Run prosody on the trimmed audio (better signal for F0)
        prosody = extract_f0(trimmed_samples, sample_rate)

        # ---- score -----------------------------------------------------------
        word_scores = _score_words(word_ipas, segment_groups, prosody)

        # ---- clean-up --------------------------------------------------------
        audio_deleted = False
        if delete_audio:
            if own_path or audio_path is not None:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                    audio_deleted = True
                except OSError:
                    pass

        return PipelineResult(
            target_text=target_text,
            transcribed_text=transcription.text,
            language=lang,
            words=tuple(word_scores),
            prosody=prosody,
            audio_deleted=audio_deleted,
            alignment_method=alignment_result.method,
        )

    finally:
        # Always clean the temp dir (trimmed WAV, possible recording WAV).
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def _trim(
    vad: VoiceActivityDetector, audio: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Run VAD trim, return trimmed audio (or original if fully silent)."""
    from phoneta.core.audio.vad import trim_silence

    return trim_silence(audio, sample_rate, threshold=vad.threshold)