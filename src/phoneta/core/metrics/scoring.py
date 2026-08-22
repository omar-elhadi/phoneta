"""Convert phoneme alignments into per-word colours and per-phoneme flags.

Colour rules (from the spec):

* **green** — high phoneme-match accuracy (≥ 85%) with no errors.
* **yellow** — minor variance: mostly right but below the green bar (or a
  low-confidence phoneme that is otherwise correct).
* **red** — a substitution, insertion or deletion, a confidence below the
  threshold, or accuracy below the red bar.

A phoneme is **flagged** when its kind is a substitution/insertion/deletion
or its alignment confidence is below ``CONFIDENCE_THRESHOLD``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phoneta.core.alignment.sequence import ERROR_KINDS, MATCH, AlignedPair

GREEN = "green"
YELLOW = "yellow"
RED = "red"

GREEN_THRESHOLD = 0.85  # ≥ this accuracy => green
RED_THRESHOLD = 0.65  # < this accuracy => red
CONFIDENCE_THRESHOLD = 0.65  # flag phonemes with lower alignment confidence


@dataclass(frozen=True)
class PhonemeFeedback:
    """Per-phoneme evaluation for one alignment column."""

    ref: str | None  # target phoneme (None for insertions)
    user: str | None  # spoken phoneme (None for deletions)
    kind: str  # match | substitution | insertion | deletion
    confidence: float
    flagged: bool

    @classmethod
    def from_pair(cls, pair: AlignedPair, confidence: float) -> PhonemeFeedback:
        flagged = pair.kind in ERROR_KINDS or confidence < CONFIDENCE_THRESHOLD
        return cls(
            ref=pair.ref,
            user=pair.user,
            kind=pair.kind,
            confidence=confidence,
            flagged=flagged,
        )


@dataclass(frozen=True)
class WordScore:
    """Score for a single word of the target text."""

    word: str
    accuracy: float  # 0..1 — matched phonemes / total alignment columns
    color: str  # green | yellow | red
    feedback: tuple[PhonemeFeedback, ...]

    @property
    def has_error(self) -> bool:
        return any(f.flagged for f in self.feedback)


def score_word(
    word: str,
    pairs: Sequence[AlignedPair],
    confidences: Sequence[float] | None = None,
    prosody_issue: bool = False,
) -> WordScore:
    """Score one word from its alignment columns.

    Parameters
    ----------
    word:
        The word being scored.
    pairs:
        Alignment columns produced by :func:`phoneta.core.alignment.sequence.align`
        for this word's phonemes.
    confidences:
        Per-column alignment confidence in ``[0, 1]``, parallel to *pairs*.
        Defaults to ``1.0`` everywhere (word-level fallback has no
        phoneme-level confidence signal).
    prosody_issue:
        ``True`` when prosody analysis flagged a minor stress/intonation
        variance for this word (e.g. a flat or missing boundary rise).  This
        is the only path to *yellow* — phoneme errors are always *red*.
    """
    confs = [1.0] * len(pairs) if confidences is None else list(confidences)
    if len(confs) != len(pairs):
        raise ValueError("confidences must be parallel to pairs")

    feedback = tuple(
        PhonemeFeedback.from_pair(pair, conf)
        for pair, conf in zip(pairs, confs, strict=True)
    )
    matches = sum(1 for p in pairs if p.kind == MATCH)
    accuracy = matches / len(pairs) if pairs else 1.0
    return WordScore(
        word=word,
        accuracy=accuracy,
        color=_color(accuracy, feedback, prosody_issue),
        feedback=feedback,
    )


def _color(
    accuracy: float,
    feedback: tuple[PhonemeFeedback, ...],
    prosody_issue: bool = False,
) -> str:
    if not feedback:
        return GREEN  # nothing to score (e.g. OOV) — treated as neutral
    if any(f.flagged for f in feedback) or accuracy < RED_THRESHOLD:
        return RED
    if prosody_issue:
        return YELLOW  # phonemes fine, minor intonation/stress variance
    if accuracy >= GREEN_THRESHOLD:
        return GREEN
    return YELLOW
