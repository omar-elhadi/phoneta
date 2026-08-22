"""Sequence alignment over IPA phoneme token lists.

IPA phonemes are multi-character strings (``tʃ``, ``əʊ``, ``ɛ̃``), so the
alignment works on **lists of tokens** rather than raw characters.  The module
is pure Python + stdlib and has no runtime dependencies, which keeps it fully
deterministic and unit-testable without any ML stack.

Two entry points:

* :func:`levenshtein` — classic edit distance over tokens.
* :func:`align` — Needleman-Wunsch global alignment that additionally labels
  every position as ``match`` / ``substitution`` / ``insertion`` /
  ``deletion`` so the scorer can flag mispronunciations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

MATCH = "match"
SUBSTITUTION = "substitution"
INSERTION = "insertion"  # user spoke a phoneme not in the target
DELETION = "deletion"  # target phoneme the user skipped

DEFAULT_MATCH_SCORE = 2
DEFAULT_MISMATCH_SCORE = -1
DEFAULT_GAP_SCORE = -1

# Kinds that count as a mispronunciation for scoring purposes.
ERROR_KINDS = frozenset({SUBSTITUTION, INSERTION, DELETION})


@dataclass(frozen=True)
class AlignedPair:
    """One column of a global alignment between reference and user phonemes.

    Exactly one of ``ref`` / ``user`` is ``None`` for insertions (user-only)
    and deletions (ref-only); both are set for matches/substitutions.
    """

    ref: Optional[str]
    user: Optional[str]
    kind: str

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS


def levenshtein(ref: Sequence[str], user: Sequence[str]) -> int:
    """Return the Levenshtein (edit) distance between two token sequences.

    Parameters
    ----------
    ref:
        Reference phoneme tokens (e.g. ``["t", "ə", "m", "eɪ", "t", "ə"]``).
    user:
        User-spoken phoneme tokens.

    Returns
    -------
    int
        Minimum number of single-token insertions, deletions or substitutions
        needed to turn ``user`` into ``ref``.
    """
    n, m = len(ref), len(user)
    # Single row is enough for the distance.
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == user[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def align(
    ref: Sequence[str],
    user: Sequence[str],
    match_score: int = DEFAULT_MATCH_SCORE,
    mismatch_score: int = DEFAULT_MISMATCH_SCORE,
    gap_score: int = DEFAULT_GAP_SCORE,
) -> list[AlignedPair]:
    """Globally align ``ref`` to ``user`` with Needleman-Wunsch.

    Returns one :class:`AlignedPair` per alignment column, in reference order.
    Ties are broken deterministically (diagonal first, then gap-in-user, then
    gap-in-ref) so results are stable across runs.

    Parameters
    ----------
    ref:
        Reference phoneme tokens.
    user:
        User-spoken phoneme tokens.
    match_score / mismatch_score / gap_score:
        Scoring parameters.  The defaults reward exact matches and penalise
        mismatches and gaps equally, which suits short pronunciation targets.
    """
    n, m = len(ref), len(user)

    # DP table: score[i][j] = best score aligning ref[:i] with user[:j].
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_score
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_score

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (
                match_score if ref[i - 1] == user[j - 1] else mismatch_score
            )
            up = score[i - 1][j] + gap_score  # consume ref, gap in user -> deletion
            left = score[i][j - 1] + gap_score  # gap in ref, consume user -> insertion
            score[i][j] = max(diag, up, left)

    # Traceback — deterministic tie-break: diag > up > left.
    pairs: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = score[i - 1][j - 1] + (
                match_score if ref[i - 1] == user[j - 1] else mismatch_score
            )
            if score[i][j] == diag:
                r, u = ref[i - 1], user[j - 1]
                kind = MATCH if r == u else SUBSTITUTION
                pairs.append(AlignedPair(ref=r, user=u, kind=kind))
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + gap_score:
            pairs.append(AlignedPair(ref=ref[i - 1], user=None, kind=DELETION))
            i -= 1
            continue
        # Must be a gap in ref (insertion).
        pairs.append(AlignedPair(ref=None, user=user[j - 1], kind=INSERTION))
        j -= 1

    pairs.reverse()
    return pairs
