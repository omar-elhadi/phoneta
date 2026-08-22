"""Tests for IPA-token sequence alignment (pure, deterministic)."""

from __future__ import annotations

import pytest

from phoneta.core.alignment.sequence import (
    DELETION,
    INSERTION,
    MATCH,
    SUBSTITUTION,
    AlignedPair,
    align,
    levenshtein,
)


class TestLevenshtein:
    @pytest.mark.parametrize(
        "ref,user,distance",
        [
            ([], [], 0),
            (["t"], [], 1),
            ([], ["t"], 1),
            (["t", "e", "s", "t"], ["t", "e", "s", "t"], 0),
            (["t", "e", "s", "t"], ["t", "e", "z", "t"], 1),  # substitution
            (["t", "e", "s", "t"], ["t", "e", "s"], 1),  # deletion
            (["t", "e", "s"], ["t", "e", "s", "t"], 1),  # insertion
            (["k", "æ", "t"], ["k", "ʌ", "p"], 2),  # 2 substitutions
        ],
    )
    def test_distance(self, ref, user, distance) -> None:
        assert levenshtein(ref, user) == distance


class TestAlign:
    def test_identical(self) -> None:
        result = align(["t", "ə", "m", "eɪ"], ["t", "ə", "m", "eɪ"])
        assert [p.kind for p in result] == [MATCH] * 4

    def test_empty_ref(self) -> None:
        result = align([], ["t", "ə"])
        assert [p.kind for p in result] == [INSERTION, INSERTION]

    def test_empty_user(self) -> None:
        result = align(["t", "ə"], [])
        assert [p.kind for p in result] == [DELETION, DELETION]

    def test_substitution(self) -> None:
        # user says "kæt" instead of "kʌt"
        result = align(["k", "ʌ", "t"], ["k", "æ", "t"])
        kinds = [p.kind for p in result]
        assert kinds == [MATCH, SUBSTITUTION, MATCH]
        assert result[1].ref == "ʌ"
        assert result[1].user == "æ"

    def test_deletion(self) -> None:
        # user drops the second phoneme
        result = align(["t", "ə"], ["t"])
        kinds = [p.kind for p in result]
        assert kinds == [MATCH, DELETION]

    def test_insertion(self) -> None:
        # user inserts an extra phoneme
        result = align(["t", "ə"], ["t", "s", "ə"])
        kinds = [p.kind for p in result]
        assert kinds == [MATCH, INSERTION, MATCH]
        assert result[1].user == "s"
        assert result[1].ref is None

    def test_mixed_french_nasal(self) -> None:
        # French "un bon vin" with nasal vowels
        ref = ["œ̃", "b", "ɔ̃", "v", "ɛ̃"]
        user = ["œ̃", "b", "ɔ̃", "v", "æ̃"]  # wrong nasal
        result = align(ref, user)
        assert result[-1].kind == SUBSTITUTION

    def test_alignedpair_is_error(self) -> None:
        assert AlignedPair(ref="t", user="t", kind=MATCH).is_error is False
        assert AlignedPair(ref="t", user="z", kind=SUBSTITUTION).is_error is True
        assert AlignedPair(ref=None, user="t", kind=INSERTION).is_error is True
        assert AlignedPair(ref="t", user=None, kind=DELETION).is_error is True

    def test_ipa_multichar_tokens(self) -> None:
        # IPA tokens are multi-character strings — alignment must not split them
        ref = ["tʃ", "əʊ", "p"]
        user = ["tʃ", "aʊ", "p"]
        result = align(ref, user)
        assert result[1].ref == "əʊ"
        assert result[1].user == "aʊ"
        assert result[1].kind == SUBSTITUTION