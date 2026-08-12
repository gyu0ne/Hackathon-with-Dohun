from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

Item = TypeVar("Item")


@dataclass(frozen=True)
class MatchScores:
    precision: float
    recall: float
    f1: float
    matched: int
    reference_count: int
    hypothesis_count: int


def _edit_distance(reference: Sequence[Item], hypothesis: Sequence[Item]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def normalize_characters(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", "", normalized)


def normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text)
    return normalized.split()


def normalize_alphanumeric_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def normalize_numeric_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    return re.findall(r"\d+(?:[.,:/-]\d+)*%?", normalized)


def _counter_scores(reference: Sequence[str], hypothesis: Sequence[str]) -> MatchScores:
    expected = Counter(reference)
    actual = Counter(hypothesis)
    matched = sum((expected & actual).values())
    reference_count = sum(expected.values())
    hypothesis_count = sum(actual.values())
    precision = matched / hypothesis_count if hypothesis_count else float(not reference_count)
    recall = matched / reference_count if reference_count else float(not hypothesis_count)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return MatchScores(
        precision=precision,
        recall=recall,
        f1=f1,
        matched=matched,
        reference_count=reference_count,
        hypothesis_count=hypothesis_count,
    )


def order_independent_word_scores(reference: str, hypothesis: str) -> MatchScores:
    return _counter_scores(
        normalize_alphanumeric_tokens(reference),
        normalize_alphanumeric_tokens(hypothesis),
    )


def numeric_value_scores(reference: str, hypothesis: str) -> MatchScores:
    return _counter_scores(
        normalize_numeric_tokens(reference),
        normalize_numeric_tokens(hypothesis),
    )


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_characters(reference)
    actual = normalize_characters(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_words(reference)
    actual = normalize_words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)
