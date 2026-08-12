from __future__ import annotations

import pytest

from research.ocr_metrics import (
    character_error_rate,
    numeric_value_scores,
    order_independent_word_scores,
    word_error_rate,
)


def test_character_error_rate_normalizes_whitespace() -> None:
    assert character_error_rate("가 나 다", "가나다") == 0.0
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)


def test_word_error_rate_counts_token_substitution() -> None:
    assert word_error_rate("신청 기간 확인", "신청 일자 확인") == pytest.approx(1 / 3)


def test_error_rates_handle_empty_reference() -> None:
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "문자") == 1.0


def test_order_independent_scores_ignore_layout_reading_order() -> None:
    scores = order_independent_word_scores(
        "이름 홍길동 날짜 2026-08-12",
        "날짜 2026-08-12 이름 홍길동",
    )

    assert scores.f1 == 1.0
    assert scores.matched == scores.reference_count == scores.hypothesis_count


def test_order_independent_scores_count_duplicate_words() -> None:
    scores = order_independent_word_scores("세금 세금 합계", "세금 합계")

    assert scores.precision == 1.0
    assert scores.recall == 2 / 3


def test_numeric_scores_preserve_document_values() -> None:
    scores = numeric_value_scores(
        "발행일 2026-08-12 합계 12,500원 세율 10%",
        "합계 12,500 발행일 2026-08-12 세율 10%",
    )

    assert scores.f1 == 1.0
