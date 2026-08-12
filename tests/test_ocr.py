from __future__ import annotations

from app.config import Settings
from app.ocr import OcrCandidate, choose_candidate


def test_primary_candidate_wins_small_internal_score_difference() -> None:
    settings = Settings(ocr_primary_bias=3.0)
    primary = OcrCandidate(strategy="primary", lines=[], score=70.0)
    alternative = OcrCandidate(strategy="alternative", lines=[], score=72.0)

    selected = choose_candidate(
        [primary, alternative], settings, primary_strategy="primary"
    )

    assert selected.strategy == "primary"


def test_alternative_candidate_wins_material_score_difference() -> None:
    settings = Settings(ocr_primary_bias=3.0)
    primary = OcrCandidate(strategy="primary", lines=[], score=70.0)
    alternative = OcrCandidate(strategy="alternative", lines=[], score=74.0)

    selected = choose_candidate(
        [primary, alternative], settings, primary_strategy="primary"
    )

    assert selected.strategy == "alternative"
