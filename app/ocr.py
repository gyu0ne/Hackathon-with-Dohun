from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytesseract
from numpy.typing import NDArray
from pytesseract import Output
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from app.config import Settings


class OcrUnavailableError(RuntimeError):
    """Raised when the OCR runtime is unavailable."""


@dataclass(frozen=True)
class OcrLineData:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrCandidate:
    strategy: str
    lines: list[OcrLineData]
    score: float


@dataclass(frozen=True)
class OcrSelection:
    strategy: str
    lines: list[OcrLineData]
    score: float
    candidate_count: int


def run_ocr(
    image: NDArray[np.uint8],
    settings: Settings,
    *,
    psm: int | None = None,
    lang: str | None = None,
) -> list[OcrLineData]:
    try:
        data: dict[str, list[Any]] = pytesseract.image_to_data(
            image,
            lang=lang or settings.tesseract_lang,
            config=f"--oem 1 --psm {psm if psm is not None else settings.tesseract_psm}",
            output_type=Output.DICT,
        )
    except (TesseractError, TesseractNotFoundError) as exc:
        raise OcrUnavailableError("OCR runtime is unavailable") from exc

    grouped: dict[tuple[int, int, int, int], list[tuple[str, float, int, int, int, int]]] = (
        defaultdict(list)
    )
    for index, raw_text in enumerate(data["text"]):
        word = unicodedata.normalize("NFC", str(raw_text).strip())
        confidence = float(data["conf"][index])
        if not word or confidence < 0:
            continue
        key = (
            int(data["page_num"][index]),
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped[key].append(
            (
                word,
                confidence,
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
            )
        )

    lines: list[OcrLineData] = []
    for words in grouped.values():
        left = min(item[2] for item in words)
        top = min(item[3] for item in words)
        right = max(item[2] + item[4] for item in words)
        bottom = max(item[3] + item[5] for item in words)
        character_weights = [max(1, len(item[0].replace(" ", ""))) for item in words]
        weighted_confidence = sum(
            item[1] * weight for item, weight in zip(words, character_weights, strict=True)
        ) / sum(character_weights)
        lines.append(
            OcrLineData(
                text=" ".join(item[0] for item in words),
                confidence=round(weighted_confidence, 2),
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
        )
    return lines


def weighted_confidence(lines: list[OcrLineData]) -> float:
    weights = [max(1, len(line.text.replace(" ", ""))) for line in lines]
    if not weights:
        return 0.0
    return round(
        sum(line.confidence * weight for line, weight in zip(lines, weights, strict=True))
        / sum(weights),
        2,
    )


def _valid_character_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    valid = sum(
        character.isalnum()
        or "가" <= character <= "힣"
        or character in ".,·:;!?%+-/()[]{}<>₩원년월일시분초"
        for character in visible
    )
    return valid / len(visible)


def score_ocr(lines: list[OcrLineData]) -> float:
    """Rank candidates without pretending that Tesseract confidence is accuracy."""
    text = "\n".join(line.text for line in lines)
    characters = len("".join(text.split()))
    if characters == 0:
        return 0.0
    confidence = weighted_confidence(lines)
    valid_ratio = _valid_character_ratio(text)
    coverage = min(1.0, math.log1p(characters) / math.log(501))
    return round(0.80 * confidence + 15.0 * valid_ratio + 5.0 * coverage, 3)


def _candidate(
    strategy: str,
    image: NDArray[np.uint8],
    settings: Settings,
    *,
    psm: int,
) -> OcrCandidate:
    lines = run_ocr(image, settings, psm=psm)
    return OcrCandidate(strategy=strategy, lines=lines, score=score_ocr(lines))


def choose_candidate(
    candidates: list[OcrCandidate], settings: Settings, *, primary_strategy: str
) -> OcrCandidate:
    if not candidates:
        raise ValueError("At least one OCR candidate is required")

    def adjusted_score(candidate: OcrCandidate) -> float:
        bias = settings.ocr_primary_bias if candidate.strategy == primary_strategy else 0.0
        return candidate.score + bias

    return max(candidates, key=adjusted_score)


def run_adaptive_ocr(
    grayscale: NDArray[np.uint8],
    normalized: NDArray[np.uint8],
    binary: NDArray[np.uint8],
    settings: Settings,
    *,
    foreground_ratio: float,
    illumination_variation: float,
) -> OcrSelection:
    primary_strategy = "정확도 모델 · 회색조 · 자동 레이아웃(PSM 3)"
    primary = _candidate(primary_strategy, grayscale, settings, psm=3)
    candidates = [primary]

    # The primary path handles ordinary pages. Extra OCR runs are reserved for
    # hard, unevenly lit, or sparse pages so common uploads stay responsive.
    difficult = primary.score < settings.ocr_candidate_trigger
    if difficult or illumination_variation >= 0.055:
        candidates.append(
            _candidate(
                "정확도 모델 · 대비 보정 · 자동 레이아웃(PSM 3)",
                normalized,
                settings,
                psm=3,
            )
        )

    if difficult:
        if foreground_ratio <= settings.sparse_foreground_max:
            candidates.append(
                _candidate(
                    "정확도 모델 · 회색조 · 희소 텍스트(PSM 11)",
                    grayscale,
                    settings,
                    psm=11,
                )
            )
        else:
            candidates.append(
                _candidate("정확도 모델 · 회색조 · 단일 블록(PSM 6)", grayscale, settings, psm=6)
            )

    if difficult and illumination_variation >= 0.075:
        candidates.append(
            _candidate(
                "정확도 모델 · 적응형 이진화 · 자동 레이아웃(PSM 3)",
                binary,
                settings,
                psm=3,
            )
        )

    selected = choose_candidate(candidates, settings, primary_strategy=primary_strategy)
    return OcrSelection(
        strategy=selected.strategy,
        lines=selected.lines,
        score=selected.score,
        candidate_count=len(candidates),
    )
