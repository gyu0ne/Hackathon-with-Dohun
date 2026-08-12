from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.ocr import (
    OcrLineData,
    OcrSelection,
    OcrUnavailableError,
    run_adaptive_ocr,
    run_ocr,
    score_ocr,
)
from app.paddle_ocr import run_paddle_ocr
from app.quality import QualityAssessment, assess_quality


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a supported image."""


@dataclass(frozen=True)
class PipelineResult:
    width: int
    height: int
    quality: QualityAssessment
    lines: list[OcrLineData] | None
    ocr_strategy: str | None
    ocr_candidate_count: int
    quality_ms: float
    ocr_ms: float


@dataclass(frozen=True)
class BaselineResult:
    width: int
    height: int
    lines: list[OcrLineData]
    ocr_strategy: str
    ocr_ms: float


@dataclass(frozen=True)
class PhotoFilterResult:
    width: int
    height: int
    lines: list[OcrLineData]
    ocr_strategy: str
    ocr_ms: float


def decode_image(payload: bytes, settings: Settings) -> NDArray[np.uint8]:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise InvalidImageError("The uploaded file is not a valid image")
    height, width = image.shape[:2]
    if min(height, width) < 32:
        raise InvalidImageError("The image is too small")
    if height * width > settings.max_image_pixels:
        raise InvalidImageError("The image dimensions are too large")
    return cast(NDArray[np.uint8], image)


def _run_document_ocr(
    image: NDArray[np.uint8],
    quality: QualityAssessment,
    settings: Settings,
) -> OcrSelection:
    if settings.ocr_engine == "paddle":
        try:
            lines = run_paddle_ocr(image, settings)
            return OcrSelection(
                strategy="한국어 문서 모델 · PaddleOCR PP-OCRv5",
                lines=lines,
                score=score_ocr(lines),
                candidate_count=1,
            )
        except OcrUnavailableError:
            pass

    fallback = run_adaptive_ocr(
        quality.grayscale,
        quality.normalized,
        quality.binary,
        settings,
        foreground_ratio=quality.foreground_ratio,
        illumination_variation=quality.illumination_variation,
    )
    if settings.ocr_engine == "paddle":
        return OcrSelection(
            strategy=f"PaddleOCR 장애 폴백 · {fallback.strategy}",
            lines=fallback.lines,
            score=fallback.score,
            candidate_count=fallback.candidate_count,
        )
    return fallback


def analyze_image(payload: bytes, settings: Settings) -> PipelineResult:
    image = decode_image(payload, settings)
    height, width = image.shape[:2]
    quality_started = perf_counter()
    quality = assess_quality(image, settings)
    quality_ms = (perf_counter() - quality_started) * 1000
    ocr_started = perf_counter()
    selection = None
    if quality.status != "retake":
        selection = _run_document_ocr(image, quality, settings)
    ocr_ms = 0.0 if selection is None else (perf_counter() - ocr_started) * 1000
    return PipelineResult(
        width=width,
        height=height,
        quality=quality,
        lines=None if selection is None else selection.lines,
        ocr_strategy=None if selection is None else selection.strategy,
        ocr_candidate_count=0 if selection is None else selection.candidate_count,
        quality_ms=round(quality_ms, 2),
        ocr_ms=round(ocr_ms, 2),
    )


def analyze_baseline(payload: bytes, settings: Settings) -> BaselineResult:
    image = decode_image(payload, settings)
    height, width = image.shape[:2]
    grayscale = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    started = perf_counter()
    lines = run_ocr(
        grayscale,
        settings,
        psm=3,
        lang=settings.baseline_tesseract_lang,
    )
    return BaselineResult(
        width=width,
        height=height,
        lines=lines,
        ocr_strategy="기본 모델 · 회색조 · 자동 레이아웃(PSM 3)",
        ocr_ms=round((perf_counter() - started) * 1000, 2),
    )


def analyze_photo_filter(payload: bytes, settings: Settings) -> PhotoFilterResult:
    image = decode_image(payload, settings)
    height, width = image.shape[:2]
    started = perf_counter()
    lines = run_paddle_ocr(image, settings, use_doc_unwarping=True)
    return PhotoFilterResult(
        width=width,
        height=height,
        lines=lines,
        ocr_strategy="실험 필터 · Paddle UVDoc 문서 펴기 · PP-OCRv5",
        ocr_ms=round((perf_counter() - started) * 1000, 2),
    )
