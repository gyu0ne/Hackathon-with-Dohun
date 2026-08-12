from __future__ import annotations

from time import perf_counter
from typing import Literal
from uuid import uuid4

from app.config import Settings
from app.llm import (
    AiServiceError,
    extractive_fallback,
    generate_cited_summary,
    generate_plain_summary,
)
from app.ocr import OcrLineData, OcrUnavailableError, weighted_confidence
from app.pipeline import analyze_baseline, analyze_image, analyze_photo_filter
from app.quality import QualityAssessment
from app.schemas import (
    AnalyzeResponse,
    BoundingBox,
    CompareResponse,
    ComparisonMetrics,
    ComparisonVariant,
    ImageInfo,
    OcrLine,
    OcrResult,
    QualityMetrics,
    QualityResult,
    SummaryResult,
    TimingMetrics,
)


def _ocr_result(
    lines: list[OcrLineData], strategy: str, candidate_count: int = 1
) -> OcrResult:
    result_lines = [
        OcrLine(
            id=index,
            text=line.text,
            confidence=line.confidence,
            bbox=BoundingBox(
                x=line.x,
                y=line.y,
                width=line.width,
                height=line.height,
            ),
        )
        for index, line in enumerate(lines, start=1)
    ]
    return OcrResult(
        text="\n".join(line.text for line in result_lines),
        lines=result_lines,
        strategy=strategy,
        candidate_count=candidate_count,
    )


def _quality_result(quality: QualityAssessment) -> QualityResult:
    return QualityResult(
        status=quality.status,
        reasons=quality.reasons,
        metrics=QualityMetrics(
            focus_score=quality.focus_score,
            text_contrast=quality.text_contrast,
            foreground_ratio=quality.foreground_ratio,
        ),
    )


def _summarize(lines: list[OcrLineData], settings: Settings) -> tuple[SummaryResult, float]:
    started = perf_counter()
    try:
        summary = generate_cited_summary(lines, settings)
    except AiServiceError:
        summary = extractive_fallback(lines, settings)
    return summary, round((perf_counter() - started) * 1000, 2)


def analyze_document(payload: bytes, settings: Settings) -> AnalyzeResponse:
    started = perf_counter()
    result = analyze_image(payload, settings)
    ocr = None
    summary = None
    ai_ms = 0.0
    if result.lines is not None:
        ocr = _ocr_result(
            result.lines,
            result.ocr_strategy or "OCR 전략 정보 없음",
            result.ocr_candidate_count,
        )
        summary, ai_ms = _summarize(result.lines, settings)
    return AnalyzeResponse(
        analysis_id=str(uuid4()),
        image=ImageInfo(width=result.width, height=result.height),
        quality=_quality_result(result.quality),
        ocr=ocr,
        summary=summary,
        timings=TimingMetrics(
            quality_ms=result.quality_ms,
            ocr_ms=result.ocr_ms,
            ai_ms=ai_ms,
            total_ms=round((perf_counter() - started) * 1000, 2),
        ),
    )


def _average_confidence(ocr: OcrResult | None) -> float:
    if ocr is None or not ocr.lines:
        return 0.0
    lines = [
        OcrLineData(
            text=line.text,
            confidence=line.confidence,
            x=line.bbox.x,
            y=line.bbox.y,
            width=line.bbox.width,
            height=line.bbox.height,
        )
        for line in ocr.lines
    ]
    return weighted_confidence(lines)


def _citation_coverage(summary: SummaryResult | None) -> float | None:
    if summary is None:
        return None
    facts = [summary.overview_source_line_ids]
    facts.extend(fact.source_line_ids for fact in summary.key_points)
    facts.extend(fact.source_line_ids for fact in summary.actions)
    return round(100 * sum(bool(ids) for ids in facts) / len(facts), 1) if facts else 0.0


def compare_document(payload: bytes, settings: Settings) -> CompareResponse:
    ours_started = perf_counter()
    ours_result = analyze_image(payload, settings)
    ours_ocr = (
        _ocr_result(
            ours_result.lines,
            ours_result.ocr_strategy or "OCR 전략 정보 없음",
            ours_result.ocr_candidate_count,
        )
        if ours_result.lines is not None
        else None
    )
    ours_summary = None
    ours_ai_ms = 0.0
    if ours_result.lines is not None:
        ours_summary, ours_ai_ms = _summarize(ours_result.lines, settings)
    ours_total_ms = round((perf_counter() - ours_started) * 1000, 2)

    photo_filter_status: Literal[
        "completed", "skipped_quality", "not_available", "failed"
    ] = "not_available"
    photo_filter_ocr = None
    photo_filter_ocr_ms = 0.0
    photo_filter_total_ms = 0.0
    if settings.ocr_engine == "paddle":
        photo_filter_started = perf_counter()
        try:
            photo_filter_result = analyze_photo_filter(payload, settings)
            photo_filter_ocr = _ocr_result(
                photo_filter_result.lines,
                photo_filter_result.ocr_strategy,
            )
            photo_filter_ocr_ms = photo_filter_result.ocr_ms
            photo_filter_status = "completed"
        except OcrUnavailableError:
            photo_filter_status = "failed"
        photo_filter_total_ms = round(
            (perf_counter() - photo_filter_started) * 1000,
            2,
        )

    baseline_started = perf_counter()
    baseline_result = analyze_baseline(payload, settings)
    baseline_ocr = _ocr_result(baseline_result.lines, baseline_result.ocr_strategy)
    baseline_ai_started = perf_counter()
    try:
        baseline_summary = generate_plain_summary(baseline_ocr.text, settings)
    except AiServiceError:
        baseline_summary = " ".join(line.text for line in baseline_ocr.lines[:5])
    baseline_ai_ms = round((perf_counter() - baseline_ai_started) * 1000, 2)
    baseline_total_ms = round((perf_counter() - baseline_started) * 1000, 2)

    return CompareResponse(
        image=ImageInfo(width=ours_result.width, height=ours_result.height),
        quality=_quality_result(ours_result.quality),
        ours=ComparisonVariant(
            status="completed" if ours_ocr is not None else "skipped_quality",
            ocr=ours_ocr,
            cited_summary=ours_summary,
            plain_summary=None,
            metrics=ComparisonMetrics(
                character_count=len(ours_ocr.text) if ours_ocr else 0,
                average_confidence=_average_confidence(ours_ocr),
                citation_coverage=_citation_coverage(ours_summary),
                ocr_ms=ours_result.ocr_ms,
                ai_ms=ours_ai_ms,
                total_ms=ours_total_ms,
            ),
        ),
        photo_filter=ComparisonVariant(
            status=photo_filter_status,
            ocr=photo_filter_ocr,
            cited_summary=None,
            plain_summary=None,
            metrics=ComparisonMetrics(
                character_count=len(photo_filter_ocr.text) if photo_filter_ocr else 0,
                average_confidence=_average_confidence(photo_filter_ocr),
                citation_coverage=None,
                ocr_ms=photo_filter_ocr_ms,
                ai_ms=0.0,
                total_ms=photo_filter_total_ms,
            ),
        ),
        baseline=ComparisonVariant(
            status="completed",
            ocr=baseline_ocr,
            cited_summary=None,
            plain_summary=baseline_summary,
            metrics=ComparisonMetrics(
                character_count=len(baseline_ocr.text),
                average_confidence=_average_confidence(baseline_ocr),
                citation_coverage=None,
                ocr_ms=baseline_result.ocr_ms,
                ai_ms=baseline_ai_ms,
                total_ms=baseline_total_ms,
            ),
        ),
    )
