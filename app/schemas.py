from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class QualityMetrics(BaseModel):
    focus_score: float
    text_contrast: float
    foreground_ratio: float


class QualityResult(BaseModel):
    status: Literal["pass", "retake"]
    reasons: list[str]
    metrics: QualityMetrics


class OcrLine(BaseModel):
    id: int
    text: str
    confidence: float
    bbox: BoundingBox


class OcrResult(BaseModel):
    text: str
    lines: list[OcrLine]
    strategy: str
    candidate_count: int = 1


class ImageInfo(BaseModel):
    width: int
    height: int


class CitedFact(BaseModel):
    text: str
    source_line_ids: list[int]


class SummaryResult(BaseModel):
    title: str
    overview: str
    overview_source_line_ids: list[int]
    key_points: list[CitedFact]
    actions: list[CitedFact]
    provider: Literal["gemini", "local"]
    model: str
    status: Literal["generated", "fallback"]


class TimingMetrics(BaseModel):
    quality_ms: float = 0.0
    ocr_ms: float = 0.0
    ai_ms: float = 0.0
    total_ms: float = 0.0


class AnalyzeResponse(BaseModel):
    analysis_id: str
    image: ImageInfo
    quality: QualityResult
    ocr: OcrResult | None
    summary: SummaryResult | None
    timings: TimingMetrics


class ComparisonMetrics(BaseModel):
    character_count: int
    average_confidence: float
    citation_coverage: float | None
    ocr_ms: float
    ai_ms: float
    total_ms: float


class ComparisonVariant(BaseModel):
    status: Literal["completed", "skipped_quality", "not_available", "failed"]
    ocr: OcrResult | None
    cited_summary: SummaryResult | None
    plain_summary: str | None
    metrics: ComparisonMetrics


class CompareResponse(BaseModel):
    image: ImageInfo
    quality: QualityResult
    ours: ComparisonVariant
    photo_filter: ComparisonVariant
    baseline: ComparisonVariant


class FeedbackRequest(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=64)
    accepted: bool
    corrected_text: str | None = Field(default=None, max_length=100_000)


class FeedbackResponse(BaseModel):
    saved: bool
