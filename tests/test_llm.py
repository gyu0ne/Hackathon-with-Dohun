from __future__ import annotations

from app.config import Settings
from app.llm import CitedFactDraft, _clean_facts, extractive_fallback
from app.ocr import OcrLineData


def _line(text: str) -> OcrLineData:
    return OcrLineData(text=text, confidence=95.0, x=0, y=0, width=100, height=20)


def test_invalid_citation_ids_are_removed() -> None:
    facts = [
        CitedFactDraft(text="유효한 사실", source_line_ids=[1, 1, 9]),
        CitedFactDraft(text="근거 없는 사실", source_line_ids=[0, 9]),
    ]

    cleaned = _clean_facts(facts, line_count=2)

    assert len(cleaned) == 1
    assert cleaned[0].source_line_ids == [1]


def test_fallback_summary_keeps_source_line_ids() -> None:
    summary = extractive_fallback(
        [_line("지원 사업 안내"), _line("신청 기간은 8월 20일까지입니다")],
        Settings(),
    )

    assert summary.status == "fallback"
    assert summary.overview_source_line_ids == [1, 2]
    assert summary.actions[0].source_line_ids == [2]
