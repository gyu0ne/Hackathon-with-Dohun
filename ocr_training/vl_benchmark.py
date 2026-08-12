from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.ocr_metrics import (
    character_error_rate,
    numeric_value_scores,
    order_independent_word_scores,
    word_error_rate,
)


@dataclass(frozen=True)
class VlScores:
    cer: float
    wer: float
    word_f1: float
    numeric_f1: float
    elapsed_ms: float
    characters: int


def markdown_to_text(markdown: str) -> str:
    text = unicodedata.normalize("NFC", markdown)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-+>]\s+", "", text)
    text = re.sub(r"[`*_~|]+", " ", text)
    return " ".join(text.split())


def result_markdown(result: Any) -> str:
    markdown = result.markdown
    if isinstance(markdown, dict):
        value = markdown.get("markdown_texts", "")
        if isinstance(value, list | tuple):
            return "\n".join(str(item) for item in value)
        return str(value)
    return str(markdown)


def score(reference: str, hypothesis: str, elapsed_ms: float) -> VlScores:
    word_scores = order_independent_word_scores(reference, hypothesis)
    numeric_scores = numeric_value_scores(reference, hypothesis)
    return VlScores(
        cer=round(character_error_rate(reference, hypothesis), 4),
        wer=round(word_error_rate(reference, hypothesis), 4),
        word_f1=round(word_scores.f1, 4),
        numeric_f1=round(numeric_scores.f1, 4),
        elapsed_ms=round(elapsed_ms, 2),
        characters=len("".join(hypothesis.split())),
    )


def create_pipeline(version: str, device: str) -> Any:
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(
        pipeline_version=version,
        vl_rec_backend="native",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
    )


def predict_text(
    pipeline: Any,
    image: NDArray[np.uint8] | Path,
    *,
    max_pixels: int | None = None,
) -> tuple[str, float]:
    started = perf_counter()
    options = {"max_pixels": max_pixels} if max_pixels is not None else {}
    results: Iterable[Any] = pipeline.predict(image, **options)
    markdown = "\n".join(result_markdown(result) for result in results)
    elapsed_ms = (perf_counter() - started) * 1000
    return markdown_to_text(markdown), elapsed_ms
