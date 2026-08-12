from __future__ import annotations

import unicodedata
from functools import lru_cache
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray
from paddleocr import PaddleOCR

from app.config import Settings
from app.ocr import OcrLineData, OcrUnavailableError
from app.reading_order import sort_reading_order

_INFERENCE_LOCK = Lock()


@lru_cache(maxsize=4)
def _engine(
    cpu_threads: int,
    detection_limit: int,
    recognition_batch_size: int,
    recognition_model_dir: str | None,
    text_det_thresh: float | None,
    text_det_box_thresh: float | None,
    detection_model_name: str,
    use_doc_orientation_classify: bool,
    use_doc_unwarping: bool,
    use_textline_orientation: bool,
) -> PaddleOCR:
    model_options: dict[str, Any] = {}
    if recognition_model_dir:
        model_options["text_recognition_model_dir"] = recognition_model_dir
    if text_det_thresh is not None:
        model_options["text_det_thresh"] = text_det_thresh
    if text_det_box_thresh is not None:
        model_options["text_det_box_thresh"] = text_det_box_thresh
    return PaddleOCR(
        text_detection_model_name=detection_model_name,
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        text_recognition_batch_size=recognition_batch_size,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=cpu_threads,
        text_det_limit_side_len=detection_limit,
        text_det_limit_type="max",
        **model_options,
    )


def _box_at(result: dict[str, Any], index: int) -> tuple[int, int, int, int]:
    boxes = result.get("rec_boxes")
    if boxes is not None and index < len(boxes):
        values = np.asarray(boxes[index]).reshape(-1)
        if len(values) >= 4:
            left, top, right, bottom = (int(round(float(value))) for value in values[:4])
            return left, top, max(1, right - left), max(1, bottom - top)

    polygons = result.get("rec_polys")
    if polygons is not None and index < len(polygons):
        polygon = np.asarray(polygons[index], dtype=np.float64).reshape(-1, 2)
        if polygon.size:
            left, top = np.floor(polygon.min(axis=0)).astype(int)
            right, bottom = np.ceil(polygon.max(axis=0)).astype(int)
            return int(left), int(top), max(1, int(right - left)), max(1, int(bottom - top))
    return 0, 0, 1, 1


def _parse_result(raw_result: Any) -> list[OcrLineData]:
    payload = raw_result.json
    result = payload.get("res", payload)
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    lines: list[OcrLineData] = []
    for index, raw_text in enumerate(texts):
        text = unicodedata.normalize("NFC", str(raw_text).strip())
        if not text:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        x, y, width, height = _box_at(result, index)
        lines.append(
            OcrLineData(
                text=text,
                confidence=round(max(0.0, min(100.0, score * 100)), 2),
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )
    return lines


def run_paddle_ocr(
    image: NDArray[np.uint8],
    settings: Settings,
    *,
    detection_model_name: str = "PP-OCRv5_mobile_det",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    sort_output: bool = True,
) -> list[OcrLineData]:
    try:
        with _INFERENCE_LOCK:
            engine = _engine(
                settings.paddle_cpu_threads,
                settings.paddle_det_limit_side_len,
                settings.paddle_recognition_batch_size,
                settings.paddle_recognition_model_dir,
                settings.paddle_text_det_thresh,
                settings.paddle_text_det_box_thresh,
                detection_model_name,
                use_doc_orientation_classify,
                use_doc_unwarping,
                use_textline_orientation,
            )
            results = list(engine.predict(image))
    except Exception as exc:
        raise OcrUnavailableError("PaddleOCR runtime is unavailable") from exc

    if not results:
        raise OcrUnavailableError("PaddleOCR returned no result")
    lines = _parse_result(results[0])
    if not lines:
        raise OcrUnavailableError("PaddleOCR detected no readable text")
    return sort_reading_order(lines) if sort_output else lines
