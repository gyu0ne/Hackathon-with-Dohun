from __future__ import annotations

from typing import Any

import numpy as np

from app.config import Settings
from app.paddle_ocr import _parse_result, run_paddle_ocr


class _FakeResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.json = {"res": payload}


def test_parse_paddle_result_keeps_text_confidence_and_box() -> None:
    result = _FakeResult(
        {
            "rec_texts": ["  시행일 2026. 8. 12.  ", ""],
            "rec_scores": np.array([0.9234, 0.4]),
            "rec_boxes": np.array([[10, 20, 210, 52], [0, 0, 1, 1]]),
        }
    )

    lines = _parse_result(result)

    assert len(lines) == 1
    assert lines[0].text == "시행일 2026. 8. 12."
    assert lines[0].confidence == 92.34
    assert (lines[0].x, lines[0].y, lines[0].width, lines[0].height) == (
        10,
        20,
        200,
        32,
    )


def test_parse_paddle_result_supports_polygon_only_output() -> None:
    result = _FakeResult(
        {
            "rec_texts": ["접수 번호"],
            "rec_scores": [0.8],
            "rec_polys": np.array([[[4, 7], [84, 6], [85, 25], [3, 26]]]),
        }
    )

    line = _parse_result(result)[0]

    assert (line.x, line.y, line.width, line.height) == (3, 6, 82, 20)


def test_run_paddle_ocr_passes_custom_recognition_model(
    monkeypatch: object,
) -> None:
    captured: list[object] = []

    class FakeEngine:
        def predict(self, image: np.ndarray) -> list[_FakeResult]:
            return [
                _FakeResult(
                    {
                        "rec_texts": ["공문서"],
                        "rec_scores": [0.9],
                        "rec_boxes": [[0, 0, 20, 10]],
                    }
                )
            ]

    def fake_engine(*args: object) -> FakeEngine:
        captured.extend(args)
        return FakeEngine()

    monkeypatch.setattr("app.paddle_ocr._engine", fake_engine)  # type: ignore[attr-defined]
    settings = Settings(paddle_recognition_model_dir="/models/public-doc")

    lines = run_paddle_ocr(np.zeros((20, 40, 3), dtype=np.uint8), settings)

    assert lines[0].text == "공문서"
    assert captured[3] == "/models/public-doc"
    assert captured[4:6] == [None, None]
