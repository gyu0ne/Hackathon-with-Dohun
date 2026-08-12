from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _blank_png() -> bytes:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def _sharp_png() -> bytes:
    image = np.full((260, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "APPLICATION 2026-08-20",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.55,
        (0, 0, 0),
        4,
    )
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "ai_configured" in response.json()


def test_user_and_comparison_interfaces_are_served() -> None:
    user_page = client.get("/")
    test_page = client.get("/test")

    assert user_page.status_code == 200
    assert "문서 한눈에" in user_page.text
    assert test_page.status_code == 200
    assert "같은 문서, 다른 처리 방식" in test_page.text


def test_analyze_rejects_non_image() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_analyze_stops_before_ocr_when_retake_is_needed() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("blank.png", _blank_png(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality"]["status"] == "retake"
    assert body["ocr"] is None
    assert body["summary"] is None


def test_analyze_uses_cited_fallback_when_ai_key_is_missing(monkeypatch: object) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("GOOGLE_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("AI_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("OCR_ENGINE", "tesseract")  # type: ignore[attr-defined]

    response = client.post(
        "/api/analyze",
        files={"file": ("sharp.png", _sharp_png(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["quality"]["status"] == "pass"
    assert body["ocr"]["strategy"]
    assert body["ocr"]["candidate_count"] >= 1
    assert body["summary"]["status"] == "fallback"
    assert body["summary"]["overview_source_line_ids"]


def test_comparison_api_runs_both_paths_without_ai_key(monkeypatch: object) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("GOOGLE_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("AI_API_KEY", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("OCR_ENGINE", "tesseract")  # type: ignore[attr-defined]

    response = client.post(
        "/api/compare",
        files={"file": ("sharp.png", _sharp_png(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ours"]["status"] == "completed"
    assert "정확도 모델" in body["ours"]["ocr"]["strategy"]
    assert body["ours"]["metrics"]["citation_coverage"] == 100.0
    assert body["photo_filter"]["status"] == "not_available"
    assert body["photo_filter"]["ocr"] is None
    assert body["baseline"]["status"] == "completed"
    assert "기본 모델" in body["baseline"]["ocr"]["strategy"]
    assert body["baseline"]["metrics"]["citation_coverage"] is None


def test_feedback_is_saved(tmp_path: Path, monkeypatch: object) -> None:
    database = tmp_path / "feedback.db"
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(database))  # type: ignore[attr-defined]

    response = client.post(
        "/api/feedback",
        json={"analysis_id": "case-1", "accepted": True, "corrected_text": None},
    )

    assert response.status_code == 200
    assert response.json() == {"saved": True}
    assert database.exists()
