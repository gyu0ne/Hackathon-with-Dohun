from __future__ import annotations

from app.config import get_settings


def test_quoted_api_key_is_normalized(monkeypatch: object) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", '"example-key"')  # type: ignore[attr-defined]

    settings = get_settings()

    assert settings.gemini_api_key == "example-key"


def test_custom_paddle_model_directory_is_optional(monkeypatch: object) -> None:
    monkeypatch.setenv("PADDLE_RECOGNITION_MODEL_DIR", "  /models/public-doc  ")  # type: ignore[attr-defined]

    settings = get_settings()

    assert settings.paddle_recognition_model_dir == "/models/public-doc"
