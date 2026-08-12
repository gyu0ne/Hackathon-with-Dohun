from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings, get_settings
from app.feedback import save_feedback
from app.ocr import OcrUnavailableError
from app.pipeline import InvalidImageError
from app.schemas import (
    AnalyzeResponse,
    CompareResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.service import analyze_document, compare_document

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(title="Public Document Assistant MVP", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def user_interface() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/test", include_in_schema=False)
def test_interface() -> FileResponse:
    return FileResponse(STATIC_DIR / "test.html")


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "ai_configured": bool(settings.gemini_api_key),
        "ai_model": settings.gemini_model,
    }


async def _read_image(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")
    payload = await file.read(settings.max_upload_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file is too large")
    return payload


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(file: Annotated[UploadFile, File()]) -> AnalyzeResponse:
    settings = get_settings()
    payload = await _read_image(file, settings)
    try:
        return analyze_document(payload, settings)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OcrUnavailableError as exc:
        raise HTTPException(status_code=503, detail="OCR is temporarily unavailable") from exc


@app.post("/api/compare", response_model=CompareResponse)
async def compare(file: Annotated[UploadFile, File()]) -> CompareResponse:
    settings = get_settings()
    payload = await _read_image(file, settings)
    try:
        return compare_document(payload, settings)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OcrUnavailableError as exc:
        raise HTTPException(status_code=503, detail="OCR is temporarily unavailable") from exc


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    if not request.accepted and not request.corrected_text:
        raise HTTPException(status_code=422, detail="Corrected text is required when rejected")
    save_feedback(request, get_settings().feedback_db_path)
    return FeedbackResponse(saved=True)

