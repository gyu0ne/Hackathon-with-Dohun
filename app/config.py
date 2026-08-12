from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _secret_from_environment() -> str:
    value = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("AI_API_KEY")
        or ""
    ).strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    blur_min: float = 80.0
    binarization_ssim_min: float = 0.50
    sauvola_window: int = 25
    sauvola_k: float = 0.20
    tesseract_lang: str = "kor_best+eng_best"
    baseline_tesseract_lang: str = "kor+eng"
    tesseract_psm: int = 3
    ocr_candidate_trigger: float = 72.0
    ocr_primary_bias: float = 3.0
    sparse_foreground_max: float = 0.035
    ocr_engine: str = "paddle"
    paddle_det_limit_side_len: int = 1536
    paddle_cpu_threads: int = 4
    paddle_recognition_batch_size: int = 1
    paddle_recognition_model_dir: str | None = None
    paddle_text_det_thresh: float | None = None
    paddle_text_det_box_thresh: float | None = None
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_ms: int = 45_000
    feedback_db_path: str = "data/feedback.db"


def get_settings() -> Settings:
    return Settings(
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)),
        max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", 20_000_000)),
        blur_min=float(os.getenv("QUALITY_BLUR_MIN", "80.0")),
        binarization_ssim_min=float(os.getenv("QUALITY_SSIM_MIN", "0.50")),
        sauvola_window=int(os.getenv("SAUVOLA_WINDOW", "25")),
        sauvola_k=float(os.getenv("SAUVOLA_K", "0.20")),
        tesseract_lang=os.getenv("TESSERACT_LANG", "kor_best+eng_best"),
        baseline_tesseract_lang=os.getenv("BASELINE_TESSERACT_LANG", "kor+eng"),
        tesseract_psm=int(os.getenv("TESSERACT_PSM", "3")),
        ocr_candidate_trigger=float(os.getenv("OCR_CANDIDATE_TRIGGER", "72.0")),
        ocr_primary_bias=float(os.getenv("OCR_PRIMARY_BIAS", "3.0")),
        sparse_foreground_max=float(os.getenv("SPARSE_FOREGROUND_MAX", "0.035")),
        ocr_engine=os.getenv("OCR_ENGINE", "paddle").strip().lower(),
        paddle_det_limit_side_len=int(os.getenv("PADDLE_DET_LIMIT_SIDE_LEN", "1536")),
        paddle_cpu_threads=int(os.getenv("PADDLE_CPU_THREADS", "4")),
        paddle_recognition_batch_size=int(
            os.getenv("PADDLE_RECOGNITION_BATCH_SIZE", "1")
        ),
        paddle_recognition_model_dir=(
            os.getenv("PADDLE_RECOGNITION_MODEL_DIR", "").strip() or None
        ),
        paddle_text_det_thresh=(
            float(os.environ["PADDLE_TEXT_DET_THRESH"])
            if os.getenv("PADDLE_TEXT_DET_THRESH")
            else None
        ),
        paddle_text_det_box_thresh=(
            float(os.environ["PADDLE_TEXT_DET_BOX_THRESH"])
            if os.getenv("PADDLE_TEXT_DET_BOX_THRESH")
            else None
        ),
        gemini_api_key=_secret_from_environment(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_timeout_ms=int(os.getenv("GEMINI_TIMEOUT_MS", "45000")),
        feedback_db_path=os.getenv("FEEDBACK_DB_PATH", "data/feedback.db"),
    )
