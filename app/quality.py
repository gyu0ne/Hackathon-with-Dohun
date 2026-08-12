from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_sauvola

from app.config import Settings

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class QualityAssessment:
    status: Literal["pass", "retake"]
    reasons: list[str]
    focus_score: float
    text_contrast: float
    foreground_ratio: float
    illumination_variation: float
    grayscale: ImageArray
    normalized: ImageArray
    binary: ImageArray


def _odd_window(preferred: int, image: ImageArray) -> int:
    maximum = min(image.shape[:2])
    window = min(preferred, maximum if maximum % 2 else maximum - 1)
    return max(3, window)


def _normalize_background(grayscale: ImageArray) -> ImageArray:
    """Improve local contrast without turning every scan artifact into black ink."""
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    return cast(ImageArray, clahe.apply(grayscale))


def _illumination_variation(grayscale: ImageArray) -> float:
    height, width = grayscale.shape
    sample = cv2.resize(
        grayscale,
        (max(2, min(16, width)), max(2, min(16, height))),
        interpolation=cv2.INTER_AREA,
    )
    return float(np.std(sample.astype(np.float32)) / 255.0)


def assess_quality(image: ImageArray, settings: Settings) -> QualityAssessment:
    grayscale = cast(ImageArray, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    normalized = _normalize_background(grayscale)
    focus_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    _, foreground = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    foreground_mask = foreground > 0
    foreground_ratio = float(np.mean(foreground_mask))
    if 0.0 < foreground_ratio < 1.0:
        ink_mean = float(np.mean(normalized[foreground_mask]))
        paper_mean = float(np.mean(normalized[~foreground_mask]))
        text_contrast = max(0.0, 100.0 * (paper_mean - ink_mean) / 255.0)
    else:
        text_contrast = 0.0

    threshold = threshold_sauvola(
        normalized,
        window_size=_odd_window(settings.sauvola_window, normalized),
        k=settings.sauvola_k,
    )
    binary = np.where(normalized > threshold, 255, 0).astype(np.uint8)
    illumination_variation = _illumination_variation(grayscale)

    blank = foreground_ratio < 0.001 or foreground_ratio > 0.70
    blurry = focus_score < settings.blur_min
    low_contrast = text_contrast < 12.0
    reasons: list[str] = []
    if blank:
        reasons.append("blank")
    if blurry:
        reasons.append("blur")
    if low_contrast:
        reasons.append("low_contrast")

    # A blurred but high-contrast scan can still be recoverable. Only stop when
    # there is effectively no text, or both focus and contrast are inadequate.
    retake = blank or (blurry and low_contrast)
    return QualityAssessment(
        status="retake" if retake else "pass",
        reasons=reasons,
        focus_score=round(focus_score, 4),
        text_contrast=round(text_contrast, 2),
        foreground_ratio=round(foreground_ratio, 5),
        illumination_variation=round(illumination_variation, 5),
        grayscale=grayscale,
        normalized=normalized,
        binary=binary,
    )
