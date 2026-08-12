from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.quality import assess_quality


def test_blank_image_requires_retake() -> None:
    image = np.full((160, 240, 3), 255, dtype=np.uint8)

    result = assess_quality(image, Settings())

    assert result.status == "retake"
    assert "blur" in result.reasons


def test_sharp_text_image_passes_initial_thresholds() -> None:
    image = np.full((240, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "TEST 2026", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    result = assess_quality(image, Settings())

    assert result.status == "pass"
    assert result.binary.shape == image.shape[:2]
    assert result.normalized.shape == image.shape[:2]
    assert result.text_contrast > 0
