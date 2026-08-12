import cv2
import numpy as np

from app.ocr import OcrLineData
from app.photo import map_lines_to_original, prepare_document_photo


def _photographed_page() -> np.ndarray:
    image = np.full((700, 900, 3), 70, dtype=np.uint8)
    corners = np.array([[160, 60], [790, 120], [720, 650], [80, 590]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (245, 245, 245))
    cv2.polylines(image, [corners], True, (255, 255, 255), 5)
    cv2.putText(
        image,
        "DOCUMENT TEST 2026",
        (190, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )
    return image


def test_prepare_document_photo_rectifies_visible_page() -> None:
    source = _photographed_page()

    prepared = prepare_document_photo(source, illumination_trigger=1.0)

    assert prepared.rectified is True
    assert prepared.inverse_transform is not None
    assert prepared.image.shape[0] < source.shape[0]
    assert prepared.image.shape[1] < source.shape[1]
    assert prepared.illumination_normalized is False


def test_map_lines_to_original_keeps_box_inside_source() -> None:
    prepared = prepare_document_photo(_photographed_page(), illumination_trigger=1.0)
    line = OcrLineData(
        text="DOCUMENT TEST 2026",
        confidence=97.0,
        x=100,
        y=150,
        width=300,
        height=60,
    )

    mapped = map_lines_to_original([line], prepared)

    assert mapped[0].text == line.text
    assert 0 <= mapped[0].x < prepared.original_width
    assert 0 <= mapped[0].y < prepared.original_height
    assert mapped[0].x + mapped[0].width <= prepared.original_width
    assert mapped[0].y + mapped[0].height <= prepared.original_height
