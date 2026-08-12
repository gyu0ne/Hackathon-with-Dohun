from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.ocr import OcrLineData

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class PreparedDocument:
    image: ImageArray
    inverse_transform: NDArray[np.float64] | None
    original_width: int
    original_height: int
    rectified: bool
    illumination_normalized: bool
    illumination_variation: float


def _order_corners(points: NDArray[np.float32]) -> NDArray[np.float32]:
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]
    return ordered


def _document_corners(image: ImageArray) -> NDArray[np.float32] | None:
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )
    preview_area = float(preview.shape[0] * preview.shape[1])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < preview_area * 0.35:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        corners = polygon.reshape(4, 2).astype(np.float32) / scale
        ordered = _order_corners(corners)
        margin = max(height, width) * 0.025
        full_page = (
            np.linalg.norm(ordered[0] - (0, 0)) < margin
            and np.linalg.norm(ordered[1] - (width - 1, 0)) < margin
            and np.linalg.norm(ordered[2] - (width - 1, height - 1)) < margin
            and np.linalg.norm(ordered[3] - (0, height - 1)) < margin
        )
        return None if full_page else ordered
    return None


def _rectify(
    image: ImageArray,
) -> tuple[ImageArray, NDArray[np.float64] | None]:
    corners = _document_corners(image)
    if corners is None:
        return image.copy(), None
    top_left, top_right, bottom_right, bottom_left = corners
    output_width = int(
        round(
            max(
                float(np.linalg.norm(top_right - top_left)),
                float(np.linalg.norm(bottom_right - bottom_left)),
            )
        )
    )
    output_height = int(
        round(
            max(
                float(np.linalg.norm(bottom_left - top_left)),
                float(np.linalg.norm(bottom_right - top_right)),
            )
        )
    )
    if output_width < 128 or output_height < 128:
        return image.copy(), None
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    inverse = cv2.getPerspectiveTransform(destination, corners)
    rectified = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cast(ImageArray, rectified), inverse.astype(np.float64)


def _illumination_variation(image: ImageArray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 512.0 / max(gray.shape))
    sample = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sigma = max(9.0, min(sample.shape) / 18.0)
    background = cv2.GaussianBlur(sample, (0, 0), sigmaX=sigma, sigmaY=sigma)
    low, high = np.percentile(background.astype(np.float32), (10, 90))
    return float((high - low) / 255.0)


def _normalize_illumination(image: ImageArray) -> ImageArray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigma = max(15.0, min(gray.shape) / 24.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=245)
    enhanced = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(normalized)
    return cast(ImageArray, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))


def prepare_document_photo(
    image: ImageArray,
    *,
    rectify: bool = True,
    normalize_illumination: bool = True,
    illumination_trigger: float = 0.10,
) -> PreparedDocument:
    height, width = image.shape[:2]
    rectified, inverse = _rectify(image) if rectify else (image.copy(), None)
    variation = _illumination_variation(rectified)
    normalize = normalize_illumination and variation >= illumination_trigger
    prepared = _normalize_illumination(rectified) if normalize else rectified
    return PreparedDocument(
        image=prepared,
        inverse_transform=inverse,
        original_width=width,
        original_height=height,
        rectified=inverse is not None,
        illumination_normalized=normalize,
        illumination_variation=round(variation, 4),
    )


def map_lines_to_original(
    lines: list[OcrLineData],
    prepared: PreparedDocument,
) -> list[OcrLineData]:
    if prepared.inverse_transform is None:
        return lines
    mapped: list[OcrLineData] = []
    for line in lines:
        corners = np.array(
            [
                [line.x, line.y],
                [line.x + line.width, line.y],
                [line.x + line.width, line.y + line.height],
                [line.x, line.y + line.height],
            ],
            dtype=np.float32,
        ).reshape(1, 4, 2)
        projected = cv2.perspectiveTransform(corners, prepared.inverse_transform)[0]
        left, top = np.floor(projected.min(axis=0)).astype(int)
        right, bottom = np.ceil(projected.max(axis=0)).astype(int)
        left = max(0, min(prepared.original_width - 1, int(left)))
        top = max(0, min(prepared.original_height - 1, int(top)))
        right = max(left + 1, min(prepared.original_width, int(right)))
        bottom = max(top + 1, min(prepared.original_height, int(bottom)))
        mapped.append(
            OcrLineData(
                text=line.text,
                confidence=line.confidence,
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
        )
    return mapped
