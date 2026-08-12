from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import Settings, get_settings
from app.ocr import OcrLineData, run_adaptive_ocr, run_ocr, weighted_confidence
from app.paddle_ocr import run_paddle_ocr
from app.quality import QualityAssessment, assess_quality
from app.reading_order import sort_reading_order
from research.ocr_metrics import (
    character_error_rate,
    numeric_value_scores,
    order_independent_word_scores,
    word_error_rate,
)

PROFILE_NAMES = (
    "baseline",
    "best_gray_psm3",
    "best_normalized_psm3",
    "best_sauvola_psm3",
    "best_gray_psm4",
    "best_gray_psm6",
    "best_gray_psm11",
    "adaptive",
    "paddle",
    "paddle_native_order",
)


def _read_manifest(path: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if split in row.get("splits", [])]
    return selected if limit is None else selected[:limit]


def _decode_image(payload: bytes) -> NDArray[np.uint8]:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("AI-Hub image could not be decoded")
    return cast(NDArray[np.uint8], image)


def _reference_text(label: dict[str, Any]) -> str:
    annotations = sorted(
        label.get("annotations", []),
        key=lambda item: int(item.get("id", 0)),
    )
    return " ".join(str(item.get("annotation.text", "")) for item in annotations)


def _lines_text(lines: list[OcrLineData]) -> str:
    return "\n".join(line.text for line in lines)


def _run_profile(
    profile: str,
    image: NDArray[np.uint8],
    quality: QualityAssessment,
    settings: Settings,
) -> tuple[list[OcrLineData], str, int]:
    if profile in {"paddle", "paddle_native_order"}:
        sorted_output = profile == "paddle"
        return (
            run_paddle_ocr(image, settings, sort_output=sorted_output),
            f"PaddleOCR PP-OCRv5 한국어 ({'visual' if sorted_output else 'native'} order)",
            1,
        )

    if profile == "adaptive":
        if quality.status == "retake":
            return [], "품질 게이트 중단", 0
        selected = run_adaptive_ocr(
            quality.grayscale,
            quality.normalized,
            quality.binary,
            settings,
            foreground_ratio=quality.foreground_ratio,
            illumination_variation=quality.illumination_variation,
        )
        return selected.lines, selected.strategy, selected.candidate_count

    image = quality.grayscale
    psm = 3
    lang = settings.tesseract_lang
    if profile == "baseline":
        lang = settings.baseline_tesseract_lang
    elif profile == "best_normalized_psm3":
        image = quality.normalized
    elif profile == "best_sauvola_psm3":
        image = quality.binary
    elif profile == "best_gray_psm4":
        psm = 4
    elif profile == "best_gray_psm6":
        psm = 6
    elif profile == "best_gray_psm11":
        psm = 11
    elif profile != "best_gray_psm3":
        raise ValueError(f"Unknown profile: {profile}")
    return run_ocr(image, settings, psm=psm, lang=lang), profile, 1


def _aggregate(rows: list[dict[str, Any]], profiles: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in profiles:
        samples = [row["profiles"][profile] for row in rows]
        review_ready = [
            item for item in samples if not item["alignment_review_required"]
        ]
        result[profile] = {
            "documents": len(samples),
            "mean_cer": round(statistics.fmean(item["cer"] for item in samples), 4),
            "median_cer": round(statistics.median(item["cer"] for item in samples), 4),
            "mean_wer": round(statistics.fmean(item["wer"] for item in samples), 4),
            "mean_word_f1": round(
                statistics.fmean(item["word_f1"] for item in samples), 4
            ),
            "mean_numeric_f1": round(
                statistics.fmean(item["numeric_f1"] for item in samples), 4
            ),
            "mean_confidence": round(
                statistics.fmean(item["model_confidence"] for item in samples), 2
            ),
            "mean_ocr_ms": round(statistics.fmean(item["ocr_ms"] for item in samples), 2),
            "alignment_review_required": len(samples) - len(review_ready),
            "mean_cer_without_review_flags": (
                round(statistics.fmean(item["cer"] for item in review_ready), 4)
                if review_ready
                else None
            ),
            "strategies": dict(Counter(item["strategy"] for item in samples)),
        }
    return result


def evaluate(
    test_data: Path,
    manifest: Path,
    split: str,
    profiles: list[str],
    limit: int | None,
) -> dict[str, Any]:
    settings = get_settings()
    cases = _read_manifest(manifest, split, limit)
    results: list[dict[str, Any]] = []
    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}

        def archive(relative_path: str) -> zipfile.ZipFile:
            path = test_data / relative_path
            if path not in archives:
                archives[path] = stack.enter_context(zipfile.ZipFile(path))
            return archives[path]

        for index, case in enumerate(cases, start=1):
            image_payload = archive(case["image_zip"]).read(case["image_member"])
            label = json.loads(archive(case["label_zip"]).read(case["label_member"]))
            image = _decode_image(image_payload)
            quality = assess_quality(image, settings)
            reference = _reference_text(label)
            profile_results: dict[str, Any] = {}
            paddle_cache: tuple[list[OcrLineData], float] | None = None
            for profile in profiles:
                if profile in {"paddle", "paddle_native_order"}:
                    if paddle_cache is None:
                        started = perf_counter()
                        native_lines = run_paddle_ocr(
                            image, settings, sort_output=False
                        )
                        paddle_cache = (
                            native_lines,
                            (perf_counter() - started) * 1000,
                        )
                    native_lines, elapsed_ms = paddle_cache
                    sorted_output = profile == "paddle"
                    lines = (
                        sort_reading_order(native_lines)
                        if sorted_output
                        else native_lines
                    )
                    strategy = (
                        "PaddleOCR PP-OCRv5 한국어 "
                        f"({'visual' if sorted_output else 'native'} order)"
                    )
                    candidate_count = 1
                else:
                    started = perf_counter()
                    lines, strategy, candidate_count = _run_profile(
                        profile, image, quality, settings
                    )
                    elapsed_ms = (perf_counter() - started) * 1000
                hypothesis = _lines_text(lines)
                reference_characters = len("".join(reference.split()))
                hypothesis_characters = len("".join(hypothesis.split()))
                cer = round(character_error_rate(reference, hypothesis), 4)
                length_ratio = (
                    hypothesis_characters / reference_characters
                    if reference_characters
                    else 0.0
                )
                profile_results[profile] = {
                    "cer": cer,
                    "wer": round(word_error_rate(reference, hypothesis), 4),
                    "word_f1": round(
                        order_independent_word_scores(reference, hypothesis).f1, 4
                    ),
                    "numeric_f1": round(
                        numeric_value_scores(reference, hypothesis).f1, 4
                    ),
                    "model_confidence": weighted_confidence(lines),
                    "characters": hypothesis_characters,
                    "reference_characters": reference_characters,
                    "output_reference_length_ratio": round(length_ratio, 3),
                    "alignment_review_required": cer > 1.0 and length_ratio >= 2.0,
                    "ocr_ms": round(elapsed_ms, 2),
                    "strategy": strategy,
                    "candidate_count": candidate_count,
                }
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "quality_status": quality.status,
                    "profiles": profile_results,
                }
            )
            print(json.dumps({"completed": index, "total": len(cases), "id": case["id"]}))

    return {
        "split": split,
        "profiles": profiles,
        "aggregate": _aggregate(results, profiles),
        "documents": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR profiles on AI-Hub labels")
    parser.add_argument("--test-data", type=Path, default=Path("Test Data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("research/manifests/ocr_manifest.jsonl")
    )
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--profile", action="append", choices=PROFILE_NAMES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = args.profile or ["baseline", "adaptive"]
    report = evaluate(
        args.test_data.resolve(),
        args.manifest.resolve(),
        args.split,
        profiles,
        args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
