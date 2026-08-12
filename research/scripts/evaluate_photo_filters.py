from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.ocr import OcrLineData, weighted_confidence
from app.paddle_ocr import run_paddle_ocr
from app.photo import prepare_document_photo
from research.ocr_metrics import (
    character_error_rate,
    numeric_value_scores,
    order_independent_word_scores,
    word_error_rate,
)

DEFORMATIONS = (
    "perspective",
    "curled",
    "fewfold",
    "multifold",
    "crumpleseasy",
    "crumpleshard",
)
LIGHTING = ("bright", "color", "shadow")
VARIANTS = ("raw", "perspective", "illumination", "combined", "paddle_unwarp")
DETECTION_MODELS = {
    "mobile": "PP-OCRv5_mobile_det",
    "server": "PP-OCRv5_server_det",
}
DATASET_URL = "https://felixhertlein.github.io/inv3d/"

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class PhotoCase:
    archive: Path
    template_id: str
    deformation: str
    lighting: str
    image_member: str
    label_member: str

    @property
    def case_id(self) -> str:
        return f"{self.template_id}-{self.deformation}-{self.lighting}"


def _decode_image(payload: bytes) -> ImageArray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Inv3D image could not be decoded")
    return cast(ImageArray, image)


def _reference_text(payload: bytes) -> str:
    words = json.loads(payload)
    return " ".join(str(word["text"]) for word in words if str(word["text"]).strip())


def _recognized_text(lines: list[OcrLineData]) -> str:
    return "\n".join(line.text for line in lines)


def _template_members(archive: zipfile.ZipFile) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for entry in archive.namelist():
        parts = entry.split("/")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        members.setdefault(parts[1], set()).add(entry)
    return members


def select_cases(
    archives: list[Path],
    *,
    templates_per_archive: int,
) -> list[PhotoCase]:
    cases: list[PhotoCase] = []
    template_offset = 0
    for archive_path in sorted(archives):
        with zipfile.ZipFile(archive_path) as archive:
            members_by_template = _template_members(archive)
            template_ids = sorted(members_by_template)[:templates_per_archive]
            for local_index, template_id in enumerate(template_ids):
                members = members_by_template[template_id]
                label_member = next(
                    member
                    for member in members
                    if member.endswith("/ground_truth_words.json")
                )
                for deformation_index, deformation in enumerate(DEFORMATIONS):
                    lighting = LIGHTING[
                        (template_offset + local_index + deformation_index)
                        % len(LIGHTING)
                    ]
                    filename = f"warped_document_{deformation}_{lighting}.jpg"
                    image_member = next(
                        member for member in members if member.endswith(f"/{filename}")
                    )
                    cases.append(
                        PhotoCase(
                            archive=archive_path,
                            template_id=template_id,
                            deformation=deformation,
                            lighting=lighting,
                            image_member=image_member,
                            label_member=label_member,
                        )
                    )
        template_offset += templates_per_archive
    return cases


def _prepare_variant(
    image: ImageArray,
    variant: str,
) -> tuple[ImageArray, dict[str, Any], bool]:
    if variant == "raw" or variant == "paddle_unwarp":
        return image, {
            "rectified": False,
            "illumination_normalized": False,
            "illumination_variation": None,
        }, variant == "paddle_unwarp"
    if variant == "perspective":
        prepared = prepare_document_photo(image, normalize_illumination=False)
    elif variant == "illumination":
        prepared = prepare_document_photo(
            image,
            rectify=False,
            illumination_trigger=0.0,
        )
    elif variant == "combined":
        prepared = prepare_document_photo(image)
    else:
        raise ValueError(f"Unknown photo variant: {variant}")
    return prepared.image, {
        "rectified": prepared.rectified,
        "illumination_normalized": prepared.illumination_normalized,
        "illumination_variation": prepared.illumination_variation,
    }, False


def _measure_variant(
    image: ImageArray,
    reference: str,
    settings: Settings,
    *,
    variant: str,
    detection_model_name: str,
) -> dict[str, Any]:
    started = perf_counter()
    ocr_image, preparation, use_doc_unwarping = _prepare_variant(image, variant)
    lines = run_paddle_ocr(
        ocr_image,
        settings,
        detection_model_name=detection_model_name,
        use_doc_unwarping=use_doc_unwarping,
    )
    elapsed_ms = (perf_counter() - started) * 1000
    hypothesis = _recognized_text(lines)
    word_scores = order_independent_word_scores(reference, hypothesis)
    number_scores = numeric_value_scores(reference, hypothesis)
    return {
        "reading_order_cer": round(character_error_rate(reference, hypothesis), 4),
        "reading_order_wer": round(word_error_rate(reference, hypothesis), 4),
        "word_precision": round(word_scores.precision, 4),
        "word_recall": round(word_scores.recall, 4),
        "word_f1": round(word_scores.f1, 4),
        "matched_words": word_scores.matched,
        "reference_words": word_scores.reference_count,
        "recognized_words": word_scores.hypothesis_count,
        "numeric_precision": round(number_scores.precision, 4),
        "numeric_recall": round(number_scores.recall, 4),
        "numeric_f1": round(number_scores.f1, 4),
        "matched_numeric_values": number_scores.matched,
        "reference_numeric_values": number_scores.reference_count,
        "model_confidence": weighted_confidence(lines),
        "ocr_ms": round(elapsed_ms, 2),
        "characters": len("".join(hypothesis.split())),
        "lines": len(lines),
        "input_size": [ocr_image.shape[1], ocr_image.shape[0]],
        **preparation,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return round(statistics.fmean(float(row[field]) for row in rows), 4)


def _variant_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "mean_reading_order_cer": _mean(rows, "reading_order_cer"),
        "median_reading_order_cer": round(
            statistics.median(row["reading_order_cer"] for row in rows), 4
        ),
        "mean_word_precision": _mean(rows, "word_precision"),
        "mean_word_recall": _mean(rows, "word_recall"),
        "mean_word_f1": _mean(rows, "word_f1"),
        "mean_numeric_f1": _mean(rows, "numeric_f1"),
        "mean_confidence": _mean(rows, "model_confidence"),
        "mean_ocr_ms": round(_mean(rows, "ocr_ms"), 2),
        "rectified": sum(bool(row["rectified"]) for row in rows),
        "illumination_normalized": sum(
            bool(row["illumination_normalized"]) for row in rows
        ),
    }


def _aggregate(
    documents: list[dict[str, Any]],
    variants: list[str],
) -> dict[str, Any]:
    result = {
        variant: _variant_aggregate([document["variants"][variant] for document in documents])
        for variant in variants
    }
    if "raw" not in variants:
        return result
    raw = [document["variants"]["raw"] for document in documents]
    effects: dict[str, Any] = {}
    for variant in variants:
        if variant == "raw":
            continue
        candidate = [document["variants"][variant] for document in documents]
        effects[variant] = {
            "mean_word_f1_gain": round(
                _mean(candidate, "word_f1") - _mean(raw, "word_f1"), 4
            ),
            "mean_numeric_f1_gain": round(
                _mean(candidate, "numeric_f1") - _mean(raw, "numeric_f1"), 4
            ),
            "mean_reading_order_cer_reduction": round(
                _mean(raw, "reading_order_cer")
                - _mean(candidate, "reading_order_cer"),
                4,
            ),
            "word_f1_wins": sum(
                selected["word_f1"] > baseline["word_f1"]
                for baseline, selected in zip(raw, candidate, strict=True)
            ),
            "word_f1_losses": sum(
                selected["word_f1"] < baseline["word_f1"]
                for baseline, selected in zip(raw, candidate, strict=True)
            ),
        }
    result["effects_vs_raw"] = effects
    return result


def _selection_heuristics(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare deployable raw/unwarp selectors with a ground-truth oracle."""
    rules = {
        "always_raw": None,
        "always_unwarp": None,
        "higher_confidence": "model_confidence",
        "more_words": "recognized_words",
        "more_characters": "characters",
        "oracle_word_f1": "word_f1",
    }
    oracle_choices = [
        "paddle_unwarp"
        if document["variants"]["paddle_unwarp"]["word_f1"]
        > document["variants"]["raw"]["word_f1"]
        else "raw"
        for document in documents
    ]
    results: dict[str, Any] = {}
    for rule, field in rules.items():
        choices: list[str] = []
        selected: list[dict[str, Any]] = []
        for document in documents:
            raw = document["variants"]["raw"]
            unwarp = document["variants"]["paddle_unwarp"]
            choice = "raw"
            if rule == "always_unwarp" or (
                field is not None and float(unwarp[field]) > float(raw[field])
            ):
                choice = "paddle_unwarp"
            choices.append(choice)
            selected.append(document["variants"][choice])
        results[rule] = {
            "mean_word_f1": _mean(selected, "word_f1"),
            "mean_numeric_f1": _mean(selected, "numeric_f1"),
            "unwarp_picks": choices.count("paddle_unwarp"),
            "oracle_matches": sum(
                choice == oracle
                for choice, oracle in zip(choices, oracle_choices, strict=True)
            ),
            "documents": len(documents),
        }
    return results


def _condition_aggregate(
    documents: list[dict[str, Any]],
    variants: list[str],
    field: str,
) -> dict[str, Any]:
    values = sorted({str(document[field]) for document in documents})
    return {
        value: _aggregate(
            [document for document in documents if str(document[field]) == value],
            variants,
        )
        for value in values
    }


def evaluate(
    archives: list[Path],
    *,
    templates_per_archive: int,
    variants: list[str],
    detection_limit: int,
    detection_model: str,
) -> dict[str, Any]:
    settings = Settings(paddle_det_limit_side_len=detection_limit)
    cases = select_cases(
        archives,
        templates_per_archive=templates_per_archive,
    )
    documents: list[dict[str, Any]] = []
    open_archives: dict[Path, zipfile.ZipFile] = {}
    try:
        for index, case in enumerate(cases, start=1):
            if case.archive not in open_archives:
                open_archives[case.archive] = zipfile.ZipFile(case.archive)
            archive = open_archives[case.archive]
            image = _decode_image(archive.read(case.image_member))
            reference = _reference_text(archive.read(case.label_member))
            variant_results = {
                variant: _measure_variant(
                    image,
                    reference,
                    settings,
                    variant=variant,
                    detection_model_name=DETECTION_MODELS[detection_model],
                )
                for variant in variants
            }
            documents.append(
                {
                    "id": case.case_id,
                    "template_id": case.template_id,
                    "deformation": case.deformation,
                    "lighting": case.lighting,
                    "variants": variant_results,
                }
            )
            print(
                json.dumps(
                    {"completed": index, "total": len(cases), "id": case.case_id}
                ),
                flush=True,
            )
    finally:
        for archive in open_archives.values():
            archive.close()
    report = {
        "dataset": "Inv3DReal",
        "dataset_url": DATASET_URL,
        "selection": {
            "templates_per_archive": templates_per_archive,
            "archives": [archive.name for archive in archives],
            "deformations": list(DEFORMATIONS),
            "lighting_rotation": list(LIGHTING),
        },
        "configuration": {
            "variants": variants,
            "detection_limit": detection_limit,
            "detection_model": DETECTION_MODELS[detection_model],
            "recognition_model": "korean_PP-OCRv5_mobile_rec",
        },
        "aggregate": _aggregate(documents, variants),
        "by_deformation": _condition_aggregate(documents, variants, "deformation"),
        "by_lighting": _condition_aggregate(documents, variants, "lighting"),
        "documents": documents,
    }
    if {"raw", "paddle_unwarp"}.issubset(variants):
        report["selection_heuristics"] = _selection_heuristics(documents)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate PaddleOCR photo preprocessing on real Inv3D phone photos"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("research/external_photos/inv3d"),
    )
    parser.add_argument("--templates-per-archive", type=int, default=1)
    parser.add_argument("--archive-limit", type=int, choices=(1, 2), default=2)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    parser.add_argument("--det-limit", type=int, default=960)
    parser.add_argument("--det-model", choices=tuple(DETECTION_MODELS), default="mobile")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/photo_filter_inv3d.json"),
    )
    args = parser.parse_args()
    if args.templates_per_archive < 1:
        parser.error("--templates-per-archive must be at least 1")
    variants = list(dict.fromkeys(args.variant or ["raw", "combined"]))
    archives = sorted(args.data_dir.resolve().glob("inv3d_real_part_*_of_2.zip"))
    if len(archives) != 2:
        parser.error("Both Inv3DReal archive parts are required")
    report = evaluate(
        archives[: args.archive_limit],
        templates_per_archive=args.templates_per_archive,
        variants=variants,
        detection_limit=args.det_limit,
        detection_model=args.det_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
