from __future__ import annotations

import json
import unicodedata
import zipfile
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from ocr_training.aihub import DatasetSplit, resolve_dataset_root, stable_number

ImageVariant = Literal["clean", "camera", "camera_filtered"]

PROGRESS_EVERY_DOCUMENTS = 100


@dataclass(frozen=True)
class BuildOptions:
    data_root: Path
    training_manifest: Path
    validation_manifest: Path
    output: Path
    max_training_documents: int | None = None
    final_documents: int = 120
    max_text_length: int = 25
    padding_ratio: float = 0.06


def document_split(row: dict[str, Any]) -> DatasetSplit:
    source = row.get("source_split")
    split = row.get("dataset_split")

    if source == "training" and split in {"train", "dev"}:
        return cast(DatasetSplit, split)
    if source == "validation" and split == "final":
        return "final"

    raise ValueError(
        f"Invalid source/dataset split contract: {source!r} -> {split!r}"
    )


def assigned_variant(document_id: str, annotation_id: str) -> ImageVariant:
    bucket = stable_number(f"{document_id}:{annotation_id}") % 100
    return "clean" if bucket < 55 else "camera"


def normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.replace("\t", " ").split())


def crop_annotation(
    image: NDArray[np.uint8],
    bbox: Iterable[float],
    padding_ratio: float,
) -> NDArray[np.uint8] | None:
    values = list(bbox)
    if len(values) < 4:
        return None

    try:
        x, y, width, height = (float(value) for value in values[:4])
    except (TypeError, ValueError):
        return None

    if not all(np.isfinite(value) for value in (x, y, width, height)):
        return None
    if width < 2 or height < 2:
        return None

    pad_x = max(2, int(round(width * padding_ratio)))
    pad_y = max(2, int(round(height * padding_ratio)))

    image_height, image_width = image.shape[:2]
    left = max(0, int(np.floor(x)) - pad_x)
    top = max(0, int(np.floor(y)) - pad_y)
    right = min(image_width, int(np.ceil(x + width)) + pad_x)
    bottom = min(image_height, int(np.ceil(y + height)) + pad_y)

    if right - left < 2 or bottom - top < 2:
        return None

    return image[top:bottom, left:right].copy()


def _camera_degrade(
    image: NDArray[np.uint8],
    seed: int,
) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    result: NDArray[np.uint8] = image.copy()

    if width >= 16 and height >= 10:
        jitter_x = max(1.0, width * float(rng.uniform(0.01, 0.035)))
        jitter_y = max(1.0, height * float(rng.uniform(0.01, 0.05)))
        source = np.asarray(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        target = source + np.asarray(
            [
                [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
                [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
                [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
                [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            ],
            dtype=np.float32,
        )
        result = cast(
            NDArray[np.uint8],
            cv2.warpPerspective(
                result,
                cv2.getPerspectiveTransform(source, target),
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ),
        )

    axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    if bool(rng.integers(0, 2)):
        axis = axis[::-1]

    lighting = 1.0 + axis * float(rng.uniform(0.08, 0.24))
    result = cast(
        NDArray[np.uint8],
        np.clip(
            result.astype(np.float32) * lighting[None, :, None],
            0,
            255,
        ).astype(np.uint8),
    )

    if min(width, height) >= 8 and bool(rng.integers(0, 2)):
        scale = float(rng.uniform(0.55, 0.85))
        reduced = cv2.resize(
            result,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        result = cast(
            NDArray[np.uint8],
            cv2.resize(
                reduced,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            ),
        )

    result = cast(
        NDArray[np.uint8],
        cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=float(rng.uniform(0.35, 1.0)),
        ),
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        result,
        [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 86))],
    )
    if ok and (decoded := cv2.imdecode(encoded, cv2.IMREAD_COLOR)) is not None:
        result = cast(NDArray[np.uint8], decoded)

    return result


def _mild_scan_filter(
    image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    ).apply(gray)

    sharpened = cv2.addWeighted(
        contrast,
        1.35,
        cv2.GaussianBlur(contrast, (0, 0), sigmaX=0.8),
        -0.35,
        0,
    )
    return cast(
        NDArray[np.uint8],
        cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR),
    )


def make_variant(
    crop: NDArray[np.uint8],
    variant: ImageVariant,
    seed: int,
) -> NDArray[np.uint8]:
    if variant == "clean":
        return crop

    degraded = _camera_degrade(crop, seed)
    return (
        _mild_scan_filter(degraded)
        if variant == "camera_filtered"
        else degraded
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL in {path} at line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Manifest row is not an object: {path}:{line_number}"
            )
        rows.append(payload)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    return rows


def _balanced_sample(
    rows: list[dict[str, Any]],
    maximum: int | None,
) -> list[dict[str, Any]]:
    if maximum is None or maximum >= len(rows):
        return sorted(rows, key=lambda row: stable_number(str(row["id"])))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            str(row.get("category", "unknown")),
            [],
        ).append(row)

    for values in grouped.values():
        values.sort(key=lambda row: stable_number(str(row["id"])))

    selected: list[dict[str, Any]] = []

    while len(selected) < maximum and any(grouped.values()):
        for category in sorted(grouped):
            if grouped[category] and len(selected) < maximum:
                selected.append(grouped[category].pop(0))

    return selected


def _decode_image(
    payload: bytes,
    *,
    source: str,
) -> NDArray[np.uint8]:
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Image payload could not be decoded: {source}")

    return cast(NDArray[np.uint8], image)


def _write_image(
    path: Path,
    image: NDArray[np.uint8],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    ):
        raise OSError(f"Could not write image: {path}")


def _validate_manifests(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    for row in training_rows:
        document_split(row)
        if row["source_split"] != "training":
            raise ValueError(
                "Training manifest contains non-Training data"
            )

    for row in validation_rows:
        document_split(row)
        if row["source_split"] != "validation":
            raise ValueError(
                "Validation manifest contains non-Validation data"
            )

    overlap = {
        str(row["id"]) for row in training_rows
    } & {
        str(row["id"]) for row in validation_rows
    }

    if overlap:
        raise ValueError(
            "Training/Validation document leakage detected: "
            f"{sorted(overlap)[:10]}"
        )


def _load_label_from_archive(
    archive: zipfile.ZipFile,
    member: str,
    *,
    archive_path: Path,
) -> dict[str, Any]:
    try:
        raw = archive.read(member)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Label member not found: {archive_path} :: {member}"
        ) from exc

    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid label JSON: {archive_path} :: {member}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Label JSON root is not an object: {archive_path} :: {member}"
        )

    return payload


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def build_dataset(
    options: BuildOptions,
) -> dict[str, Any]:
    print("[dataset] loading manifests", flush=True)

    training_rows = _read_manifest(options.training_manifest)
    validation_rows = _read_manifest(options.validation_manifest)
    _validate_manifests(training_rows, validation_rows)

    print(
        f"[dataset] manifests loaded: "
        f"training={len(training_rows):,}, "
        f"validation={len(validation_rows):,}",
        flush=True,
    )

    if options.max_training_documents is None:
        selected_training = _balanced_sample(training_rows, None)
    else:
        if options.max_training_documents < 1:
            raise ValueError("max_training_documents must be >= 1")

        dev_maximum = (
            max(1, round(options.max_training_documents * 0.1))
            if options.max_training_documents > 1
            else 0
        )
        train_maximum = options.max_training_documents - dev_maximum

        selected_training = _balanced_sample(
            [
                row
                for row in training_rows
                if document_split(row) == "train"
            ],
            train_maximum,
        ) + _balanced_sample(
            [
                row
                for row in training_rows
                if document_split(row) == "dev"
            ],
            dev_maximum,
        )

    selected_final = _balanced_sample(
        validation_rows,
        options.final_documents,
    )

    rows = selected_training + selected_final

    print(
        f"[dataset] selected documents: "
        f"training={len(selected_training):,}, "
        f"final={len(selected_final):,}, "
        f"total={len(rows):,}",
        flush=True,
    )

    dataset_root = resolve_dataset_root(options.data_root)
    print(
        f"[dataset] resolved source root: {dataset_root}",
        flush=True,
    )

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    label_rows: dict[str, list[str]] = {
        "train": [],
        "dev": [],
    }

    final_rows: dict[ImageVariant, list[str]] = {
        "clean": [],
        "camera": [],
        "camera_filtered": [],
    }

    counts = {
        "documents": 0,
        "train_documents": sum(
            document_split(row) == "train"
            for row in selected_training
        ),
        "dev_documents": sum(
            document_split(row) == "dev"
            for row in selected_training
        ),
        "final_documents": len(selected_final),
        "train": 0,
        "dev": 0,
        "final_per_variant": 0,
        "empty_skipped": 0,
        "too_long_skipped": 0,
        "invalid_bbox_skipped": 0,
        "invalid_annotation_skipped": 0,
    }

    metadata_path = output / "metadata.jsonl"
    metadata_tmp = metadata_path.with_suffix(
        metadata_path.suffix + ".tmp"
    )

    try:
        with (
            ExitStack() as stack,
            metadata_tmp.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as metadata,
        ):
            open_archives: dict[Path, zipfile.ZipFile] = {}

            def archive(
                relative_path: str,
            ) -> tuple[Path, zipfile.ZipFile]:
                path = (
                    dataset_root / relative_path
                ).resolve()

                if path not in open_archives:
                    try:
                        open_archives[path] = stack.enter_context(
                            zipfile.ZipFile(path)
                        )
                    except (
                        zipfile.BadZipFile,
                        OSError,
                    ) as exc:
                        raise RuntimeError(
                            f"Could not open ZIP archive: {path}"
                        ) from exc

                return path, open_archives[path]

            for row_index, row in enumerate(rows, start=1):
                split = document_split(row)

                image_archive_path, image_archive = archive(
                    str(row["image_zip"])
                )

                try:
                    image_payload = image_archive.read(
                        str(row["image_member"])
                    )
                except KeyError as exc:
                    raise FileNotFoundError(
                        "Image member not found: "
                        f"{image_archive_path} :: "
                        f"{row['image_member']}"
                    ) from exc

                image = _decode_image(
                    image_payload,
                    source=(
                        f"{image_archive_path} :: "
                        f"{row['image_member']}"
                    ),
                )

                label_archive_path, label_archive = archive(
                    str(row["label_zip"])
                )
                label = _load_label_from_archive(
                    label_archive,
                    str(row["label_member"]),
                    archive_path=label_archive_path,
                )

                counts["documents"] += 1

                annotations = label.get("annotations", [])
                if not isinstance(annotations, list):
                    raise ValueError(
                        "annotations is not a list: "
                        f"{label_archive_path} :: "
                        f"{row['label_member']}"
                    )

                for index, annotation in enumerate(annotations):
                    if not isinstance(annotation, dict):
                        counts["invalid_annotation_skipped"] += 1
                        continue

                    text = normalize_label(
                        annotation.get("annotation.text")
                    )
                    if not text:
                        counts["empty_skipped"] += 1
                        continue

                    if len(text) > options.max_text_length:
                        counts["too_long_skipped"] += 1
                        continue

                    crop = crop_annotation(
                        image,
                        annotation.get(
                            "annotation.bbox",
                            [],
                        ),
                        options.padding_ratio,
                    )

                    if crop is None:
                        counts["invalid_bbox_skipped"] += 1
                        continue

                    annotation_id = str(
                        annotation.get("id", index)
                    )

                    if split in {"train", "dev"}:
                        train_variant: ImageVariant = (
                            assigned_variant(
                                str(row["id"]),
                                annotation_id,
                            )
                            if split == "train"
                            else "clean"
                        )

                        relative = (
                            Path("images")
                            / split
                            / str(row["id"])
                            / f"{annotation_id}_{train_variant}.jpg"
                        )

                        seed = stable_number(
                            f"{row['id']}:{annotation_id}:"
                            f"{train_variant}"
                        )

                        _write_image(
                            output / relative,
                            make_variant(
                                crop,
                                train_variant,
                                seed,
                            ),
                        )

                        label_rows[split].append(
                            f"{relative.as_posix()}\t{text}"
                        )
                        counts[split] += 1

                        metadata.write(
                            json.dumps(
                                {
                                    "document_id": row["id"],
                                    "annotation_id": annotation_id,
                                    "source_split": row["source_split"],
                                    "dataset_split": split,
                                    "variant": train_variant,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    for final_variant in final_rows:
                        relative = (
                            Path("images")
                            / "final"
                            / final_variant
                            / str(row["id"])
                            / f"{annotation_id}.jpg"
                        )

                        seed = stable_number(
                            f"{row['id']}:{annotation_id}:"
                            f"{final_variant}:final"
                        )

                        _write_image(
                            output / relative,
                            make_variant(
                                crop,
                                final_variant,
                                seed,
                            ),
                        )

                        final_rows[final_variant].append(
                            f"{relative.as_posix()}\t{text}"
                        )

                    counts["final_per_variant"] += 1

                    metadata.write(
                        json.dumps(
                            {
                                "document_id": row["id"],
                                "annotation_id": annotation_id,
                                "source_split": "validation",
                                "dataset_split": "final",
                                "variant": "all-final-variants",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if (
                    row_index == 1
                    or row_index == len(rows)
                    or row_index % PROGRESS_EVERY_DOCUMENTS == 0
                ):
                    print(
                        f"[dataset] documents "
                        f"{row_index:,}/{len(rows):,} "
                        f"(train crops={counts['train']:,}, "
                        f"dev crops={counts['dev']:,}, "
                        f"final crops={counts['final_per_variant']:,})",
                        flush=True,
                    )

        metadata_tmp.replace(metadata_path)

    finally:
        if metadata_tmp.exists():
            metadata_tmp.unlink(missing_ok=True)

    for label_split, lines in label_rows.items():
        _atomic_write_text(
            output / f"{label_split}.txt",
            ("\n".join(lines) + "\n") if lines else "",
        )

    for output_variant, lines in final_rows.items():
        _atomic_write_text(
            output / f"final_{output_variant}.txt",
            ("\n".join(lines) + "\n") if lines else "",
        )

    if not label_rows["train"]:
        raise ValueError(
            "No training crops were generated; refusing to write a "
            "successful dataset summary"
        )

    if not label_rows["dev"]:
        raise ValueError(
            "No dev crops were generated; refusing to write a "
            "successful dataset summary"
        )

    summary = {
        **counts,
        "max_text_length": options.max_text_length,
        "contract": (
            "AI-Hub Training -> train/dev; "
            "AI-Hub Validation -> final only"
        ),
        "validation_used_by_training": False,
        "train_variant_policy": (
            "55% clean, 45% simulated camera"
        ),
        "dev_variant_policy": "clean only",
        "final_variants": list(final_rows),
    }

    _atomic_write_text(
        output / "summary.json",
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    print(
        "[dataset] complete: "
        f"documents={counts['documents']:,}, "
        f"train crops={counts['train']:,}, "
        f"dev crops={counts['dev']:,}, "
        f"final/variant={counts['final_per_variant']:,}",
        flush=True,
    )

    return summary