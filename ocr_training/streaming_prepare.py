from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from array import array
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_training.aihub import (
    resolve_dataset_root,
    stable_number,
    valid_annotation_indices,
)
from ocr_training.dataset import (
    ImageVariant,
    crop_annotation,
    make_variant,
    normalize_label,
)
from ocr_training.streaming_index import INDEX_SCHEMA_VERSION


@dataclass(frozen=True)
class PrepareOptions:
    data_root: Path
    training_manifest: Path
    validation_manifest: Path
    output: Path
    final_documents: int = 120
    dev_max_samples: int = 10_000
    max_text_length: int = 25
    padding_ratio: float = 0.06


def _manifest_rows(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid manifest JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row is not an object at {path}:{line_number}")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        temporary.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.flush()
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _save_array(path: Path, values: array[int]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, np.frombuffer(values, dtype=np.int64), allow_pickle=False)
    temporary.replace(path)


def _insert_document(
    connection: sqlite3.Connection, row: dict[str, Any]
) -> int:
    sample_count = int(row.get("sample_count", 0))
    if sample_count < 0:
        raise ValueError(f"Invalid sample count for document {row.get('id')}")
    cursor = connection.execute(
        """
        INSERT INTO documents (
            document_id, source_split, dataset_split, category,
            image_zip, image_member, label_zip, label_member,
            sample_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["id"]),
            str(row["source_split"]),
            str(row["dataset_split"]),
            str(row.get("category", "unknown")),
            str(row["image_zip"]),
            str(row["image_member"]),
            str(row["label_zip"]),
            str(row["label_member"]),
            sample_count,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an inserted document key")
    return int(cursor.lastrowid)


def _build_index(options: PrepareOptions, dataset_root: Path) -> dict[str, int]:
    output = options.output
    output.mkdir(parents=True, exist_ok=True)
    database = output / "documents.sqlite3"
    database_tmp = database.with_suffix(".sqlite3.tmp")
    database_tmp.unlink(missing_ok=True)

    train_documents: array[int] = array("q")
    train_ends: array[int] = array("q")
    dev_candidates: dict[str, list[tuple[int, int, int]]] = {}
    counts = {
        "documents": 0,
        "train_documents": 0,
        "dev_documents": 0,
        "train": 0,
        "dev": 0,
    }

    connection = sqlite3.connect(database_tmp)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE documents (
                document_key INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL,
                source_split TEXT NOT NULL,
                dataset_split TEXT NOT NULL,
                category TEXT NOT NULL,
                image_zip TEXT NOT NULL,
                image_member TEXT NOT NULL,
                label_zip TEXT NOT NULL,
                label_member TEXT NOT NULL,
                sample_count INTEGER NOT NULL CHECK(sample_count >= 0)
            );
            CREATE UNIQUE INDEX documents_source_id
                ON documents(source_split, document_id);
            """
        )
        for row in _manifest_rows(options.training_manifest):
            if row.get("source_split") != "training":
                raise ValueError("Training manifest contains non-Training data")
            split = str(row.get("dataset_split"))
            if split not in {"train", "dev"}:
                raise ValueError(f"Invalid Training dataset split: {split}")
            document_key = _insert_document(connection, row)
            sample_count = int(row.get("sample_count", 0))
            counts["documents"] += 1
            counts[f"{split}_documents"] += 1
            if not sample_count:
                continue
            if split == "train":
                train_documents.append(document_key)
                counts["train"] += sample_count
                train_ends.append(counts["train"])
            else:
                priority = stable_number(f"dev:{row.get('category')}:{row['id']}")
                category = str(row.get("category", "unknown"))
                dev_candidates.setdefault(category, []).append(
                    (priority, document_key, sample_count)
                )
            if counts["documents"] % 10_000 == 0:
                connection.commit()
                print(f"[prepare] indexed {counts['documents']:,} Training documents", flush=True)
        connection.commit()
    finally:
        connection.close()

    dev_documents: array[int] = array("q")
    dev_ends: array[int] = array("q")
    ordered_dev = {
        category: deque(sorted(candidates))
        for category, candidates in dev_candidates.items()
    }
    while counts["dev"] < options.dev_max_samples and any(ordered_dev.values()):
        for category in sorted(ordered_dev):
            candidates = ordered_dev[category]
            if not candidates or counts["dev"] >= options.dev_max_samples:
                continue
            _, document_key, sample_count = candidates.popleft()
            dev_documents.append(document_key)
            counts["dev"] += sample_count
            dev_ends.append(counts["dev"])

    if not train_ends or not dev_ends:
        database_tmp.unlink(missing_ok=True)
        raise ValueError("Training and development streaming indexes must both be non-empty")

    database_tmp.replace(database)
    _save_array(output / "train_documents.npy", train_documents)
    _save_array(output / "train_sample_ends.npy", train_ends)
    _save_array(output / "dev_documents.npy", dev_documents)
    _save_array(output / "dev_sample_ends.npy", dev_ends)
    return counts


def _decode_image(payload: bytes, source: str) -> np.ndarray[Any, Any]:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {source}")
    return image


def _write_image(path: Path, image: np.ndarray[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write final evaluation image: {path}")


def _build_final(options: PrepareOptions, dataset_root: Path) -> dict[str, int]:
    rows = sorted(
        _manifest_rows(options.validation_manifest),
        key=lambda row: stable_number(f"final:{row.get('category')}:{row['id']}"),
    )[: options.final_documents]
    variants: tuple[ImageVariant, ...] = ("clean", "camera", "camera_filtered")
    labels: dict[ImageVariant, list[str]] = {variant: [] for variant in variants}
    count = 0

    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}

        def archive(relative: str) -> zipfile.ZipFile:
            path = (dataset_root / relative).resolve()
            if path not in archives:
                archives[path] = stack.enter_context(zipfile.ZipFile(path))
            return archives[path]

        for row in rows:
            if row.get("source_split") != "validation" or row.get("dataset_split") != "final":
                raise ValueError("Validation manifest may only contain final data")
            image = _decode_image(
                archive(str(row["image_zip"])).read(str(row["image_member"])),
                f"{row['image_zip']}::{row['image_member']}",
            )
            label = json.loads(
                archive(str(row["label_zip"])).read(str(row["label_member"]))
            )
            annotations = label.get("annotations", []) if isinstance(label, dict) else []
            valid_indices = valid_annotation_indices(annotations, options.max_text_length)
            for annotation_index in valid_indices:
                annotation = annotations[int(annotation_index)]
                if not isinstance(annotation, dict):
                    continue
                text = normalize_label(annotation.get("annotation.text"))
                crop = crop_annotation(
                    image, annotation.get("annotation.bbox", []), options.padding_ratio
                )
                if not text or crop is None or len(text) > options.max_text_length:
                    continue
                annotation_id = str(annotation.get("id", annotation_index))
                for variant in variants:
                    relative = (
                        Path("images")
                        / "final"
                        / variant
                        / str(row["id"])
                        / f"{annotation_id}.jpg"
                    )
                    transformed = make_variant(
                        crop,
                        variant,
                        stable_number(f"{row['id']}:{annotation_id}:{variant}:final"),
                    )
                    _write_image(options.output / relative, transformed)
                    labels[variant].append(f"{relative.as_posix()}\t{text}")
                count += 1

    for variant, lines in labels.items():
        path = options.output / f"final_{variant}.txt"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if not count:
        raise ValueError("Final evaluation selection produced no valid crops")
    return {"final_documents": len(rows), "final_per_variant": count}


def prepare_streaming_dataset(options: PrepareOptions) -> dict[str, Any]:
    if options.final_documents < 1 or options.dev_max_samples < 1:
        raise ValueError("final_documents and dev_max_samples must be >= 1")
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / "prepare_state.json").unlink(missing_ok=True)
    final_images = options.output.resolve() / "images" / "final"
    if final_images.is_dir() and final_images.is_relative_to(options.output.resolve()):
        shutil.rmtree(final_images)
    dataset_root = resolve_dataset_root(options.data_root)
    index_counts = _build_index(options, dataset_root)
    final_counts = _build_final(options, dataset_root)
    state = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "training_manifest_sha256": _sha256(options.training_manifest),
        "validation_manifest_sha256": _sha256(options.validation_manifest),
        "max_text_length": options.max_text_length,
        "padding_ratio": options.padding_ratio,
        "dev_max_samples": options.dev_max_samples,
        **index_counts,
        **final_counts,
    }
    state["fingerprint"] = hashlib.sha256(
        json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = options.output.resolve()
    for legacy in (output / "images" / "train", output / "images" / "dev"):
        if legacy.is_dir() and legacy.is_relative_to(output):
            shutil.rmtree(legacy)
    for name in ("train.txt", "dev.txt", "metadata.jsonl", "summary.json"):
        legacy_file = output / name
        if legacy_file.is_file() and legacy_file.is_relative_to(output):
            legacy_file.unlink()
    _atomic_json(options.output / "prepare_state.json", state)
    return state
