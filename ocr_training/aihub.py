from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

SourceSplit = Literal["training", "validation"]
DatasetSplit = Literal["train", "dev", "final"]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
PROGRESS_EVERY_LABELS = 5000


@dataclass(frozen=True)
class ArchiveSet:
    split: SourceSplit
    directory: Path
    image_archives: tuple[Path, ...]
    label_archives: tuple[Path, ...]


def stable_number(value: str) -> int:
    """Return a stable deterministic integer for splitting/sampling."""
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        "big",
    )


def _portable_path(path: Path, root: Path) -> str:
    """Prefer a path relative to dataset root, falling back to an absolute POSIX path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _basename(member: str) -> str:
    """Return the basename of a ZIP member independent of slash style."""
    return PurePosixPath(member.replace("\\", "/")).name


def _relative_stem(member: str) -> str:
    """
    Build a normalized relative key for matching image/label members.

    AI-Hub ZIPs commonly add one dataset-root directory inside the archive.
    Dropping the first path component makes image and label members easier to pair.
    """
    parts = PurePosixPath(member.replace("\\", "/")).parts
    relative = PurePosixPath(*parts[1:]) if len(parts) > 1 else PurePosixPath(*parts)
    return relative.with_suffix("").as_posix().casefold()


def _find_split_directory(root: Path, split: SourceSplit) -> Path:
    aliases = {
        "training": ("training", "train", "학습"),
        "validation": ("validation", "valid", "검증"),
    }[split]

    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and any(alias in path.name.casefold() for alias in aliases)
    ]

    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {split!r} directory under {root}; found: {names}"
        )

    return matches[0]


def resolve_dataset_root(path: Path) -> Path:
    """
    Accept either the AI-Hub dataset directory or its immediate parent.

    A valid dataset root must contain a Validation-like directory directly,
    or exactly one immediate child that contains such a directory.
    """
    root = path.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"AI-Hub data root does not exist: {root}")

    validation_aliases = ("validation", "valid", "검증")

    if any(
        child.is_dir()
        and any(alias in child.name.casefold() for alias in validation_aliases)
        for child in root.iterdir()
    ):
        print(f"[dataset] resolved root: {root}", flush=True)
        return root

    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and any(
            grandchild.is_dir()
            and any(alias in grandchild.name.casefold() for alias in validation_aliases)
            for grandchild in child.iterdir()
        )
    ]

    if len(candidates) != 1:
        candidate_names = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            "Could not uniquely locate the AI-Hub OCR dataset directory "
            f"under {root}; candidates: {candidate_names}"
        )

    resolved = candidates[0]
    print(f"[dataset] resolved root: {resolved}", flush=True)
    return resolved


def _inspect_zip_contents(archive_path: Path) -> tuple[int, int]:
    """Return (image_member_count, json_member_count) for one ZIP archive."""
    image_count = 0
    label_count = 0

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue

                lowered = info.filename.casefold()
                if lowered.endswith(IMAGE_SUFFIXES):
                    image_count += 1
                if lowered.endswith(".json"):
                    label_count += 1
    except (zipfile.BadZipFile, OSError) as exc:
        raise RuntimeError(f"Could not inspect ZIP archive: {archive_path}") from exc

    return image_count, label_count


def discover_archives(dataset_root: Path, split: SourceSplit) -> ArchiveSet:
    split_directory = _find_split_directory(dataset_root, split)

    print(
        f"[discover:{split}] scanning ZIP files under: {split_directory}",
        flush=True,
    )

    image_archives: list[Path] = []
    label_archives: list[Path] = []
    zip_paths = sorted(split_directory.rglob("*.zip"))

    if not zip_paths:
        raise FileNotFoundError(f"No ZIP archives found under {split_directory}")

    for index, archive_path in enumerate(zip_paths, start=1):
        image_count, label_count = _inspect_zip_contents(archive_path)

        if image_count and label_count:
            raise ValueError(
                "Mixed image/label ZIP is not supported: "
                f"{archive_path} (images={image_count}, json={label_count})"
            )

        if image_count:
            image_archives.append(archive_path)
        elif label_count:
            label_archives.append(archive_path)

        if index == 1 or index == len(zip_paths) or index % 10 == 0:
            print(
                f"[discover:{split}] checked {index}/{len(zip_paths)} ZIPs "
                f"(image ZIPs={len(image_archives)}, label ZIPs={len(label_archives)})",
                flush=True,
            )

    if not image_archives or not label_archives:
        raise FileNotFoundError(
            f"{split_directory} must contain at least one image ZIP "
            "and one JSON label ZIP"
        )

    print(
        f"[discover:{split}] complete: "
        f"{len(image_archives)} image ZIP(s), {len(label_archives)} label ZIP(s)",
        flush=True,
    )

    return ArchiveSet(
        split=split,
        directory=split_directory,
        image_archives=tuple(image_archives),
        label_archives=tuple(label_archives),
    )


def _image_index(
    archives: Iterable[Path],
) -> tuple[
    dict[str, tuple[Path, str]],
    dict[str, tuple[Path, str]],
    set[str],
]:
    archive_list = tuple(archives)

    relative_index: dict[str, tuple[Path, str]] = {}
    basename_index: dict[str, tuple[Path, str]] = {}
    ambiguous: set[str] = set()

    print(
        f"[images] building index from {len(archive_list)} image ZIP(s)",
        flush=True,
    )

    for archive_index, archive_path in enumerate(archive_list, start=1):
        archive_image_count = 0

        try:
            with zipfile.ZipFile(archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or not info.filename.casefold().endswith(
                        IMAGE_SUFFIXES
                    ):
                        continue

                    archive_image_count += 1
                    relative_key = _relative_stem(info.filename)

                    if relative_key in relative_index:
                        existing_archive, existing_member = relative_index[relative_key]
                        raise ValueError(
                            "Duplicate relative image path detected: "
                            f"{relative_key}\n"
                            f"  existing: {existing_archive} :: {existing_member}\n"
                            f"  duplicate: {archive_path} :: {info.filename}"
                        )

                    relative_index[relative_key] = (archive_path, info.filename)

                    name = _basename(info.filename)
                    if name in basename_index:
                        ambiguous.add(name)
                        basename_index.pop(name, None)
                    elif name not in ambiguous:
                        basename_index[name] = (archive_path, info.filename)

        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError(
                f"Could not read image ZIP archive: {archive_path}"
            ) from exc

        print(
            f"[images] indexed ZIP {archive_index}/{len(archive_list)}: "
            f"{archive_path.name} ({archive_image_count:,} image(s); "
            f"total={len(relative_index):,})",
            flush=True,
        )

    print(
        f"[images] index complete: {len(relative_index):,} image(s), "
        f"{len(ambiguous):,} ambiguous basename(s)",
        flush=True,
    )

    return relative_index, basename_index, ambiguous


def _load_json_bytes(raw: bytes) -> Any:
    """
    Parse JSON directly from bytes.

    json.loads(bytes) performs JSON encoding detection (UTF-8/16/32 where valid),
    so we intentionally do not force UTF-8 here.
    """
    return json.loads(raw)


def build_split_rows(
    archives: ArchiveSet,
    dataset_root: Path,
    dev_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")

    print(
        f"[manifest:{archives.split}] starting manifest build",
        flush=True,
    )

    relative_images, basename_images, ambiguous = _image_index(
        archives.image_archives
    )

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    invalid_labels: list[str] = []
    empty_image_records: list[str] = []
    seen_ids: set[str] = set()

    label_count = 0

    for zip_index, label_path in enumerate(archives.label_archives, start=1):
        zip_json_count = 0
        zip_valid_count = 0
        zip_invalid_count = 0

        print(
            f"[manifest:{archives.split}] reading label ZIP "
            f"{zip_index}/{len(archives.label_archives)}: {label_path.name}",
            flush=True,
        )

        try:
            with zipfile.ZipFile(label_path) as labels:
                for info in labels.infolist():
                    if info.is_dir() or not info.filename.casefold().endswith(".json"):
                        continue

                    label_count += 1
                    zip_json_count += 1

                    try:
                        raw = labels.read(info)
                        payload = _load_json_bytes(raw)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        KeyError,
                        RuntimeError,
                    ) as exc:
                        zip_invalid_count += 1
                        invalid_labels.append(f"{label_path}::{info.filename}")

                        print(
                            "WARNING: skipping invalid JSON: "
                            f"{label_path} :: {info.filename} ({exc})",
                            flush=True,
                        )
                        continue

                    if not isinstance(payload, dict):
                        zip_invalid_count += 1
                        invalid_labels.append(f"{label_path}::{info.filename}")
                        print(
                            "WARNING: skipping JSON whose root is not an object: "
                            f"{label_path} :: {info.filename}",
                            flush=True,
                        )
                        continue

                    image_records = payload.get("images", [])
                    if not isinstance(image_records, list) or not image_records:
                        empty_image_records.append(info.filename)
                        continue

                    image_record = image_records[0]
                    if not isinstance(image_record, dict):
                        empty_image_records.append(info.filename)
                        continue

                    image_name = _basename(
                        str(image_record.get("image.file.name", ""))
                    )

                    matched = relative_images.get(_relative_stem(info.filename))
                    if matched is None and image_name:
                        matched = basename_images.get(image_name)

                    if matched is None:
                        missing_images.append(image_name or info.filename)
                        continue

                    document_id = Path(image_name).stem if image_name else ""
                    if not document_id:
                        document_id = _relative_stem(info.filename).replace("/", "__")

                    if document_id in seen_ids:
                        document_id = _relative_stem(info.filename).replace("/", "__")

                    if document_id in seen_ids:
                        raise ValueError(
                            f"Duplicate AI-Hub document id: {document_id}"
                        )

                    seen_ids.add(document_id)
                    image_path, image_member = matched

                    annotations = payload.get("annotations", [])
                    annotation_count = (
                        len(annotations) if isinstance(annotations, list) else 0
                    )

                    rows.append(
                        {
                            "id": document_id,
                            "category": str(
                                image_record.get("image.category", "unknown")
                            ),
                            "source_split": archives.split,
                            "dataset_split": (
                                "final"
                                if archives.split == "validation"
                                else None
                            ),
                            "splits": (
                                ["final"]
                                if archives.split == "validation"
                                else []
                            ),
                            "image_zip": _portable_path(
                                image_path,
                                dataset_root,
                            ),
                            "image_member": image_member,
                            "label_zip": _portable_path(
                                label_path,
                                dataset_root,
                            ),
                            "label_member": info.filename,
                            "annotation_count": annotation_count,
                        }
                    )

                    zip_valid_count += 1

                    if label_count % PROGRESS_EVERY_LABELS == 0:
                        print(
                            f"[manifest:{archives.split}] "
                            f"labels={label_count:,}, paired={len(rows):,}, "
                            f"invalid={len(invalid_labels):,}, "
                            f"missing-image={len(missing_images):,}",
                            flush=True,
                        )

        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError(
                f"Could not read label ZIP archive: {label_path}"
            ) from exc

        print(
            f"[manifest:{archives.split}] finished {label_path.name}: "
            f"json={zip_json_count:,}, paired={zip_valid_count:,}, "
            f"invalid={zip_invalid_count:,}",
            flush=True,
        )

    if archives.split == "training":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in rows:
            grouped[row["category"]].append(row)

        for category_rows in grouped.values():
            category_rows.sort(
                key=lambda row: stable_number(
                    f"{row['category']}:{row['id']}"
                )
            )

            dev_count = (
                max(1, round(len(category_rows) * dev_fraction))
                if len(category_rows) > 1
                else 0
            )

            dev_ids = {row["id"] for row in category_rows[:dev_count]}

            for row in category_rows:
                row["dataset_split"] = (
                    "dev" if row["id"] in dev_ids else "train"
                )

    counts = {
        split: sum(row["dataset_split"] == split for row in rows)
        for split in ("train", "dev", "final")
    }

    stats = {
        "source_split": archives.split,
        "image_archives": [
            _portable_path(path, dataset_root)
            for path in archives.image_archives
        ],
        "label_archives": [
            _portable_path(path, dataset_root)
            for path in archives.label_archives
        ],
        "images_indexed": len(relative_images),
        "labels_seen": label_count,
        "paired": len(rows),
        "invalid_labels": len(invalid_labels),
        "invalid_label_examples": invalid_labels[:20],
        "empty_image_records": len(empty_image_records),
        "empty_image_record_examples": empty_image_records[:20],
        "ambiguous_image_names": len(ambiguous),
        "ambiguous_examples": sorted(ambiguous)[:20],
        "missing_images": len(missing_images),
        "missing_examples": missing_images[:20],
        "dataset_split_counts": counts,
    }

    print(
        f"[manifest:{archives.split}] complete: "
        f"labels={label_count:,}, paired={len(rows):,}, "
        f"invalid={len(invalid_labels):,}, "
        f"missing-image={len(missing_images):,}, splits={counts}",
        flush=True,
    )

    return sorted(rows, key=lambda row: row["id"]), stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """
    Write JSONL atomically.

    A temporary file is written first, then replaced into place only after the
    complete output has been generated. This prevents a half-written manifest
    from looking valid after interruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    row_count = 0

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            for row in rows:
                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                row_count += 1

        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)

    print(
        f"[write] {path}: {row_count:,} row(s)",
        flush=True,
    )