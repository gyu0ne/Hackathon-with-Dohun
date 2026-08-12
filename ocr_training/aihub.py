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


@dataclass(frozen=True)
class ArchiveSet:
    split: SourceSplit
    directory: Path
    image_archives: tuple[Path, ...]
    label_archives: tuple[Path, ...]


def stable_number(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _basename(member: str) -> str:
    return PurePosixPath(member.replace("\\", "/")).name


def _relative_stem(member: str) -> str:
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
        if path.is_dir() and any(alias in path.name.casefold() for alias in aliases)
    ]
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {split!r} directory under {root}; found: {names}"
        )
    return matches[0]


def resolve_dataset_root(path: Path) -> Path:
    """Accept either the AI-Hub dataset directory or its immediate parent."""
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AI-Hub data root does not exist: {root}")
    validation_aliases = ("validation", "valid", "검증")
    if any(
        child.is_dir()
        and any(alias in child.name.casefold() for alias in validation_aliases)
        for child in root.iterdir()
    ):
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
        raise FileNotFoundError(
            f"Could not uniquely locate the AI-Hub OCR dataset directory under {root}"
        )
    return candidates[0]


def discover_archives(dataset_root: Path, split: SourceSplit) -> ArchiveSet:
    split_directory = _find_split_directory(dataset_root, split)
    image_archives: list[Path] = []
    label_archives: list[Path] = []
    for archive_path in sorted(split_directory.rglob("*.zip")):
        image_count = 0
        label_count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                lowered = info.filename.casefold()
                image_count += lowered.endswith(IMAGE_SUFFIXES)
                label_count += lowered.endswith(".json")
        if image_count and label_count:
            raise ValueError(f"Mixed image/label ZIP is not supported: {archive_path}")
        if image_count:
            image_archives.append(archive_path)
        elif label_count:
            label_archives.append(archive_path)
    if not image_archives or not label_archives:
        raise FileNotFoundError(
            f"{split_directory} must contain at least one image ZIP and one JSON label ZIP"
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
    relative_index: dict[str, tuple[Path, str]] = {}
    basename_index: dict[str, tuple[Path, str]] = {}
    ambiguous: set[str] = set()
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.casefold().endswith(IMAGE_SUFFIXES):
                    continue
                relative_key = _relative_stem(info.filename)
                if relative_key in relative_index:
                    raise ValueError(f"Duplicate relative image path: {relative_key}")
                relative_index[relative_key] = (archive_path, info.filename)
                name = _basename(info.filename)
                if name in basename_index:
                    ambiguous.add(name)
                    basename_index.pop(name, None)
                elif name not in ambiguous:
                    basename_index[name] = (archive_path, info.filename)
    return relative_index, basename_index, ambiguous


def build_split_rows(
    archives: ArchiveSet,
    dataset_root: Path,
    dev_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < dev_fraction < 0.5:
        raise ValueError("dev_fraction must be between 0 and 0.5")
    relative_images, basename_images, ambiguous = _image_index(archives.image_archives)
    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    seen_ids: set[str] = set()
    label_count = 0
    for label_path in archives.label_archives:
        with zipfile.ZipFile(label_path) as labels:
            for info in labels.infolist():
                if info.is_dir() or not info.filename.casefold().endswith(".json"):
                    continue
                label_count += 1
                payload = json.loads(labels.read(info))
                image_records = payload.get("images", [])
                if not image_records:
                    missing_images.append(info.filename)
                    continue
                image_record = image_records[0]
                image_name = _basename(str(image_record.get("image.file.name", "")))
                matched = relative_images.get(_relative_stem(info.filename))
                if matched is None:
                    matched = basename_images.get(image_name)
                if matched is None:
                    missing_images.append(image_name or info.filename)
                    continue
                document_id = Path(image_name).stem
                if document_id in seen_ids:
                    document_id = _relative_stem(info.filename).replace("/", "__")
                if document_id in seen_ids:
                    raise ValueError(f"Duplicate AI-Hub document id: {document_id}")
                seen_ids.add(document_id)
                image_path, image_member = matched
                rows.append(
                    {
                        "id": document_id,
                        "category": str(image_record.get("image.category", "unknown")),
                        "source_split": archives.split,
                        "dataset_split": "final" if archives.split == "validation" else None,
                        "splits": ["final"] if archives.split == "validation" else [],
                        "image_zip": _portable_path(image_path, dataset_root),
                        "image_member": image_member,
                        "label_zip": _portable_path(label_path, dataset_root),
                        "label_member": info.filename,
                        "annotation_count": len(payload.get("annotations", [])),
                    }
                )

    if archives.split == "training":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["category"]].append(row)
        for category_rows in grouped.values():
            category_rows.sort(key=lambda row: stable_number(f"{row['category']}:{row['id']}"))
            dev_count = (
                max(1, round(len(category_rows) * dev_fraction))
                if len(category_rows) > 1
                else 0
            )
            dev_ids = {row["id"] for row in category_rows[:dev_count]}
            for row in category_rows:
                row["dataset_split"] = "dev" if row["id"] in dev_ids else "train"

    counts = {
        split: sum(row["dataset_split"] == split for row in rows)
        for split in ("train", "dev", "final")
    }
    stats = {
        "source_split": archives.split,
        "image_archives": [_portable_path(path, dataset_root) for path in archives.image_archives],
        "label_archives": [_portable_path(path, dataset_root) for path in archives.label_archives],
        "images_indexed": len(relative_images),
        "labels_seen": label_count,
        "paired": len(rows),
        "ambiguous_image_names": len(ambiguous),
        "ambiguous_examples": sorted(ambiguous)[:20],
        "missing_images": len(missing_images),
        "missing_examples": missing_images[:20],
        "dataset_split_counts": counts,
    }
    return sorted(rows, key=lambda row: row["id"]), stats


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
