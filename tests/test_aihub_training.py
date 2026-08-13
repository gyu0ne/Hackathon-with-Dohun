from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

from ocr_training.aihub import build_split_rows, discover_archives


def _write_zip(path: Path, members: Mapping[str, bytes | object]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            payload = (
                value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            )
            archive.writestr(name, payload)


def _label(name: str, category: str) -> dict[str, object]:
    return {
        "images": [{"image.file.name": name, "image.category": category}],
        "annotations": [
            {"id": 1, "annotation.text": "접수 2026", "annotation.bbox": [1, 2, 30, 10]}
        ],
    }


def test_archive_discovery_uses_contents_not_aihub_zip_filename(tmp_path: Path) -> None:
    training = tmp_path / "Training"
    training.mkdir()
    _write_zip(training / "part-a.zip", {"02.원천데이터(Jpg)/업무/doc-1.jpg": b"jpeg"})
    _write_zip(
        training / "part-b.zip",
        {"01.라벨링데이터(Json)/업무/doc-1.json": _label("doc-1.jpg", "일반행정")},
    )

    archives = discover_archives(tmp_path, "training")
    assert archives.image_archives == (training / "part-a.zip",)
    assert archives.label_archives == (training / "part-b.zip",)


def test_training_is_stratified_into_train_and_dev(tmp_path: Path) -> None:
    training = tmp_path / "Training"
    training.mkdir()
    images = {f"02.원천데이터(Jpg)/업무/doc-{index}.jpg": b"jpeg" for index in range(20)}
    labels = {
        f"01.라벨링데이터(Json)/업무/doc-{index}.json": _label(
            f"doc-{index}.jpg", "일반행정"
        )
        for index in range(20)
    }
    _write_zip(training / "images.zip", images)
    _write_zip(training / "labels.zip", labels)

    rows, stats = build_split_rows(discover_archives(tmp_path, "training"), tmp_path)
    assert {row["dataset_split"] for row in rows} == {"train", "dev"}
    assert stats["dataset_split_counts"] == {"train": 18, "dev": 2, "final": 0}
    assert all(row["source_split"] == "training" for row in rows)
    assert all(row["sample_count"] == 1 for row in rows)


def test_aihub_validation_is_final_only(tmp_path: Path) -> None:
    validation = tmp_path / "Validation"
    validation.mkdir()
    _write_zip(validation / "anything-1.zip", {"source/doc.jpg": b"jpeg"})
    _write_zip(validation / "anything-2.zip", {"labels/doc.json": _label("doc.jpg", "민원")})

    rows, stats = build_split_rows(discover_archives(tmp_path, "validation"), tmp_path)
    assert [row["dataset_split"] for row in rows] == ["final"]
    assert rows[0]["splits"] == ["final"]
    assert stats["dataset_split_counts"] == {"train": 0, "dev": 0, "final": 1}


def test_duplicate_basenames_are_paired_by_actual_aihub_relative_path(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "Validation"
    validation.mkdir()
    _write_zip(
        validation / "sources.zip",
        {
            "02.원천데이터(Jpg)/일반행정/기관/2026/doc.jpg": b"first",
            "02.원천데이터(Jpg)/보건복지/기관/2026/doc.jpg": b"second",
        },
    )
    _write_zip(
        validation / "labels.zip",
        {
            "01.라벨링데이터(Json)/일반행정/기관/2026/doc.json": _label(
                "doc.jpg", "일반행정"
            )
        },
    )

    rows, stats = build_split_rows(
        discover_archives(tmp_path, "validation"), tmp_path
    )
    assert rows[0]["image_member"].startswith("02.원천데이터(Jpg)/일반행정/")
    assert stats["ambiguous_image_names"] == 1
