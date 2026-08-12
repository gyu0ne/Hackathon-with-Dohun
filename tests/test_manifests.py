from __future__ import annotations

import json
import zipfile
from pathlib import Path

from research.scripts.build_manifests import _find_unique, build_ocr_rows, build_summary_rows


def _write_zip(path: Path, members: dict[str, object | bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            archive.writestr(name, payload)


def test_find_unique_treats_brackets_as_literal_filename(tmp_path: Path) -> None:
    expected = tmp_path / "nested" / "[원천]validation.zip"
    expected.parent.mkdir()
    expected.touch()

    assert _find_unique(tmp_path, "[원천]validation.zip") == expected


def test_ocr_manifest_pairs_by_declared_image_name(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    label_zip = tmp_path / "label.zip"
    _write_zip(source_zip, {"odd/path/doc-1.jpg": b"jpeg", "odd/path/orphan.jpg": b"jpeg"})
    label = {
        "images": [{"image.file.name": "doc-1.jpg", "image.category": "일반행정"}],
        "annotations": [{"annotation.text": "신청 기간 2026년"}],
    }
    _write_zip(label_zip, {"labels/doc-1.json": label, "labels/missing.json": label})

    rows, stats = build_ocr_rows(source_zip, label_zip, tmp_path)

    assert len(rows) == 2
    assert rows[0]["image_member"] == "odd/path/doc-1.jpg"
    assert "smoke" in rows[0]["splits"]
    assert stats["image_without_label"] == ["orphan"]


def test_summary_manifest_selects_validation_and_reads_type5(tmp_path: Path) -> None:
    development_zip = tmp_path / "development.zip"
    validation_zip = tmp_path / "validation.zip"
    sample = {
        "annotation": {
            "corrected_summary": {
                "corrected_type5": {"errors": [{"type": "type5"}]},
                "corrected_all": {"text": "corrected"},
            }
        }
    }
    _write_zip(development_zip, {"dev.json": sample})
    _write_zip(validation_zip, {"val.json": sample})

    rows, stats = build_summary_rows(development_zip, validation_zip, tmp_path)

    assert len(rows) == 2
    assert all(row["selected"] for row in rows)
    assert all(row["error_count"] == 1 for row in rows)
    assert stats["selected_claim_cases"] == 4
