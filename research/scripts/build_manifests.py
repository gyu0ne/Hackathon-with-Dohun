from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

SEED = "kcode2026-v1"
SLOT_PATTERNS = {
    "date": re.compile(r"(?:신청|접수|기간|20\d{2}\s*[.\-/년])"),
    "amount": re.compile(r"(?:지원금|금액|\d[\d,]*\s*(?:원|만원|억원))"),
    "contact": re.compile(r"(?:문의|연락처|0\d{1,2}[- )]\d{3,4}-\d{4})"),
}


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _basename(member: str) -> str:
    return PurePosixPath(member.replace("\\", "/")).name


def _portable_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_ocr_rows(
    source_zip: Path,
    label_zip: Path,
    test_data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(source_zip) as source_archive:
        image_members = {
            _basename(info.filename): info.filename
            for info in source_archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        }

    rows: list[dict[str, Any]] = []
    label_basenames: set[str] = set()
    with zipfile.ZipFile(label_zip) as label_archive:
        for info in label_archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            label_basenames.add(Path(_basename(info.filename)).stem)
            data = json.loads(label_archive.read(info))
            image_info = data["images"][0]
            image_name = str(image_info["image.file.name"])
            image_member = image_members.get(image_name)
            if image_member is None:
                continue
            texts = [
                str(annotation.get("annotation.text", ""))
                for annotation in data.get("annotations", [])
            ]
            joined_text = " ".join(texts)
            slot_candidates = [
                slot for slot, pattern in SLOT_PATTERNS.items() if pattern.search(joined_text)
            ]
            rows.append(
                {
                    "id": Path(image_name).stem,
                    "category": str(image_info.get("image.category", "unknown")),
                    "image_zip": _portable_path(source_zip, test_data_root),
                    "image_member": image_member,
                    "label_zip": _portable_path(label_zip, test_data_root),
                    "label_member": info.filename,
                    "annotation_count": len(data.get("annotations", [])),
                    "slot_candidates": slot_candidates,
                    "splits": [],
                    "slot_benchmark": [],
                    "golden_review_candidate": False,
                }
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    for category_rows in grouped.values():
        ordered = sorted(category_rows, key=lambda row: _stable_key(row["id"]))
        for index, row in enumerate(ordered):
            if index < 2:
                row["splits"].append("smoke")
            if index < 10:
                row["splits"].append("regression")
            if index < 50:
                row["splits"].append("research")

    for slot in SLOT_PATTERNS:
        candidates = sorted(
            (row for row in rows if slot in row["slot_candidates"]),
            key=lambda row: _stable_key(f"{slot}:{row['id']}"),
        )
        for row in candidates[:10]:
            row["slot_benchmark"].append(slot)

    golden_candidates = sorted(
        (row for row in rows if row["slot_benchmark"]),
        key=lambda row: _stable_key(f"golden:{row['id']}"),
    )
    for row in golden_candidates[:8]:
        row["golden_review_candidate"] = True

    image_basenames = {Path(name).stem for name in image_members}
    stats = {
        "paired": len(rows),
        "label_without_image": sorted(label_basenames - image_basenames),
        "image_without_label": sorted(image_basenames - label_basenames),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "split_counts": {
            split: sum(split in row["splits"] for row in rows)
            for split in ("smoke", "regression", "research")
        },
        "slot_counts": {
            slot: sum(slot in row["slot_benchmark"] for row in rows)
            for slot in SLOT_PATTERNS
        },
        "golden_review_candidates": sum(row["golden_review_candidate"] for row in rows),
    }
    return sorted(rows, key=lambda row: row["id"]), stats


def _summary_row(
    archive_path: Path,
    member: str,
    data: dict[str, Any],
    split: str,
    selected: bool,
    test_data_root: Path,
) -> dict[str, Any]:
    annotation = data.get("annotation", {})
    corrected = annotation.get("corrected_summary", {}) or {}
    type5 = corrected.get("corrected_type5", {}) or {}
    corrected_all = corrected.get("corrected_all", {}) or {}
    return {
        "id": Path(_basename(member)).stem,
        "split": split,
        "zip": _portable_path(archive_path, test_data_root),
        "member": member,
        "error_count": len(type5.get("errors", [])),
        "has_corrected_all": bool(corrected_all.get("text")),
        "selected": selected,
    }


def build_summary_rows(
    development_zip: Path,
    validation_zip: Path,
    test_data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    development_members: list[tuple[str, dict[str, Any]]] = []
    with zipfile.ZipFile(development_zip) as archive:
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith(".json"):
                development_members.append((info.filename, json.loads(archive.read(info))))

    selected_development = {
        member
        for member, _ in sorted(
            development_members,
            key=lambda item: _stable_key(_basename(item[0])),
        )[:2000]
    }
    rows = [
        _summary_row(
            development_zip,
            member,
            data,
            "development",
            member in selected_development,
            test_data_root,
        )
        for member, data in development_members
    ]

    with zipfile.ZipFile(validation_zip) as archive:
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith(".json"):
                rows.append(
                    _summary_row(
                        validation_zip,
                        info.filename,
                        json.loads(archive.read(info)),
                        "validation",
                        True,
                        test_data_root,
                    )
                )

    stats = {
        "development_documents": sum(row["split"] == "development" for row in rows),
        "development_selected_documents": sum(
            row["split"] == "development" and row["selected"] for row in rows
        ),
        "validation_documents": sum(row["split"] == "validation" for row in rows),
        "validation_selected_documents": sum(
            row["split"] == "validation" and row["selected"] for row in rows
        ),
        "selected_claim_cases": 2
        * sum(row["selected"] for row in rows),
    }
    return sorted(rows, key=lambda row: (row["split"], row["id"])), stats


def _find_unique(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and path.name == filename]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename!r}, found {len(matches)}")
    return matches[0]


def create_manifests(test_data_root: Path, output_dir: Path) -> dict[str, Any]:
    ocr_root = test_data_root / "공공행정문서 OCR"
    summary_root = test_data_root / "157.추상 요약 사실성 검증 데이터"
    source_zip = _find_unique(ocr_root, "[원천]validation.zip")
    label_zip = _find_unique(ocr_root, "[라벨]validation.zip")
    development_zip = _find_unique(
        summary_root, "TL_기계요약문_사실전달형_corrected_type5.zip"
    )
    validation_zip = _find_unique(
        summary_root, "VL_기계요약문_사실전달형_corrected_type5.zip"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_rows, ocr_stats = build_ocr_rows(source_zip, label_zip, test_data_root)
    summary_rows, summary_stats = build_summary_rows(
        development_zip, validation_zip, test_data_root
    )
    _write_jsonl(output_dir / "ocr_manifest.jsonl", ocr_rows)
    _write_jsonl(output_dir / "summary_manifest.jsonl", summary_rows)
    stats = {"seed": SEED, "ocr": ocr_stats, "summary": summary_stats}
    (output_dir / "manifest_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic AI-Hub test manifests")
    parser.add_argument("--test-data", type=Path, default=Path("Test Data"))
    parser.add_argument("--output", type=Path, default=Path("research/manifests"))
    args = parser.parse_args()
    stats = create_manifests(args.test_data.resolve(), args.output.resolve())
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
