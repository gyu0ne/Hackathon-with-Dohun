from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

from ocr_training.checkpoints import (
    checkpoint_data_fingerprint,
    find_latest_checkpoint,
    is_complete_checkpoint,
)
from ocr_training.dataset import (
    assigned_variant,
    crop_annotation,
    document_split,
    make_variant,
    normalize_label,
)
from ocr_training.patch_paddleocr_data import patch_data_loader
from ocr_training.scripts.package_model import main as package_model
from ocr_training.scripts.train import build_command
from ocr_training.streaming_index import sample_location
from ocr_training.streaming_prepare import PrepareOptions, prepare_streaming_dataset
from ocr_training.vl_benchmark import markdown_to_text, score


def test_aihub_source_contract_is_enforced() -> None:
    assert document_split(
        {"source_split": "training", "dataset_split": "train"}
    ) == "train"
    assert document_split(
        {"source_split": "training", "dataset_split": "dev"}
    ) == "dev"
    assert document_split(
        {"source_split": "validation", "dataset_split": "final"}
    ) == "final"


def test_validation_cannot_become_training_data() -> None:
    try:
        document_split({"source_split": "validation", "dataset_split": "train"})
    except ValueError as error:
        assert "Invalid source/dataset split contract" in str(error)
    else:
        raise AssertionError("Validation leaked into training")


def _write_checkpoint(
    prefix: Path,
    epoch: int,
    step: int,
    complete: bool = True,
    fingerprint: str | None = None,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".pdparams").write_bytes(b"params")
    prefix.with_suffix(".pdopt").write_bytes(b"optimizer")
    prefix.with_suffix(".states").write_bytes(
        pickle.dumps(
            {
                "epoch": epoch,
                "global_step": step,
                "best_model_dict": {"data_fingerprint": fingerprint},
            }
        )
    )
    if complete:
        prefix.with_suffix(".complete").write_text(str(step), encoding="utf-8")


def test_resume_uses_newest_complete_generation(tmp_path: Path) -> None:
    latest = tmp_path / "latest"
    complete = tmp_path / "recovery_step_300"
    incomplete = tmp_path / "recovery_step_400"
    _write_checkpoint(latest, epoch=1, step=200)
    _write_checkpoint(complete, epoch=1, step=300)
    _write_checkpoint(incomplete, epoch=1, step=400, complete=False)

    assert not is_complete_checkpoint(incomplete)
    assert find_latest_checkpoint(tmp_path) == complete


def test_checkpoint_keeps_data_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest"
    _write_checkpoint(checkpoint, epoch=2, step=100, fingerprint="dataset-a")
    assert checkpoint_data_fingerprint(checkpoint) == "dataset-a"


def test_resume_command_uses_checkpoint_instead_of_pretrained(tmp_path: Path) -> None:
    checkpoint = tmp_path / "recovery_step_300"
    command = build_command(
        Path("config.yml"),
        Path("pretrained.pdparams"),
        tmp_path,
        checkpoint,
        epochs=20,
        batch_size=32,
        save_batch_step=100,
    )
    assert f"Global.checkpoints={checkpoint}" in command
    assert not any(value.startswith("Global.pretrained_model=") for value in command)
    assert "Train.sampler.batch_size=32" in command
    assert "Train.sampler.resume_batch=0" in command


def test_streaming_sample_location_uses_compact_document_arrays() -> None:
    documents = np.array([11, 22, 33], dtype=np.int64)
    ends = np.array([2, 5, 6], dtype=np.int64)
    assert sample_location(documents, ends, 0) == (11, 0)
    assert sample_location(documents, ends, 2) == (22, 0)
    assert sample_location(documents, ends, 5) == (33, 0)


def test_paddle_data_patch_registers_streaming_components(tmp_path: Path) -> None:
    target = tmp_path / "data_init.py"
    target.write_text(
        "from ppocr.data.multi_scale_sampler import MultiScaleSampler\n"
        "support_dict = [\n"
        '        "LaTeXOCRDataSet",\n'
        "]\n",
        encoding="utf-8",
    )
    assert patch_data_loader(target)
    patched = target.read_text(encoding="utf-8")
    assert "AIHubStreamDataSet" in patched
    assert "DocumentBatchSampler" in patched


def test_prepare_writes_streaming_index_without_train_crops(tmp_path: Path) -> None:
    data_root = tmp_path / "AIHub"
    training = data_root / "Training"
    validation = data_root / "Validation"
    training.mkdir(parents=True)
    validation.mkdir()
    image = np.full((40, 120, 3), 255, dtype=np.uint8)
    ok, encoded = __import__("cv2").imencode(".jpg", image)
    assert ok

    def write_sources(directory: Path, document_id: str) -> tuple[Path, Path]:
        image_zip = directory / "images.zip"
        label_zip = directory / "labels.zip"
        with zipfile.ZipFile(image_zip, "w") as archive:
            archive.writestr(f"source/{document_id}.jpg", encoded.tobytes())
        label = {
            "annotations": [
                {"id": 1, "annotation.text": "접수 2026", "annotation.bbox": [2, 2, 80, 20]}
            ]
        }
        with zipfile.ZipFile(label_zip, "w") as archive:
            archive.writestr(f"labels/{document_id}.json", json.dumps(label))
        return image_zip, label_zip

    train_image_zip, train_label_zip = write_sources(training, "train-1")
    dev_image_zip, dev_label_zip = write_sources(training, "dev-1")
    final_image_zip, final_label_zip = write_sources(validation, "final-1")
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    def row(document_id: str, split: str, image_zip: Path, label_zip: Path) -> dict[str, object]:
        source_split = "validation" if split == "final" else "training"
        return {
            "id": document_id,
            "category": "일반행정",
            "source_split": source_split,
            "dataset_split": split,
            "image_zip": image_zip.relative_to(data_root).as_posix(),
            "image_member": f"source/{document_id}.jpg",
            "label_zip": label_zip.relative_to(data_root).as_posix(),
            "label_member": f"labels/{document_id}.json",
            "sample_count": 1,
        }

    training_rows = [
        row("train-1", "train", train_image_zip, train_label_zip),
        row("dev-1", "dev", dev_image_zip, dev_label_zip),
    ]
    validation_rows = [row("final-1", "final", final_image_zip, final_label_zip)]
    training_manifest = manifests / "training.jsonl"
    validation_manifest = manifests / "validation.jsonl"
    training_manifest.write_text(
        "".join(json.dumps(value) + "\n" for value in training_rows), encoding="utf-8"
    )
    validation_manifest.write_text(
        "".join(json.dumps(value) + "\n" for value in validation_rows), encoding="utf-8"
    )
    output = tmp_path / "prepared"
    state = prepare_streaming_dataset(
        PrepareOptions(
            data_root=data_root,
            training_manifest=training_manifest,
            validation_manifest=validation_manifest,
            output=output,
            final_documents=1,
            dev_max_samples=1,
        )
    )
    assert state["train"] == 1
    assert state["dev"] == 1
    assert (output / "documents.sqlite3").is_file()
    assert not (output / "images" / "train").exists()
    assert not (output / "images" / "dev").exists()
    assert (output / "images" / "final" / "clean").is_dir()


def test_completed_export_is_packaged_for_transfer(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    model = Path("ocr_training/artifacts/inference/public_doc_rec")
    model.mkdir(parents=True)
    (model / "inference.yml").write_text("Global: {}\n", encoding="utf-8")
    (model / "inference.json").write_text("{}\n", encoding="utf-8")
    (model / "inference.pdiparams").write_bytes(b"weights")
    data = Path("ocr_training/data")
    data.mkdir(parents=True)
    (data / "prepare_state.json").write_text(
        json.dumps({"fingerprint": "dataset-a"}), encoding="utf-8"
    )
    checkpoints = Path("ocr_training/artifacts/checkpoints")
    checkpoints.mkdir(parents=True)
    (checkpoints / "training_complete.json").write_text(
        json.dumps({"data_fingerprint": "dataset-a", "epochs": 20}),
        encoding="utf-8",
    )

    package_model()

    output = Path("ocr_training/artifacts/deliverables/public_doc_rec.zip")
    assert output.is_file()
    assert output.with_suffix(".zip.sha256").is_file()
    with zipfile.ZipFile(output) as bundle:
        names = set(bundle.namelist())
    assert "public_doc_rec/model/inference.yml" in names
    assert "public_doc_rec/model/inference.json" in names
    assert "public_doc_rec/model/inference.pdiparams" in names
    assert "public_doc_rec/model_manifest.json" in names
    assert "public_doc_rec/checksums.sha256" in names


def test_variant_assignment_is_stable_and_known() -> None:
    first = assigned_variant("doc-1", "annotation-9")
    assert first == assigned_variant("doc-1", "annotation-9")
    assert first in {"clean", "camera"}


def test_crop_annotation_clamps_padding_to_image() -> None:
    image = np.zeros((30, 50, 3), dtype=np.uint8)
    crop = crop_annotation(image, [0, 0, 20, 10], padding_ratio=0.1)
    assert crop is not None
    assert crop.shape[0] == 12
    assert crop.shape[1] == 22


def test_camera_variants_keep_crop_shape() -> None:
    image = np.full((32, 120, 3), 220, dtype=np.uint8)
    for variant in ("clean", "camera", "camera_filtered"):
        assert make_variant(image, variant, seed=42).shape == image.shape


def test_training_variant_never_uses_filter_that_hurt_baseline() -> None:
    variants = {assigned_variant(f"doc-{index}", "1") for index in range(100)}
    assert variants == {"clean", "camera"}


def test_labels_are_normalized_for_paddle_list_format() -> None:
    assert normalize_label("  접수\t 번호\n2026  ") == "접수 번호 2026"


def test_vl_markdown_is_converted_before_scoring() -> None:
    hypothesis = markdown_to_text("# 시행일\n- **2026-08-12** [근거](https://example.invalid)")
    measured = score("시행일 2026-08-12 근거", hypothesis, 120.0)
    assert "시행일" in hypothesis
    assert measured.word_f1 == 1.0
    assert measured.numeric_f1 == 1.0
