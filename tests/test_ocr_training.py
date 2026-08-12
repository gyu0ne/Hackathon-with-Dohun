from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ocr_training.checkpoints import find_latest_checkpoint, is_complete_checkpoint
from ocr_training.dataset import (
    assigned_variant,
    crop_annotation,
    document_split,
    make_variant,
    normalize_label,
)
from ocr_training.scripts.train import build_command
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


def _write_checkpoint(prefix: Path, epoch: int, step: int, complete: bool = True) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".pdparams").write_bytes(b"params")
    prefix.with_suffix(".pdopt").write_bytes(b"optimizer")
    prefix.with_suffix(".states").write_bytes(
        pickle.dumps({"epoch": epoch, "global_step": step, "best_model_dict": {}})
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
