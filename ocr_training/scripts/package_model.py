from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluation_summary(directory: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(report, dict) and isinstance(report.get("aggregate"), dict):
            summary[path.stem] = report["aggregate"]
    return summary


def main() -> None:
    model_dir = Path("ocr_training/artifacts/inference/public_doc_rec")
    prepare_state_path = Path("ocr_training/data/prepare_state.json")
    completion_path = Path("ocr_training/artifacts/checkpoints/training_complete.json")
    evaluation_dir = Path("ocr_training/artifacts/evaluation")
    deliverables = Path("ocr_training/artifacts/deliverables")
    output = deliverables / "public_doc_rec.zip"

    if not model_dir.is_dir() or not any(path.is_file() for path in model_dir.rglob("*")):
        raise SystemExit(f"Exported inference model is missing or empty: {model_dir}")
    if not prepare_state_path.is_file() or not completion_path.is_file():
        raise SystemExit("Prepare state or training completion marker is missing")

    prepare_state = json.loads(prepare_state_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if prepare_state.get("fingerprint") != completion.get("data_fingerprint"):
        raise SystemExit("Training completion marker does not match the prepared dataset")

    files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    required_names = {"inference.yml", "inference.pdiparams"}
    model_names = {path.name for path in files}
    if not required_names.issubset(model_names) or not (
        {"inference.json", "inference.pdmodel"} & model_names
    ):
        raise SystemExit(
            "Exported model is incomplete; expected inference.yml, parameters, and graph"
        )
    checksums = {
        path.relative_to(model_dir).as_posix(): _sha256(path) for path in files
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model": "korean_PP-OCRv5_mobile_rec_finetuned",
        "data_fingerprint": prepare_state["fingerprint"],
        "training": completion,
        "model_files": checksums,
    }
    evaluation = _evaluation_summary(evaluation_dir)
    readme = """한국어 공문서 OCR 추론 모델

1. 이 ZIP을 원하는 폴더에 풉니다.
2. model 폴더의 절대 경로를 PADDLE_RECOGNITION_MODEL_DIR에 지정합니다.
3. 앱을 다시 시작하면 기본 글자 찾기 모델과 이 글자 인식 모델을 함께 사용합니다.

예시:
PADDLE_RECOGNITION_MODEL_DIR=C:\\models\\public_doc_rec\\model

checksums.sha256으로 복사 중 파일이 손상되지 않았는지 확인할 수 있습니다.
"""

    deliverables.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=deliverables, delete=False, suffix=".zip.tmp"
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                relative = path.relative_to(model_dir).as_posix()
                bundle.write(path, f"public_doc_rec/model/{relative}")
            bundle.writestr(
                "public_doc_rec/model_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            bundle.writestr(
                "public_doc_rec/evaluation_summary.json",
                json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            )
            bundle.writestr(
                "public_doc_rec/checksums.sha256",
                "".join(f"{digest}  model/{name}\n" for name, digest in checksums.items()),
            )
            bundle.writestr("public_doc_rec/README_KO.txt", readme)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)

    digest = _sha256(output)
    output.with_suffix(".zip.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {"package": str(output), "bytes": output.stat().st_size, "sha256": digest},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
