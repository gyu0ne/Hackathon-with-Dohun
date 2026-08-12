from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
from paddleocr import TextRecognition

from research.ocr_metrics import character_error_rate


def _read_labels(data_dir: Path, label_file: Path, limit: int | None) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        relative, expected = line.split("\t", maxsplit=1)
        rows.append((data_dir / relative, expected))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PaddleOCR recognition model on crops")
    parser.add_argument("--data-dir", type=Path, default=Path("ocr_training/data"))
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options: dict[str, Any] = {
        "model_name": "korean_PP-OCRv5_mobile_rec",
        "device": args.device,
        "enable_mkldnn": False,
    }
    if args.model_dir:
        options["model_dir"] = str(args.model_dir.resolve())
    predictor = TextRecognition(**options)
    documents: list[dict[str, Any]] = []
    for index, (image_path, reference) in enumerate(
        _read_labels(args.data_dir, args.label_file, args.limit), start=1
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image could not be decoded: {image_path}")
        started = perf_counter()
        result = list(predictor.predict(image))[0].json["res"]
        elapsed_ms = (perf_counter() - started) * 1000
        hypothesis = str(result.get("rec_text", ""))
        documents.append(
            {
                "image": image_path.as_posix(),
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": round(character_error_rate(reference, hypothesis), 4),
                "exact": reference == hypothesis,
                "confidence": round(float(result.get("rec_score", 0.0)), 4),
                "elapsed_ms": round(elapsed_ms, 2),
            }
        )
        if index % 100 == 0:
            print(json.dumps({"completed": index}))
    aggregate = {
        "images": len(documents),
        "mean_cer": round(statistics.fmean(item["cer"] for item in documents), 4),
        "exact_accuracy": round(
            sum(item["exact"] for item in documents) / len(documents), 4
        ),
        "mean_confidence": round(
            statistics.fmean(item["confidence"] for item in documents), 4
        ),
        "mean_ms": round(statistics.fmean(item["elapsed_ms"] for item in documents), 2),
    }
    report = {
        "model": str(args.model_dir) if args.model_dir else "korean_PP-OCRv5_mobile_rec",
        "label_file": str(args.label_file),
        "aggregate": aggregate,
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

