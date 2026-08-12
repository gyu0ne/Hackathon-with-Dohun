from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_training.vl_benchmark import create_pipeline, predict_text, score


def _reference_text(label: dict[str, Any]) -> str:
    annotations = sorted(label.get("annotations", []), key=lambda item: int(item.get("id", 0)))
    return " ".join(str(item.get("annotation.text", "")) for item in annotations)


def _decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("AI-Hub image could not be decoded")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full PaddleOCR-VL on locked AI-Hub docs"
    )
    parser.add_argument("--test-data", type=Path, default=Path("Test Data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("research/manifests/ocr_manifest.jsonl")
    )
    parser.add_argument("--split", default="smoke")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--version", choices=("v1.5", "v1.6"), default="v1.6")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_638_400,
        help="Maximum VLM crop pixels; lower values reduce CPU/GPU memory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    cases = [row for row in rows if args.split in row.get("splits", [])][: args.limit]
    pipeline = create_pipeline(args.version, args.device)
    documents: list[dict[str, Any]] = []
    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}

        def archive(relative_path: str) -> zipfile.ZipFile:
            path = (args.test_data / relative_path).resolve()
            if path not in archives:
                archives[path] = stack.enter_context(zipfile.ZipFile(path))
            return archives[path]

        for index, case in enumerate(cases, start=1):
            image = _decode(archive(case["image_zip"]).read(case["image_member"]))
            label = json.loads(archive(case["label_zip"]).read(case["label_member"]))
            reference = _reference_text(label)
            hypothesis, elapsed_ms = predict_text(
                pipeline, image, max_pixels=args.max_pixels
            )
            item = {
                "id": case["id"],
                "category": case["category"],
                **asdict(score(reference, hypothesis, elapsed_ms)),
                "hypothesis": hypothesis,
            }
            documents.append(item)
            print(json.dumps({"completed": index, "total": len(cases), "id": case["id"]}))

    aggregate = {
        "documents": len(documents),
        "mean_cer": round(statistics.fmean(item["cer"] for item in documents), 4),
        "mean_wer": round(statistics.fmean(item["wer"] for item in documents), 4),
        "mean_word_f1": round(statistics.fmean(item["word_f1"] for item in documents), 4),
        "mean_numeric_f1": round(
            statistics.fmean(item["numeric_f1"] for item in documents), 4
        ),
        "mean_ms": round(statistics.fmean(item["elapsed_ms"] for item in documents), 2),
    }
    report = {
        "model": f"PaddleOCR-VL-{args.version.removeprefix('v')}",
        "pipeline": "full layout analysis + VLM recognition",
        "device": args.device,
        "split": args.split,
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
