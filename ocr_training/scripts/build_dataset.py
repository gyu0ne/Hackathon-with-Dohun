from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_training.dataset import BuildOptions, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leak-safe AI-Hub recognition crops")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("ocr_training/work/manifests/training.jsonl"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("ocr_training/work/manifests/validation.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("ocr_training/data"))
    parser.add_argument("--max-training-documents", type=int)
    parser.add_argument("--final-documents", type=int, default=120)
    parser.add_argument("--max-text-length", type=int, default=25)
    args = parser.parse_args()
    summary = build_dataset(
        BuildOptions(
            data_root=args.data_root,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            output=args.output,
            max_training_documents=args.max_training_documents,
            final_documents=args.final_documents,
            max_text_length=args.max_text_length,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
