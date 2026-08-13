from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_training.streaming_prepare import PrepareOptions, prepare_streaming_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact AI-Hub streaming index and bounded final set"
    )
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
    parser.add_argument("--final-documents", type=int, default=120)
    parser.add_argument("--dev-max-samples", type=int, default=10_000)
    parser.add_argument("--max-text-length", type=int, default=25)
    args = parser.parse_args()
    summary = prepare_streaming_dataset(
        PrepareOptions(
            data_root=args.data_root,
            training_manifest=args.training_manifest,
            validation_manifest=args.validation_manifest,
            output=args.output,
            final_documents=args.final_documents,
            dev_max_samples=args.dev_max_samples,
            max_text_length=args.max_text_length,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
