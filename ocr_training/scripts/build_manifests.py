from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from ocr_training.aihub import (
    SourceSplit,
    build_split_rows,
    discover_archives,
    resolve_dataset_root,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manifests from the actual AI-Hub ZIP schema"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ocr_training/work/manifests"))
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Inspect local Validation data without requiring Training",
    )
    args = parser.parse_args()
    dataset_root = resolve_dataset_root(args.data_root)
    splits = ("validation",) if args.validation_only else ("training", "validation")
    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "contract": "AI-Hub Training -> train/dev; AI-Hub Validation -> final only",
    }
    all_ids: dict[str, set[str]] = {}
    for split in splits:
        archives = discover_archives(dataset_root, cast(SourceSplit, split))
        rows, stats = build_split_rows(archives, dataset_root, args.dev_fraction)
        write_jsonl(args.output / f"{split}.jsonl", rows)
        report[split] = stats
        all_ids[split] = {row["id"] for row in rows}
    if "training" in all_ids and "validation" in all_ids:
        overlap = all_ids["training"] & all_ids["validation"]
        if overlap:
            raise ValueError(
                f"Training/Validation document leakage detected: {sorted(overlap)[:10]}"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
