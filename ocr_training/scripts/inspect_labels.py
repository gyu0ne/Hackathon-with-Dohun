from __future__ import annotations

import argparse
import json
import statistics
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ocr_training.dataset import normalize_label


def inspect(test_data: Path, manifest: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    lengths: list[int] = []
    empty = 0
    with ExitStack() as stack:
        archives: dict[Path, zipfile.ZipFile] = {}
        for row in rows:
            path = (test_data / row["label_zip"]).resolve()
            if path not in archives:
                archives[path] = stack.enter_context(zipfile.ZipFile(path))
            label = json.loads(archives[path].read(row["label_member"]))
            for annotation in label.get("annotations", []):
                text = normalize_label(annotation.get("annotation.text"))
                if text:
                    lengths.append(len(text))
                else:
                    empty += 1
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        if not ordered:
            return 0
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]

    return {
        "documents": len(rows),
        "labels": len(lengths),
        "empty": empty,
        "mean_length": round(statistics.fmean(lengths), 3) if lengths else 0.0,
        "p95_length": percentile(0.95),
        "p99_length": percentile(0.99),
        "max_length": max(lengths, default=0),
        "over_25": sum(length > 25 for length in lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AI-Hub OCR label lengths")
    parser.add_argument("--test-data", type=Path, default=Path("Test Data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("research/manifests/ocr_manifest.jsonl")
    )
    args = parser.parse_args()
    print(json.dumps(inspect(args.test_data, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

