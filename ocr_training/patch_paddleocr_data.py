from __future__ import annotations

import argparse
from pathlib import Path

PATCH_MARKER = "KCODE_AIHUB_STREAM_DATASET_PATCH_V1"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"PaddleOCR v3.7.0 data patch anchor count was {source.count(old)}, expected 1"
        )
    return source.replace(old, new, 1)


def patch_data_loader(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    source = _replace_once(
        source,
        "from ppocr.data.multi_scale_sampler import MultiScaleSampler\n",
        "from ppocr.data.multi_scale_sampler import MultiScaleSampler\n"
        "# KCODE_AIHUB_STREAM_DATASET_PATCH_V1\n"
        "from ocr_training.streaming_dataset import AIHubStreamDataSet, DocumentBatchSampler\n",
    )
    source = _replace_once(
        source,
        '        "LaTeXOCRDataSet",\n',
        '        "LaTeXOCRDataSet",\n'
        '        "AIHubStreamDataSet",\n',
    )
    path.write_text(source, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the AI-Hub streaming dataset")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print("patched" if patch_data_loader(args.path) else "already patched")


if __name__ == "__main__":
    main()
