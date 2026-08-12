from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_training.aihub import discover_archives, resolve_dataset_root


def inspect_environment(data_root: Path, require_gpu: bool = True) -> dict[str, object]:
    dataset_root = resolve_dataset_root(data_root)
    discovered: dict[str, object] = {}
    for split in ("training", "validation"):
        archives = discover_archives(dataset_root, split)
        discovered[split] = {
            "image_zip_count": len(archives.image_archives),
            "label_zip_count": len(archives.label_archives),
        }
    gpu_count = 0
    paddle_version = "not checked"
    if require_gpu:
        import paddle

        paddle_version = paddle.__version__
        gpu_count = paddle.device.cuda.device_count()
        if not paddle.is_compiled_with_cuda() or gpu_count < 1:
            raise RuntimeError("A CUDA-enabled PaddlePaddle runtime and NVIDIA GPU are required")
    report = {
        "dataset_root": str(dataset_root),
        "paddle_version": paddle_version,
        "gpu_count": gpu_count,
        "archives": discovered,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AI-Hub data and GPU runtime")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect_environment(args.data_root, require_gpu=not args.skip_gpu),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
