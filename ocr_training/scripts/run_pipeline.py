from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from ocr_training.scripts.check_environment import inspect_environment


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command)

    stop_requested = False

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        if process.poll() is None:
            process.send_signal(signum)

    previous_term = signal.signal(signal.SIGTERM, forward_signal)
    previous_int = signal.signal(signal.SIGINT, forward_signal)
    try:
        return_code = process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    if stop_requested:
        print("Pipeline stopped safely. Run the same command to resume.", flush=True)
        raise SystemExit(0)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible AI-Hub OCR GPU pipeline")
    parser.add_argument("--data-root", type=Path, default=Path("/datasets/aihub_ocr"))
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "export", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("OCR_EPOCHS", "20")))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.getenv("OCR_BATCH_SIZE", "64"))
    )
    parser.add_argument(
        "--save-batch-step",
        type=int,
        default=int(os.getenv("OCR_SAVE_BATCH_STEP", "500")),
    )
    parser.add_argument("--max-training-documents", type=int)
    parser.add_argument("--final-documents", type=int, default=120)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    inspect_environment(args.data_root, require_gpu=args.stage in {"train", "export", "all"})

    if args.stage in {"prepare", "all"}:
        manifests_ready = Path("ocr_training/work/manifests/report.json").is_file()
        dataset_ready = Path("ocr_training/data/summary.json").is_file()
        pretrained_ready = Path(
            "ocr_training/artifacts/pretrained/"
            "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
        ).is_file()
        if args.force_prepare or not manifests_ready:
            run(
                "ocr_training.scripts.build_manifests",
                "--data-root",
                str(args.data_root),
                "--output",
                "ocr_training/work/manifests",
            )
        if args.force_prepare or not dataset_ready:
            dataset_args = [
                "--data-root",
                str(args.data_root),
                "--output",
                "ocr_training/data",
                "--final-documents",
                str(args.final_documents),
            ]
            if args.max_training_documents is not None:
                dataset_args.extend(
                    ["--max-training-documents", str(args.max_training_documents)]
                )
            run("ocr_training.scripts.build_dataset", *dataset_args)
        if not pretrained_ready:
            run("ocr_training.scripts.download_pretrained")
    if args.stage in {"train", "all"}:
        run(
            "ocr_training.scripts.train",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--save-batch-step",
            str(args.save_batch_step),
        )
    if args.stage in {"export", "all"}:
        run("ocr_training.scripts.export_model")
    if args.stage in {"evaluate", "all"}:
        output = Path("ocr_training/artifacts/evaluation")
        custom_model = "ocr_training/artifacts/inference/public_doc_rec"
        for variant in ("clean", "camera", "camera_filtered"):
            label_file = f"ocr_training/data/final_{variant}.txt"
            for model_name, model_args in (
                ("baseline", []),
                ("finetuned", ["--model-dir", custom_model]),
            ):
                run(
                    "ocr_training.scripts.evaluate_recognition",
                    "--data-dir",
                    "ocr_training/data",
                    "--label-file",
                    label_file,
                    "--device",
                    "gpu",
                    "--output",
                    str(output / f"{model_name}_{variant}.json"),
                    *model_args,
                )


if __name__ == "__main__":
    main()
