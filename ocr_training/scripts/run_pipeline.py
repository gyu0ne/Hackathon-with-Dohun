from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from ocr_training.scripts.check_environment import inspect_environment


MANIFEST_FILES = (
    Path("ocr_training/work/manifests/training.jsonl"),
    Path("ocr_training/work/manifests/validation.jsonl"),
    Path("ocr_training/work/manifests/report.json"),
)

DATASET_FILES = (
    Path("ocr_training/data/train.txt"),
    Path("ocr_training/data/dev.txt"),
    Path("ocr_training/data/final_clean.txt"),
    Path("ocr_training/data/final_camera.txt"),
    Path("ocr_training/data/final_camera_filtered.txt"),
    Path("ocr_training/data/metadata.jsonl"),
    Path("ocr_training/data/summary.json"),
)

PRETRAINED_MODEL = Path(
    "ocr_training/artifacts/pretrained/"
    "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
)


def _files_ready(
    paths: tuple[Path, ...],
) -> bool:
    return all(path.is_file() for path in paths)


def _print_missing(
    label: str,
    paths: tuple[Path, ...],
) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if not missing:
        print(f"[pipeline] {label}: ready", flush=True)
        return

    print(
        f"[pipeline] {label}: incomplete; "
        f"missing {len(missing)} file(s)",
        flush=True,
    )

    for path in missing[:10]:
        print(f"  - {path}", flush=True)

    if len(missing) > 10:
        print(
            f"  ... and {len(missing) - 10} more",
            flush=True,
        )


def run(
    module: str,
    *arguments: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    print(
        "[pipeline] running:",
        " ".join(command),
        flush=True,
    )

    process = subprocess.Popen(command)
    stop_requested = False

    def forward_signal(
        signum: int,
        _frame: object,
    ) -> None:
        nonlocal stop_requested
        stop_requested = True

        if process.poll() is None:
            process.send_signal(signum)

    previous_term = signal.signal(
        signal.SIGTERM,
        forward_signal,
    )
    previous_int = signal.signal(
        signal.SIGINT,
        forward_signal,
    )

    try:
        return_code = process.wait()
    finally:
        signal.signal(
            signal.SIGTERM,
            previous_term,
        )
        signal.signal(
            signal.SIGINT,
            previous_int,
        )

    if stop_requested:
        print(
            "[pipeline] stopped safely. "
            "Run the same command to resume.",
            flush=True,
        )
        raise SystemExit(0)

    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )

    print(
        f"[pipeline] completed: {module}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible AI-Hub OCR GPU pipeline"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/datasets/aihub_ocr"),
    )
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "train",
            "export",
            "evaluate",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(
            os.getenv(
                "OCR_EPOCHS",
                "20",
            )
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(
            os.getenv(
                "OCR_BATCH_SIZE",
                "64",
            )
        ),
    )
    parser.add_argument(
        "--save-batch-step",
        type=int,
        default=int(
            os.getenv(
                "OCR_SAVE_BATCH_STEP",
                "500",
            )
        ),
    )
    parser.add_argument(
        "--max-training-documents",
        type=int,
    )
    parser.add_argument(
        "--final-documents",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
    )

    args = parser.parse_args()

    print(
        f"[pipeline] stage={args.stage}, "
        f"epochs={args.epochs}, "
        f"batch_size={args.batch_size}, "
        f"save_batch_step={args.save_batch_step}",
        flush=True,
    )

    if args.max_training_documents is not None:
        print(
            "[pipeline] max training documents: "
            f"{args.max_training_documents}",
            flush=True,
        )

    print(
        f"[pipeline] checking environment for: "
        f"{args.data_root}",
        flush=True,
    )

    environment = inspect_environment(
        args.data_root,
        require_gpu=(
            args.stage
            in {
                "train",
                "export",
                "evaluate",
                "all",
            }
        ),
    )

    print(
        "[pipeline] environment OK: "
        f"dataset_root={environment['dataset_root']}, "
        f"gpu_count={environment['gpu_count']}, "
        f"paddle={environment['paddle_version']}",
        flush=True,
    )

    if args.stage in {"prepare", "all"}:
        manifests_ready = _files_ready(
            MANIFEST_FILES
        )
        dataset_ready = _files_ready(
            DATASET_FILES
        )
        pretrained_ready = PRETRAINED_MODEL.is_file()

        _print_missing(
            "manifests",
            MANIFEST_FILES,
        )
        _print_missing(
            "prepared dataset",
            DATASET_FILES,
        )

        print(
            "[pipeline] pretrained model: "
            f"{'ready' if pretrained_ready else 'missing'}",
            flush=True,
        )

        if args.force_prepare or not manifests_ready:
            print(
                "[pipeline] building manifests",
                flush=True,
            )
            run(
                "ocr_training.scripts.build_manifests",
                "--data-root",
                str(args.data_root),
                "--output",
                "ocr_training/work/manifests",
            )
        else:
            print(
                "[pipeline] manifests already complete; skipping",
                flush=True,
            )

        if args.force_prepare or not dataset_ready:
            print(
                "[pipeline] building recognition dataset",
                flush=True,
            )

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
                    [
                        "--max-training-documents",
                        str(args.max_training_documents),
                    ]
                )

            run(
                "ocr_training.scripts.build_dataset",
                *dataset_args,
            )
        else:
            print(
                "[pipeline] prepared dataset already complete; skipping",
                flush=True,
            )

        if not pretrained_ready:
            print(
                "[pipeline] downloading pretrained model",
                flush=True,
            )
            run(
                "ocr_training.scripts.download_pretrained"
            )
        else:
            print(
                "[pipeline] pretrained model already present; skipping",
                flush=True,
            )

        _print_missing(
            "manifests after prepare",
            MANIFEST_FILES,
        )
        _print_missing(
            "prepared dataset after prepare",
            DATASET_FILES,
        )

        if not _files_ready(MANIFEST_FILES):
            raise RuntimeError(
                "Prepare stage finished but manifest files are incomplete"
            )

        if not _files_ready(DATASET_FILES):
            raise RuntimeError(
                "Prepare stage finished but dataset files are incomplete"
            )

        if not PRETRAINED_MODEL.is_file():
            raise RuntimeError(
                "Prepare stage finished but pretrained model is missing"
            )

        print(
            "[pipeline] prepare stage verified successfully",
            flush=True,
        )

    if args.stage in {"train", "all"}:
        if not _files_ready(DATASET_FILES):
            _print_missing(
                "training prerequisites",
                DATASET_FILES,
            )
            raise SystemExit(
                "Training dataset is incomplete. "
                "Run --stage prepare first."
            )

        if not PRETRAINED_MODEL.is_file():
            raise SystemExit(
                "Pretrained model is missing. "
                "Run --stage prepare first."
            )

        print(
            "[pipeline] starting training",
            flush=True,
        )

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
        print(
            "[pipeline] exporting model",
            flush=True,
        )
        run(
            "ocr_training.scripts.export_model"
        )

    if args.stage in {"evaluate", "all"}:
        output = Path(
            "ocr_training/artifacts/evaluation"
        )
        custom_model = (
            "ocr_training/artifacts/inference/public_doc_rec"
        )

        print(
            "[pipeline] evaluating baseline and finetuned models",
            flush=True,
        )

        for variant in (
            "clean",
            "camera",
            "camera_filtered",
        ):
            label_file = Path(
                f"ocr_training/data/final_{variant}.txt"
            )

            if not label_file.is_file():
                raise FileNotFoundError(
                    f"Evaluation label file missing: {label_file}"
                )

            if label_file.stat().st_size == 0:
                raise ValueError(
                    f"Evaluation label file is empty: {label_file}"
                )

            for model_name, model_args in (
                ("baseline", []),
                (
                    "finetuned",
                    [
                        "--model-dir",
                        custom_model,
                    ],
                ),
            ):
                print(
                    f"[pipeline] evaluate "
                    f"{model_name}/{variant}",
                    flush=True,
                )

                run(
                    "ocr_training.scripts.evaluate_recognition",
                    "--data-dir",
                    "ocr_training/data",
                    "--label-file",
                    str(label_file),
                    "--device",
                    "gpu",
                    "--output",
                    str(
                        output
                        / f"{model_name}_{variant}.json"
                    ),
                    *model_args,
                )

    print(
        f"[pipeline] stage '{args.stage}' finished successfully",
        flush=True,
    )


if __name__ == "__main__":
    main()