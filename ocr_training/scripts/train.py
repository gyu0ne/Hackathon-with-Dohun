from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from ocr_training.checkpoints import find_latest_checkpoint, is_complete_checkpoint


def build_command(
    config: Path,
    pretrained: Path,
    checkpoint_dir: Path,
    checkpoint: Path | None,
    epochs: int,
    batch_size: int,
    save_batch_step: int,
) -> list[str]:
    command = [
        sys.executable,
        "/opt/PaddleOCR/tools/train.py",
        "-c",
        str(config),
        "-o",
        f"Global.epoch_num={epochs}",
        f"Global.save_model_dir={checkpoint_dir}",
        f"Global.save_batch_step={save_batch_step}",
        f"Train.sampler.first_bs={batch_size}",
        f"Train.loader.batch_size_per_card={batch_size}",
    ]
    if checkpoint is not None:
        command.append(f"Global.checkpoints={checkpoint}")
    else:
        command.append(f"Global.pretrained_model={pretrained}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Korean PP-OCRv5 with automatic checkpoint resume"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/ocr_training/configs/korean_public_doc_rec.yml"),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path(
            "/workspace/ocr_training/artifacts/pretrained/"
            "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/workspace/ocr_training/artifacts/checkpoints"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--save-batch-step",
        type=int,
        default=500,
        help="Save a crash-safe recovery generation every N batches; 0 disables it",
    )
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint
    if checkpoint is not None and not is_complete_checkpoint(checkpoint):
        raise SystemExit(f"Incomplete checkpoint: {checkpoint}")
    if checkpoint is None and not args.no_resume:
        checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if checkpoint is None and not args.pretrained.is_file():
        raise SystemExit(f"Pretrained model not found: {args.pretrained}")
    if checkpoint:
        print(f"Resuming complete checkpoint: {checkpoint}", flush=True)
    else:
        print(f"Starting from pretrained model: {args.pretrained}", flush=True)
    command = build_command(
        args.config,
        args.pretrained,
        args.checkpoint_dir,
        checkpoint,
        args.epochs,
        args.batch_size,
        args.save_batch_step,
    )
    process = subprocess.Popen(command)

    def forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
