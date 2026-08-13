from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from ocr_training.checkpoints import (
    checkpoint_data_fingerprint,
    find_latest_checkpoint,
    is_complete_checkpoint,
)


def build_command(
    config: Path,
    pretrained: Path,
    checkpoint_dir: Path,
    checkpoint: Path | None,
    epochs: int,
    batch_size: int,
    save_batch_step: int,
    data_root: Path = Path("/datasets/aihub_ocr"),
    data_fingerprint: str = "",
    resume_batch: int = 0,
    resume_epoch: int = 0,
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
        f"Train.loader.batch_size_per_card={batch_size}",
        f"Train.sampler.batch_size={batch_size}",
        f"Train.dataset.data_root={data_root}",
        f"Eval.dataset.data_root={data_root}",
        f"Global.data_fingerprint={data_fingerprint}",
        f"Global.resume_batch={resume_batch}",
        f"Global.resume_epoch={resume_epoch}",
        f"Train.sampler.resume_batch={resume_batch}",
        f"Train.sampler.resume_epoch={resume_epoch}",
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
    parser.add_argument(
        "--prepare-state",
        type=Path,
        default=Path("/workspace/ocr_training/data/prepare_state.json"),
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
    if args.epochs < 1 or args.batch_size < 1 or args.save_batch_step < 0:
        raise SystemExit("epochs and batch-size must be >= 1; save-batch-step must be >= 0")
    if not args.prepare_state.is_file():
        raise SystemExit(f"Prepare state not found: {args.prepare_state}")
    state = json.loads(args.prepare_state.read_text(encoding="utf-8"))
    data_fingerprint = str(state.get("fingerprint", ""))
    data_root = Path(str(state.get("dataset_root", "")))
    if not data_fingerprint or not data_root.is_dir():
        raise SystemExit("Prepare state has no valid fingerprint or dataset root")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.checkpoint_dir / "training_complete.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (
            completion.get("data_fingerprint") == data_fingerprint
            and int(completion.get("epochs", 0)) == args.epochs
        ):
            print("Full training is already complete; skipping", flush=True)
            return
    checkpoint = args.checkpoint
    if checkpoint is not None and not is_complete_checkpoint(checkpoint):
        raise SystemExit(f"Incomplete checkpoint: {checkpoint}")
    if checkpoint is None and not args.no_resume:
        checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if args.no_resume and any(args.checkpoint_dir.glob("*.complete")):
        raise SystemExit(
            "--no-resume cannot overwrite an existing completed checkpoint directory"
        )
    if checkpoint is not None:
        checkpoint_fingerprint = checkpoint_data_fingerprint(checkpoint)
        if checkpoint_fingerprint != data_fingerprint:
            raise SystemExit(
                "Checkpoint data fingerprint does not match the prepared dataset. "
                "Move the old checkpoint directory before starting a different dataset."
            )
    if checkpoint is None and not args.pretrained.is_file():
        raise SystemExit(f"Pretrained model not found: {args.pretrained}")
    if checkpoint:
        print(f"Resuming complete checkpoint: {checkpoint}", flush=True)
    else:
        print(f"Starting from pretrained model: {args.pretrained}", flush=True)
    resume_batch = 0
    resume_epoch = 0
    if checkpoint is not None:
        import pickle

        with checkpoint.with_suffix(".states").open("rb") as source:
            checkpoint_state = pickle.load(source)  # noqa: S301 - trusted local output
        best = checkpoint_state.get("best_model_dict", {})
        if isinstance(best, dict):
            resume_batch = int(best.get("resume_batch", 0))
            resume_epoch = int(best.get("start_epoch", 0))
    command = build_command(
        args.config,
        args.pretrained,
        args.checkpoint_dir,
        checkpoint,
        args.epochs,
        args.batch_size,
        args.save_batch_step,
        data_root,
        data_fingerprint,
        resume_batch,
        resume_epoch,
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
    completed = args.checkpoint_dir / f"iter_epoch_{args.epochs}"
    if not is_complete_checkpoint(completed):
        raise SystemExit(
            "Training stopped before the final epoch was completely saved; "
            "run the same command to resume"
        )
    from ocr_training.checkpoints import checkpoint_progress

    global_step, completed_epoch, _ = checkpoint_progress(completed)
    if completed_epoch != args.epochs:
        raise SystemExit(
            f"Training stopped safely at epoch {completed_epoch}/{args.epochs}; "
            "run the same command to resume"
        )
    completion = {
        "epochs": args.epochs,
        "global_step": global_step,
        "checkpoint": str(completed),
        "data_fingerprint": data_fingerprint,
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=completion_path.parent, delete=False, suffix=".tmp"
    ) as temporary:
        json.dump(completion, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(completion_path)


if __name__ == "__main__":
    main()
