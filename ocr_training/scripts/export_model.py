from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ocr_training.checkpoints import find_latest_checkpoint, is_complete_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the best checkpoint for app inference")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/ocr_training/configs/korean_public_doc_rec.yml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/workspace/ocr_training/artifacts/checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/ocr_training/artifacts/inference/public_doc_rec"),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        best = args.checkpoint_dir / "best_accuracy"
        checkpoint = best if is_complete_checkpoint(best) else find_latest_checkpoint(
            args.checkpoint_dir
        )
    if checkpoint is None or not is_complete_checkpoint(checkpoint):
        raise SystemExit("No complete checkpoint is available for export")
    print(f"Exporting checkpoint: {checkpoint}", flush=True)
    command = [
        sys.executable,
        "/opt/PaddleOCR/tools/export_model.py",
        "-c",
        str(args.config),
        "-o",
        f"Global.checkpoints={checkpoint}",
        f"Global.save_inference_dir={args.output}",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
