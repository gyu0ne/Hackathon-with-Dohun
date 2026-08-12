#!/usr/bin/env bash
set -euo pipefail

docker compose --env-file .env.ocr -f compose.ocr-gpu.yaml build ocr-trainer
docker compose --env-file .env.ocr -f compose.ocr-gpu.yaml run --rm ocr-trainer \
  python -m ocr_training.scripts.run_pipeline "$@"
