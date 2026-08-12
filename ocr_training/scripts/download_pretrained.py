from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "korean_PP-OCRv5_mobile_rec_pretrained.pdparams"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Korean PP-OCRv5 weights")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr_training/artifacts/pretrained/korean_PP-OCRv5_mobile_rec_pretrained.pdparams"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(MODEL_URL, timeout=120) as response,
        args.output.open("wb") as target,
    ):
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    receipt = {
        "url": MODEL_URL,
        "bytes": args.output.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
