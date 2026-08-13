from __future__ import annotations

import argparse
import json
from pathlib import Path

import paddle


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate existing Paddle pretrained weights")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.is_file() or args.path.stat().st_size == 0:
        raise SystemExit(f"Pretrained model is missing or empty: {args.path}")
    try:
        payload = paddle.load(str(args.path))
    except Exception as error:
        raise SystemExit(f"Pretrained model cannot be loaded: {args.path}: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise SystemExit(f"Pretrained model contains no parameter dictionary: {args.path}")
    print(
        json.dumps(
            {
                "path": str(args.path),
                "bytes": args.path.stat().st_size,
                "parameters": len(payload),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
