from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main"
MODELS = {
    "kor_best.traineddata": (
        "kor.traineddata",
        "f888d4038348a0c3d25151e7f452bda0d74ca275b18cab146798bcbb94084fff",
    ),
    "eng_best.traineddata": (
        "eng.traineddata",
        "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba",
    ),
}


def install_models(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for target_name, (source_name, expected_sha256) in MODELS.items():
        with urllib.request.urlopen(f"{BASE_URL}/{source_name}") as response:
            payload = response.read()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {source_name}")
        (target_dir / target_name).write_bytes(payload)


def main() -> None:
    install_models(Path("/usr/share/tesseract-ocr/5/tessdata"))


if __name__ == "__main__":
    main()
