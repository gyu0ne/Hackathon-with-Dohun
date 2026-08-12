from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

RECOVERY_PATTERN = re.compile(r"recovery_step_(\d+)$")
REQUIRED_SUFFIXES = (".pdparams", ".pdopt", ".states")


def is_complete_checkpoint(prefix: Path) -> bool:
    if not all(prefix.with_suffix(suffix).is_file() for suffix in REQUIRED_SUFFIXES):
        return False
    if not prefix.with_suffix(".complete").is_file():
        return False
    return checkpoint_progress(prefix)[0] >= 0


def checkpoint_progress(prefix: Path) -> tuple[int, int, int]:
    epoch = -1
    global_step = -1
    try:
        with prefix.with_suffix(".states").open("rb") as source:
            state: dict[str, Any] = pickle.load(source)  # noqa: S301 - trusted local output
        epoch = int(state.get("epoch", -1))
        global_step = int(state.get("global_step", -1))
        best = state.get("best_model_dict", {})
        if prefix.name.startswith("recovery_step_"):
            epoch = int(best.get("start_epoch", epoch))
        global_step = max(global_step, int(best.get("global_step", -1)))
    except (OSError, ValueError, TypeError, pickle.PickleError, EOFError):
        return (-1, -1, -1)
    recovery_match = RECOVERY_PATTERN.fullmatch(prefix.name)
    recovery_step = int(recovery_match.group(1)) if recovery_match else -1
    return (global_step, epoch, recovery_step)


def find_latest_checkpoint(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    names = {path.stem for path in directory.glob("*.complete")}
    candidates = [directory / name for name in names]
    candidates = [prefix for prefix in candidates if is_complete_checkpoint(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda prefix: (checkpoint_progress(prefix), prefix.name))
