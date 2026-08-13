from __future__ import annotations

from typing import Any

import numpy as np

INDEX_SCHEMA_VERSION = 1


def sample_location(
    document_ids: np.ndarray[Any, Any],
    sample_ends: np.ndarray[Any, Any],
    sample_index: int,
) -> tuple[int, int]:
    """Map one global sample number to (document primary key, local position)."""
    if sample_index < 0 or not len(sample_ends) or sample_index >= int(sample_ends[-1]):
        raise IndexError(sample_index)
    document_position = int(np.searchsorted(sample_ends, sample_index, side="right"))
    previous_end = 0 if document_position == 0 else int(sample_ends[document_position - 1])
    return int(document_ids[document_position]), sample_index - previous_end
