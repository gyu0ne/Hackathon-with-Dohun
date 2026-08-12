from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol


class BoxLike(Protocol):
    @property
    def x(self) -> int: ...

    @property
    def y(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


@dataclass
class _Row[BoxItem: BoxLike]:
    items: list[BoxItem]
    center_y: float


def sort_reading_order[BoxItem: BoxLike](items: list[BoxItem]) -> list[BoxItem]:
    """Order arbitrary document text boxes by visual row, then from left to right."""
    if len(items) < 2:
        return list(items)
    median_height = statistics.median(max(1, item.height) for item in items)
    center_tolerance = max(3.0, median_height * 0.55)
    rows: list[_Row[BoxItem]] = []
    for item in sorted(items, key=lambda value: (value.y + value.height / 2, value.x)):
        center = item.y + item.height / 2
        nearest = min(rows, key=lambda row: abs(row.center_y - center), default=None)
        if nearest is None or abs(nearest.center_y - center) > center_tolerance:
            rows.append(_Row(items=[item], center_y=center))
            continue
        nearest.items.append(item)
        nearest.center_y = statistics.fmean(
            value.y + value.height / 2 for value in nearest.items
        )
    rows.sort(key=lambda row: min(item.y for item in row.items))
    ordered: list[BoxItem] = []
    for row in rows:
        ordered.extend(sorted(row.items, key=lambda item: (item.x, item.y)))
    return ordered
