from __future__ import annotations

from dataclasses import dataclass

from app.reading_order import sort_reading_order


@dataclass(frozen=True)
class Box:
    name: str
    x: int
    y: int
    width: int
    height: int


def test_visual_rows_are_sorted_top_to_bottom_and_left_to_right() -> None:
    boxes = [
        Box("bottom", 20, 60, 50, 12),
        Box("top-right", 120, 10, 40, 14),
        Box("top-left", 15, 12, 50, 12),
    ]

    ordered = sort_reading_order(boxes)

    assert [box.name for box in ordered] == ["top-left", "top-right", "bottom"]


def test_single_box_order_is_unchanged() -> None:
    box = Box("only", 0, 0, 20, 10)

    assert sort_reading_order([box]) == [box]
