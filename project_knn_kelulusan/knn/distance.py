from __future__ import annotations

import math


def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    squared_differences = [(x - y) ** 2 for x, y in zip(a, b)]
    return math.sqrt(sum(squared_differences))
