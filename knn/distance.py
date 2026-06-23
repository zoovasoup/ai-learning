from __future__ import annotations

import math


def euclidean(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return sum(abs(x - y) for x, y in zip(a, b))
