from __future__ import annotations

import csv
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def encode_labels(
    records: list[dict[str, Any]], col: str = "label"
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    classes = {}
    result = deepcopy(records)
    for row in result:
        val = row[col]
        if val not in classes:
            classes[val] = len(classes)
        row[col] = classes[val]
    return result, classes


def min_max_normalize(
    records: list[dict[str, Any]], cols: list[str]
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    min_vals: dict[str, float] = {}
    max_vals: dict[str, float] = {}

    for col in cols:
        values = [float(row[col]) for row in records]
        min_vals[col] = min(values)
        max_vals[col] = max(values)

    result = deepcopy(records)
    for row in result:
        for col in cols:
            val = float(row[col])
            mn = min_vals[col]
            mx = max_vals[col]
            if mx - mn == 0:
                row[col] = 0.0
            else:
                row[col] = (val - mn) / (mx - mn)

    return result, min_vals, max_vals


def train_test_split(
    records: list[dict[str, Any]], test_ratio: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = deepcopy(records)
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]
