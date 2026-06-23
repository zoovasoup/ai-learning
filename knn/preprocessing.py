from __future__ import annotations

import csv
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def encode_labels(
    records: list[dict[str, Any]], col: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    encoded_records = deepcopy(records)
    classes: dict[str, int] = {}

    for row in encoded_records:
        label = str(row[col])
        if label not in classes:
            classes[label] = len(classes)
        row[col] = classes[label]

    return encoded_records, classes


def min_max_normalize(
    records: list[dict[str, Any]], cols: list[str]
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    normalized_records = deepcopy(records)
    min_vals: dict[str, float] = {}
    max_vals: dict[str, float] = {}

    for col in cols:
        values = [float(row[col]) for row in records]
        min_vals[col] = min(values)
        max_vals[col] = max(values)

    for row in normalized_records:
        for col in cols:
            value = float(row[col])
            min_value = min_vals[col]
            max_value = max_vals[col]
            value_range = max_value - min_value
            row[col] = 0.0 if value_range == 0 else (value - min_value) / value_range

    return normalized_records, min_vals, max_vals


def train_test_split(
    records: list[dict[str, Any]], test_ratio: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    shuffled_records = deepcopy(records)
    random.shuffle(shuffled_records)

    test_size = max(1, int(len(shuffled_records) * test_ratio))
    split_index = len(shuffled_records) - test_size

    return shuffled_records[:split_index], shuffled_records[split_index:]
