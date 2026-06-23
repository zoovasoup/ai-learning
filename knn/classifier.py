from __future__ import annotations

from collections import Counter
from typing import Callable

from .distance import euclidean

DistanceFunc = Callable[[list[float], list[float]], float]


def predict_one(
    test_row: list[float],
    train_X: list[list[float]],
    train_y: list[int],
    k: int,
    dist_func: DistanceFunc = euclidean,
) -> int:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if len(train_X) != len(train_y):
        raise ValueError("train_X and train_y must have the same length")
    if not train_X:
        raise ValueError("train_X must not be empty")

    distances = [
        (dist_func(test_row, train_row), label)
        for train_row, label in zip(train_X, train_y)
    ]
    distances.sort(key=lambda item: item[0])

    nearest_labels = [label for _, label in distances[:k]]
    vote_counts = Counter(nearest_labels)

    return max(nearest_labels, key=lambda label: vote_counts[label])


def predict(
    test_X: list[list[float]],
    train_X: list[list[float]],
    train_y: list[int],
    k: int,
    dist_func: DistanceFunc = euclidean,
) -> list[int]:
    return [predict_one(test_row, train_X, train_y, k, dist_func) for test_row in test_X]


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true must not be empty")

    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    return correct / len(y_true) * 100


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    n_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1
    cm = [[0] * n_classes for _ in range(n_classes)]
    for actual, predicted in zip(y_true, y_pred):
        cm[actual][predicted] += 1
    return cm
