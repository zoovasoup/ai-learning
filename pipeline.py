from __future__ import annotations

import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from knn.classifier import accuracy_score, confusion_matrix, predict
from knn.distance import euclidean
from knn.preprocessing import (
    encode_labels,
    min_max_normalize,
    read_csv,
    train_test_split,
)

DistanceFunc = Callable[[list[float], list[float]], float]

FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
LABEL_COL = "mood"


def _extract_xy(
    records: list[dict[str, Any]],
) -> tuple[list[list[float]], list[int]]:
    X = [[float(row[c]) for c in FEATURE_COLS] for row in records]
    y = [int(row[LABEL_COL]) for row in records]
    return X, y


def run_pipeline(csv_path: str | Path, k: int) -> dict[str, Any]:
    records = read_csv(csv_path)
    records_enc, classes = encode_labels(records, LABEL_COL)
    records_norm, mins, maxs = min_max_normalize(records_enc, FEATURE_COLS)
    train, test = train_test_split(records_norm, test_ratio=0.2)

    X_train, y_train = _extract_xy(train)
    X_test, y_test = _extract_xy(test)

    y_pred = predict(X_test, X_train, y_train, k)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "k": k,
        "accuracy": acc,
        "cm": cm,
        "y_pred": y_pred,
        "y_test": y_test,
        "X_test": X_test,
        "test_data": test,
        "train_data": train,
        "classes": classes,
        "mins": mins,
        "maxs": maxs,
        "raw_records": records_enc,
    }


def cross_validate(
    X: list[list[float]],
    y: list[int],
    k: int,
    n_folds: int = 5,
    dist_func: DistanceFunc = euclidean,
) -> dict[str, Any]:
    fold_size = len(X) // n_folds
    accuracies: list[float] = []
    all_cms: list[list[list[int]]] = []

    indices = list(range(len(X)))
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(X)

        val_idx = set(range(val_start, val_end))
        train_idx = [i for i in indices if i not in val_idx]

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        y_pred = predict(X_val, X_train, y_train, k, dist_func)
        acc = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        accuracies.append(acc)
        all_cms.append(cm)

    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

    return {
        "k": k,
        "n_folds": n_folds,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "accuracies": accuracies,
        "cms": all_cms,
        "dist_func_name": dist_func.__name__,
    }
