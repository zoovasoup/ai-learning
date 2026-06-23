from __future__ import annotations

from pathlib import Path
from typing import Any

from knn.classifier import accuracy_score, confusion_matrix, predict
from knn.preprocessing import (
    encode_labels,
    min_max_normalize,
    read_csv,
    train_test_split,
)

FEATURE_COLS = [
    "nilai_tugas",
    "nilai_kuis",
    "nilai_uts",
    "nilai_uas",
    "absensi",
    "jam_belajar",
]
LABEL_COL = "label"


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
        "classes": classes,
    }
