from __future__ import annotations

from pathlib import Path
from typing import Any

from knn.distance import euclidean, manhattan
from knn.preprocessing import encode_labels, min_max_normalize, read_csv
from pipeline import cross_validate, run_pipeline
from reporting import plot_confusion_matrix, plot_scatter, save_cv_log, save_log

K_VALUES = [3, 5, 7]
N_FOLDS = 5
CSV_PATH = Path("data/tracks_mood.csv")
OUTPUT_DIR = Path("output") / "tracks_mood"
LOG_DIR = OUTPUT_DIR / "logs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
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


def _print_detail(result: dict[str, Any]) -> None:
    k = result["k"]
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    cm = result["cm"]
    acc = result["accuracy"]
    classes = result.get("classes", {})
    rev_classes = {v: k for k, v in classes.items()}

    print(f"\n{'=' * 62}")
    print(f"HASIL K = {k}")
    print(f"{'-' * 62}")
    for i in range(len(y_test)):
        actual = rev_classes.get(y_test[i], str(y_test[i]))
        predicted = rev_classes.get(y_pred[i], str(y_pred[i]))
        status = "Benar" if y_test[i] == y_pred[i] else "Salah"
        print(f"Data ke-{i+1} | Actual: {actual:<12} | Predicted: {predicted:<12} | {status}")

    print(f"\nAKURASI: {acc:.2f}%")
    print(f"K = {k}  | Accuracy: {acc:.2f}%")
    print(f"{'=' * 62}")
    print(f"\nConfusion Matrix:")
    rev_list = [rev_classes[i] for i in range(len(cm))]
    header = f"{'':20s}"
    for c in rev_list:
        header += f"{c:>16s}"
    print(header)
    for i, row in enumerate(cm):
        line = f"{rev_list[i]:20s}"
        for v in row:
            line += f"{v:>16d}"
        print(line)
    print(f"{'=' * 62}")


def main() -> None:
    results = []
    for k in K_VALUES:
        result = run_pipeline(CSV_PATH, k)
        results.append(result)
        _print_detail(result)
        rev_classes = {v: k for k, v in result["classes"].items()}
        class_names = [rev_classes[i] for i in range(len(result["cm"]))]
        plot_confusion_matrix(
            result["cm"],
            k,
            GRAPH_DIR / f"confusion_matrix_k{k}.png",
            classes=class_names,
        )
        plot_scatter(
            result["train_data"],
            result["test_data"],
            result["y_test"],
            result["y_pred"],
            result["classes"],
            k,
            GRAPH_DIR / f"scatter_k{k}.png",
        )
    save_log(results, LOG_DIR / "log.txt")
    print(f"\nOutput saved to {LOG_DIR / 'log.txt'}")

    print(f"\n{'=' * 62}")
    print(f"K-FOLD CROSS VALIDATION ({N_FOLDS} FOLDS)")
    print(f"{'=' * 62}")

    records = read_csv(CSV_PATH)
    records_enc, classes = encode_labels(records, LABEL_COL)
    records_norm, mins, maxs = min_max_normalize(records_enc, FEATURE_COLS)
    X_all = [[float(row[c]) for c in FEATURE_COLS] for row in records_norm]
    y_all = [int(row[LABEL_COL]) for row in records_norm]

    for dist_name, dist_func in [("Euclidean", euclidean), ("Manhattan", manhattan)]:
        print(f"\n--- Distance: {dist_name} ---")
        print(f"{'K':>3s}  {'Rata-rata':>10s}  {'Std Dev':>8s}  {'Per-Fold':>30s}")
        print(f"{'-' * 55}")

        cv_all = []
        for k in K_VALUES:
            cv = cross_validate(X_all, y_all, k, n_folds=N_FOLDS, dist_func=dist_func)
            cv_all.append(cv)
            acc_str = "  ".join(f"{a:6.2f}" for a in cv["accuracies"])
            print(f"{k:3d}  {cv['mean_accuracy']:8.2f}%  {cv['std_accuracy']:6.2f}%    {acc_str}")

        save_cv_log(cv_all, LOG_DIR / f"cv_{dist_name.lower()}.txt")

    print(f"\nCV logs saved to {LOG_DIR / 'cv_euclidean.txt'} & {LOG_DIR / 'cv_manhattan.txt'}")


if __name__ == "__main__":
    main()
