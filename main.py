from __future__ import annotations

from pathlib import Path
from typing import Any

from knn.distance import euclidean, manhattan
from knn.preprocessing import encode_labels, min_max_normalize, read_csv
from pipeline import cross_validate, run_pipeline
from reporting import plot_confusion_matrix, plot_scatter, save_cv_log, save_log

K_VALUES = [3, 5, 7]
N_FOLDS = 5
CSV_PATH = Path("data/dataset_mahasiswa.csv")
OUTPUT_DIR = Path("output")
LOG_DIR = OUTPUT_DIR / "logs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
FEATURE_COLS = [
    "nilai_tugas",
    "nilai_kuis",
    "nilai_uts",
    "nilai_uas",
    "absensi",
    "jam_belajar",
]
LABEL_COL = "label"


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
    print(f"{'':20s}{'Pred Lulus':>12s}{'Pred Tdk Lulus':>16s}")
    print(f"{'Lulus':20s}{cm[0][0]:>12d}{cm[0][1]:>16d}")
    print(f"{'Tidak Lulus':20s}{cm[1][0]:>12d}{cm[1][1]:>16d}")
    print(f"{'=' * 62}")


def main() -> None:
    results = []
    for k in K_VALUES:
        result = run_pipeline(CSV_PATH, k)
        results.append(result)
        _print_detail(result)
        plot_confusion_matrix(
            result["cm"],
            k,
            GRAPH_DIR / f"confusion_matrix_k{k}.png",
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
