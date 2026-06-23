from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline import run_pipeline
from reporting import plot_confusion_matrix, plot_scatter, save_log

K_VALUES = [3, 5, 7]
CSV_PATH = Path("data/dataset_mahasiswa.csv")
OUTPUT_DIR = Path("output")
LOG_DIR = OUTPUT_DIR / "logs"
GRAPH_DIR = OUTPUT_DIR / "graphs"


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


if __name__ == "__main__":
    main()
