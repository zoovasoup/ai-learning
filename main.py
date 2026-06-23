from __future__ import annotations

from pathlib import Path

from pipeline import run_pipeline
from reporting import plot_confusion_matrix, save_log

K_VALUES = [3, 5, 7]
CSV_PATH = Path("data/dataset_mahasiswa.csv")
OUTPUT_DIR = Path("output")
LOG_DIR = OUTPUT_DIR / "logs"
GRAPH_DIR = OUTPUT_DIR / "graphs"


def main() -> None:
    results = []
    for k in K_VALUES:
        result = run_pipeline(CSV_PATH, k)
        results.append(result)
        print(f"K={k} | Accuracy: {result['accuracy']:.2f}%")
        plot_confusion_matrix(
            result["cm"],
            k,
            GRAPH_DIR / f"confusion_matrix_k{k}.png",
        )
    save_log(results, LOG_DIR / "log.txt")
    print("\nOutput saved to output/logs/log.txt")


if __name__ == "__main__":
    main()
