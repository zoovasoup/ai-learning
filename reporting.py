from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_log(results: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("KNN - PREDIKSI KELULUSAN MAHASISWA")
    lines.append("=" * 62)
    lines.append(f"Dataset        : dataset_mahasiswa.csv")
    lines.append(f"Total Data     : ...")
    lines.append("=" * 62)
    lines.append("")

    for result in results:
        k = result["k"]
        y_test = result["y_test"]
        y_pred = result["y_pred"]
        cm = result["cm"]
        acc = result["accuracy"]
        classes = result.get("classes", {})
        rev_classes = {v: k for k, v in classes.items()}

        lines.append(f"HASIL K = {k}")
        lines.append("-" * 62)
        benar = 0
        for i in range(len(y_test)):
            actual = rev_classes.get(y_test[i], str(y_test[i]))
            predicted = rev_classes.get(y_pred[i], str(y_pred[i]))
            status = "Benar" if y_test[i] == y_pred[i] else "Salah"
            if y_test[i] == y_pred[i]:
                benar += 1
            lines.append(
                f"Data ke-{i+1} | Actual: {actual:<12} | "
                f"Predicted: {predicted:<12} | {status}"
            )
        lines.append("")
        lines.append(f"AKURASI: {acc:.2f}%")
        lines.append(f"K = {k}  | Accuracy: {acc:.2f}%")
        lines.append("=" * 62)
        lines.append("")
        lines.append("Confusion Matrix:")
        lines.append(f"{'':20s}{'Pred Lulus':>12s}{'Pred Tdk Lulus':>16s}")
        lines.append(f"{'Lulus':20s}{cm[0][0]:>12d}{cm[0][1]:>16d}")
        lines.append(f"{'Tidak Lulus':20s}{cm[1][0]:>12d}{cm[1][1]:>16d}")
        lines.append("=" * 62)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(
    cm: list[list[int]], k: int, save_path: str | Path
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = ["Lulus", "Tidak Lulus"]
    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, cmap="Blues", vmin=0, vmax=cm_array.max() + 1)

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (K = {k})")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm_array[i, j]),
                ha="center", va="center",
                color="white" if cm_array[i, j] > cm_array.max() / 2 else "black",
                fontsize=14,
            )

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
