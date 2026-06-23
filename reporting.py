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
        rev_list = [rev_classes[i] for i in range(len(cm))]
        header = f"{'':20s}"
        for c in rev_list:
            header += f"{c:>16s}"
        lines.append(header)
        for i, row in enumerate(cm):
            line = f"{rev_list[i]:20s}"
            for v in row:
                line += f"{v:>16d}"
            lines.append(line)
        lines.append("=" * 62)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(
    cm: list[list[int]], k: int, save_path: str | Path,
    classes: list[str] | None = None,
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if classes is None:
        classes = [str(i) for i in range(len(cm))]
    cm_array = np.array(cm)
    n = len(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array, cmap="Blues", vmin=0, vmax=cm_array.max() + 1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (K = {k})")

    for i in range(n):
        for j in range(n):
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


def save_cv_log(cv_results: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("K-FOLD CROSS VALIDATION")
    lines.append("=" * 62)
    lines.append("")

    for result in cv_results:
        n_folds = result["n_folds"]
        k = result["k"]
        mean_acc = result["mean_accuracy"]
        std_acc = result["std_accuracy"]
        accs = result["accuracies"]
        dist_name = result["dist_func_name"]

        lines.append(f"K        = {k}")
        lines.append(f"Folds    = {n_folds}-fold")
        lines.append(f"Distance = {dist_name}")
        lines.append("")
        for i, acc in enumerate(accs, 1):
            lines.append(f"  Fold-{i} : {acc:.2f}%")
        lines.append("")
        lines.append(f"  Rata-rata : {mean_acc:.2f}%")
        lines.append(f"  Std Dev   : {std_acc:.2f}%")
        lines.append(f"  Interval  : {mean_acc - 2*std_acc:.2f}% - {mean_acc + 2*std_acc:.2f}%")
        lines.append("")
        lines.append("-" * 62)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_scatter(
    train_data: list[dict[str, Any]],
    test_data: list[dict[str, Any]],
    y_test: list[int],
    y_pred: list[int],
    classes: dict[str, int],
    k: int,
    save_path: str | Path,
    x_col: str = "valence",
    y_col: str = "energy",
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rev_classes = {v: k for k, v in classes.items()}
    colors = {"Energetic": "#d64933", "Happy": "#f4a261", "Chill": "#2e86ab", "Sad": "#6b4c85"}
    feat_labels = {
        "danceability": "Danceability",
        "energy": "Energy",
        "loudness": "Loudness (dB)",
        "speechiness": "Speechiness",
        "acousticness": "Acousticness",
        "instrumentalness": "Instrumentalness",
        "liveness": "Liveness",
        "valence": "Valence",
        "tempo": "Tempo (BPM)",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for row in train_data:
        x = float(row[x_col])
        y = float(row[y_col])
        label = rev_classes.get(int(row["mood"]), "Unknown")
        ax.scatter(x, y, c=colors.get(label, "#888"), s=40, alpha=0.35, edgecolors="none", zorder=1)

    correct_x, correct_y = [], []
    wrong_x, wrong_y = [], []
    for i, row in enumerate(test_data):
        x = float(row[x_col])
        y = float(row[y_col])
        if y_test[i] == y_pred[i]:
            correct_x.append(x)
            correct_y.append(y)
        else:
            wrong_x.append(x)
            wrong_y.append(y)

    ax.scatter(correct_x, correct_y, c="#1b5e20", s=110, alpha=0.9,
               edgecolors="black", linewidths=1.2, marker="o", zorder=3, label="Prediksi Benar")
    ax.scatter(wrong_x, wrong_y, c="#b71c1c", s=150, alpha=0.9,
               edgecolors="black", linewidths=1.5, marker="X", zorder=4, label="Prediksi Salah")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", c=colors[m], alpha=0.35, label=f"{m} (latih)")
        for m in rev_classes.values()
    ] + [
        plt.Line2D([], [], marker="o", linestyle="none", c="#1b5e20", markeredgecolor="black",
                   markeredgewidth=1.2, label="Uji - Benar"),
        plt.Line2D([], [], marker="X", linestyle="none", c="#b71c1c", markersize=8,
                   markeredgewidth=1.5, label="Uji - Salah"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)

    ax.set_xlabel(feat_labels.get(x_col, x_col))
    ax.set_ylabel(feat_labels.get(y_col, y_col))
    ax.set_title(f"Sebaran Mood Lagu (K = {k})")
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
