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
    lines.append("KNN - SONG MOOD CLASSIFIER (SPOTIFY 2019)")
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


def plot_pca(
    X_all: list[list[float]],
    y_all: list[int],
    test_idx_list: list[int],
    y_pred: list[int],
    classes: dict[str, int],
    k: int,
    save_path: str | Path,
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rev_classes = {v: k for k, v in classes.items()}
    colors = {"Energetic": "#d64933", "Happy": "#f4a261", "Chill": "#2e86ab", "Sad": "#6b4c85"}

    X = np.array(X_all)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt.T[:, :2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # background: all points colored by mood
    for i in range(len(X_pca)):
        label = rev_classes.get(y_all[i], "Unknown")
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=colors.get(label, "#888"),
                   s=12, alpha=0.2, edgecolors="none", zorder=1)

    # overlay test points — idx order matches y_pred order
    correct_x, correct_y = [], []
    wrong_x, wrong_y = [], []
    for pred_i, idx in enumerate(test_idx_list):
        if y_all[idx] == y_pred[pred_i]:
            correct_x.append(X_pca[idx, 0])
            correct_y.append(X_pca[idx, 1])
        else:
            wrong_x.append(X_pca[idx, 0])
            wrong_y.append(X_pca[idx, 1])

    ax.scatter(correct_x, correct_y, c="#1b5e20", s=110, alpha=0.9,
               edgecolors="black", linewidths=1.2, marker="o", zorder=3, label="Uji - Benar")
    ax.scatter(wrong_x, wrong_y, c="#b71c1c", s=150, alpha=0.9,
               edgecolors="black", linewidths=1.5, marker="X", zorder=4, label="Uji - Salah")

    var_ratio = (S[:2] ** 2) / (S ** 2).sum()
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%} varians)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%} varians)")
    ax.set_title(f"PCA — Proyeksi 9 Fitur Audio (K = {k})")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", c=colors[m], alpha=0.5, label=m)
        for m in rev_classes.values()
    ] + [
        plt.Line2D([], [], marker="o", linestyle="none", c="#1b5e20", markeredgecolor="black",
                   markeredgewidth=1.2, label="Uji - Benar"),
        plt.Line2D([], [], marker="X", linestyle="none", c="#b71c1c", markersize=8,
                   markeredgewidth=1.5, label="Uji - Salah"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_radar(
    records_norm: list[dict[str, Any]],
    feature_cols: list[str],
    classes: dict[str, int],
    save_path: str | Path,
    label_col: str = "mood",
    input_vector: list[float] | None = None,
    input_label: str = "Input",
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rev_classes = {v: k for k, v in classes.items()}
    colors = {"Energetic": "#d64933", "Happy": "#f4a261", "Chill": "#2e86ab", "Sad": "#6b4c85"}

    # mean per mood per feature (denormalize for intelligible scale)
    mood_means: dict[str, list[float]] = {}
    for class_id, mood in rev_classes.items():
        group = [r for r in records_norm if int(r[label_col]) == class_id]
        means = [float(np.mean([float(r[c]) for r in group])) for c in feature_cols]
        mood_means[mood] = means

    n_feat = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for mood, means in mood_means.items():
        values = means + means[:1]
        ax.plot(angles, values, color=colors[mood], linewidth=1.8, label=mood)
        ax.fill(angles, values, color=colors[mood], alpha=0.08)

    if input_vector is not None:
        values = input_vector + input_vector[:1]
        ax.plot(angles, values, color="black", linewidth=2.5, linestyle="--", label=input_label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, size=9)
    ax.set_title("Profil Fitur per Mood", pad=25, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_predict_scatter(
    X_all: list[list[float]],
    y_all: list[int],
    classes: dict[str, int],
    input_vector: list[float],
    save_path: str | Path,
) -> None:
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rev_classes = {v: k for k, v in classes.items()}
    colors = {"Energetic": "#d64933", "Happy": "#f4a261", "Chill": "#2e86ab", "Sad": "#6b4c85"}

    X = np.array(X_all)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt.T[:, :2]

    # project input vector using same PCA
    inp = np.array([input_vector])
    inp_centered = inp - X.mean(axis=0)
    inp_pca = inp_centered @ Vt.T[:, :2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # all training points
    for i in range(len(X_pca)):
        label = rev_classes.get(y_all[i], "Unknown")
        ax.scatter(X_pca[i, 0], X_pca[i, 1], c=colors.get(label, "#888"),
                   s=12, alpha=0.2, edgecolors="none", zorder=1)

    # input point
    ax.scatter(inp_pca[0, 0], inp_pca[0, 1], c="#e63946", s=200, alpha=0.95,
               edgecolors="black", linewidths=2, marker="*", zorder=5, label="Input")

    var_ratio = (S[:2] ** 2) / (S ** 2).sum()
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%} varians)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%} varians)")
    ax.set_title("PCA — Posisi Input di Ruang 9 Fitur Audio")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", c=colors[m], alpha=0.5, label=m)
        for m in rev_classes.values()
    ] + [
        plt.Line2D([], [], marker="*", linestyle="none", c="#e63946", markersize=10,
                   markeredgecolor="black", markeredgewidth=1.5, label="Input"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
