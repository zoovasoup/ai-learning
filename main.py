from __future__ import annotations

import math
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from knn.compute import DistanceFunc, accuracy_score, confusion_matrix, euclidean, manhattan, predict, predict_one
from knn.preprocessing import encode_labels, min_max_normalize, read_csv
from pipeline import cross_validate
from reporting import (
    plot_confusion_matrix,
    plot_pca,
    plot_predict_scatter,
    plot_radar,
    save_cv_log,
    save_log,
)

K_VALUES = [3, 5, 7]
N_FOLDS = 5
CSV_PATH = Path("data") / "tracks_mood.csv"
SAVE_DIR = Path("output")
LOG_DIR = SAVE_DIR / "logs"
GRAPH_DIR = SAVE_DIR / "graphs"
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

FEATURE_NAMES_ID = {
    "danceability": "Danceability (0.0-1.0)",
    "energy": "Energy (0.0-1.0)",
    "loudness": "Loudness (-60 - 0 dB)",
    "speechiness": "Speechiness (0.0-1.0)",
    "acousticness": "Acousticness (0.0-1.0)",
    "instrumentalness": "Instrumentalness (0-1)",
    "liveness": "Liveness (0.0-1.0)",
    "valence": "Valence (0.0-1.0)",
    "tempo": "Tempo (BPM, 0-250)",
}


def _separator(char: str = "=", width: int = 62) -> None:
    print(char * width)


def load_data() -> tuple:
    records = read_csv(CSV_PATH)
    records_enc, classes = encode_labels(records, LABEL_COL)
    records_norm, mins, maxs = min_max_normalize(records_enc, FEATURE_COLS)
    X_all = [[float(row[c]) for c in FEATURE_COLS] for row in records_norm]
    y_all = [int(row[LABEL_COL]) for row in records_norm]
    return records_norm, classes, X_all, y_all, mins, maxs


def _dataset_report(
    records_norm: list[dict],
    classes: dict[str, int],
    X_all: list[list[float]],
    y_all: list[int],
) -> None:
    rev_classes = {v: k for k, v in classes.items()}

    print("=" * 62)
    print("DATASET EXPLORATION")
    print("=" * 62)
    print(f"Total data: {len(records_norm)} lagu (2019)")
    print(f"Sumber: tracks.csv (586k lagu 1922-2021) → filter year=2019 →")
    print(f"         derive mood via Russell's Circumplex (valence×energy median split)")
    print(f"         → shuffle + sample 1.500 baris")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Moods: {len(classes)}")
    for f in FEATURE_COLS:
        print(f"  - {f}")

    print(f"\nMood distribution:")
    counts: dict[str, int] = {}
    for r in records_norm:
        mood = rev_classes[int(r[LABEL_COL])]
        counts[mood] = counts.get(mood, 0) + 1
    max_bar = max(counts.values())
    for mood in sorted(counts, key=lambda m: classes[m]):
        cnt = counts[mood]
        pct = 100.0 * cnt / len(records_norm)
        bar_len = int(cnt / max_bar * 15)
        print(f"  {mood:16s} {cnt:>4d} ({pct:>5.1f}%)  {'█' * bar_len}")

    # feature means per mood
    print(f"\nFeature means per mood:")
    sorted_moods = sorted(classes, key=lambda m: classes[m])
    header = f"{'Feature':18s}"
    for m in sorted_moods:
        header += f"{m:>14s}"
    print(header)
    print("-" * (18 + 14 * len(classes)))
    for f in FEATURE_COLS:
        line = f"{f:18s}"
        for m in sorted_moods:
            cid = classes[m]
            group = [float(r[f]) for r in records_norm if int(r[LABEL_COL]) == cid]
            mean = sum(group) / len(group)
            line += f"{mean:>14.3f}"
        print(line)

    # overall feature ranges
    print(f"\nOverall feature ranges:")
    for f in FEATURE_COLS:
        vals = [float(r[f]) for r in records_norm]
        print(
            f"  {f:20s} min={min(vals):>8.3f}  max={max(vals):>8.3f}  "
            f"mean={np.mean(vals):>8.3f}  median={np.median(vals):>8.3f}"
        )

    # pca exploration
    print("-" * 62)
    print("Generating visualizations...")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    plot_pca(X_all, y_all, [], [], classes, 0, GRAPH_DIR / "pca_explore.png")
    print(f"  PCA plot \u2192 {GRAPH_DIR / 'pca_explore.png'}")
    plot_radar(records_norm, FEATURE_COLS, classes, GRAPH_DIR / "radar.png")
    print(f"  Radar chart \u2192 {GRAPH_DIR / 'radar.png'}")
    print("=" * 62)


def _print_detail(k: int, y_test: list[int], y_pred: list[int], cm: list[list[int]], acc: float, classes: dict[str, int]) -> None:
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


def _run_evaluation(
    records_norm: list[dict],
    classes: dict[str, int],
    X_all: list[list[float]],
    y_all: list[int],
) -> None:
    random.seed(42)
    rev_classes = {v: k for k, v in classes.items()}

    print("=" * 62)
    print("EVALUATION \u2014 80:20 SPLIT")
    print("=" * 62)

    idx = list(range(len(records_norm)))
    random.shuffle(idx)
    test_size = max(1, int(len(idx) * 0.2))
    test_idx_list = sorted(idx[:test_size])
    train_idx = idx[test_size:]

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_test = [X_all[i] for i in test_idx_list]
    y_test = [y_all[i] for i in test_idx_list]

    for dist_name, dist_func in [("Euclidean", euclidean), ("Manhattan", manhattan)]:
        print(f"\n--- Distance: {dist_name} ---")
        results = []
        for k in K_VALUES:
            y_pred = predict(X_test, X_train, y_train, k, dist_func)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            res = {"k": k, "accuracy": acc, "cm": cm, "y_pred": y_pred, "y_test": y_test, "classes": classes}
            results.append(res)

            print(f"\nK = {k}  |  Accuracy: {acc:.2f}%")
            print("Confusion Matrix:")
            sorted_moods = sorted(classes, key=lambda m: classes[m])
            header = f"{'':20s}"
            for m in sorted_moods:
                header += f"{m:>16s}"
            print(header)
            rev_list = [rev_classes[i] for i in range(len(cm))]
            for i, row in enumerate(cm):
                line = f"{rev_list[i]:20s}"
                for v in row:
                    line += f"{v:>16d}"
                print(line)

            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            class_names = [rev_classes[i] for i in range(len(cm))]
            plot_confusion_matrix(cm, k, GRAPH_DIR / f"confusion_matrix_k{k}_{dist_name.lower()}.png", classes=class_names)
            print(f"  Saved: {GRAPH_DIR / f'confusion_matrix_k{k}_{dist_name.lower()}.png'}")
            plot_pca(X_all, y_all, test_idx_list, y_pred, classes, k, GRAPH_DIR / f"pca_k{k}_{dist_name.lower()}.png")
            print(f"  Saved: {GRAPH_DIR / f'pca_k{k}_{dist_name.lower()}.png'}")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        save_log(results, LOG_DIR / f"log_{dist_name.lower()}.txt")

    print(f"\n{'=' * 62}")
    print(f"EVALUATION \u2014 {N_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'=' * 62}")

    for dist_name, dist_func in [("Euclidean", euclidean), ("Manhattan", manhattan)]:
        print(f"\nDistance: {dist_name}")
        print(f"{'K':>3s}  {'Rata-rata':>10s}  {'Std Dev':>8s}  {'Per-Fold':>30s}")
        print(f"{'-' * 55}")

        cv_all = []
        for k in K_VALUES:
            cv = cross_validate(X_all, y_all, k, n_folds=N_FOLDS, dist_func=dist_func)
            cv_all.append(cv)
            acc_str = "  ".join(f"{a:6.2f}" for a in cv["accuracies"])
            print(f"{k:3d}  {cv['mean_accuracy']:8.2f}%  {cv['std_accuracy']:6.2f}%    {acc_str}")

        save_cv_log(cv_all, LOG_DIR / f"cv_{dist_name.lower()}.txt")

    print("=" * 62)


def _predict_interactive() -> None:
    records = read_csv(CSV_PATH)
    records_enc, classes = encode_labels(records, LABEL_COL)
    records_norm, mins, maxs = min_max_normalize(records_enc, FEATURE_COLS)
    X_all = [[float(row[c]) for c in FEATURE_COLS] for row in records_norm]
    y_all = [int(row[LABEL_COL]) for row in records_norm]
    rev_classes = {v: k for k, v in classes.items()}
    n_train = len(X_all)

    while True:
        print("\n=== INPUT FITUR LAGU ===")
        raw = {}
        raw["danceability"] = _ask_float("Danceability (0.0-1.0): ")
        raw["energy"] = _ask_float("Energy (0.0-1.0)       : ")
        raw["loudness"] = _ask_float("Loudness (-60 - 0 dB) : ")
        raw["speechiness"] = _ask_float("Speechiness (0.0-1.0) : ")
        raw["acousticness"] = _ask_float("Acousticness (0.0-1.0): ")
        raw["instrumentalness"] = _ask_float("Instrumentalness (0-1): ")
        raw["liveness"] = _ask_float("Liveness (0.0-1.0)    : ")
        raw["valence"] = _ask_float("Valence (0.0-1.0)      : ")
        raw["tempo"] = _ask_float("Tempo (BPM, 0-250)     : ")

        norm = []
        for col in FEATURE_COLS:
            v = raw[col]
            mn = mins[col]
            mx = maxs[col]
            norm.append(0.0 if mx - mn == 0 else (v - mn) / (mx - mn))

        _predict_both(norm, raw, X_all, y_all, records_norm, classes, mins, maxs, n_train, rev_classes)

        lanjut = input("\nInput lagu lain? (y/n): ").strip().lower()
        if lanjut != "y":
            break


def _predict_both(
    norm: list[float],
    raw: dict[str, float],
    X_all: list[list[float]],
    y_all: list[int],
    records_norm: list[dict],
    classes: dict[str, int],
    mins: dict[str, float],
    maxs: dict[str, float],
    n_train: int,
    rev_classes: dict[int, str],
) -> None:
    print("\n" + "=" * 62)
    print("  STEP-BY-STEP PREDIKSI KNN")
    print("=" * 62)

    # ── STEP 1 ──
    print("\n[STEP 1] INPUT FITUR LAGU")
    print("-" * 40)
    for col in FEATURE_COLS:
        print(f"  {col:20s} = {raw[col]:>10.3f}")

    # ── STEP 2 ──
    print(f"\n[STEP 2] NORMALISASI (MIN-MAX) KE {n_train} DATA LATIH")
    print(f"  Rumus: norm = (nilai - min) / (max - min)")
    print("-" * 65)
    print(f"  {'Fitur':20s} {'Nilai':>8s} {'Min':>8s} {'Max':>8s} {'Hasil':>8s}")
    print(f"  {'-' * 55}")
    for col, nv in zip(FEATURE_COLS, norm):
        mn = mins[col]
        mx = maxs[col]
        print(f"  {col:20s} {raw[col]:>8.3f} {mn:>8.3f} {mx:>8.3f} {nv:>8.3f}")
    print(f"\n  Vector ternormalisasi:")
    print(f"  [{', '.join(f'{v:.3f}' for v in norm)}]")

    # ── COMPUTE BOTH DISTANCES ──
    dists_euclidean = [
        (euclidean(norm, row), idx)
        for idx, row in enumerate(X_all)
    ]
    dists_manhattan = [
        (manhattan(norm, row), idx)
        for idx, row in enumerate(X_all)
    ]

    # ── STEP 3 ──
    print(f"\n[STEP 3] HITUNG JARAK KE {n_train} DATA LATIH")
    print(f"  Euclidean: d = sqrt( SUM (a_i - b_i)^2 )")
    print(f"  Manhattan: d = SUM |a_i - b_i|")
    print(f"\n  Contoh perhitungan untuk baris #1 (Euclidean):")
    print(f"  {'Fitur':20s} {'Input(a)':>8s} {'Train(b)':>8s} {'a-b':>8s} {'(a-b)^2':>8s}")
    print(f"  {'-' * 55}")
    sumsq = 0.0
    for j in range(len(FEATURE_COLS)):
        diff = norm[j] - X_all[0][j]
        sq = diff ** 2
        sumsq += sq
        print(f"  {FEATURE_COLS[j]:20s} {norm[j]:>8.3f} {X_all[0][j]:>8.3f} {diff:>+8.4f} {sq:>8.4f}")
    print(f"  {'-' * 55}")
    print(f"  {'Jumlah kuadrat (SUM)':>47s} = {sumsq:.4f}")
    print(f"  {'d = sqrt(SUM)':>47s} = {math.sqrt(sumsq):.4f}")

    print(f"\n  Tabel 5 baris pertama + baris terakhir:")
    print(f"  {'No':>4s}  {'Selisih (a-b)':>35s}  {'Euclid':>8s}  {'Manh':>8s}  {'Label':>10s}")
    print(f"  {'-' * 68}")

    rows_to_show = min(5, n_train)
    for i in range(rows_to_show):
        de, idx_e = dists_euclidean[i]
        dm, idx_m = dists_manhattan[i]
        diffs = [norm[j] - X_all[i][j] for j in range(len(FEATURE_COLS))]
        diff_str = ", ".join(f"{d:+.3f}" for d in diffs[:3])
        label = rev_classes.get(y_all[i], '?')
        print(f"  {i+1:4d}  {diff_str:>35s}  {de:>8.4f}  {dm:>8.4f}  {label:>10s}")

    if n_train > 5:
        print(f"  {'...':>4s}  {'(skip ' + str(n_train - 6) + ' baris)':>35s}  {'...':>8s}  {'...':>8s}")
        de, idx_e = dists_euclidean[-1]
        dm, idx_m = dists_manhattan[-1]
        diffs = [norm[j] - X_all[-1][j] for j in range(len(FEATURE_COLS))]
        diff_str = ", ".join(f"{d:+.3f}" for d in diffs[:3])
        label = rev_classes.get(y_all[-1], '?')
        print(f"  {n_train:4d}  {diff_str:>35s}  {de:>8.4f}  {dm:>8.4f}  {label:>10s}")

    # ── STEP 4 ──
    print(f"\n[STEP 4] URUTKAN DARI JARAK TERKECIL")
    dists_euclidean.sort(key=lambda x: x[0])
    dists_manhattan.sort(key=lambda x: x[0])

    idx1 = dists_euclidean[0][1]
    print(f"\n  Setelah diurutkan, peringkat #1 (Euclidean):")
    print(f"  Data ke-{idx1+1} dari dataset")
    print(f"    Mood : {rev_classes.get(y_all[idx1], '?')}")
    print(f"    Jarak: {dists_euclidean[0][0]:.4f} (Euclidean)")
    print(f"    Jarak: {dists_manhattan[0][0]:.4f} (Manhattan)")
    print(f"  Vector data tsb: [{', '.join(f'{v:.3f}' for v in X_all[idx1])}]")

    print("-" * 40)

    print(f"  {'Rank':>4s}  {'Euclid':>8s}  {'Manh':>8s}  {'Label':>10s}")
    print(f"  {'-' * 35}")
    for i in range(min(5, n_train)):
        de, idx_e = dists_euclidean[i]
        dm, idx_m = dists_manhattan[i]
        print(f"  {i+1:4d}  {de:>8.4f}  {dm:>8.4f}  {rev_classes.get(y_all[idx_e], '?'):>10s}")
    if n_train > 5:
        print(f"  {'...':>4s}  {'...':>8s}  {'...':>8s}")
        de, idx_e = dists_euclidean[-1]
        dm, idx_m = dists_manhattan[-1]
        print(f"  {n_train:4d}  {de:>8.4f}  {dm:>8.4f}  {rev_classes.get(y_all[idx_e], '?'):>10s}")

    # ── STEP 5 ──
    print(f"\n[STEP 5] AMBIL K TETANGGA + VOTING")
    print("-" * 62)

    results = {}
    for dist_name, dists in [("Euclidean", dists_euclidean), ("Manhattan", dists_manhattan)]:
        print(f"\n  --- {dist_name} ---")
        for k in K_VALUES:
            nearest = dists[:k]
            idxs = [idx for _, idx in nearest]
            labels = [y_all[idx] for idx in idxs]
            counts = Counter(labels)
            winner = max(labels, key=lambda lbl: counts[lbl])
            results[(dist_name, k)] = rev_classes.get(winner, str(winner))

            tetangga_str = ", ".join(
                f"#{i+1}={rev_classes.get(y_all[idx], '?')}({d:.4f})"
                for i, (d, idx) in enumerate(nearest)
            )
            print(f"  K={k}: {tetangga_str}")
            for mood_name, cid in classes.items():
                cnt = counts.get(cid, 0)
                if cnt > 0:
                    print(f"    {mood_name:16s}: {'#' * cnt} ({cnt} suara)")
            print(f"    -> {results[(dist_name, k)]}")

    # ── STEP 6 ──
    print(f"\n{'=' * 68}")
    print(f"{'HASIL AKHIR':>40s}")
    print(f"{'=' * 68}")
    print(f"{'':>20s}  {'Euclidean':>26s}  {'Manhattan':>26s}")
    print(f"{'':>20s}  {'K=3':>8s}  {'K=5':>8s}  {'K=7':>8s}  {'K=3':>8s}  {'K=5':>8s}  {'K=7':>8s}")
    print(f"{'-' * 81}")
    print(f"{'Mood':>20s}  {results[('Euclidean',3)]:>8s}  {results[('Euclidean',5)]:>8s}  {results[('Euclidean',7)]:>8s}  {results[('Manhattan',3)]:>8s}  {results[('Manhattan',5)]:>8s}  {results[('Manhattan',7)]:>8s}")
    print(f"{'=' * 68}")

    # ── CHARTS ──
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    scatter_path = GRAPH_DIR / "predict_scatter.png"
    radar_path = GRAPH_DIR / "radar_input.png"

    print(f"\n  Scatter plot \u2192 {scatter_path}")
    plot_predict_scatter(X_all, y_all, classes, norm, scatter_path)

    print(f"  Radar chart \u2192 {radar_path}")
    plot_radar(records_norm, FEATURE_COLS, classes, radar_path, input_vector=norm, input_label="Input")

    os.startfile(str(scatter_path))
    os.startfile(str(radar_path))


def _ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Input harus angka. Coba lagi.")


def main() -> None:
    records_norm = classes = X_all = y_all = mins = maxs = None

    while True:
        print("\n" + "=" * 55)
        print("   SONG MOOD CLASSIFIER (Spotify 2019)")
        print("=" * 55)
        print("  1. Learn the dataset")
        print("  2. Check the output compare to the dataset")
        print("  3. Input and find the mood")
        print("  0. Exit")
        print("-" * 55)
        choice = input("  Pilihan: ").strip()

        if choice == "1":
            if records_norm is None:
                records_norm, classes, X_all, y_all, mins, maxs = load_data()
            _dataset_report(records_norm, classes, X_all, y_all)

        elif choice == "2":
            if records_norm is None:
                records_norm, classes, X_all, y_all, mins, maxs = load_data()
            _run_evaluation(records_norm, classes, X_all, y_all)

        elif choice == "3":
            _predict_interactive()

        elif choice == "0":
            print("Selesai.")
            break

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
