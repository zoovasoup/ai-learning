from __future__ import annotations

from pathlib import Path

from knn.classifier import predict_one
from knn.distance import euclidean, manhattan
from knn.preprocessing import encode_labels, min_max_normalize, read_csv

FEATURE_COLS = [
    "nilai_tugas",
    "nilai_kuis",
    "nilai_uts",
    "nilai_uas",
    "absensi",
    "jam_belajar",
]
LABEL_COL = "label"
K_VALUES = [3, 5, 7]
CSV_PATH = Path("data/dataset_mahasiswa.csv")

DIST_CHOICES = {"1": ("Euclidean", euclidean), "2": ("Manhattan", manhattan)}


def _normalize_input(
    values: dict[str, float],
    mins: dict[str, float],
    maxs: dict[str, float],
) -> list[float]:
    result = []
    for col in FEATURE_COLS:
        v = values[col]
        mn = mins[col]
        mx = maxs[col]
        if mx - mn == 0:
            result.append(0.0)
        else:
            result.append((v - mn) / (mx - mn))
    return result


def _ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Input harus angka. Coba lagi.")


def _ask_data() -> dict[str, float]:
    print("\n=== INPUT DATA MAHASISWA ===")
    return {
        "nilai_tugas": _ask_float("Nilai Tugas (0-100)   : "),
        "nilai_kuis": _ask_float("Nilai Kuis (0-100)    : "),
        "nilai_uts": _ask_float("Nilai UTS (0-100)     : "),
        "nilai_uas": _ask_float("Nilai UAS (0-100)     : "),
        "absensi": _ask_float("Absensi (0-100)       : "),
        "jam_belajar": _ask_float("Jam Belajar/Minggu (0-40): "),
    }


def _ask_dist() -> tuple[str, object]:
    print("\nPilih jarak:")
    print("  1. Euclidean (L2)")
    print("  2. Manhattan (L1)")
    while True:
        c = input("Pilihan (1/2): ").strip()
        if c in DIST_CHOICES:
            return DIST_CHOICES[c]
        print("Pilihan tidak valid.")


def main() -> None:
    records = read_csv(CSV_PATH)
    records_enc, classes = encode_labels(records, LABEL_COL)
    records_norm, mins, maxs = min_max_normalize(records_enc, FEATURE_COLS)

    train_X = [[float(row[c]) for c in FEATURE_COLS] for row in records_norm]
    train_y = [int(row[LABEL_COL]) for row in records_norm]

    rev_classes = {v: k for k, v in classes.items()}

    while True:
        raw = _ask_data()
        norm = _normalize_input(raw, mins, maxs)
        dist_name, dist_func = _ask_dist()

        print(f"\n{'=' * 55}")
        print(f"{'Data':>25s}  {'K=3':>8s}  {'K=5':>8s}  {'K=7':>8s}")
        print(f"{'Jarak':>25s}  {dist_name:>8s}")
        print(f"{'-' * 55}")

        results = {}
        for k in K_VALUES:
            pred = predict_one(norm, train_X, train_y, k, dist_func)
            results[k] = rev_classes.get(pred, str(pred))

        print(f"{'Hasil':>25s}  {results[3]:>8s}  {results[5]:>8s}  {results[7]:>8s}")
        print(f"{'=' * 55}")

        lanjut = input("\nInput lagi? (y/n): ").strip().lower()
        if lanjut != "y":
            print("Selesai.")
            break


if __name__ == "__main__":
    main()
