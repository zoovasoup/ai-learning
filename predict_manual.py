from __future__ import annotations

from pathlib import Path

from knn.classifier import predict_one
from knn.distance import euclidean, manhattan
from knn.preprocessing import encode_labels, min_max_normalize, read_csv

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
K_VALUES = [3, 5, 7]
CSV_PATH = Path("data/tracks_mood.csv")

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
    print("\n=== INPUT FITUR LAGU ===")
    return {
        "danceability": _ask_float("Danceability (0.0-1.0): "),
        "energy": _ask_float("Energy (0.0-1.0)       : "),
        "loudness": _ask_float("Loudness (-60 - 0 dB) : "),
        "speechiness": _ask_float("Speechiness (0.0-1.0) : "),
        "acousticness": _ask_float("Acousticness (0.0-1.0): "),
        "instrumentalness": _ask_float("Instrumentalness (0-1): "),
        "liveness": _ask_float("Liveness (0.0-1.0)    : "),
        "valence": _ask_float("Valence (0.0-1.0)      : "),
        "tempo": _ask_float("Tempo (BPM, 0-250)     : "),
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
        print(f"{'Data':>25s}  {'K=3':>10s}  {'K=5':>10s}  {'K=7':>10s}")
        print(f"{'Jarak':>25s}  {dist_name:>10s}")
        print(f"{'-' * 55}")

        results = {}
        for k in K_VALUES:
            pred = predict_one(norm, train_X, train_y, k, dist_func)
            results[k] = rev_classes.get(pred, str(pred))

        print(f"{'Mood':>25s}  {results[3]:>10s}  {results[5]:>10s}  {results[7]:>10s}")
        print(f"{'=' * 55}")

        lanjut = input("\nInput lagu lain? (y/n): ").strip().lower()
        if lanjut != "y":
            print("Selesai.")
            break


if __name__ == "__main__":
    main()
