"""=============================================================
FILTER + LABEL DATASET SPOTIFY
=============================================================
Langkah-langkah pembuatan tracks_mood.csv:
  1. Baca tracks.csv (586.672 lagu, 1922-2021)
  2. Filter: ambil hanya tahun 2019
  3. Hitung median valence & energy dari data 2019
  4. Tentukan mood tiap lagu pakai Russell's Circumplex:
       ┌────────────┬───────────┬──────────┐
       │ Mood       │ Valence   │ Energy   │
       ├────────────┼───────────┼──────────┤
       │ Happy      │ ≥ median  │ ≥ median │
       │ Chill      │ ≥ median  │ < median │
       │ Energetic  │ < median  │ ≥ median │
       │ Sad        │ < median  │ < median │
       └────────────┴───────────┴──────────┘
  5. Shuffle + ambil 1.500 baris pertama
  6. Simpan ke tracks_mood.csv

Contoh (Russell's Circumplex Model):
  valence=0.75, energy=0.82
    → valence ≥ median (0.50) ✓, energy ≥ median (0.64) ✓
    → Mood: HAPPY

  valence=0.30, energy=0.55
    → valence < median (0.50) ✓, energy < median (0.64) ✓
    → Mood: SAD
============================================================="""

import argparse
import csv
import random
from pathlib import Path

SRC = Path("data/tracks.csv")
DST = Path("data/tracks_mood.csv")
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


def _derive_mood(valence: float, energy: float, v_med: float, e_med: float) -> str:
    v_high = valence >= v_med
    e_high = energy >= e_med
    if v_high and e_high:
        return "Happy"
    if v_high and not e_high:
        return "Chill"
    if not v_high and e_high:
        return "Energetic"
    return "Sad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2019, help="Filter year")
    parser.add_argument("--max-rows", type=int, default=1500, help="Max rows to output")
    args = parser.parse_args()

    print(f"Reading {SRC}...")
    with open(SRC, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"Total: {len(reader)}")
    filtered = []
    for row in reader:
        yr = row.get("release_date", "")[:4]
        try:
            if int(yr) == args.year:
                filtered.append(row)
        except ValueError:
            pass
    print(f"Filtered ({args.year}): {len(filtered)}")

    # compute medians for mood boundary
    valences = [float(r["valence"]) for r in filtered]
    energies = [float(r["energy"]) for r in filtered]
    v_med = sorted(valences)[len(valences) // 2]
    e_med = sorted(energies)[len(energies) // 2]
    print(f"Median: valence={v_med:.3f}, energy={e_med:.3f}")

    # derive mood
    rows_out = []
    for r in filtered:
        valence = float(r["valence"])
        energy = float(r["energy"])
        mood = _derive_mood(valence, energy, v_med, e_med)
        out = {col: r[col] for col in FEATURE_COLS}
        out["mood"] = mood
        rows_out.append(out)

    # shuffle + limit
    random.seed(42)
    random.shuffle(rows_out)
    rows_out = rows_out[: args.max_rows]

    mood_counts = {}
    for r in rows_out:
        mood_counts[r["mood"]] = mood_counts.get(r["mood"], 0) + 1
    print("Mood distribution:")
    for m, c in sorted(mood_counts.items()):
        print(f"  {m}: {c} ({c/len(rows_out):.0%})")

    output_cols = FEATURE_COLS + ["mood"]
    with open(DST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=output_cols)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\nWritten {len(rows_out)} rows → {DST}")


if __name__ == "__main__":
    main()
