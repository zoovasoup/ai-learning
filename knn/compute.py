from __future__ import annotations

import math
from collections import Counter
from typing import Callable

"""=============================================================
KNN — SEMUA FUNGSI HITUNGAN + STEP-BY-STEP
=============================================================

KONSEP DASAR KNN (K-Nearest Neighbor):
  Algoritma ini bekerja dengan prinsip "seperti apa tetangganya,
  seperti itu pula dirinya". Sebuah data baru diklasifikasikan
  berdasarkan mayoritas label dari k tetangga terdekat.

LANGKAH-LANGKAH PREDIKSI (K=3, Euclidean):
  1. Masukkan fitur lagu baru (test_row), misal:
       [0.71, 0.82, -4.50, 0.05, 0.20, 0.00, 0.10, 0.60, 120.0]

  2. Normalisasi pakai min-max dari data latih supaya skala fitur
     seimbang (ga ada fitur yg mendominasi hanya karena angkanya besar).

  3. Hitung jarak test_row ke SETIAP baris data latih (1500 lagu):
       Setiap baris → satu nilai jarak (Euclidean / Manhattan)

  4. Urutkan hasil dari jarak TERKECIL ke terbesar.

  5. Ambil k baris teratas = TETANGGA TERDEKAT.

  6. Ambil label mood dari k tetangga itu.

  7. Voting: label dengan suara terbanyak = HASIL PREDIKSI.

CONTOH ILUSTRASI (K=3, Euclidean):
  Setelah dihitung dan diurutkan:
    ┌──────┬──────────┬──────────┐
    │ Rank │ Distance │  Mood    │
    ├──────┼──────────┼──────────┤
    │  1   │  0.084   │  Happy   │
    │  2   │  0.112   │  Happy   │
    │  3   │  0.156   │  Sad     │
    │  4   │  0.189   │  Chill   │
    │ ...  │   ...    │  ...     │
    │ 1500 │  2.345   │  Energet │
    └──────┴──────────┴──────────┘

  K=3 tetangga terdekat: [Happy, Happy, Sad]
  Voting: Happy = 2, Sad = 1
  HASIL: Happy

"""

DistanceFunc = Callable[[list[float], list[float]], float]


def euclidean(a: list[float], b: list[float]) -> float:
    """=========================================================
    EUCLIDEAN DISTANCE (L2) — Jarak Garis Lurus
    =========================================================
    Rumus:
        d(a,b) = sqrt( (a1-b1)^2 + (a2-b2)^2 + ... + (a9-b9)^2 )

    Penjelasan:
      Jarak Euclidean mengukur panjang garis lurus antara dua
      titik dalam ruang n-dimensi.
      > Setiap fitur dikurangkan (a_i - b_i)
      > Hasilnya dikuadratkan agar tidak ada nilai negatif
      > Semua dijumlahkan
      > Di-akar-kuadratkan untuk mengembalikan ke skala asli

    Contoh step-by-step dengan 9 fitur audio:
      Fitur               Test(a)   Train(b)   a-b     (a-b)^2
      ────────────────────────────────────────────────────────
      danceability        0.710     0.650     0.060    0.0036
      energy              0.820     0.780     0.040    0.0016
      loudness            0.750     0.800    -0.050    0.0025
      speechiness         0.050     0.080    -0.030    0.0009
      acousticness        0.200     0.150     0.050    0.0025
      instrumentalness    0.000     0.000     0.000    0.0000
      liveness            0.100     0.120    -0.020    0.0004
      valence             0.600     0.550     0.050    0.0025
      tempo               0.620     0.580     0.040    0.0016
      ────────────────────────────────────────────────────────
      Jumlah (sum of squares)                     = 0.0156
      d = sqrt(0.0156)                            = 0.1249

    Parameter:
      a, b — dua vektor dengan panjang yang sama (9 fitur)

    Return:
      Jarak Euclidean (float ≥ 0)
    ========================================================="""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    # Langkah 1: hitung selisih tiap fitur → kuadratkan → jumlahkan
    sum_sq = sum((x - y) ** 2 for x, y in zip(a, b))

    # Langkah 2: akar kuadrat dari total
    return math.sqrt(sum_sq)


def manhattan(a: list[float], b: list[float]) -> float:
    """=========================================================
    MANHATTAN DISTANCE (L1) — Jarak Grid / Kota
    =========================================================
    Rumus:
        d(a,b) = |a1-b1| + |a2-b2| + ... + |a9-b9|

    Penjelasan:
      Jarak Manhattan menjumlahkan selisih mutlak tiap dimensi.
      Disebut "Manhattan" karena seperti berjalan di grid jalan
      kota Manhattan — hanya bisa lurus, tidak bisa diagonal.

    Contoh step-by-step:
      Fitur               Test(a)   Train(b)   |a-b|
      ──────────────────────────────────────────────
      danceability        0.710     0.650     0.060
      energy              0.820     0.780     0.040
      loudness            0.750     0.800     0.050
      speechiness         0.050     0.080     0.030
      acousticness        0.200     0.150     0.050
      instrumentalness    0.000     0.000     0.000
      liveness            0.100     0.120     0.020
      valence             0.600     0.550     0.050
      tempo               0.620     0.580     0.040
      ──────────────────────────────────────────────
      Jumlah                                     0.340

    Parameter:
      a, b — dua vektor dengan panjang yang sama (9 fitur)

    Return:
      Jarak Manhattan (float ≥ 0)
    ========================================================="""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")

    # Langkah 1: hitung selisih mutlak tiap fitur → jumlahkan
    return sum(abs(x - y) for x, y in zip(a, b))


def predict_one(
    test_row: list[float],
    train_X: list[list[float]],
    train_y: list[int],
    k: int,
    dist_func: DistanceFunc = euclidean,
) -> int:
    """=========================================================
    PREDIKSI 1 DATA TEST DENGAN KNN
    =========================================================
    Langkah-langkah:
      1. Hitung jarak test_row ke SETIAP baris di train_X
      2. Simpan hasil sebagai daftar (jarak, label)
      3. Urutkan daftar dari jarak terkecil
      4. Ambil k baris teratas (k tetangga terdekat)
      5. Ambil label dari k tetangga
      6. Hitung voting (label mana paling sering muncul)
      7. Kembalikan label pemenang

    Contoh hasil sebelum diurutkan (beberapa baris pertama):
      ┌──────────┬───────┐
      │ Jarak    │ Label │
      ├──────────┼───────┤
      │ 0.084    │   2   │  (Happy)
      │ 0.450    │   0   │  (Energetic)
      │ 0.112    │   2   │  (Happy)
      │ 0.890    │   1   │  (Chill)
      │ 0.156    │   3   │  (Sad)
      │ ...      │  ...  │
      └──────────┴───────┘

    Setelah diurutkan (ascending by jarak):
      Rank  Jarak    Label
      1     0.084      2  (Happy)
      2     0.112      2  (Happy)
      3     0.156      3  (Sad)
      4     0.450      0  (Energetic)
      ...

    K=3 → ambil 3 teratas: label [2, 2, 3]
    Voting: label 2 = 2 suara, label 3 = 1 suara
    Pemenang: label 2 (Happy)

    Parameter:
      test_row — vektor fitur data yang akan diprediksi (len=9)
      train_X  — daftar vektor fitur data latih
      train_y  — daftar label data latih (int 0-3)
      k        — jumlah tetangga yang dipertimbangkan
      dist_func — fungsi jarak yang dipakai (euclidean / manhattan)

    Return:
      Label hasil prediksi (int 0-3)
    ========================================================="""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if len(train_X) != len(train_y):
        raise ValueError("train_X and train_y must have the same length")
    if not train_X:
        raise ValueError("train_X must not be empty")

    # Langkah 1-2: hitung jarak ke semua data latih
    distances = [
        (dist_func(test_row, train_row), label)
        for train_row, label in zip(train_X, train_y)
    ]

    # Langkah 3: urutkan dari jarak terdekat
    distances.sort(key=lambda item: item[0])

    # Langkah 4-5: ambil k label tetangga terdekat
    nearest_labels = [label for _, label in distances[:k]]

    # Langkah 6-7: voting mayoritas
    vote_counts = Counter(nearest_labels)
    return max(nearest_labels, key=lambda label: vote_counts[label])


def predict(
    test_X: list[list[float]],
    train_X: list[list[float]],
    train_y: list[int],
    k: int,
    dist_func: DistanceFunc = euclidean,
) -> list[int]:
    """=========================================================
    PREDIKSI BANYAK DATA TEST
    =========================================================
    Menjalankan predict_one untuk SETIAP baris di test_X.

    Parameter:
      test_X — daftar vektor fitur yang akan diprediksi
      train_X — daftar vektor fitur data latih
      train_y — daftar label data latih
      k       — jumlah tetangga
      dist_func — fungsi jarak

    Return:
      Daftar label hasil prediksi (int), urut sesuai test_X
    ========================================================="""
    return [predict_one(test_row, train_X, train_y, k, dist_func) for test_row in test_X]


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    """=========================================================
    AKURASI
    =========================================================
    Rumus:
        accuracy = (jumlah prediksi benar / total data) × 100%

    Contoh:
      Data ke    Actual   Predicted  Status
      ─────────────────────────────────────
        1         2          2       ✅ (Happy → Happy)
        2         2          3       ❌ (Happy → Sad)
        3         3          3       ✅ (Sad → Sad)
        4         0          0       ✅ (Energetic → Energetic)
        5         1          2       ❌ (Chill → Happy)
       ...       ...        ...     ...
      ─────────────────────────────────────
      Total benar = 255, Total data = 300
      Accuracy = 255/300 × 100% = 85.00%

    Parameter:
      y_true — label asli (ground truth)
      y_pred — label hasil prediksi

    Return:
      Akurasi dalam persen (0.00 - 100.00)
    ========================================================="""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true must not be empty")

    # Hitung berapa banyak prediksi yang cocok dengan label asli
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))

    # Konversi ke persen
    return correct / len(y_true) * 100


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    """=========================================================
    CONFUSION MATRIX (MATRIKS KEKELIRUAN) — n×n
    =========================================================
    Confusion matrix menunjukkan perbandingan antara label asli
    (baris) dengan label prediksi (kolom).

    Baris = ACTUAL (label yang sebenarnya)
    Kolom = PREDICTED (label yang diprediksi oleh model)

    Contoh (K=3, Euclidean):
                      PREDICTED
                    Ener  Chill  Sad   Happy
    ACTUAL  Energetic  46     1    10      4
            Chill       0    40     0      8
            Sad         5     5    77      1
            Happy       5     6     1     91

    Cara baca:
      - Energetic ditebak Energetic = 46 ✅
      - Energetic salah ditebak Sad = 10 ❌
      - Chill semua benar ditebak Chill = 40 ✅
      - Happy salah ditebak Chill = 6 ❌

    Parameter:
      y_true — label asli (list of int)
      y_pred — label hasil prediksi (list of int)

    Return:
      Matriks n×n, dimana cm[actual][predicted] = jumlah data
    ========================================================="""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Tentukan jumlah kelas (dari label terbesar)
    n_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1

    # Buat matriks n×n, isi 0
    cm = [[0] * n_classes for _ in range(n_classes)]

    # Isi matriks: untuk tiap pasangan (actual, predicted), increment
    for actual, predicted in zip(y_true, y_pred):
        cm[actual][predicted] += 1

    return cm
