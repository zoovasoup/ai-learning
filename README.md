# KNN Mood Classifier — Spotify Dataset

**Mata Kuliah:** Kecerdasan Buatan — S1 Rekayasa Perangkat Lunak  
**Semester:** Genap 2025/2026  
**Algoritma:** K-Nearest Neighbor (KNN) — implementasi manual (tanpa Scikit-learn)

---

## Dataset

| Sumber  | Kaggle — Spotify Dataset 1922–2021 (600k tracks)                                |
| ------- | ------------------------------------------------------------------------------- |
| URL     | https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212021-600k-tracks |
| Raw     | `data/tracks.csv` — 586.672 baris, 20 kolom (1922–2021), **gitignored**         |
| Working | `data/tracks_mood.csv` — 1.500 baris, 10 kolom (tahun 2019 saja)                |

### Pipeline Filtering

1. Filter tahun = 2019 (~29.000 baris)
2. Hapus baris dengan nilai null
3. Hitung median **valence** dan **energy** dari data 2019
4. Label mood berdasarkan Russell's Circumplex Model:
   - **Happy** → valence ≥ median, energy ≥ median
   - **Chill** → valence ≥ median, energy < median
   - **Energetic** → valence < median, energy ≥ median
   - **Sad** → valence < median, energy < median
5. Acak (random shuffle)
6. Ambil 1.500 baris pertama sebagai sampel kerja

### Kolom

| Kolom            | Rentang                         | Keterangan                       |
| ---------------- | ------------------------------- | -------------------------------- |
| danceability     | 0–1                             |                                  |
| energy           | 0–1                             |                                  |
| loudness         | -60–0 dB                        |                                  |
| speechiness      | 0–1                             |                                  |
| acousticness     | 0–1                             |                                  |
| instrumentalness | 0–1                             |                                  |
| liveness         | 0–1                             | Kehadiran penonton               |
| valence          | 0–1                             | Positivitas musik                |
| tempo            | 0–250 BPM                       |                                  |
| **mood**         | Happy / Chill / Energetic / Sad | Label hasil Russell's Circumplex |

---

## Algoritma: K-Nearest Neighbor

KNN bekerja dengan prinsip **"seperti apa tetangganya, seperti itu pula dirinya"**.
Data baru diklasifikasikan berdasarkan mayoritas label dari _k_ tetangga terdekat.

### Langkah-langkah (detail di `knn/compute.py`)

1. **Input fitur** — 9 fitur numerik lagu yang akan diprediksi
2. **Normalisasi Min-Max** — semua fitur diskalakan ke [0, 1] agar tidak ada fitur yang mendominasi karena skala angkanya besar
3. **Hitung jarak** ke setiap data latih:
   - **Euclidean (L2):** $d = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$
   - **Manhattan (L1):** $d = \sum_{i=1}^{n} |a_i - b_i|$
4. **Urutkan** dari jarak terkecil ke terbesar
5. **Ambil _k_ tetangga terdekat**
6. **Voting** — mood dengan suara terbanyak dari _k_ tetangga menjadi hasil prediksi

### Contoh perhitungan Euclidean (baris pertama)

| Fitur            | Input(a) | Train(b) | a–b    | (a–b)²     |
| ---------------- | -------- | -------- | ------ | ---------- |
| danceability     | 0,729    | 0,795    | –0,066 | 0,0043     |
| energy           | 0,820    | 0,783    | +0,037 | 0,0014     |
| loudness         | 0,844    | 0,851    | –0,007 | 0,0000     |
| speechiness      | 0,053    | 0,090    | –0,038 | 0,0014     |
| acousticness     | 0,100    | 0,017    | +0,083 | 0,0069     |
| instrumentalness | 0,000    | 0,000    | +0,000 | 0,0000     |
| liveness         | 0,104    | 0,177    | –0,073 | 0,0053     |
| valence          | 0,664    | 0,210    | +0,454 | 0,2057     |
| tempo            | 0,577    | 0,452    | +0,125 | 0,0157     |
| **SUM**          |          |          |        | **0,2407** |
| **d = √SUM**     |          |          |        | **0,4906** |

---

## Evaluasi

### 80:20 Split

- 80% data latih (1.200 baris), 20% data uji (300 baris)
- Menjalankan KNN untuk Euclidean dan Manhattan secara terpisah
- Output: akurasi + confusion matrix per distance function

### 5-Fold Cross Validation

- Dataset dibagi 5 lipatan merata
- Setiap lipatan bergantian jadi validation set, sisanya training
- Hasil: rata-rata akurasi ± standar deviasi

### Hasil Terbaik

| Distance  | K   | Akurasi        |
| --------- | --- | -------------- |
| Manhattan | 7   | 87,80% ± 2,02% |

---

## Confusion Matrix

Confusion matrix adalah tabel yang menunjukkan **perbandingan antara prediksi model dengan label asli (ground truth)**. Berguna untuk melihat *di mana* model salah klasifikasi, bukan hanya berapa banyak yang salah.

### Format (Multi-class, 4 mood)

| Aktual \ Prediksi | Happy | Chill | Energetic | Sad |
|-------------------|-------|-------|-----------|-----|
| **Happy** | 72 | 3 | 4 | 1 |
| **Chill** | 2 | 58 | 0 | 0 |
| **Energetic** | 7 | 0 | 43 | 2 |
| **Sad** | 1 | 2 | 3 | 102 |

- **Diagonal** (kiri atas ke kanan bawah) = prediksi **BENAR** — model menebak sesuai label asli
- **Off-diagonal** = prediksi **SALAH** — model keliru membedakan mood
- Contoh: baris Energetic kolom Happy = 7 → ada *7 lagu Energetic* yang diprediksi sebagai *Happy*
- Semakin pekat warna diagonal, semakin baik model

Dalam output program, confusion matrix divisualisasikan sebagai **heatmap** (warna biru-putih) dengan angka di setiap sel.

---

## PCA (Principal Component Analysis)

PCA adalah teknik reduksi dimensi yang mengubah fitur asli menjadi **komponen utama (principal components)** — kombinasi linear dari fitur asli yang menangkap variasi terbesar dalam data.

### Apa itu PC1 dan PC2?

- **PC1 (Principal Component 1):** sumbu yang menangkap **varian terbesar** dari data. Semakin besar % varian yang dijelaskan PC1, semakin banyak informasi yang terkandung di dalamnya.
- **PC2 (Principal Component 2):** sumbu kedua terbesar, **ortogonal (tegak lurus)** terhadap PC1. Menangkap varian terbesar yang tersisa setelah PC1.
- PC1 dan PC2 digunakan untuk memproyeksikan data 9 dimensi ke **bidang 2D** untuk visualisasi.
- Dalam output program, *Scatter Plot PCA* menampilkan 1.500 titik data latih diproyeksikan ke PC1–PC2, di-overlay dengan *input pengguna* (bintang merah ⭐) sehingga terlihat di sekitar mood mana input tersebut berada.

### Cara Menghitung PCA (via SVD)

Implementasi dalam program (`reporting.py:271-277`) menggunakan **Singular Value Decomposition (SVD)**:

#### Langkah 1: Center data
Kurangkan setiap fitur dengan rata-ratanya agar data berpusat di (0,0):

$$X_{\text{centered}} = X - \bar{X}$$

Di mana $\bar{X}$ = vektor rata-rata tiap fitur (9 nilai).

#### Langkah 2: Dekomposisi SVD

$$X_{\text{centered}} = U \cdot S \cdot V^T$$

- $U$ = matriks ortogonal (baris = data point, kolom = komponen)
- $S$ = vektor **singular values** (diagonal, urut menurun): $s_1 \ge s_2 \ge ... \ge s_9$
- $V^T$ = matriks **singular vectors** (baris = komponen utama, kolom = fitur asli)

#### Langkah 3: Proyeksi ke PC1 dan PC2

$$X_{PCA} = X_{\text{centered}} \cdot V^T[:2]^T$$

Ini mengambil 2 baris pertama dari $V^T$ (PC1 dan PC2) sebagai sumbu baru:

- **PC1** = baris pertama $V^T$ = vektor bobot $[w_{1,1}, w_{1,2}, ..., w_{1,9}]$
  - Nilai PC1 untuk data ke-$i$: $pc1_i = \sum_{j=1}^{9} w_{1,j} \cdot (x_{i,j} - \bar{x}_j)$
- **PC2** = baris kedua $V^T$ = vektor bobot $[w_{2,1}, w_{2,2}, ..., w_{2,9}]$
  - Nilai PC2 untuk data ke-$i$: $pc2_i = \sum_{j=1}^{9} w_{2,j} \cdot (x_{i,j} - \bar{x}_j)$

#### Langkah 4: Variance explained ratio

$$R^2_1 = \frac{s_1^2}{\sum_{j=1}^{9} s_j^2}, \quad R^2_2 = \frac{s_2^2}{\sum_{j=1}^{9} s_j^2}$$

- $R^2_1$ = proporsi varian data yang dijelaskan PC1 (misal 58%)
- $R^2_2$ = proporsi varian yang dijelaskan PC2 (misal 18%)
- Total varian yang dipertahankan = $R^2_1 + R^2_2$ (misal 76% — informasi yang "ditangkap" dari 9 dimensi asli)

#### Contoh ilustrasi bobot PC (nilai fiktif)

Setelah SVD, $V^T$ bisa menghasilkan bobot seperti:

| Fitur | Bobot PC1 | Bobot PC2 |
|-------|-----------|-----------|
| danceability | +0.52 | -0.11 |
| energy | +0.48 | +0.09 |
| loudness | +0.41 | +0.03 |
| speechiness | -0.12 | +0.58 |
| acousticness | -0.38 | -0.15 |
| instrumentalness | -0.05 | +0.62 |
| liveness | -0.01 | +0.32 |
| valence | +0.39 | -0.28 |
| tempo | +0.13 | +0.21 |

Artinya:
- **PC1** didominasi danceability, energy, loudness, valence (bobot positif besar) → merepresentasikan "seberapa aktif/positif sebuah lagu"
- **PC2** didominasi speechiness, instrumentalness (bobot positif besar) → merepresentasikan "seberapa banyak elemen vokal/instrumen"

### Interpretasi Scatter Plot

- Sumbu X = PC1, sumbu Y = PC2
- Setiap titik adalah satu lagu, diwarnai berdasarkan mood
- Bintang merah = posisi input pengguna setelah diproyeksikan ke PC1–PC2
- Jika bintang merah berada di area titik hijau (Happy), maka wajar jika KNN memprediksi Happy

---

## Output Program

### Menu (3 opsi)

| Opsi | Fungsi                                                                                                            |
| ---- | ----------------------------------------------------------------------------------------------------------------- |
| 1    | **Eksplorasi Dataset** — tampilkan jumlah baris, distribusi mood, statistik per fitur, info pipeline filtering    |
| 2    | **Evaluasi** — 80:20 split + 5-fold CV untuk Euclidean & Manhattan, tampilkan akurasi + confusion matrix          |
| 3    | **Prediksi Manual** — input 9 fitur, tampilkan step-by-step: normalisasi → tabel jarak → ranking → voting → hasil |

### Output Opsi 3 (Step-by-Step)

1. Menampilkan **nilai input + hasil normalisasi**
2. Tabel **contoh perhitungan Euclidean** per-fitur (input, train, selisih, kuadrat selisih, SUM, sqrt)
3. Tabel **jarak** 5 baris pertama + terakhir (Euclidean dan Manhattan)
4. Ranking setelah diurutkan + data tetangga terdekat
5. **Voting** per K (3, 5, 7) untuk kedua distance function
6. **Tabel perbandingan akhir** — 6 hasil (2 distance × 3 K)
7. **Scatter plot PCA** (otomatis terbuka) — semua data latih + input sebagai bintang merah
8. **Radar chart** (otomatis terbuka) — rata-rata fitur per mood + input sebagai garis putus-putus

### File Output

| File             | Lokasi                                        |
| ---------------- | --------------------------------------------- |
| Log pipeline     | `output/logs/log.txt`                         |
| CV Euclidean     | `output/logs/cv_euclidean.txt`                |
| CV Manhattan     | `output/logs/cv_manhattan.txt`                |
| Confusion matrix | `output/logs/*.png`                           |
| Scatter PCA      | `output/graphs/pca_k*.png`, `pca_explore.png` |
| Radar chart      | `output/graphs/radar*.png`                    |
| Predict scatter  | `output/graphs/predict_scatter.png`           |

---

## Struktur Proyek

```
├── data/
│   ├── tracks.csv                 # (IGNORED) raw — 111MB
│   └── tracks_mood.csv            # 1.500 baris, 2019 + mood label
├── knn/
│   ├── __init__.py
│   ├── compute.py                 # Semua fungsi KNN + step-by-step docstring
│   └── preprocessing.py           # read_csv, encode, normalize, split
├── scripts/
│   └── generate_mood_csv.py       # Filter tahun + label mood
├── output/
│   ├── logs/
│   │   ├── log.txt
│   │   ├── cv_euclidean.txt
│   │   └── cv_manhattan.txt
│   └── graphs/
│       ├── confusion_matrix_k*.png
│       ├── pca_k*.png
│       ├── pca_explore.png
│       ├── radar*.png
│       └── predict_scatter.png
├── main.py                        # Entry point — menu 3 opsi
├── pipeline.py                    # Orchestrator (run_pipeline, cross_validate)
├── reporting.py                   # Logging + plotting (PCA, radar, confusion matrix)
└── README.md
```

---

## Cara Menjalankan

```bash
# 1. Generate dataset (jika belum ada)
python scripts/generate_mood_csv.py

# 2. Jalankan program
python main.py
```

Pilih menu:

- **1** → Lihat laporan dataset
- **2** → Jalankan evaluasi (split + CV) — output ke `output/logs/` dan `output/graphs/`
- **3** → Prediksi manual — masukkan 9 fitur lagu, lihat step-by-step + grafik
