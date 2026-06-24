# KNN Mood Classifier — Spotify Dataset

**Mata Kuliah:** Kecerdasan Buatan — S1 Rekayasa Perangkat Lunak  
**Semester:** Genap 2025/2026  
**Algoritma:** K-Nearest Neighbor (KNN) — implementasi manual (tanpa Scikit-learn)

---

## Dataset

| Sumber | Kaggle — Spotify Dataset 1922–2021 (600k tracks) |
|--------|---------------------------------------------------|
| URL    | https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212021-600k-tracks |
| Raw    | `data/tracks.csv` — 586.672 baris, 20 kolom (1922–2021), **gitignored** |
| Working | `data/tracks_mood.csv` — 1.500 baris, 10 kolom (tahun 2019 saja) |

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

| Kolom | Rentang | Keterangan |
|-------|---------|------------|
| danceability | 0–1 | |
| energy | 0–1 | |
| loudness | -60–0 dB | |
| speechiness | 0–1 | |
| acousticness | 0–1 | |
| instrumentalness | 0–1 | |
| liveness | 0–1 | Kehadiran penonton |
| valence | 0–1 | Positivitas musik |
| tempo | 0–250 BPM | |
| **mood** | Happy / Chill / Energetic / Sad | Label hasil Russell's Circumplex |

---

## Algoritma: K-Nearest Neighbor

KNN bekerja dengan prinsip **"seperti apa tetangganya, seperti itu pula dirinya"**.
Data baru diklasifikasikan berdasarkan mayoritas label dari *k* tetangga terdekat.

### Langkah-langkah (detail di `knn/compute.py`)

1. **Input fitur** — 9 fitur numerik lagu yang akan diprediksi
2. **Normalisasi Min-Max** — semua fitur diskalakan ke [0, 1] agar tidak ada fitur yang mendominasi karena skala angkanya besar
3. **Hitung jarak** ke setiap data latih:
   - **Euclidean (L2):** $d = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$
   - **Manhattan (L1):** $d = \sum_{i=1}^{n} |a_i - b_i|$
4. **Urutkan** dari jarak terkecil ke terbesar
5. **Ambil *k* tetangga terdekat**
6. **Voting** — mood dengan suara terbanyak dari *k* tetangga menjadi hasil prediksi

### Contoh perhitungan Euclidean (baris pertama)

| Fitur | Input(a) | Train(b) | a–b | (a–b)² |
|-------|----------|----------|-----|--------|
| danceability | 0,729 | 0,795 | –0,066 | 0,0043 |
| energy | 0,820 | 0,783 | +0,037 | 0,0014 |
| loudness | 0,844 | 0,851 | –0,007 | 0,0000 |
| speechiness | 0,053 | 0,090 | –0,038 | 0,0014 |
| acousticness | 0,100 | 0,017 | +0,083 | 0,0069 |
| instrumentalness | 0,000 | 0,000 | +0,000 | 0,0000 |
| liveness | 0,104 | 0,177 | –0,073 | 0,0053 |
| valence | 0,664 | 0,210 | +0,454 | 0,2057 |
| tempo | 0,577 | 0,452 | +0,125 | 0,0157 |
| **SUM** | | | | **0,2407** |
| **d = √SUM** | | | | **0,4906** |

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

| Distance | K | Akurasi |
|----------|---|---------|
| Manhattan | 7 | 87,80% ± 2,02% |

---

## PCA (Principal Component Analysis)

PCA adalah teknik reduksi dimensi yang mengubah fitur asli menjadi **komponen utama (principal components)** — kombinasi linear dari fitur asli yang menangkap variasi terbesar dalam data.

- **PC1 (Principal Component 1):** komponen yang menangkap **varian terbesar** dari data. Semakin besar variansi yang dijelaskan PC1, semakin banyak informasi yang terkandung di dalamnya.
- **PC2 (Principal Component 2):** komponen kedua terbesar, **ortogonal (tegak lurus)** terhadap PC1. Menangkap varian terbesar yang tersisa setelah PC1.
- PC1 dan PC2 digunakan untuk memproyeksikan data 9 dimensi ke **bidang 2D** untuk visualisasi.
- Dalam output program, *Scatter Plot PCA* menampilkan 1.500 titik data latih diproyeksikan ke PC1–PC2, di-overlay dengan *input pengguna* (bintang merah ⭐) sehingga terlihat di sekitar mood mana input tersebut berada.

### Interpretasi Scatter Plot

- Sumbu X = PC1, sumbu Y = PC2
- Setiap titik adalah satu lagu, diwarnai berdasarkan mood
- Bintang merah = posisi input pengguna setelah diproyeksikan ke PC1–PC2
- Jika bintang merah berada di area titik hijau (Happy), maka wajar jika KNN memprediksi Happy

---

## Output Program

### Menu (3 opsi)

| Opsi | Fungsi |
|------|--------|
| 1 | **Eksplorasi Dataset** — tampilkan jumlah baris, distribusi mood, statistik per fitur, info pipeline filtering |
| 2 | **Evaluasi** — 80:20 split + 5-fold CV untuk Euclidean & Manhattan, tampilkan akurasi + confusion matrix |
| 3 | **Prediksi Manual** — input 9 fitur, tampilkan step-by-step: normalisasi → tabel jarak → ranking → voting → hasil |

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

| File | Lokasi |
|------|--------|
| Log pipeline | `output/logs/log.txt` |
| CV Euclidean | `output/logs/cv_euclidean.txt` |
| CV Manhattan | `output/logs/cv_manhattan.txt` |
| Confusion matrix | `output/logs/*.png` |
| Scatter PCA | `output/graphs/pca_k*.png`, `pca_explore.png` |
| Radar chart | `output/graphs/radar*.png` |
| Predict scatter | `output/graphs/predict_scatter.png` |

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

---

## Aturan Penilaian PBL

| Komponen | Bobot | Detail |
|----------|-------|--------|
| **Program** | 30% | Kerapihan 10%, berjalan benar 10%, output 10% |
| **Laporan** | 50% | Preprocessing 20%, kesesuaian algoritma 20%, kesesuaian dg code 10% |
| **Presentasi** | 20% | Dinilai per individu |

**Catatan:**
- Tidak boleh menggunakan library yang langsung mengimplementasikan algoritma (contoh: Scikit-learn KNN). Pengurangan nilai jika melanggar.
- Kode program diberi komentar pada bagian/baris penting.
- Laporan dalam format PDF: deskripsi kasus, analisa preprocessing, desain algoritma, evaluasi model, screenshot hasil running.
- Format ZIP: `KELAS_KELOMPOK_NIM.zip` (contoh: `SE4801_03_1311281234.zip`).

---

## Anggota Kelompok & Peran

| Nama | NIM | Peran |
|------|-----|-------|
| Anggota 1 | 131xxxxxxx | (isi sesuai peran masing-masing) |
| Anggota 2 | 131xxxxxxx | (isi sesuai peran masing-masing) |
