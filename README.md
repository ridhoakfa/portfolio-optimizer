# Portofolio Optimizer

Aplikasi web untuk analisis dan optimasi portofolio saham menggunakan pendekatan Mean-Semivariance. Dibangun dengan Streamlit dan Python, menyediakan dua strategi optimasi berbasis Sequential Least Squares Programming (SLSQP): memaksimalkan Sortino Ratio dan meminimalkan Semivariance.

Dikembangkan sebagai bagian dari skripsi S1 Program Studi Statistika, Universitas Diponegoro.

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Fitur](#fitur)
- [Metodologi](#metodologi)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Konfigurasi Default](#konfigurasi-default)
- [Contoh Skenario Analisis](#contoh-skenario-analisis)
- [Catatan Teknis](#catatan-teknis)
- [Keterbatasan](#keterbatasan)
- [Lisensi](#lisensi)
- [Kontak](#kontak)

---

## Gambaran Umum

Aplikasi ini dirancang untuk membantu investor, analis, dan akademisi dalam membentuk portofolio saham optimal secara kuantitatif. Alih-alih mengandalkan intuisi atau estimasi return historis sederhana, aplikasi ini menggunakan matriks risiko berbasis downside (semivariance-semicovariance) dengan benchmark yang dapat dikonfigurasi.

Aplikasi dapat digunakan untuk ticker apa pun yang tersedia di Yahoo Finance, termasuk saham Indonesia (suffix `.JK`), saham Amerika Serikat, mata uang kripto, ETF, dan indeks. Konfigurasi default disetel untuk 27 saham konstituen Indeks BISNIS-27 dengan benchmark IHSG dan risk-free rate dari BI Rate.

Nilai tambah utama aplikasi ini adalah perbandingan langsung antara dua fungsi tujuan optimasi yang menggunakan matriks risiko yang sama:

1. **Max Sortino Ratio** — memaksimalkan risk-adjusted return
2. **Min Semivariance** — meminimalkan risiko downside

Perbandingan kedua strategi bersifat langsung dan setara karena keduanya menerima input risiko yang identik.

---

## Fitur

**Konfigurasi Fleksibel**
- Input daftar ticker bebas dari Yahoo Finance
- Benchmark dapat diganti (IHSG, S&P 500, Nikkei, dan lain-lain)
- Periode penelitian dapat disesuaikan
- Risk-free rate dapat diubah sesuai tingkat bunga acuan yang relevan

**Analisis Bertahap**
- Eksplorasi data: statistik deskriptif, harga ternormalisasi Base 100, heatmap korelasi
- Seleksi dua tahap: filter expected return positif dan seleksi korelasi Spearman
- Perhitungan matriks semivariance-semicovariance dengan benchmark
- Optimasi SLSQP dengan dua fungsi tujuan alternatif

**Evaluasi Kinerja**
- Sortino Ratio (harian dan tahunan)
- Cumulative Return
- Value at Risk 95% dengan Historical Simulation
- Perbandingan langsung antara kedua strategi

**Antarmuka**
- Tujuh halaman analisis yang runtut
- Dukungan tema terang dan gelap (melalui pengaturan Streamlit)
- Validasi input dan peringatan otomatis untuk konfigurasi yang tidak konsisten
- Narasi hasil otomatis untuk kedua strategi

---

## Metodologi

### Seleksi Saham Dua Tahap

**Tahap 1 — Filter Expected Return Positif**

Saham dengan rata-rata log return negatif pada periode penelitian dieliminasi. Diasumsikan saham dengan ekspektasi return negatif tidak layak masuk portofolio.

**Tahap 2 — Seleksi Korelasi Spearman**

Dari saham yang lolos tahap pertama, dipilih pasangan saham dengan korelasi Spearman di bawah threshold tertentu (default 0.2). Prinsipnya, korelasi rendah antar saham memberikan diversifikasi yang lebih efektif.

Korelasi Spearman dipilih (bukan Pearson) karena tidak memerlukan asumsi normalitas dan lebih tahan terhadap outlier.

### Matriks Risiko

Matriks semivariance-semicovariance dihitung dengan formula Estrada (2008):

<<<<<<< HEAD
$$\Sigma_{sv}[i,j] = \frac{1}{T} \sum_{t=1}^{T} \min(R_{i,t} - B_t, 0) \cdot \min(R_{j,t} - B_t, 0)$$

dengan:

=======
$\Sigma_{sv}[i, j] = \frac{1}{T} \sum_{t=1}^{T} \min(R_{i,t} - B_t, 0) \cdot \min(R_{j,t} - B_t, 0)$

dengan:
>>>>>>> 14152edf6c8c1653e22d6a2f1604b7cb472b784d
- $R_{i,t}$ = log return saham ke-$i$ pada periode $t$
- $B_t$ = log return benchmark pada periode $t$
- $T$ = jumlah observasi

Berbeda dengan covariance standar yang memperhitungkan semua deviasi, semivariance hanya memperhitungkan deviasi di bawah benchmark. Ini sesuai dengan perilaku investor yang menghindari risiko downside, bukan semua volatilitas.

### Optimasi Portofolio

Kedua strategi menggunakan SLSQP dengan kendala yang sama:

- Long-only: $0 \leq w_i \leq 1$
- Total bobot sama dengan 1: $\sum_{i=1}^{N} w_i = 1$

**Strategi 1 — Max Sortino Ratio**

<<<<<<< HEAD
$$\max_{w} \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma_{sv} w}}$$

**Strategi 2 — Min Semivariance**

$$\min_{w} w^T \Sigma_{sv} w$$
=======
$$
\max_{w} \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma_{sv} w}}
$$

**Strategi 2 — Min Semivariance**

$$
\min_{w} \quad w^T \Sigma_{sv} w
$$
>>>>>>> 14152edf6c8c1653e22d6a2f1604b7cb472b784d

Kedua strategi menggunakan matriks risiko $\Sigma_{sv}$ yang sama. Perbedaan hanya terletak pada fungsi tujuan.

### Evaluasi Kinerja

<<<<<<< HEAD
- **Sortino Ratio**: rasio excess return terhadap semideviation, dihitung harian dan tahunan:

  $$\text{Sortino} = \frac{\bar{R}_p - R_f}{\sigma_{\text{downside}}}$$

- **Cumulative Return**:

  $$R_{\text{cum}} = \exp\left(\sum_{t=1}^{T} r_t\right) - 1$$
=======
- **Sortino Ratio**: rasio excess return terhadap semideviation, dihitung harian dan tahunan
- **Cumulative Return**:
  
  $$
  R_{\text{cum}} = \exp\left(\sum_{t=1}^{T} r_t\right) - 1
  $$
>>>>>>> 14152edf6c8c1653e22d6a2f1604b7cb472b784d

- **Value at Risk 95%**: Historical Simulation, kuantil ke-5 dari return portofolio aktual, dikonversi ke nominal rupiah dengan dana awal yang dapat dikonfigurasi

---

## Persyaratan Sistem

- Python 3.10 atau lebih baru
- pip (Python package manager)
- Koneksi internet untuk mengunduh data dari Yahoo Finance

---

## Instalasi

### 1. Clone atau unduh proyek

```bash
git clone https://github.com/ridhoakfa/portfolio-optimizer.git
cd portfolio-optimizer
```

Atau unduh ZIP dan ekstrak ke folder lokal.

### 2. Buat virtual environment

Disarankan untuk mengisolasi dependensi proyek dari Python sistem.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

Proses ini akan mengunduh library Streamlit, pandas, numpy, scipy, yfinance, dan plotly.

---

## Menjalankan Aplikasi

Setelah virtual environment aktif dan dependensi terpasang:

```bash
streamlit run main.py
```

Aplikasi akan otomatis membuka browser pada alamat `http://localhost:8501`. Jika tidak terbuka otomatis, buka manual alamat tersebut di browser.

Untuk menghentikan aplikasi, tekan `Ctrl+C` di terminal.

---

## Panduan Penggunaan

### Alur Analisis

Aplikasi ini memiliki tujuh halaman yang sebaiknya dikunjungi secara berurutan:

1. **Home** — penjelasan aplikasi dan alur analisis
2. **Input Data** — konfigurasi ticker, benchmark, periode, dan parameter seleksi
3. **Eksplorasi Data** — statistik deskriptif, harga ternormalisasi, heatmap korelasi
4. **Seleksi Saham** — hasil seleksi dua tahap dengan heatmap Spearman
5. **Optimasi Portofolio** — menjalankan kedua strategi dan melihat bobot optimal
6. **Evaluasi Kinerja** — metrik kinerja, cumulative return, dan VaR untuk kedua strategi
7. **Perbandingan Strategi** — tabel dan narasi perbandingan

### Langkah Pertama

1. Buka halaman **Input Data**
2. Isi daftar ticker (satu per baris atau dipisah koma)
3. Pilih benchmark yang relevan dengan ticker
4. Tentukan periode penelitian
5. Sesuaikan parameter seleksi jika perlu
6. Klik **Muat Data & Jalankan Seleksi**

### Mengganti Konfigurasi

Untuk analisis ulang dengan konfigurasi berbeda, cukup ubah parameter di halaman **Input Data** dan klik tombol **Muat Data** lagi. Semua hasil sebelumnya akan di-reset otomatis karena data dasar berubah.

---

## Struktur Proyek

```
portfolio-optimizer/
├── main.py                    # Antarmuka Streamlit dan routing halaman
├── functions.py               # Fungsi backend perhitungan
├── requirements.txt           # Dependensi Python
├── .streamlit/
│   └── config.toml            # Konfigurasi tema dan server Streamlit
├── README.md                  # Dokumentasi proyek
└── .gitignore                 # File/folder yang diabaikan Git (opsional)
```

**Pemisahan tanggung jawab:**

- `main.py` hanya menangani rendering UI, state management, dan routing
- `functions.py` berisi fungsi murni untuk load data, seleksi, optimasi, dan evaluasi
- `config.toml` mengatur warna primary dan perilaku server

---

## Konfigurasi Default

Default disetel untuk kasus penelitian skripsi, namun semua parameter dapat diubah melalui antarmuka:

| Parameter | Nilai Default | Keterangan |
|---|---|---|
| Ticker | 27 saham BISNIS-27 | Daftar konstituen indeks |
| Benchmark | `^JKSE` | Indeks Harga Saham Gabungan |
| Risk-free Rate | 4.75% p.a. | BI Rate periode penelitian |
| Periode | 3 Nov 2025 – 29 Apr 2026 | Periode efektif konstituen |
| Threshold Spearman | 0.2 | Ambang batas korelasi rendah |
| Maksimum Saham | 12 | Batas jumlah saham terpilih |
| Dana Awal VaR | Rp 100.000.000 | Basis konversi VaR ke rupiah |

---

## Contoh Skenario Analisis

### Saham Amerika Serikat

1. Buka halaman **Input Data**
2. Ganti daftar ticker menjadi: `AAPL, MSFT, GOOGL, AMZN, META`
3. Ganti benchmark menjadi: `^GSPC` (S&P 500)
4. Ganti risk-free rate menjadi: `5.33` (Fed Funds Rate)
5. Sesuaikan periode sesuai kebutuhan
6. Klik **Muat Data & Jalankan Seleksi**

### Mata Uang Kripto

1. Ticker: `BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD`
2. Benchmark: `^GSPC` atau indeks kripto lain
3. Risk-free rate: `0`
4. Periode: sesuaikan (kripto cenderung memerlukan data 2–3 tahun)

### Saham Hong Kong

1. Ticker: `0700.HK, 0939.HK, 1299.HK, 2318.HK, 0981.HK`
2. Benchmark: `^HSI` (Hang Seng Index)
3. Risk-free rate: sesuaikan dengan HIBOR atau tingkat bunga acuan Hong Kong

---

## Catatan Teknis

### Log Return vs Simple Return

Aplikasi menggunakan log return, bukan simple return, karena:
1. Bersifat aditif antar periode
2. Lebih stabil secara statistik
3. Mendekati distribusi normal pada frekuensi tinggi

### Semivariance vs Variance

Matriks risiko yang digunakan adalah semivariance-semicovariance, bukan covariance standar. Ini konsisten dengan perilaku investor yang menghindari kerugian (downside) daripada semua volatilitas.

### Long-Only Constraint

Kedua strategi optimasi menerapkan long-only constraint melalui parameter bounds SLSQP. Tidak ada bobot negatif yang diizinkan, sehingga tidak ada skenario short-selling.

### Multiple Random Starts

Optimasi SLSQP dijalankan dengan 10 titik awal acak untuk mengurangi risiko terjebak di local optimum. Hasil terbaik (Sortino tertinggi atau Semivariance terendah) yang diambil.

### Guard Pencegahan Error

- Validasi jumlah ticker minimal 2
- Peringatan jika periode menghasilkan kurang dari 30 observasi
- Peringatan otomatis jika ticker dan benchmark tidak cocok (misal ticker `.JK` dengan benchmark `^GSPC`)
- Filter bobot di bawah 1e-6 dari pie chart untuk menghindari notasi ilmiah
- Penanganan koneksi gagal ke Yahoo Finance dengan pesan yang jelas

### Tema Terang dan Gelap

Aplikasi kompatibel dengan kedua tema Streamlit. Untuk berpindah tema, buka menu titik tiga di pojok kanan atas, pilih **Settings**, lalu ubah **Theme**. Konfigurasi tema disimpan di browser.

---

## Keterbatasan

1. **Data historis tidak menjamin kinerja masa depan.** Optimasi didasarkan pada pola historis yang mungkin tidak berulang.

2. **Model mengasumsikan return yang wajar.** Distribusi return pada data keuangan sering bersifat leptokurtik (ekor tebal) dan tidak sepenuhnya normal. Semivariance lebih tahan terhadap asumsi ini, namun tetap memiliki batasan.

3. **Periode pendek dapat menimbulkan hasil tidak stabil.** Periode di bawah 1 tahun mungkin menghasilkan estimasi korelasi dan semivariance yang sensitif terhadap kondisi pasar sesaat.

4. **Kualitas data bergantung pada Yahoo Finance.** Jika ada masalah scraping atau gap pada sumber data, hasil dapat terpengaruh.

5. **Tidak memodelkan biaya transaksi.** Hasil return yang dihitung adalah gross return, belum memperhitungkan biaya broker, pajak, atau slippage.

6. **Konstituen indeks dapat berubah.** Untuk analisis indeks tertentu, daftar konstituen pada periode penelitian harus dipastikan terlebih dahulu di sumber resmi.

---

## Lisensi

Proyek ini bersifat open-source. Penggunaan untuk keperluan akademik, pembelajaran, dan penelitian diizinkan dengan tetap mencantumkan atribusi.

---

## Kontak

**Ridho Akbar Fadhilah**
Program Studi Statistika, Universitas Diponegoro

Untuk pertanyaan, saran, atau laporan bug, silakan buka issue di repositori GitHub atau hubungi langsung melalui GitHub.
