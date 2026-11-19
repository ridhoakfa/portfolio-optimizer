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

Aplikasi dapat digunakan untuk ticker apa pun yang tersedia di Yahoo Finance, termasuk saham Indonesia (suffix `.JK`), saham Amerika Serikat, mata uang kripto, ETF, dan indeks. Konfigurasi default disetel untuk 27 saham konstituen Indeks BISNIS-27 dengan benchmark IHSG.

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

**Analisis Bertahap**
- Eksplorasi data: statistik deskriptif return harian dan harga ternormalisasi Base 100
- Seleksi dua tahap: filter expected return positif dan seleksi korelasi Spearman
- Perhitungan matriks semivariance-semicovariance dengan benchmark
- Optimasi SLSQP dengan dua fungsi tujuan alternatif

**Evaluasi Kinerja**
- Sortino Ratio harian
- Cumulative Return
- Value at Risk 95% dengan Simulasi Historis
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

Dari saham yang lolos tahap pertama, pasangan saham diurutkan dari korelasi Spearman terendah. Saham diambil dari pasangan yang korelasinya di bawah threshold (default 0,2) sampai batas maksimum saham (default 13) tercapai. Batas korelasi berlaku untuk pasangan asal setiap saham terpilih.

### Matriks Risiko

Matriks semivariance-semicovariance dihitung dengan formula Estrada (2008):

$\Sigma_{sv}[i, j] = \frac{1}{T} \sum_{t=1}^{T} \min(R_{i,t} - B_t, 0) \cdot \min(R_{j,t} - B_t, 0)$

dengan:
- $R_{i,t}$ = log return saham ke-$i$ pada periode $t$
- $B_t$ = log return benchmark pada periode $t$
- $T$ = jumlah observasi

Berbeda dengan covariance standar yang memperhitungkan semua deviasi, semivariance hanya memperhitungkan deviasi di bawah benchmark. Ini sesuai dengan perilaku investor yang menghindari risiko downside, bukan semua volatilitas.

### Optimasi Portofolio

Kedua strategi menggunakan SLSQP dengan kendala yang sama:
- Long-only: setiap bobot antara 0 dan 1
- Total bobot = 1

**Strategi 1 — Max Sortino Ratio**

$$
\max_{w} \quad \frac{w^T \mu - \bar{B}}{\sqrt{w^T \Sigma_{sv} w}}
$$

**Strategi 2 — Min Semivariance**

$$
\min_{w} \quad w^T \Sigma_{sv} w
$$

Kedua strategi menggunakan matriks risiko $\Sigma_{sv}$ yang sama. Perbedaan hanya terletak pada fungsi tujuan. Pada Strategi 1, $\bar{B}$ adalah rata-rata return benchmark harian; return benchmark berfungsi sebagai *Minimum Acceptable Return* (MAR).

### Evaluasi Kinerja

- **Sortino Ratio**: dihitung dengan rumus yang sama dengan fungsi tujuan Strategi 1, yaitu $(w^T \mu - \bar{B}) / \sqrt{w^T \Sigma_{sv} w}$, dalam skala harian.
- **Cumulative Return**:

  $$R_{cum} = e^{\sum_{t=1}^{T} r_t} - 1$$

- **Value at Risk 95%**: Simulasi Historis. Return portofolio diurutkan dari terkecil, lalu diambil return ke-$k$ dengan $k = \lceil \alpha T \rceil$ ($\alpha = 0{,}05$):

  $$VaR = V_0 \times (-R_{P,(k)}) \times \sqrt{t}$$

  dengan $V_0$ dana awal (default Rp100.000.000) dan $t$ periode holding (1 hari).

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
3. **Eksplorasi Data** — statistik deskriptif dan harga ternormalisasi
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
| Periode | 3 Nov 2025 – 30 Apr 2026 | Periode efektif konstituen; tanggal akhir ikut diunduh |
| Threshold Spearman | 0.2 | Ambang batas korelasi rendah |
| Maksimum Saham | 13 | Batas jumlah saham terpilih |
| Dana Awal VaR | Rp 100.000.000 | Basis konversi VaR ke rupiah |

---

## Contoh Skenario Analisis

### Saham Amerika Serikat

1. Buka halaman **Input Data**
2. Ganti daftar ticker menjadi: `AAPL, MSFT, GOOGL, AMZN, META`
3. Ganti benchmark menjadi: `^GSPC` (S&P 500)
4. Sesuaikan periode sesuai kebutuhan
5. Klik **Muat Data & Jalankan Seleksi**

### Mata Uang Kripto

1. Ticker: `BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD`
2. Benchmark: `^GSPC` atau indeks kripto lain
3. Periode: sesuaikan (kripto cenderung memerlukan data 2–3 tahun)

### Saham Hong Kong

1. Ticker: `0700.HK, 0939.HK, 1299.HK, 2318.HK, 0981.HK`
2. Benchmark: `^HSI` (Hang Seng Index)
3. Sesuaikan periode sesuai kebutuhan

---

## Catatan Teknis

### Log Return vs Simple Return

Aplikasi menggunakan log return karena bersifat aditif antarperiode: jumlah log return harian sama dengan log return seluruh periode. Return portofolio dihitung sebagai jumlah tertimbang log return saham, yang merupakan pendekatan.

### Semivariance vs Variance

Matriks risiko yang digunakan adalah semivariance-semicovariance, bukan covariance standar. Ini konsisten dengan perilaku investor yang menghindari kerugian (downside) daripada semua volatilitas.

### Long-Only Constraint

Kedua strategi optimasi menerapkan long-only constraint melalui parameter bounds SLSQP. Tidak ada bobot negatif yang diizinkan, sehingga tidak ada skenario short-selling.

### Titik Awal SLSQP

Optimasi SLSQP dijalankan dari 10 titik awal acak (benih 0 sampai 9) dan hasil terbaik yang diambil. Kedua fungsi tujuan tidak memiliki optimum lokal selain optimum global, sehingga kesepuluh titik awal berfungsi sebagai pemeriksaan bahwa solver memberi hasil yang sama. Bila tidak ada titik awal yang konvergen, aplikasi menampilkan peringatan dan bobot diisi sama rata.

### Tanggal Akhir

Yahoo Finance tidak menyertakan tanggal `end` yang diminta, sehingga aplikasi menggeser tanggal akhir satu hari saat mengunduh. Tanggal akhir yang diisi pengguna ikut dihitung.

### Guard Pencegahan Error

- Validasi jumlah ticker minimal 2
- Peringatan jika periode menghasilkan kurang dari 30 observasi
- Peringatan jika SLSQP tidak konvergen dari semua titik awal
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
