# Portofolio Optimizer

Aplikasi web untuk analisis dan optimasi portofolio saham menggunakan pendekatan Mean-Semivariance. Dibangun dengan Streamlit dan Python, menyediakan dua strategi optimasi berbasis Sequential Least Squares Programming (SLSQP): memaksimalkan Sortino Ratio dan meminimalkan Semivariance.

Dikembangkan sebagai bagian dari skripsi S1 Program Studi Statistika, Universitas Diponegoro.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-optimizer-akfa.streamlit.app/)

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Fitur](#fitur)
- [Metodologi](#metodologi)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Unduhan CSV](#unduhan-csv)
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

1. **Maximum Sortino Ratio**: memaksimalkan risk-adjusted return
2. **Minimum Semivariance**: meminimalkan risiko downside

Perbandingan kedua strategi bersifat langsung dan setara karena keduanya menerima input risiko yang identik.

---

## Fitur

**Konfigurasi Fleksibel**
- Input daftar ticker bebas dari Yahoo Finance
- Benchmark dapat diganti (IHSG, S&P 500, Nikkei, dan lain-lain)
- Periode penelitian dapat disesuaikan

**Analisis Bertahap**
- Satu tombol **Jalankan Analisis** menjalankan semua tahap berurutan: unduh data, seleksi saham, optimasi SLSQP, dan evaluasi
- Eksplorasi data: statistik deskriptif return harian (termasuk benchmark) dan pergerakan harga ternormalisasi Base 100
- Seleksi dua tahap: filter expected return positif dan seleksi korelasi Spearman, dengan kalimat ringkasan otomatis
- Perhitungan matriks semivariance-semicovariance dengan benchmark
- Optimasi SLSQP dengan dua fungsi tujuan alternatif

**Evaluasi Kinerja**
- Sortino Ratio harian dan Semideviation harian
- Cumulative Return
- Value at Risk 95% dengan Simulasi Historis
- Tabel perbandingan kedua strategi dengan nilai yang lebih baik ditebalkan, serta narasi hasil otomatis

**Detail Perhitungan**
- Rumus setiap tahap dengan notasi naskah skripsi
- Substitusi angka langkah demi langkah untuk Sortino Ratio, Cumulative Return, dan VaR
- Vektor dan matriks antara (log return, korelasi Spearman, $\boldsymbol{\mu}$, $\bar{B}$, matriks semivariance-semicovariance, nilai eigen, bobot)
- Unduhan CSV berpresisi penuh untuk dicocokkan dengan perhitungan manual

**Antarmuka**
- Enam halaman analisis yang runtut
- Angka dalam format Indonesia (koma desimal, titik ribuan) dan nama bulan bahasa Indonesia
- Dukungan tema terang dan gelap (melalui menu Streamlit)
- Validasi input dan peringatan otomatis untuk konfigurasi yang tidak konsisten

---

## Metodologi

### Seleksi Saham Dua Tahap

**Tahap 1: Filter Expected Return Positif**

Saham dengan rata-rata log return negatif pada periode penelitian dieliminasi. Diasumsikan saham dengan ekspektasi return negatif tidak layak masuk portofolio.

**Tahap 2: Seleksi Korelasi Spearman**

Dari saham yang lolos tahap pertama, pasangan saham diurutkan dari korelasi Spearman terendah. Saham diambil dari pasangan yang korelasinya di bawah threshold (default 0,2) sampai batas maksimum saham (default 13) tercapai. Batas korelasi berlaku untuk pasangan asal setiap saham terpilih.

Koefisien Spearman dihitung sebagai korelasi Pearson atas peringkat return; nilai yang kembar diberi peringkat rata-rata. Hasilnya sama dengan fungsi Excel `CORREL` atas `RANK.AVG`.

### Matriks Risiko

Matriks semivariance-semicovariance dihitung dengan formula Estrada (2008). Untuk saham $i$ dan saham $j$, semivariance dan semicovariance terhadap benchmark adalah:

$$
S_i^2=\frac{\sum_{t=1}^{T}\left[\min\left(R_{i,t}-B_t,0\right)\right]^2}{T}
$$

$$
S_{ij}=\frac{\sum_{t=1}^{T}\min\left(R_{i,t}-B_t,0\right)\min\left(R_{j,t}-B_t,0\right)}{T}
$$

Keduanya disusun menjadi matriks berukuran $I \times I$ dengan semivariance pada diagonal utama:

$$
\boldsymbol{\Sigma}_{sv}=\begin{bmatrix}
S_1^2 & S_{12} & \cdots & S_{1I}\\
S_{21} & S_2^2 & \cdots & S_{2I}\\
\vdots & \vdots & \ddots & \vdots\\
S_{I1} & S_{I2} & \cdots & S_I^2
\end{bmatrix}
$$

dengan:
- $R_{i,t}$ = log return saham $i$ pada waktu $t$
- $B_t$ = log return benchmark pada waktu $t$
- $T$ = banyaknya data return
- $I$ = banyaknya saham dalam portofolio

Berbeda dengan covariance standar yang memperhitungkan semua deviasi, semivariance hanya memperhitungkan deviasi di bawah benchmark. Ini sesuai dengan perilaku investor yang menghindari risiko downside, bukan semua volatilitas.

### Optimasi Portofolio

Kedua strategi menggunakan SLSQP dengan kendala yang sama, yaitu long-only (setiap bobot antara 0 dan 1) dan total bobot sama dengan 1.

**Strategi 1: Maximum Sortino Ratio**

$$
\begin{aligned}
\max_{\mathbf{w}}\quad & \frac{\mathbf{w}^\top\boldsymbol{\mu}-\bar{B}}{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}}}\\
\text{terhadap}\quad & \sum_{i=1}^{I}w_i=1,\quad 0\le w_i\le1
\end{aligned}
$$

**Strategi 2: Minimum Semivariance**

$$
\begin{aligned}
\min_{\mathbf{w}}\quad & \mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}\\
\text{terhadap}\quad & \sum_{i=1}^{I}w_i=1,\quad 0\le w_i\le1
\end{aligned}
$$

dengan:
- $\mathbf{w}$ = vektor bobot portofolio dan $\mathbf{w}^\top$ = transpos vektor bobot
- $\boldsymbol{\mu}$ = vektor expected return setiap saham (rata-rata log return)
- $\bar{B}$ = rata-rata return benchmark harian
- $\boldsymbol{\Sigma}_{sv}$ = matriks semivariance-semicovariance

Kedua strategi menggunakan matriks risiko $\boldsymbol{\Sigma}_{sv}$ yang sama. Perbedaan hanya terletak pada fungsi tujuan. Pada Strategi 1, return benchmark berfungsi sebagai *Minimum Acceptable Return* (MAR).

### Evaluasi Kinerja

**Sortino Ratio** harian, dengan rumus yang sama dengan fungsi tujuan Strategi 1:

$$
Sortino_P=\frac{\mathbf{w}^\top\boldsymbol{\mu}-\bar{B}}{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}}}
$$

**Cumulative Return**, dengan $R_{P,t}$ return portofolio pada waktu $t$:

$$
R_{cum}=\exp\left(\sum_{t=1}^{T}R_{P,t}\right)-1
$$

**Value at Risk 95%** dengan Simulasi Historis. Return portofolio diurutkan dari terkecil, lalu diambil return pada posisi $k$ dengan $\alpha = 0{,}05$ :

$$
k=\lceil\alpha T\rceil
$$

$$
VaR=V_0\times\left(-R_{P,(k)}\right)\times\sqrt{t}
$$

dengan:
- $V_0$ = dana awal (default Rp100.000.000)
- $R_{P,(k)}$ = return portofolio terurut pada posisi $k$
- $t$ = periode holding (1 hari)

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

Aplikasi dapat langsung dipakai tanpa instalasi di [portfolio-optimizer-akfa.streamlit.app](https://portfolio-optimizer-akfa.streamlit.app/).

Untuk menjalankan di komputer sendiri, setelah virtual environment aktif dan dependensi terpasang:

```bash
streamlit run main.py
```

Aplikasi akan otomatis membuka browser pada alamat `http://localhost:8501`. Jika tidak terbuka otomatis, buka manual alamat tersebut di browser.

Untuk menghentikan aplikasi, tekan `Ctrl+C` di terminal.

---

## Panduan Penggunaan

### Alur Analisis

Aplikasi ini memiliki enam halaman:

1. **Beranda**: penjelasan aplikasi, alur analisis, dan konfigurasi default
2. **Data**: konfigurasi ticker, benchmark, periode, dan parameter; tombol **Jalankan Analisis**; statistik deskriptif, pergerakan harga, dan unduhan data harga
3. **Seleksi Saham**: hasil tahap 1 untuk semua saham, pasangan berkorelasi di bawah threshold beserta saham yang masuk dari setiap pasangan, dan heatmap Spearman
4. **Optimasi Portofolio**: ringkasan bobot, pie chart kedua strategi, dan tabel perbandingan bobot
5. **Hasil & Perbandingan**: tabel perbandingan, narasi hasil, grafik Cumulative Return, dan distribusi return dengan VaR
6. **Detail Perhitungan**: rumus, angka antara, dan unduhan CSV untuk tab Return, Seleksi, Matriks Risiko, Optimasi, dan Evaluasi

Halaman 3 sampai 6 baru berisi setelah analisis dijalankan.

### Langkah Pertama

1. Buka halaman **Data**
2. Isi daftar ticker (satu per baris atau dipisah koma)
3. Pilih benchmark yang relevan dengan ticker
4. Tentukan periode penelitian
5. Sesuaikan parameter seleksi dan dana awal VaR jika perlu
6. Klik **Jalankan Analisis**, lalu tunggu status proses (Mengunduh data, Seleksi saham, Optimasi SLSQP, Evaluasi) sampai selesai

### Mengganti Konfigurasi

Untuk analisis ulang dengan konfigurasi berbeda, ubah parameter di halaman **Data** dan klik **Jalankan Analisis** lagi. Hasil sebelumnya dihapus setiap kali tombol ditekan. Bila salah satu tahap gagal, proses berhenti dengan pesan galat dan tidak ada hasil yang disimpan.

---

## Unduhan CSV

Semua file CSV memakai pemisah titik koma (`;`), koma desimal, dan encoding UTF-8 dengan BOM, sehingga dapat langsung dibuka di Excel berlokal Indonesia. Angka di dalam CSV berpresisi penuh, tidak dibulatkan seperti tampilan.

| Halaman | File | Isi |
|---|---|---|
| Data | `data_harga.csv` | Harga penutupan yang disesuaikan (adjusted close) semua ticker dan benchmark pada tanggal yang dipakai analisis |
| Detail Perhitungan, tab Return | `log_return.csv` | Log return harian semua ticker dan benchmark |
| Detail Perhitungan, tab Seleksi | `matriks_spearman.csv` | Matriks korelasi Spearman saham yang lolos tahap 1 |
| Detail Perhitungan, tab Matriks Risiko | `matriks_risiko.csv` | Vektor $\boldsymbol{\mu}$, $\bar{B}$, matriks semivariance-semicovariance, dan nilai eigen |
| Detail Perhitungan, tab Optimasi | `bobot_optimal.csv` | Vektor bobot kedua strategi |
| Detail Perhitungan, tab Evaluasi | `return_portofolio.csv` | Return portofolio harian kedua strategi |

---

## Struktur Proyek

```
portfolio-optimizer/
├── main.py                    # Antarmuka Streamlit, alur analisis, dan routing halaman
├── functions.py               # Fungsi backend perhitungan
├── requirements.txt           # Dependensi Python
├── .devcontainer/
│   └── devcontainer.json      # Konfigurasi Dev Container
├── README.md                  # Dokumentasi proyek
└── .gitignore                 # File/folder yang diabaikan Git
```

**Pemisahan tanggung jawab:**

- `main.py` menangani tampilan, state management, routing, format angka, dan unduhan CSV
- `functions.py` berisi fungsi murni untuk load data, seleksi, optimasi, dan evaluasi

---

## Konfigurasi Default

Default disetel untuk kasus penelitian skripsi, namun semua parameter dapat diubah melalui antarmuka:

| Parameter | Nilai Default | Keterangan |
|---|---|---|
| Ticker | 27 saham BISNIS-27 | Daftar konstituen indeks |
| Benchmark | `^JKSE` | Indeks Harga Saham Gabungan |
| Periode | 3 November 2025 sampai 30 April 2026 | Periode efektif konstituen; tanggal akhir ikut diunduh |
| Threshold Spearman | 0,2 | Ambang batas korelasi rendah |
| Maksimum Saham | 13 | Batas jumlah saham terpilih |
| Dana Awal VaR | Rp100.000.000 | Basis konversi VaR ke rupiah |

---

## Contoh Skenario Analisis

### Saham Amerika Serikat

1. Buka halaman **Data**
2. Ganti daftar ticker menjadi: `AAPL, MSFT, GOOGL, AMZN, META`
3. Ganti benchmark menjadi: `^GSPC` (S&P 500)
4. Naikkan threshold Spearman, misalnya menjadi 0,6. Kelima saham teknologi ini saling berkorelasi cukup tinggi, sehingga pada periode default tidak ada pasangan dengan korelasi di bawah 0,2 dan analisis berhenti di tahap seleksi
5. Sesuaikan periode sesuai kebutuhan
6. Klik **Jalankan Analisis**

Narasi hasil akan menyebut benchmark sebagai S&P 500.

### Mata Uang Kripto

1. Ticker: `BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD`
2. Benchmark: `^GSPC` atau indeks kripto lain
3. Periode: sesuaikan (kripto cenderung memerlukan data 2 sampai 3 tahun)
4. Klik **Jalankan Analisis**

### Saham Hong Kong

1. Ticker: `0700.HK, 0939.HK, 1299.HK, 2318.HK, 0981.HK`
2. Benchmark: `^HSI` (Hang Seng Index)
3. Sesuaikan periode sesuai kebutuhan
4. Klik **Jalankan Analisis**

---

## Catatan Teknis

### Log Return vs Simple Return

Aplikasi menggunakan log return karena bersifat aditif antarperiode: jumlah log return harian sama dengan log return seluruh periode. Return portofolio dihitung sebagai jumlah tertimbang log return saham, yang merupakan pendekatan.

### Semivariance vs Variance

Matriks risiko yang digunakan adalah semivariance-semicovariance, bukan covariance standar. Ini konsisten dengan perilaku investor yang menghindari kerugian (downside) daripada semua volatilitas.

### Long-Only Constraint

Kedua strategi optimasi menerapkan long-only constraint melalui parameter bounds SLSQP. Tidak ada bobot negatif yang diizinkan, sehingga tidak ada skenario short-selling.

### Titik Awal SLSQP

Optimasi SLSQP dijalankan dari 10 titik awal acak (benih 0 sampai 9), dan dipakai hasil dengan nilai fungsi tujuan terbaik. Bila tidak ada titik awal yang konvergen, aplikasi menampilkan peringatan dan bobot diisi sama rata.

### Tanggal Akhir

Yahoo Finance tidak menyertakan tanggal `end` yang diminta, sehingga aplikasi menggeser tanggal akhir satu hari saat mengunduh. Tanggal akhir yang diisi pengguna ikut dihitung.

### Guard Pencegahan Error

- Validasi jumlah ticker minimal 2
- Proses berhenti dengan pesan galat jika periode menghasilkan kurang dari 30 hari bursa atau seleksi tidak menghasilkan saham
- Peringatan jika seleksi menghasilkan kurang dari 3 saham
- Peringatan jika SLSQP tidak konvergen dari semua titik awal
- Peringatan otomatis jika ticker dan benchmark tidak cocok (misal ticker `.JK` dengan benchmark `^GSPC`)
- Filter bobot di bawah 1e-6 dari pie chart untuk menghindari notasi ilmiah
- Penanganan koneksi gagal ke Yahoo Finance dengan pesan yang jelas

### Tema Terang dan Gelap

Aplikasi kompatibel dengan kedua tema Streamlit. Untuk berpindah tema, buka menu titik tiga di pojok kanan atas, lalu pilih **System**, **Light**, atau **Dark**. Pilihan tema disimpan di browser.

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
