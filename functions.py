"""
Backend computation untuk aplikasi Portfolio Optimizer.

Berisi semua fungsi perhitungan portofolio: load data dari Yahoo Finance,
seleksi saham, matriks risiko, optimasi SLSQP, dan evaluasi kinerja.
Fungsi-fungsi ini dipanggil dari main.py.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from scipy.optimize import minimize


# Daftar default saham konstituen BISNIS-27 dalam format Yahoo Finance
DAFTAR_SAHAM_DEFAULT = [
    "ADMR.JK", "ADRO.JK", "AMRT.JK", "ANTM.JK", "ASII.JK",
    "BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK", "BRPT.JK",
    "BUMI.JK", "DSNG.JK", "HEAL.JK", "INCO.JK", "INDF.JK",
    "INKP.JK", "JPFA.JK", "KLBF.JK", "MAPI.JK", "MEDC.JK",
    "MIKA.JK", "MYOR.JK", "NCKL.JK", "PGEO.JK", "PTBA.JK",
    "TLKM.JK", "UNTR.JK",
]

BENCHMARK_DEFAULT = "^JKSE"
RISK_FREE_TAHUNAN_DEFAULT = 0.0475
START_DATE_DEFAULT = "2025-11-03"
END_DATE_DEFAULT = "2026-04-30"

# Parameter seleksi saham
THRESHOLD_KORELASI_DEFAULT = 0.2
MAX_SAHAM_DEFAULT = 12

# Parameter keuangan
DANA_AWAL_VAR_DEFAULT = 100_000_000
HARI_BURSA_SETAHUN = 252
MIN_OBSERVASI = 30


def load_data_saham(tickers, start_date, end_date):
    """
    Unduh harga penutupan historis dari Yahoo Finance untuk sekumpulan ticker.

    Parameters
    ----------
    tickers : list of str
        Daftar kode ticker, contoh: ['BBCA.JK', 'BBRI.JK'].
    start_date, end_date : str
        Rentang tanggal dalam format YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame
        Matriks harga penutupan; baris = tanggal, kolom = ticker.
    """
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        # Raise sebagai ConnectionError agar bisa ditangkap jelas di main.py
        raise ConnectionError(
            f"Gagal mengunduh data dari Yahoo Finance: {type(e).__name__}. "
            f"Periksa koneksi internet."
        )

    if data is None or data.empty:
        raise ValueError(
            f"Tidak ada data untuk ticker pada periode {start_date} sampai {end_date}."
        )

    # Yahoo Finance mengembalikan kolom MultiIndex (Price, Ticker) saat
    # mengunduh lebih dari satu ticker. Kita hanya butuh harga Close.
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # Forward fill untuk hari libur yang tidak ada data, lalu drop baris
    # yang seluruhnya kosong (biasanya di awal/akhir periode).
    return data.ffill().dropna(how="all")


def load_data_benchmark(ticker, start_date, end_date):
    """
    Unduh harga benchmark (IHSG, S&P 500, atau indeks lain) dari Yahoo Finance.

    Returns
    -------
    pd.Series
        Seri harga penutupan benchmark dengan index tanggal.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise ConnectionError(
            f"Gagal mengunduh benchmark {ticker}: {type(e).__name__}."
        )

    if data is None or data.empty:
        raise ValueError(
            f"Tidak ada data untuk benchmark {ticker} pada periode tersebut."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # squeeze() mengubah DataFrame satu kolom menjadi Series
    series = data.squeeze()
    return series.ffill().dropna()


def hitung_log_return(data):
    """
    Hitung log return harian: ln(P_t / P_{t-1}).

    Log return dipakai (bukan simple return) karena bersifat aditif
    antar periode dan lebih stabil untuk data finansial.

    Returns
    -------
    pd.DataFrame or pd.Series
        Log return dengan baris pertama (NaN) sudah dihapus.
    """
    return np.log(data / data.shift(1)).dropna()


def hitung_expected_return(log_return):
    """
    Hitung expected return sebagai rata-rata log return.
    Digunakan sebagai proksi return harian yang diharapkan.
    """
    return log_return.mean()


def seleksi_saham_dua_tahap(log_return, expected_return, threshold=0.2, max_saham=12):
    """
    Seleksi saham dua tahap:

    Tahap 1: hanya saham dengan expected return positif yang lolos.
    Tahap 2: dari kandidat tersebut, ambil pasangan saham dengan korelasi
             Spearman di bawah threshold, lalu deduplikasi sampai batas
             maksimum saham.

    Penggunaan Spearman (bukan Pearson) karena tidak memerlukan asumsi
    normalitas dan lebih tahan terhadap outlier.

    Returns
    -------
    dict berisi:
        saham_positif, saham_terpilih, spearman_matrix, pasangan_df,
        pasangan_terpilih, dan pesan (jika ada peringatan).
    """
    # Tahap 1: filter expected return positif
    saham_positif = expected_return[expected_return > 0].index.tolist()

    # Jika kandidat terlalu sedikit, kembalikan lebih awal dengan pesan
    if len(saham_positif) < 2:
        return {
            "saham_positif": saham_positif,
            "saham_terpilih": saham_positif,
            "spearman_matrix": None,
            "pasangan_df": None,
            "pasangan_terpilih": None,
            "pesan": (
                f"Hanya {len(saham_positif)} saham dengan E(R) > 0. "
                f"Perpanjang periode atau ganti ticker."
            ),
        }

    # Hitung matriks korelasi Spearman antar saham kandidat
    spearman_corr, _ = spearmanr(log_return[saham_positif].values)
    spearman_matrix = pd.DataFrame(
        spearman_corr,
        index=saham_positif,
        columns=saham_positif,
    )

    # Kumpulkan semua pasangan unik dengan nilai korelasinya
    pasangan = []
    for i in range(len(saham_positif)):
        for j in range(i + 1, len(saham_positif)):
            pasangan.append({
                "Saham A": saham_positif[i],
                "Saham B": saham_positif[j],
                "Korelasi": float(spearman_matrix.iloc[i, j]),
            })

    pasangan_df = pd.DataFrame(pasangan).sort_values("Korelasi").reset_index(drop=True)

    # Tahap 2: ambil pasangan yang lolos threshold, deduplikasi
    pasangan_terpilih = pasangan_df[pasangan_df["Korelasi"] < threshold].copy()

    saham_terpilih = []
    for _, row in pasangan_terpilih.iterrows():
        if len(saham_terpilih) >= max_saham:
            break
        # Tambahkan kedua saham dari pasangan ini jika belum ada
        for s in [row["Saham A"], row["Saham B"]]:
            if s not in saham_terpilih and len(saham_terpilih) < max_saham:
                saham_terpilih.append(s)

    # Beri peringatan jika saham terpilih terlalu sedikit untuk portofolio
    pesan = None
    if len(saham_terpilih) < 3:
        pesan = (
            f"Hanya {len(saham_terpilih)} saham lolos threshold Spearman {threshold}. "
            f"Naikkan threshold atau perpanjang periode."
        )

    return {
        "saham_positif": saham_positif,
        "saham_terpilih": saham_terpilih,
        "spearman_matrix": spearman_matrix,
        "pasangan_df": pasangan_df,
        "pasangan_terpilih": pasangan_terpilih,
        "pesan": pesan,
    }


def hitung_matriks_semivariance(returns, benchmark):
    """
    Hitung matriks semivariance-semicovariance dengan benchmark.

    Formula Estrada (2008):
        Sigma_sv[i,j] = (1/T) * sum_t [min(R_i,t - B_t, 0) * min(R_j,t - B_t, 0)]

    Hanya memperhitungkan deviasi di bawah benchmark (downside risk),
    berbeda dengan covariance standar yang memperhitungkan semua deviasi.

    Parameters
    ----------
    returns : np.ndarray
        Matriks return saham berukuran (T, n).
    benchmark : np.ndarray
        Return benchmark berukuran (T,).

    Returns
    -------
    np.ndarray
        Matriks semivariance-semicovariance berukuran (n, n).
    """
    returns = np.asarray(returns)
    benchmark = np.asarray(benchmark).reshape(-1, 1)

    T = returns.shape[0]
    # Downside = min(R - B, 0); hanya selisih negatif yang dihitung
    downside = np.minimum(returns - benchmark, 0)
    return (downside.T @ downside) / T


def optimasi_slsqp_max_sortino(mu, sigma_sv, rf_daily, n_starts=10):
    """
    Optimasi SLSQP untuk memaksimalkan Sortino Ratio portofolio.

    Sortino = (Return_portofolio - rf) / semideviation

    Kendala:
        - Long-only: setiap bobot antara 0 dan 1
        - Total bobot = 1

    Multiple random starts dipakai untuk mengurangi risiko terjebak
    di local optimum.

    Returns
    -------
    tuple (bobot_optimal, sortino_optimal)
    """
    n_assets = len(mu)
    mu = np.asarray(mu)
    sigma_sv = np.asarray(sigma_sv)

    # SLSQP hanya bisa meminimalkan, jadi Sortino dinegatifkan
    def neg_sortino(w):
        ret = w @ mu
        svar = w @ sigma_sv @ w
        # Guard agar tidak bagi nol saat optimasi menjelajah titik awal
        sd = np.sqrt(svar) if svar > 0 else 1e-8
        return -(ret - rf_daily) / sd

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    best_weights = None
    best_sortino = -np.inf

    for seed in range(n_starts):
        np.random.seed(seed)
        w0 = np.random.random(n_assets)
        w0 = w0 / w0.sum()

        try:
            result = minimize(
                neg_sortino, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
        except Exception:
            # Skip start ini kalau SLSQP gagal konvergen
            continue

        if result.success and -result.fun > best_sortino:
            best_sortino = float(-result.fun)
            best_weights = result.x

    # Fallback: kalau semua start gagal, pakai equal weight
    if best_weights is None:
        best_weights = np.ones(n_assets) / n_assets
        best_sortino = float(-neg_sortino(best_weights))

    return best_weights, best_sortino


def optimasi_slsqp_min_semivariance(sigma_sv, n_starts=10):
    """
    Optimasi SLSQP untuk meminimalkan Semivariance portofolio.

    Tidak melibatkan expected return, murni meminimalkan risiko downside.
    Hasil ini dipakai sebagai pembanding strategi max Sortino.

    Returns
    -------
    np.ndarray
        Vektor bobot optimal.
    """
    n_assets = sigma_sv.shape[0]
    sigma_sv = np.asarray(sigma_sv)

    def semivariance(w):
        return w @ sigma_sv @ w

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    best_w = None
    best_sv = np.inf

    for seed in range(n_starts):
        np.random.seed(seed)
        w0 = np.random.random(n_assets)
        w0 = w0 / w0.sum()

        try:
            result = minimize(
                semivariance, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
        except Exception:
            continue

        if result.success and result.fun < best_sv:
            best_sv = float(result.fun)
            best_w = result.x

    if best_w is None:
        best_w = np.ones(n_assets) / n_assets

    return best_w


def hitung_return_portofolio(returns, bobot):
    """
    Hitung return portofolio harian sebagai kombinasi linear dari
    return saham dan vektor bobot.
    """
    return returns @ bobot


def hitung_semideviation(returns, benchmark):
    """
    Hitung semideviation portofolio terhadap benchmark.

    Formula: sqrt(mean(min(R - B, 0)^2))

    Hanya memperhitungkan return yang di bawah benchmark,
    berbeda dengan standard deviation yang menghitung semua deviasi.
    """
    returns = np.asarray(returns)
    benchmark = np.asarray(benchmark)
    downside = np.minimum(returns - benchmark, 0)
    return float(np.sqrt(np.mean(downside ** 2)))


def hitung_sortino(returns, rf_annual, benchmark):
    """
    Hitung Sortino Ratio harian dan tahunan.

    Sortino tahunan dihitung dari return tahunan dan semideviation
    tahunan (dikonversi dengan faktor sqrt(252)).

    Returns
    -------
    dict dengan sortino_harian, sortino_tahunan, semidev_harian,
    semidev_tahunan, dan return_tahunan.
    """
    returns = np.asarray(returns)
    rf_daily = rf_annual / HARI_BURSA_SETAHUN

    semidev_harian = hitung_semideviation(returns, benchmark)
    semidev_tahunan = semidev_harian * np.sqrt(HARI_BURSA_SETAHUN)

    # Annualisasi return menggunakan compounding eksponensial dari log return
    return_tahunan = float(np.exp(np.mean(returns) * HARI_BURSA_SETAHUN) - 1)

    # Guard pembagian nol
    if semidev_harian > 0:
        sortino_harian = (np.mean(returns) - rf_daily) / semidev_harian
    else:
        sortino_harian = 0.0

    if semidev_tahunan > 0:
        sortino_tahunan = (return_tahunan - rf_annual) / semidev_tahunan
    else:
        sortino_tahunan = 0.0

    return {
        "sortino_harian": float(sortino_harian),
        "sortino_tahunan": float(sortino_tahunan),
        "semidev_harian": float(semidev_harian),
        "semidev_tahunan": float(semidev_tahunan),
        "return_tahunan": return_tahunan,
    }


def hitung_cumulative_return(returns):
    """
    Hitung cumulative return: exp(sum(returns)) - 1.

    Karena menggunakan log return, penjumlahan sederhana sudah
    menghasilkan log return kumulatif; perlu di-exponentiate untuk
    mendapat simple return.
    """
    returns = np.asarray(returns)
    return float(np.exp(np.sum(returns)) - 1)


def hitung_var_historical(returns, alpha=0.05, dana_awal=100_000_000):
    """
    VaR dengan pendekatan Historical Simulation.

    VaR = kerugian maksimum pada tingkat signifikansi alpha, dihitung
    langsung dari distribusi return historis tanpa asumsi distribusi.

    Parameters
    ----------
    returns : np.ndarray
        Seri return portofolio.
    alpha : float
        Tingkat signifikansi, default 0.05 untuk VaR 95%.
    dana_awal : float
        Dana awal untuk konversi VaR ke nominal rupiah.

    Returns
    -------
    dict berisi q_alpha, var_persen, var_rupiah, posisi_k, n_obs, pesan.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]

    n_obs = len(returns)

    # Minimal observasi untuk VaR 95%
    if n_obs < MIN_OBSERVASI:
        return {
            "q_alpha": float("nan"),
            "var_persen": float("nan"),
            "var_rupiah": float("nan"),
            "posisi_k": 0,
            "n_obs": n_obs,
            "pesan": f"Data terlalu sedikit untuk VaR. Minimal {MIN_OBSERVASI} observasi.",
        }

    # Urutkan return, ambil return pada posisi kuantil bawah
    returns_sorted = np.sort(returns)
    k = max(int(np.floor(alpha * n_obs)), 1)
    q_alpha = float(returns_sorted[k - 1])

    # q_alpha negatif = ada kerugian; positif = periode sangat baik
    if q_alpha < 0:
        var_persen = -q_alpha * 100
        var_rupiah = dana_awal * (-q_alpha)
        pesan = None
    else:
        var_persen = 0.0
        var_rupiah = 0.0
        pesan = (
            "Pada periode ini, kuantil 5% return masih positif. "
            "Tidak ada risiko loss terdeteksi."
        )

    return {
        "q_alpha": q_alpha,
        "var_persen": var_persen,
        "var_rupiah": var_rupiah,
        "posisi_k": k,
        "n_obs": n_obs,
        "pesan": pesan,
    }


def bandingkan_strategi(returns, bobot_max_sortino, bobot_min_sv, rf_annual, benchmark):
    """
    Bandingkan dua strategi optimasi SLSQP menggunakan matriks risiko
    yang sama (Semivariance-Semicovariance).

    Return DataFrame siap tampil di halaman Perbandingan Strategi.
    """
    returns = np.asarray(returns)

    return_max = returns @ np.asarray(bobot_max_sortino)
    return_min = returns @ np.asarray(bobot_min_sv)

    sortino_max = hitung_sortino(return_max, rf_annual, benchmark)
    sortino_min = hitung_sortino(return_min, rf_annual, benchmark)

    var_max = hitung_var_historical(return_max)
    var_min = hitung_var_historical(return_min)

    cumret_max = hitung_cumulative_return(return_max)
    cumret_min = hitung_cumulative_return(return_min)

    comparison = pd.DataFrame({
        "Metrik": [
            "Sortino Ratio Tahunan",
            "Sortino Ratio Harian",
            "Return Tahunan",
            "Semideviation Tahunan",
            "Cumulative Return",
            "VaR 95% (Rp)",
        ],
        "Max Sortino": [
            f"{sortino_max['sortino_tahunan']:.4f}",
            f"{sortino_max['sortino_harian']:.4f}",
            f"{sortino_max['return_tahunan'] * 100:.2f}%",
            f"{sortino_max['semidev_tahunan'] * 100:.2f}%",
            f"{cumret_max * 100:.2f}%",
            f"Rp {var_max['var_rupiah']:,.0f}",
        ],
        "Min Semivariance": [
            f"{sortino_min['sortino_tahunan']:.4f}",
            f"{sortino_min['sortino_harian']:.4f}",
            f"{sortino_min['return_tahunan'] * 100:.2f}%",
            f"{sortino_min['semidev_tahunan'] * 100:.2f}%",
            f"{cumret_min * 100:.2f}%",
            f"Rp {var_min['var_rupiah']:,.0f}",
        ],
    })

    return comparison


def deteksi_ketidakcocokan_benchmark(tickers, benchmark):
    """
    Deteksi ketidakcocokan antara jenis ticker dan benchmark yang dipilih.

    Contoh: user input saham ID (.JK) tapi benchmark = ^GSPC (S&P 500),
    hasil analisis jadi kurang relevan.

    Return string pesan peringatan jika tidak cocok, None jika OK.
    """
    if not tickers:
        return None

    n_jk = sum(1 for t in tickers if t.upper().endswith(".JK"))
    benchmark_upper = benchmark.upper()

    # Mayoritas saham ID tapi benchmark bukan IHSG
    if n_jk > len(tickers) / 2 and benchmark_upper != "^JKSE":
        return (
            "Mayoritas ticker adalah saham Indonesia (.JK), tetapi benchmark bukan IHSG (^JKSE). "
            "Pertimbangkan ganti benchmark ke ^JKSE untuk hasil yang lebih relevan."
        )

    # Mayoritas bukan saham ID tapi benchmark IHSG
    if n_jk < len(tickers) / 2 and benchmark_upper == "^JKSE":
        return (
            "Sebagian besar ticker bukan saham Indonesia, tetapi benchmark adalah IHSG (^JKSE). "
            "Pertimbangkan ganti benchmark ke indeks yang relevan (misalnya ^GSPC untuk saham US)."
        )

    return None