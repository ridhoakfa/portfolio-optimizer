"""Fungsi perhitungan aplikasi optimasi portofolio Mean-Semivariance; dipanggil dari main.py."""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import spearmanr


# Konstituen Indeks BISNIS-27 periode 3 November 2025 sampai 30 April 2026, format Yahoo Finance.
DAFTAR_SAHAM_DEFAULT = [
    "ADMR.JK", "ADRO.JK", "AMRT.JK", "ANTM.JK", "ASII.JK",
    "BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK", "BRPT.JK",
    "BUMI.JK", "DSNG.JK", "HEAL.JK", "INCO.JK", "INDF.JK",
    "INKP.JK", "JPFA.JK", "KLBF.JK", "MAPI.JK", "MEDC.JK",
    "MIKA.JK", "MYOR.JK", "NCKL.JK", "PGEO.JK", "PTBA.JK",
    "TLKM.JK", "UNTR.JK",
]

BENCHMARK_DEFAULT = "^JKSE"
START_DATE_DEFAULT = "2025-11-03"
END_DATE_DEFAULT = "2026-04-30"

THRESHOLD_KORELASI_DEFAULT = 0.2
MAX_SAHAM_DEFAULT = 13

DANA_AWAL_VAR_DEFAULT = 100_000_000
MIN_OBSERVASI = 30
JUMLAH_TITIK_AWAL = 10


def _akhir_eksklusif(end_date: str) -> str:
    """Geser tanggal akhir satu hari karena Yahoo Finance tidak menyertakan tanggal `end`."""
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _unduh_harga(tickers: list[str] | str, start_date: str, end_date: str) -> pd.DataFrame:
    """Unduh harga penutupan yang disesuaikan (adjusted close) dari Yahoo Finance."""
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=_akhir_eksklusif(end_date),
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        raise ConnectionError(
            f"Gagal mengunduh data dari Yahoo Finance: {type(e).__name__}. "
            f"Periksa koneksi internet."
        )

    if data is None or data.empty:
        raise ValueError(
            f"Tidak ada data untuk {tickers} pada periode {start_date} sampai {end_date}."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data


def load_data_saham(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Harga saham harian (baris tanggal, kolom ticker); tanggal akhir ikut diunduh."""
    return _unduh_harga(tickers, start_date, end_date).ffill().dropna(how="all")


def load_data_benchmark(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Harga benchmark harian, misalnya IHSG (^JKSE)."""
    return _unduh_harga(ticker, start_date, end_date).squeeze().ffill().dropna()


def hitung_log_return(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log return harian ln(P_t / P_{t-1}); baris pertama dibuang."""
    return np.log(data / data.shift(1)).dropna()


def hitung_expected_return(log_return: pd.DataFrame) -> pd.Series:
    """Expected return harian setiap saham, yaitu rata-rata log return."""
    return log_return.mean()


def seleksi_saham_dua_tahap(
    log_return: pd.DataFrame,
    expected_return: pd.Series,
    threshold: float = THRESHOLD_KORELASI_DEFAULT,
    max_saham: int = MAX_SAHAM_DEFAULT,
) -> dict:
    """
    Tahap 1: saham dengan expected return positif.
    Tahap 2: saham dari pasangan berkorelasi Spearman di bawah threshold,
    mulai dari pasangan berkorelasi terendah, sampai max_saham.
    """
    saham_positif = expected_return[expected_return > 0].index.tolist()

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

    spearman_corr, _ = spearmanr(log_return[saham_positif].values)
    spearman_matrix = pd.DataFrame(spearman_corr, index=saham_positif, columns=saham_positif)

    pasangan = [
        {
            "Saham A": saham_positif[i],
            "Saham B": saham_positif[j],
            "Korelasi": float(spearman_matrix.iloc[i, j]),
        }
        for i in range(len(saham_positif))
        for j in range(i + 1, len(saham_positif))
    ]
    pasangan_df = pd.DataFrame(pasangan).sort_values("Korelasi").reset_index(drop=True)
    pasangan_terpilih = pasangan_df[pasangan_df["Korelasi"] < threshold].copy()

    saham_terpilih: list[str] = []
    for _, row in pasangan_terpilih.iterrows():
        for saham in (row["Saham A"], row["Saham B"]):
            if saham not in saham_terpilih and len(saham_terpilih) < max_saham:
                saham_terpilih.append(saham)
        if len(saham_terpilih) >= max_saham:
            break

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


def hitung_matriks_semivariance(returns: pd.DataFrame | np.ndarray, benchmark: pd.Series | np.ndarray) -> np.ndarray:
    """Matriks semivariance-semicovariance terhadap benchmark B_t dengan pembagi T (Estrada, 2008)."""
    returns = np.asarray(returns)
    benchmark = np.asarray(benchmark).reshape(-1, 1)
    selisih_negatif = np.minimum(returns - benchmark, 0)
    return (selisih_negatif.T @ selisih_negatif) / returns.shape[0]


def hitung_sortino(bobot: np.ndarray, mu: np.ndarray, sigma_sv: np.ndarray, rata_benchmark: float) -> dict:
    """
    Sortino Ratio harian portofolio (w'mu - rata-rata benchmark) / sqrt(w' Sigma_sv w).

    Rumus yang sama dipakai sebagai fungsi tujuan Maximum Sortino Ratio dan untuk evaluasi.
    """
    bobot = np.asarray(bobot)
    excess_return = float(bobot @ np.asarray(mu) - rata_benchmark)
    semideviation = float(np.sqrt(max(bobot @ np.asarray(sigma_sv) @ bobot, 0.0)))
    sortino_harian = excess_return / semideviation if semideviation > 0 else 0.0
    return {
        "sortino_harian": sortino_harian,
        "excess_return": excess_return,
        "semideviation": semideviation,
    }


def _optimasi_slsqp(fungsi_tujuan, n_aset: int) -> tuple[np.ndarray, bool]:
    """
    Minimalkan fungsi_tujuan dengan SLSQP dari beberapa titik awal acak (benih 0 sampai 9).

    Kendala long-only (0 <= w_i <= 1) dan jumlah bobot sama dengan 1.
    Bila tidak ada titik awal yang konvergen, bobot sama rata dikembalikan dengan berhasil=False.
    """
    batas = [(0.0, 1.0)] * n_aset
    kendala = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    bobot_terbaik, nilai_terbaik = None, np.inf
    for benih in range(JUMLAH_TITIK_AWAL):
        w0 = np.random.RandomState(benih).random_sample(n_aset)
        hasil = minimize(
            fungsi_tujuan, w0 / w0.sum(),
            method="SLSQP",
            bounds=batas,
            constraints=kendala,
            options={"maxiter": 1000, "ftol": 1e-15},
        )
        if hasil.success and hasil.fun < nilai_terbaik:
            bobot_terbaik, nilai_terbaik = hasil.x, hasil.fun

    if bobot_terbaik is None:
        return np.ones(n_aset) / n_aset, False
    return bobot_terbaik, True


def optimasi_slsqp_max_sortino(mu: np.ndarray, sigma_sv: np.ndarray, rata_benchmark: float) -> tuple[np.ndarray, float, bool]:
    """Bobot yang memaksimalkan Sortino Ratio; mengembalikan (bobot, sortino_harian, berhasil)."""
    def negatif_sortino(w: np.ndarray) -> float:
        return -hitung_sortino(w, mu, sigma_sv, rata_benchmark)["sortino_harian"]

    bobot, berhasil = _optimasi_slsqp(negatif_sortino, len(mu))
    return bobot, -negatif_sortino(bobot), berhasil


def optimasi_slsqp_min_semivariance(sigma_sv: np.ndarray) -> tuple[np.ndarray, bool]:
    """Bobot yang meminimalkan semivariance portofolio w' Sigma_sv w; mengembalikan (bobot, berhasil)."""
    sigma_sv = np.asarray(sigma_sv)
    return _optimasi_slsqp(lambda w: w @ sigma_sv @ w, sigma_sv.shape[0])


def hitung_cumulative_return(returns: np.ndarray) -> float:
    """Cumulative Return exp(sum R_P,t) - 1 dari log return portofolio."""
    return float(np.exp(np.sum(returns)) - 1)


def hitung_var_historical(
    returns: np.ndarray,
    alpha: float = 0.05,
    dana_awal: float = DANA_AWAL_VAR_DEFAULT,
    holding_period: int = 1,
) -> dict:
    """
    VaR Simulasi Historis: V0 x (-R_P,(k)) x sqrt(t), dengan k = ceil(alpha x T)
    pada return portofolio yang diurutkan dari terkecil.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n_obs = len(returns)

    if n_obs < MIN_OBSERVASI:
        return {
            "q_alpha": float("nan"),
            "var_persen": float("nan"),
            "var_rupiah": float("nan"),
            "posisi_k": 0,
            "n_obs": n_obs,
            "holding_period": holding_period,
            "pesan": f"Data terlalu sedikit untuk VaR. Minimal {MIN_OBSERVASI} observasi.",
        }

    k = max(int(np.ceil(alpha * n_obs)), 1)
    q_alpha = float(np.sort(returns)[k - 1])
    faktor_holding = float(np.sqrt(holding_period))

    if q_alpha < 0:
        var_persen = -q_alpha * faktor_holding * 100
        var_rupiah = dana_awal * -q_alpha * faktor_holding
        pesan = None
    else:
        var_persen = var_rupiah = 0.0
        pesan = (
            "Pada periode ini, kuantil 5% return masih positif. "
            "Tidak ada risiko kerugian terdeteksi."
        )

    return {
        "q_alpha": q_alpha,
        "var_persen": var_persen,
        "var_rupiah": var_rupiah,
        "posisi_k": k,
        "n_obs": n_obs,
        "holding_period": holding_period,
        "pesan": pesan,
    }


def hitung_metrik_strategi(
    returns: np.ndarray,
    bobot: np.ndarray,
    mu: np.ndarray,
    sigma_sv: np.ndarray,
    benchmark: np.ndarray,
    dana_awal: float = DANA_AWAL_VAR_DEFAULT,
) -> dict:
    """Semua metrik evaluasi satu strategi: Sortino Ratio, Cumulative Return, dan VaR."""
    return_portofolio = np.asarray(returns) @ np.asarray(bobot)
    return {
        "return_portofolio": return_portofolio,
        "sortino": hitung_sortino(bobot, mu, sigma_sv, float(np.mean(benchmark))),
        "cumulative_return": hitung_cumulative_return(return_portofolio),
        "var": hitung_var_historical(return_portofolio, dana_awal=dana_awal),
    }


def deteksi_ketidakcocokan_benchmark(tickers: list[str], benchmark: str) -> str | None:
    """Pesan peringatan bila mayoritas ticker dan benchmark berasal dari pasar berbeda."""
    if not tickers:
        return None

    n_jk = sum(1 for t in tickers if t.upper().endswith(".JK"))
    benchmark_upper = benchmark.upper()

    if n_jk > len(tickers) / 2 and benchmark_upper != "^JKSE":
        return (
            "Mayoritas ticker adalah saham Indonesia (.JK), tetapi benchmark bukan IHSG (^JKSE). "
            "Pertimbangkan ganti benchmark ke ^JKSE untuk hasil yang lebih relevan."
        )

    if n_jk < len(tickers) / 2 and benchmark_upper == "^JKSE":
        return (
            "Sebagian besar ticker bukan saham Indonesia, tetapi benchmark adalah IHSG (^JKSE). "
            "Pertimbangkan ganti benchmark ke indeks yang relevan (misalnya ^GSPC untuk saham US)."
        )

    return None
