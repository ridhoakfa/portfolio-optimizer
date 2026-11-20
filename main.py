"""Aplikasi web Streamlit optimasi portofolio Mean-Semivariance dengan SLSQP (skripsi S1 Statistika Undip)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pandas.io.formats.style import Styler
from streamlit.delta_generator import DeltaGenerator

from functions import (
    DAFTAR_SAHAM_DEFAULT,
    BENCHMARK_DEFAULT,
    START_DATE_DEFAULT,
    END_DATE_DEFAULT,
    THRESHOLD_KORELASI_DEFAULT,
    MAX_SAHAM_DEFAULT,
    DANA_AWAL_VAR_DEFAULT,
    MIN_OBSERVASI,
    JUMLAH_TITIK_AWAL,
    load_data_saham,
    load_data_benchmark,
    hitung_log_return,
    hitung_expected_return,
    seleksi_saham_dua_tahap,
    hitung_matriks_semivariance,
    optimasi_slsqp_max_sortino,
    optimasi_slsqp_min_semivariance,
    hitung_cumulative_return,
    hitung_metrik_strategi,
    deteksi_ketidakcocokan_benchmark,
)


st.set_page_config(
    page_title="Portofolio Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


PALETTE_LINE = [
    "#1E3A8A", "#D97706", "#475569", "#047857", "#B91C1C",
    "#0369A1", "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
]

PALETTE_PIE = [
    "#1E3A8A", "#D97706", "#047857", "#B91C1C", "#0369A1",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#475569",
    "#0F1E54", "#92400E",
]

STRATEGI_MAX = "Maximum Sortino Ratio"
STRATEGI_MIN = "Minimum Semivariance"
WARNA_STRATEGI = {STRATEGI_MAX: "#1E3A8A", STRATEGI_MIN: "#D97706"}
WARNA_BENCHMARK = "#94A3B8"
WARNA_VAR = "#D97706"

BOBOT_MINIMAL_TAMPIL = 1e-6

BORDER_HAIRLINE = "1px solid rgba(128, 128, 128, 0.2)"
BORDER_DASHED = "1px dashed rgba(128, 128, 128, 0.25)"

# Pemisah desimal lalu ribuan untuk semua angka yang diformat Plotly (sumbu, hover, label).
PEMISAH_PLOTLY = ",."

BULAN_INDONESIA = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
BULAN_INDONESIA_PANJANG = (
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
)

NAMA_BENCHMARK = {"^JKSE": "IHSG", "^GSPC": "S&P 500"}

# Tingkat signifikansi VaR, sama dengan nilai bawaan alpha di hitung_var_historical.
ALPHA_VAR = 0.05
MAKS_SAHAM_BMATRIX = 13

KUNCI_HASIL_ANALISIS = [
    "data_saham", "data_benchmark", "log_return", "log_return_benchmark", "expected_return",
    "hasil_seleksi", "saham_terpilih", "returns_terpilih", "benchmark_aligned", "mu_terpilih",
    "matriks_sv", "bobot_max_sortino", "bobot_min_sv", "metrik",
]

PESAN_BELUM_ANALISIS = "Analisis belum dijalankan. Buka halaman Data, lalu tekan Jalankan Analisis."


def format_angka(nilai: float, desimal: int | None = 2, awalan: str = "", akhiran: str = "") -> str:
    """Angka format Indonesia (titik ribuan, koma desimal); desimal None memakai format ringkas."""
    teks = f"{nilai:,g}" if desimal is None else f"{nilai:,.{desimal}f}"
    teks = teks.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{awalan}{teks}{akhiran}"


def format_tabel(df: pd.DataFrame, desimal_kolom: dict[str, int]) -> Styler:
    """Styler yang menampilkan kolom angka dengan format_angka tanpa mengubah nilai aslinya."""
    return df.style.format({
        kolom: (lambda x, d=desimal: format_angka(x, d)) for kolom, desimal in desimal_kolom.items()
    })


def format_tanggal(tanggal: pd.Timestamp) -> str:
    """Tanggal dengan nama bulan Indonesia, misalnya 3 Nov 2025."""
    return f"{tanggal.day} {BULAN_INDONESIA[tanggal.month - 1]} {tanggal.year}"


def format_tanggal_panjang(tanggal: str) -> str:
    """Tanggal konfigurasi (YYYY-MM-DD) dengan nama bulan lengkap, misalnya 3 November 2025."""
    t = pd.Timestamp(tanggal)
    return f"{t.day} {BULAN_INDONESIA_PANJANG[t.month - 1]} {t.year}"


def atur_sumbu_tanggal(fig: go.Figure, tanggal: pd.DatetimeIndex) -> None:
    """Label sumbu x pada hari bursa pertama setiap bulan dengan nama bulan Indonesia."""
    awal_bulan = pd.Series(tanggal, index=tanggal).groupby(tanggal.to_period("M")).first()
    fig.update_xaxes(
        tickvals=list(awal_bulan),
        ticktext=[f"{BULAN_INDONESIA[t.month - 1]} {t.year}" for t in awal_bulan],
    )


def nama_benchmark(ticker: str) -> str:
    """Nama benchmark untuk kalimat: IHSG, S&P 500, atau ticker-nya."""
    return NAMA_BENCHMARK.get(ticker, ticker)


def label_benchmark(ticker: str) -> str:
    """Label benchmark untuk tabel dan grafik, misalnya IHSG (^JKSE)."""
    nama = nama_benchmark(ticker)
    return f"{nama} ({ticker})" if nama != ticker else ticker


def gabung_nama(daftar: list[str]) -> str:
    """Gabungkan nama menjadi "A", "A dan B", atau "A, B, dan C"."""
    if len(daftar) <= 2:
        return " dan ".join(daftar)
    return f"{', '.join(daftar[:-1])}, dan {daftar[-1]}"


def tinggi_tabel(n_baris: int) -> int:
    """Tinggi st.dataframe dalam piksel agar semua baris tampil tanpa gulir."""
    return 35 * (n_baris + 1) + 3


def tombol_unduh_csv(df: pd.DataFrame, label: str, nama_file: str) -> None:
    """Tombol unduh CSV berpemisah titik koma dengan koma desimal (UTF-8 BOM)."""
    # Pemisah titik koma dan koma desimal agar CSV langsung terbaca di Excel berlokal Indonesia.
    isi_csv = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label,
        data=isi_csv,
        file_name=nama_file,
        mime="text/csv",
        key=f"unduh_{nama_file}",
        icon=":material/download:",
    )


def init_state() -> None:
    """Isi nilai awal session_state sekali per sesi."""
    defaults = {kunci: None for kunci in KUNCI_HASIL_ANALISIS}
    defaults.update({
        "solver_gagal": [],
        "config": {
            "tickers": DAFTAR_SAHAM_DEFAULT,
            "benchmark": BENCHMARK_DEFAULT,
            "start_date": START_DATE_DEFAULT,
            "end_date": END_DATE_DEFAULT,
            "threshold": THRESHOLD_KORELASI_DEFAULT,
            "max_saham": MAX_SAHAM_DEFAULT,
            "dana_awal": DANA_AWAL_VAR_DEFAULT,
        },
    })
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Hasil dari versi kode sebelumnya (nama strategi berbeda) tidak bisa dipakai halaman sekarang.
    metrik = st.session_state.metrik
    if metrik is not None and set(metrik) != set(WARNA_STRATEGI):
        _hapus_hasil_analisis()


def require_analisis() -> None:
    """Guard: hentikan halaman bila analisis belum dijalankan."""
    if st.session_state.metrik is None:
        empty_state(PESAN_BELUM_ANALISIS)
        st.stop()


def hairline_header(title: str, metadata: str = "") -> None:
    """Judul halaman dengan garis atas dan bawah serta metadata di kanan."""
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline;
                    border-top: {BORDER_HAIRLINE};
                    border-bottom: {BORDER_HAIRLINE};
                    padding: 12px 0; margin-bottom: 24px;">
            <div style="font-size: 0.85rem; font-weight: 600;
                        letter-spacing: 0.08em; text-transform: uppercase;">
                {title}
            </div>
            <div style="font-family: monospace; font-size: 0.75rem; opacity: 0.7;">
                {metadata or "&nbsp;"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    """Judul section dengan garis bawah tipis dan letter-spacing lebar."""
    st.markdown(
        f"""
        <div style="font-size: 0.72rem; font-weight: 600;
                    letter-spacing: 0.08em; text-transform: uppercase;
                    margin-top: 28px; margin-bottom: 12px;
                    padding-bottom: 6px;
                    border-bottom: {BORDER_HAIRLINE};
                    opacity: 0.75;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, accent: str = "#1E3A8A", sublabel: str | None = None) -> None:
    """Kartu satu nilai metrik dengan garis aksen di kiri."""
    sublabel_html = ""
    if sublabel:
        sublabel_html = (
            f'<div style="font-size: 0.72rem; opacity: 0.7; margin-top: 6px;">'
            f'{sublabel}</div>'
        )

    st.markdown(
        f"""
        <div style="border: {BORDER_HAIRLINE};
                    border-left: 3px solid {accent}; border-radius: 8px;
                    padding: 16px 18px;">
            <div style="font-size: 0.68rem; font-weight: 500;
                        letter-spacing: 0.06em; text-transform: uppercase;
                        opacity: 0.7; margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-family: monospace; font-size: 1.4rem; font-weight: 500;">
                {value}
            </div>
            {sublabel_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, kind: str = "info") -> None:
    """Kotak informasi berwarna sesuai jenisnya (info, success, warning, error)."""
    color_map = {
        "info":    ("#0369A1", "rgba(3, 105, 161, 0.12)"),
        "success": ("#047857", "rgba(4, 120, 87, 0.12)"),
        "warning": ("#92400E", "rgba(146, 64, 14, 0.12)"),
        "error":   ("#991B1B", "rgba(153, 27, 27, 0.12)"),
    }
    color, bg = color_map.get(kind, color_map["info"])

    st.markdown(
        f"""
        <div style="background: {bg}; border-left: 3px solid {color};
                    border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
                    font-size: 0.85rem; line-height: 1.55;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message: str) -> None:
    """Tampilan saat tidak ada data untuk ditampilkan."""
    st.markdown(
        f"""
        <div style="border: {BORDER_DASHED}; border-radius: 8px;
                    padding: 40px 24px; text-align: center; margin-top: 24px;">
            <div style="font-size: 0.9rem; opacity: 0.7;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pie_bobot(saham_list: list[str], bobot: np.ndarray, warna_saham: dict[str, str]) -> go.Figure | None:
    """Pie chart bobot dengan warna tetap per saham; bobot di bawah BOBOT_MINIMAL_TAMPIL tidak digambar."""
    mask = bobot > BOBOT_MINIMAL_TAMPIL
    labels = [s for s, m in zip(saham_list, mask) if m]
    values = bobot[mask] * 100
    colors = [warna_saham[s] for s in labels]

    if len(labels) == 0:
        return None

    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textposition="auto",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Bobot: %{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        separators=PEMISAH_PLOTLY,
        height=380,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _kartu_status_analisis() -> None:
    """Kartu Status Analisis di sidebar: ringkasan data atau keterangan belum dijalankan."""
    if st.session_state.metrik is not None:
        n_saham = st.session_state.data_saham.shape[1]
        n_hari = st.session_state.data_saham.shape[0]
        bench = st.session_state.config["benchmark"]
        n_terpilih = len(st.session_state.saham_terpilih)
        threshold = format_angka(st.session_state.config["threshold"], None)

        st.markdown(
            f"""
            <div style="border: {BORDER_HAIRLINE};
                        border-radius: 6px; padding: 12px 14px;
                        font-family: monospace;
                        font-size: 0.72rem; line-height: 1.7;">
                <div style="font-weight: 500;">
                    {n_saham} saham · {n_hari} hari bursa
                </div>
                <div style="opacity: 0.8;">Benchmark: {bench}</div>
                <div style="opacity: 0.8;">Terpilih: {n_terpilih} saham</div>
                <div style="opacity: 0.8;">Threshold: Spearman &lt; {threshold}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="border: {BORDER_DASHED};
                        border-radius: 6px; padding: 12px 14px;
                        font-family: monospace;
                        font-size: 0.72rem; opacity: 0.5;
                        text-align: center;">
                Belum dijalankan
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> tuple[str, DeltaGenerator]:
    """Sidebar navigasi dan status analisis; mengembalikan nama halaman dan wadah kartu status."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 0 16px;">
                <div style="font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;">
                    Portofolio Optimizer
                </div>
                <div style="font-size: 0.72rem; opacity: 0.7; margin-top: 2px;">
                    Mean-Semivariance · SLSQP
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_raw = st.radio(
            "Navigasi",
            options=[
                ":material/home: Beranda",
                ":material/database: Data",
                ":material/filter_alt: Seleksi Saham",
                ":material/tune: Optimasi Portofolio",
                ":material/assessment: Hasil & Perbandingan",
                ":material/function: Detail Perhitungan",
            ],
            label_visibility="collapsed",
        )

        page = page_raw.split(" ", 1)[-1] if " " in page_raw else page_raw

        st.divider()

        st.markdown(
            """
            <div style="font-size: 0.65rem; font-weight: 600;
                        letter-spacing: 0.1em; text-transform: uppercase;
                        opacity: 0.6; margin-bottom: 10px;">
                Status Analisis
            </div>
            """,
            unsafe_allow_html=True,
        )

        wadah_status = st.empty()
        with wadah_status.container():
            _kartu_status_analisis()

    return page, wadah_status


def render_home() -> None:
    """Halaman Beranda: penjelasan aplikasi, alur analisis, dan konfigurasi default."""
    hairline_header("Beranda", "By: Ridho Akbar Fadhilah")

    st.markdown(
        """
        <div style="font-size: 1.55rem; font-weight: 600;
                    letter-spacing: -0.015em; margin: 32px 0 10px;">
            Optimasi Portofolio Mean-Semivariance
        </div>
        <div style="font-size: 1rem; opacity: 0.75;
                    margin-bottom: 8px; max-width: 760px; line-height: 1.6;">
            Aplikasi untuk analisis dan optimasi portofolio saham menggunakan
            pendekatan Mean-Semivariance. Menyediakan dua strategi optimasi:
            memaksimalkan Sortino Ratio dan meminimalkan Semivariance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Tentang Aplikasi")

    st.markdown(
        """
        Aplikasi ini membentuk portofolio optimal melalui dua tahap. Tahap pertama
        menyeleksi saham dengan kriteria expected return positif dan diversifikasi
        korelasi rendah. Tahap kedua mengoptimasi bobot menggunakan Sequential
        Least Squares Programming dengan dua fungsi tujuan alternatif.

        Dua strategi optimasi yang tersedia:

        - **Max Sortino Ratio**: memaksimalkan risk-adjusted return dengan
          menekan risiko downside terhadap benchmark.
        - **Min Semivariance**: meminimalkan risiko downside tanpa memperhatikan
          expected return secara langsung.

        Kedua strategi menggunakan matriks risiko yang sama, sehingga perbandingan
        kinerja bersifat langsung dan setara.
        """
    )

    section_title("Alur Analisis")

    steps = [
        ("01", "Data", "Atur ticker, benchmark, dan periode, lalu tekan Jalankan Analisis."),
        ("02", "Eksplorasi", "Statistik deskriptif dan pergerakan harga di halaman Data."),
        ("03", "Seleksi Saham", "Filter E(R) > 0, lalu korelasi Spearman di bawah threshold."),
        ("04", "Optimasi Portofolio", "SLSQP Maximum Sortino Ratio dan Minimum Semivariance."),
        ("05", "Hasil & Perbandingan", "Sortino Ratio harian, Cumulative Return, dan VaR 95% Simulasi Historis."),
        ("06", "Detail Perhitungan", "Rumus dan angka antara setiap tahap."),
    ]

    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="border: {BORDER_HAIRLINE}; border-radius: 8px;
                            padding: 18px; margin-bottom: 16px; min-height: 150px;">
                    <div style="font-family: monospace; font-size: 0.7rem;
                                color: #D97706; font-weight: 500; margin-bottom: 8px;">
                        {num}
                    </div>
                    <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 6px;">
                        {title}
                    </div>
                    <div style="font-size: 0.78rem; opacity: 0.7; line-height: 1.55;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Konfigurasi Default")

    st.markdown(
        f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
            <tr style="border-bottom: {BORDER_HAIRLINE};">
                <td style="padding: 10px 0; opacity: 0.7; width: 40%;">Ticker</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    27 saham BISNIS-27
                </td>
            </tr>
            <tr style="border-bottom: {BORDER_HAIRLINE};">
                <td style="padding: 10px 0; opacity: 0.7;">Benchmark</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    {BENCHMARK_DEFAULT}
                </td>
            </tr>
            <tr style="border-bottom: {BORDER_HAIRLINE};">
                <td style="padding: 10px 0; opacity: 0.7;">Periode</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    {START_DATE_DEFAULT} → {END_DATE_DEFAULT}
                </td>
            </tr>
            <tr style="border-bottom: {BORDER_HAIRLINE};">
                <td style="padding: 10px 0; opacity: 0.7;">Threshold Spearman</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    {format_angka(THRESHOLD_KORELASI_DEFAULT, None)}
                </td>
            </tr>
            <tr>
                <td style="padding: 10px 0; opacity: 0.7;">Maksimum saham</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    {MAX_SAHAM_DEFAULT}
                </td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _parse_tickers(text: str) -> list[str]:
    """Pisahkan input ticker (baris baru atau koma) menjadi daftar unik huruf kapital."""
    tickers = []
    for line in text.replace(",", "\n").split("\n"):
        t = line.strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def _pesan_seleksi(hasil_seleksi: dict, threshold: float) -> str | None:
    """Pesan seleksi dari functions.py dengan angka threshold dalam format Indonesia."""
    pesan = hasil_seleksi.get("pesan")
    if not pesan:
        return None
    return pesan.replace(f"Spearman {threshold}", f"Spearman {format_angka(threshold, None)}")


def _hapus_hasil_analisis() -> None:
    """Kosongkan semua hasil analisis sebelumnya di session_state."""
    for kunci in KUNCI_HASIL_ANALISIS:
        st.session_state[kunci] = None
    st.session_state.solver_gagal = []


def _jalankan_pipeline(tickers: list[str], benchmark: str, start_date: str, end_date: str,
                       threshold: float, max_saham: int, dana_awal: int) -> None:
    """Jalankan unduh data, seleksi, optimasi, dan evaluasi; hasil disimpan hanya bila semua tahap berhasil."""
    _hapus_hasil_analisis()

    if len(tickers) < 2:
        st.error("Minimal 2 ticker diperlukan untuk membentuk portofolio.")
        return

    pesan_galat = None
    with st.status("Mengunduh data", expanded=False) as status:
        try:
            data_saham = load_data_saham(tickers, start_date, end_date)
            data_benchmark = load_data_benchmark(benchmark, start_date, end_date)
        except (ConnectionError, ValueError) as e:
            pesan_galat = str(e)

        if pesan_galat is None:
            idx_bersama = data_saham.index.intersection(data_benchmark.index)
            data_saham = data_saham.loc[idx_bersama]
            data_benchmark = data_benchmark.loc[idx_bersama]
            if len(idx_bersama) < MIN_OBSERVASI:
                pesan_galat = (
                    f"Hanya {len(idx_bersama)} hari bursa yang beririsan antara saham dan benchmark. "
                    f"Minimal {MIN_OBSERVASI} hari diperlukan."
                )

        if pesan_galat is None:
            status.update(label="Seleksi saham")
            log_return = hitung_log_return(data_saham)
            log_return_benchmark = hitung_log_return(data_benchmark)
            expected_return = hitung_expected_return(log_return)
            hasil_seleksi = seleksi_saham_dua_tahap(
                log_return, expected_return,
                threshold=threshold, max_saham=max_saham,
            )
            saham_terpilih = hasil_seleksi["saham_terpilih"]
            if not saham_terpilih:
                pesan_galat = _pesan_seleksi(hasil_seleksi, threshold) or "Seleksi saham tidak menghasilkan saham."

        if pesan_galat is None:
            status.update(label="Optimasi SLSQP")
            returns_terpilih = log_return[saham_terpilih].values
            benchmark_aligned = log_return_benchmark.loc[log_return.index].values
            matriks_sv = hitung_matriks_semivariance(returns_terpilih, benchmark_aligned)
            mu_terpilih = log_return[saham_terpilih].mean().values
            bobot_max, _, berhasil_max = optimasi_slsqp_max_sortino(
                mu_terpilih, matriks_sv, float(np.mean(benchmark_aligned))
            )
            bobot_min, berhasil_min = optimasi_slsqp_min_semivariance(matriks_sv)

            status.update(label="Evaluasi")
            metrik = {
                nama: hitung_metrik_strategi(
                    returns_terpilih, bobot, mu_terpilih, matriks_sv, benchmark_aligned, dana_awal
                )
                for nama, bobot in ((STRATEGI_MAX, bobot_max), (STRATEGI_MIN, bobot_min))
            }
            status.update(label="Analisis selesai", state="complete")
        else:
            status.update(state="error")

    if pesan_galat is not None:
        st.error(pesan_galat)
        return

    st.session_state.update({
        "data_saham": data_saham,
        "data_benchmark": data_benchmark,
        "log_return": log_return,
        "log_return_benchmark": log_return_benchmark,
        "expected_return": expected_return,
        "hasil_seleksi": hasil_seleksi,
        "saham_terpilih": saham_terpilih,
        "returns_terpilih": returns_terpilih,
        "benchmark_aligned": benchmark_aligned,
        "matriks_sv": matriks_sv,
        "mu_terpilih": mu_terpilih,
        "bobot_max_sortino": bobot_max,
        "bobot_min_sv": bobot_min,
        "metrik": metrik,
        "solver_gagal": [
            nama for nama, berhasil in ((STRATEGI_MAX, berhasil_max), (STRATEGI_MIN, berhasil_min))
            if not berhasil
        ],
    })
    st.session_state.config.update({
        "tickers": tickers,
        "benchmark": benchmark,
        "start_date": start_date,
        "end_date": end_date,
        "threshold": threshold,
        "max_saham": max_saham,
        "dana_awal": dana_awal,
    })


def render_data(wadah_status: DeltaGenerator) -> None:
    """Halaman Data: konfigurasi aset dan parameter, tombol analisis, lalu eksplorasi data."""
    hairline_header("Data")

    info_box(
        "Nilai bawaan: 27 saham BISNIS-27 dengan benchmark IHSG (^JKSE). "
        "Untuk aset lain, ubah daftar ticker dan benchmark.",
        kind="info",
    )

    section_title("Konfigurasi Aset")

    col1, col2 = st.columns([2, 1])

    with col1:
        default_tickers = "\n".join(st.session_state.config["tickers"])
        tickers_input = st.text_area(
            "Daftar Ticker (satu per baris atau pisahkan dengan koma)",
            value=default_tickers,
            height=220,
            help="Contoh: BBCA.JK, BBRI.JK (saham ID) · AAPL, MSFT (saham US) · BTC-USD (kripto)",
        )

    with col2:
        benchmark_input = st.text_input(
            "Benchmark",
            value=st.session_state.config["benchmark"],
            help="Contoh: ^JKSE (IHSG), ^GSPC (S&P 500), ^N225 (Nikkei)",
        )

    section_title("Periode dan Parameter Seleksi")

    col3, col4, col5 = st.columns(3)

    with col3:
        start_input = st.date_input(
            "Tanggal Mulai",
            value=pd.to_datetime(st.session_state.config["start_date"]),
        )
        end_input = st.date_input(
            "Tanggal Akhir",
            value=pd.to_datetime(st.session_state.config["end_date"]),
        )

    with col4:
        threshold_input = st.number_input(
            "Threshold Korelasi Spearman",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.config["threshold"]),
            step=0.05,
            help="Pasangan dengan korelasi di bawah threshold dianggap terdiversifikasi",
        )
        max_saham_input = st.number_input(
            "Maksimum Saham Terpilih",
            min_value=2,
            max_value=30,
            value=int(st.session_state.config["max_saham"]),
            step=1,
        )

    with col5:
        dana_awal_input = st.number_input(
            "Dana Awal untuk VaR (Rp)",
            min_value=0,
            max_value=10_000_000_000,
            value=int(st.session_state.config["dana_awal"]),
            step=1_000_000,
            format="%d",
        )

    tickers_parsed = _parse_tickers(tickers_input)
    warning = deteksi_ketidakcocokan_benchmark(tickers_parsed, benchmark_input)
    if warning:
        info_box(warning, kind="warning")

    st.markdown("")
    run = st.button(
        "Jalankan Analisis",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
    )

    if run:
        _jalankan_pipeline(
            tickers=tickers_parsed,
            benchmark=benchmark_input.strip().upper(),
            start_date=str(start_input),
            end_date=str(end_input),
            threshold=threshold_input,
            max_saham=max_saham_input,
            dana_awal=dana_awal_input,
        )
        with wadah_status.container():
            _kartu_status_analisis()

    if st.session_state.metrik is not None:
        _ringkasan_analisis()
        _render_eksplorasi()


def _ringkasan_analisis() -> None:
    """Kotak ringkasan data dan peringatan seleksi setelah analisis berhasil."""
    data_saham = st.session_state.data_saham
    saham_terpilih = st.session_state.saham_terpilih

    info_box(
        f"Data dimuat: <strong>{data_saham.shape[1]} saham</strong> · "
        f"<strong>{data_saham.shape[0]} hari bursa</strong>. "
        f"Seleksi menghasilkan <strong>{len(saham_terpilih)} saham</strong>.",
        kind="success",
    )

    if len(saham_terpilih) < 3:
        st.warning(
            f"Hanya {len(saham_terpilih)} saham lolos seleksi. "
            f"Portofolio terlalu terkonsentrasi. "
            f"Naikkan threshold atau perpanjang periode."
        )
        pesan = _pesan_seleksi(st.session_state.hasil_seleksi, st.session_state.config["threshold"])
        if pesan:
            st.info(pesan)


def _render_eksplorasi() -> None:
    """Statistik deskriptif dan pergerakan harga di halaman Data, beserta unduhan data harga."""
    log_return = st.session_state.log_return
    data_saham = st.session_state.data_saham
    data_benchmark = st.session_state.data_benchmark
    benchmark = st.session_state.config["benchmark"]

    section_title("Eksplorasi Data")

    tab_statistik, tab_harga = st.tabs(["Statistik Deskriptif", "Pergerakan Harga"])

    with tab_statistik:
        kolom_persen = ["Expected Return (%)", "Standar Deviasi (%)", "Min (%)", "Max (%)"]

        def ringkasan(seri: pd.DataFrame | pd.Series) -> pd.DataFrame:
            return pd.DataFrame({
                "Expected Return (%)": seri.mean() * 100,
                "Standar Deviasi (%)": seri.std() * 100,
                "Min (%)": seri.min() * 100,
                "Max (%)": seri.max() * 100,
            }, index=None if isinstance(seri, pd.DataFrame) else [0])

        statistik_saham = ringkasan(log_return).rename_axis("Kode Saham").reset_index()
        statistik_saham = statistik_saham.sort_values("Expected Return (%)", ascending=False)
        statistik_benchmark = ringkasan(st.session_state.log_return_benchmark)
        statistik_benchmark.insert(0, "Kode Saham", label_benchmark(benchmark))
        statistik = pd.concat([statistik_saham, statistik_benchmark], ignore_index=True)

        st.dataframe(
            format_tabel(statistik, {kolom: 4 for kolom in kolom_persen}),
            hide_index=True, width="stretch", height=tinggi_tabel(len(statistik)),
        )

    with tab_harga:
        st.caption(
            "Base 100 memungkinkan perbandingan performa antar aset dengan skala harga berbeda. "
            "Semua aset mulai dari nilai 100 pada hari pertama periode. Nilai 150 berarti naik 50% "
            "dari hari pertama; nilai 80 berarti turun 20%."
        )

        normalized = data_saham / data_saham.iloc[0] * 100
        normalized_benchmark = data_benchmark / data_benchmark.iloc[0] * 100
        tanggal_teks = [format_tanggal(t) for t in normalized.index]
        template_hover = "<b>%{fullData.name}</b><br>Tanggal: %{customdata}<br>Nilai: %{y:.2f}<extra></extra>"

        fig = go.Figure()
        for i, ticker in enumerate(normalized.columns):
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized[ticker],
                mode="lines",
                name=ticker,
                line=dict(width=1.5, color=PALETTE_LINE[i % len(PALETTE_LINE)]),
                customdata=tanggal_teks,
                hovertemplate=template_hover,
            ))
        fig.add_trace(go.Scatter(
            x=normalized_benchmark.index,
            y=normalized_benchmark,
            mode="lines",
            name=label_benchmark(benchmark),
            line=dict(width=2, color=WARNA_BENCHMARK, dash="dash"),
            customdata=tanggal_teks,
            hovertemplate=template_hover,
        ))

        fig.update_layout(
            template="plotly_white",
            separators=PEMISAH_PLOTLY,
            height=520,
            xaxis_title="",
            yaxis_title="Nilai Ternormalisasi (Base 100)",
            hovermode="closest",
            margin=dict(l=60, r=180, t=30, b=60),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10),
                itemsizing="constant",
            ),
        )
        atur_sumbu_tanggal(fig, normalized.index)
        st.plotly_chart(fig, width="stretch")

    harga = data_saham.copy()
    harga[benchmark] = data_benchmark
    harga.index = harga.index.strftime("%Y-%m-%d")
    tombol_unduh_csv(harga.rename_axis("Tanggal").reset_index(), "Unduh data harga (CSV)", "data_harga.csv")


def _tabel_pasangan_terpilih(pasangan_terpilih: pd.DataFrame, saham_terpilih: list[str]) -> pd.DataFrame:
    """Pasangan di bawah threshold beserta saham yang pertama kali masuk dari pasangan itu."""
    # Saham terpilih selalu masuk pada pasangan pertama tempat ia muncul, sesuai urutan seleksi tahap 2.
    sudah_masuk: set[str] = set()
    saham_baru = []
    for _, baris in pasangan_terpilih.iterrows():
        baru = [s for s in (baris["Saham A"], baris["Saham B"]) if s in saham_terpilih and s not in sudah_masuk]
        sudah_masuk.update(baru)
        saham_baru.append(", ".join(baru) if baru else "-")

    tabel = pasangan_terpilih.reset_index(drop=True).copy()
    tabel.insert(0, "No", np.arange(1, len(tabel) + 1))
    tabel["Saham Baru Terpilih"] = saham_baru
    return tabel


def _kalimat_seleksi(hasil: dict, saham_terpilih: list[str], n_saham: int, threshold: float, max_saham: int) -> str:
    """Ringkasan otomatis seleksi dua tahap beserta alasan saham lolos tahap 1 yang tidak terpilih."""
    teks_threshold = format_angka(threshold, None)
    saham_positif = hasil["saham_positif"]
    kalimat = [f"{len(saham_positif)} dari {n_saham} saham lolos tahap 1."]

    if hasil["pasangan_terpilih"] is None:
        kalimat.append(f"Terpilih {len(saham_terpilih)} saham.")
        return " ".join(kalimat)

    pasangan = hasil["pasangan_terpilih"]
    kalimat.append(
        f"Dari {len(pasangan)} pasangan berkorelasi &lt; {teks_threshold}, terpilih {len(saham_terpilih)} saham."
    )

    di_pasangan = set(pasangan["Saham A"]) | set(pasangan["Saham B"])
    tidak_terpilih = [s for s in saham_positif if s not in saham_terpilih]
    tanpa_pasangan = [s for s in tidak_terpilih if s not in di_pasangan]
    kena_batas = [s for s in tidak_terpilih if s in di_pasangan]

    if tanpa_pasangan:
        kalimat.append(
            f"{gabung_nama(tanpa_pasangan)} tidak terpilih karena korelasinya dengan setiap saham lain "
            f"bernilai {teks_threshold} atau lebih."
        )
    if kena_batas:
        kalimat.append(
            f"{gabung_nama(kena_batas)} tidak terpilih karena batas maksimum {max_saham} saham sudah tercapai."
        )
    return " ".join(kalimat)


def render_seleksi() -> None:
    """Halaman Seleksi Saham: hasil tahap 1, tahap 2, dan heatmap Spearman."""
    hairline_header("Seleksi Saham")
    require_analisis()

    hasil = st.session_state.hasil_seleksi
    saham_terpilih = st.session_state.saham_terpilih
    expected_return = st.session_state.expected_return
    threshold = st.session_state.config["threshold"]
    teks_threshold = format_angka(threshold, None)

    info_box(
        _kalimat_seleksi(
            hasil, saham_terpilih, len(expected_return), threshold, st.session_state.config["max_saham"]
        ),
        kind="info",
    )

    section_title("Tahap 1: Expected Return Positif")

    tahap_1 = pd.DataFrame({
        "Kode Saham": expected_return.index,
        "Expected Return (%)": expected_return.values * 100,
        "Status": ["Lolos" if t in hasil["saham_positif"] else "Tidak lolos" for t in expected_return.index],
    }).sort_values("Expected Return (%)", ascending=False)
    st.dataframe(
        format_tabel(tahap_1, {"Expected Return (%)": 4}),
        hide_index=True, width="stretch", height=tinggi_tabel(len(tahap_1)),
    )

    section_title(f"Tahap 2: Korelasi Spearman < {teks_threshold}")

    if hasil["pasangan_terpilih"] is not None:
        n_pasangan = len(hasil["pasangan_terpilih"])
        st.markdown(f"**{n_pasangan} pasangan memenuhi threshold < {teks_threshold}**")
        tabel_pasangan = _tabel_pasangan_terpilih(hasil["pasangan_terpilih"], saham_terpilih)
        st.dataframe(
            format_tabel(tabel_pasangan, {"Korelasi": 4}),
            hide_index=True, width="stretch", height=tinggi_tabel(len(tabel_pasangan)),
        )

        section_title("Heatmap Spearman")

        spearman_matrix = hasil["spearman_matrix"]

        fig_heat = go.Figure(data=go.Heatmap(
            z=spearman_matrix.values,
            x=spearman_matrix.columns.tolist(),
            y=spearman_matrix.index.tolist(),
            colorscale=[
                [0.0, "#1E3A8A"],
                [0.5, "#FFFFFF"],
                [1.0, "#B91C1C"],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[format_angka(v, 2) for v in baris] for baris in spearman_matrix.values],
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="Spearman(%{x}, %{y}) = %{z:.3f}<extra></extra>",
        ))

        posisi = {saham: i for i, saham in enumerate(spearman_matrix.columns)}
        for _, baris in hasil["pasangan_terpilih"].iterrows():
            a, b = posisi[baris["Saham A"]], posisi[baris["Saham B"]]
            for kolom, baris_sel in ((a, b), (b, a)):
                fig_heat.add_shape(
                    type="rect",
                    x0=kolom - 0.5, x1=kolom + 0.5, y0=baris_sel - 0.5, y1=baris_sel + 0.5,
                    line=dict(color="#D97706", width=2),
                )

        fig_heat.update_layout(
            template="plotly_white",
            separators=PEMISAH_PLOTLY,
            height=max(400, 22 * len(spearman_matrix.columns)),
            xaxis=dict(tickfont=dict(size=10), side="bottom"),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, width="stretch")

    section_title("Saham Terpilih")

    df_terpilih = pd.DataFrame({
        "Kode Saham": saham_terpilih,
        "Expected Return (%)": [expected_return[t] * 100 for t in saham_terpilih],
    })
    st.dataframe(
        format_tabel(df_terpilih, {"Expected Return (%)": 4}),
        hide_index=True, width="stretch", height=tinggi_tabel(len(df_terpilih)),
    )


def render_optimasi() -> None:
    """Halaman Optimasi Portofolio: ringkasan bobot, pie chart, dan tabel perbandingan bobot."""
    hairline_header("Optimasi Portofolio")
    require_analisis()

    saham_terpilih = st.session_state.saham_terpilih
    bobot_strategi = {
        STRATEGI_MAX: st.session_state.bobot_max_sortino,
        STRATEGI_MIN: st.session_state.bobot_min_sv,
    }

    info_box(
        f"Optimasi dijalankan untuk <strong>{len(saham_terpilih)} saham</strong> "
        f"dengan matriks semivariance berukuran {len(saham_terpilih)}×{len(saham_terpilih)}. "
        f"Dua strategi: {STRATEGI_MAX} dan {STRATEGI_MIN} (keduanya long-only).",
        kind="info",
    )

    if st.session_state.solver_gagal:
        st.warning(
            "SLSQP tidak konvergen dari semua titik awal untuk strategi "
            f"{', '.join(st.session_state.solver_gagal)}. Bobot diisi sama rata "
            "dan bukan hasil optimasi; periksa data sebelum memakai hasil ini."
        )

    warna_saham = {saham: PALETTE_PIE[i % len(PALETTE_PIE)] for i, saham in enumerate(saham_terpilih)}

    for kolom, (nama, bobot) in zip(st.columns(2), bobot_strategi.items()):
        with kolom:
            section_title(nama)
            terbesar = int(np.argmax(bobot))
            kartu_kiri, kartu_kanan = st.columns([3, 2])
            with kartu_kiri:
                metric_card(
                    "Jumlah saham berbobot &gt; 0",
                    str(int(np.sum(bobot > BOBOT_MINIMAL_TAMPIL))),
                    accent=WARNA_STRATEGI[nama],
                )
            with kartu_kanan:
                metric_card(
                    "Bobot terbesar",
                    format_angka(bobot[terbesar] * 100, 2, akhiran="%"),
                    accent=WARNA_STRATEGI[nama],
                    sublabel=saham_terpilih[terbesar],
                )

            fig_pie = _pie_bobot(saham_terpilih, bobot, warna_saham)
            if fig_pie is not None:
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("Tidak ada bobot signifikan untuk ditampilkan.")

    section_title("Perbandingan Bobot")

    kolom_max, kolom_min = f"{STRATEGI_MAX} (%)", f"{STRATEGI_MIN} (%)"
    df_compare = pd.DataFrame({
        "Kode Saham": saham_terpilih,
        kolom_max: bobot_strategi[STRATEGI_MAX] * 100,
        kolom_min: bobot_strategi[STRATEGI_MIN] * 100,
    }).sort_values([kolom_max, kolom_min], ascending=False)
    st.dataframe(
        format_tabel(df_compare, {kolom_max: 2, kolom_min: 2}),
        hide_index=True, width="stretch", height=tinggi_tabel(len(df_compare)),
    )
    st.caption("Bobot 0,00% berarti saham tidak masuk portofolio.")


def _ukuran_kinerja(metrik: dict) -> list[tuple[str, float, str, bool]]:
    """Daftar (nama ukuran, nilai, teks tampilan, lebih kecil lebih baik) untuk satu strategi."""
    sortino = metrik["sortino"]["sortino_harian"]
    semideviation = metrik["sortino"]["semideviation"] * 100
    cumulative_return = metrik["cumulative_return"] * 100
    var_rupiah = metrik["var"]["var_rupiah"]
    return [
        ("Sortino Ratio Harian", sortino, format_angka(sortino, 4), False),
        ("Semideviation Harian", semideviation, format_angka(semideviation, 2, akhiran="%"), True),
        ("Cumulative Return", cumulative_return, format_angka(cumulative_return, 2, akhiran="%"), False),
        ("VaR 95% (1 hari)", var_rupiah, format_angka(var_rupiah, 0, awalan="Rp"), True),
    ]


def _pemenang(ukuran_max: tuple, ukuran_min: tuple) -> str | None:
    """Nama strategi dengan nilai lebih baik pada satu ukuran, atau None bila sama setelah pembulatan tampilan."""
    _, nilai_max, teks_max, lebih_kecil_lebih_baik = ukuran_max
    _, nilai_min, teks_min, _ = ukuran_min
    if teks_max == teks_min:
        return None
    max_unggul = nilai_max < nilai_min if lebih_kecil_lebih_baik else nilai_max > nilai_min
    return STRATEGI_MAX if max_unggul else STRATEGI_MIN


def _tabel_perbandingan(metrik_max: dict, metrik_min: dict) -> Styler:
    """Tabel perbandingan kedua strategi; nilai yang lebih baik ditebalkan."""
    ukuran_max, ukuran_min = _ukuran_kinerja(metrik_max), _ukuran_kinerja(metrik_min)
    tabel = pd.DataFrame({
        "Ukuran": [u[0] for u in ukuran_max],
        STRATEGI_MAX: [u[2] for u in ukuran_max],
        STRATEGI_MIN: [u[2] for u in ukuran_min],
    })
    pemenang = [_pemenang(a, b) for a, b in zip(ukuran_max, ukuran_min)]

    def tebal(kolom: pd.Series) -> list[str]:
        return ["font-weight: bold" if p == kolom.name else "" for p in pemenang]

    return tabel.style.apply(tebal, subset=[STRATEGI_MAX, STRATEGI_MIN]).hide(axis="index")


def _klausa_ukuran(nama_ukuran: str, ukuran_max: tuple, ukuran_min: tuple, kata_unggul: str) -> str:
    """Satu klausa perbandingan untuk ukuran yang dimenangkan strategi berbeda atau bernilai sama."""
    pemenang = _pemenang(ukuran_max, ukuran_min)
    if pemenang is None:
        return f"{nama_ukuran} kedua portofolio sama, yaitu {ukuran_max[2]}"
    unggul, kalah = (ukuran_max, ukuran_min) if pemenang == STRATEGI_MAX else (ukuran_min, ukuran_max)
    return f"{nama_ukuran} portofolio {pemenang} lebih {kata_unggul} ({unggul[2]} dibandingkan {kalah[2]})"


def _narasi_hasil(metrik_max: dict, metrik_min: dict) -> str:
    """Narasi otomatis 2 sampai 3 kalimat yang hanya menyebut angka hasil analisis."""
    config = st.session_state.config
    periode = f"{format_tanggal_panjang(config['start_date'])} sampai {format_tanggal_panjang(config['end_date'])}"
    teks_dana = format_angka(config["dana_awal"], 0, awalan="Rp")
    ukuran_max, ukuran_min = _ukuran_kinerja(metrik_max), _ukuran_kinerja(metrik_min)
    sortino_max, semidev_max, cr_max, var_max = ukuran_max
    sortino_min, semidev_min, cr_min, var_min = ukuran_min
    pemenang = [_pemenang(a, b) for a, b in zip(ukuran_max, ukuran_min)]

    def nilai(strategi: str, ukuran_a: tuple, ukuran_b: tuple) -> str:
        return ukuran_a[2] if strategi == STRATEGI_MAX else ukuran_b[2]

    kalimat = []
    if pemenang[0] is not None and len(set(pemenang)) == 1:
        unggul = pemenang[0]
        kalah = STRATEGI_MIN if unggul == STRATEGI_MAX else STRATEGI_MAX
        kalimat.append(
            f"Pada periode {periode}, portofolio {unggul} unggul pada semua ukuran dibandingkan "
            f"portofolio {kalah}: Sortino Ratio harian {nilai(unggul, sortino_max, sortino_min)} dibandingkan "
            f"{nilai(kalah, sortino_max, sortino_min)}, Cumulative Return {nilai(unggul, cr_max, cr_min)} "
            f"dibandingkan {nilai(kalah, cr_max, cr_min)}, Semideviation harian "
            f"{nilai(unggul, semidev_max, semidev_min)} dibandingkan {nilai(kalah, semidev_max, semidev_min)}, "
            f"dan VaR 95% satu hari {nilai(unggul, var_max, var_min)} dibandingkan "
            f"{nilai(kalah, var_max, var_min)} untuk dana {teks_dana}."
        )
    else:
        pemenang_return = {pemenang[0], pemenang[2]}
        if len(pemenang_return) == 1 and None not in pemenang_return:
            unggul = pemenang[0]
            kalah = STRATEGI_MIN if unggul == STRATEGI_MAX else STRATEGI_MAX
            kalimat.append(
                f"Pada periode {periode}, portofolio {unggul} menghasilkan Sortino Ratio harian "
                f"{nilai(unggul, sortino_max, sortino_min)} dan Cumulative Return {nilai(unggul, cr_max, cr_min)}, "
                f"lebih tinggi daripada portofolio {kalah} ({nilai(kalah, sortino_max, sortino_min)} dan "
                f"{nilai(kalah, cr_max, cr_min)})."
            )
        else:
            kalimat.append(
                f"Pada periode {periode}, "
                f"{_klausa_ukuran('Sortino Ratio harian', sortino_max, sortino_min, 'tinggi')}, sedangkan "
                f"{_klausa_ukuran('Cumulative Return', cr_max, cr_min, 'tinggi')}."
            )

        pemenang_risiko = {pemenang[1], pemenang[3]}
        if len(pemenang_risiko) == 1 and None not in pemenang_risiko:
            unggul = pemenang[1]
            kalah = STRATEGI_MIN if unggul == STRATEGI_MAX else STRATEGI_MAX
            kalimat.append(
                f"Portofolio {unggul} memiliki risiko penurunan yang lebih kecil, dengan Semideviation harian "
                f"{nilai(unggul, semidev_max, semidev_min)} dibandingkan {nilai(kalah, semidev_max, semidev_min)} "
                f"dan VaR 95% satu hari {nilai(unggul, var_max, var_min)} dibandingkan "
                f"{nilai(kalah, var_max, var_min)} untuk dana {teks_dana}."
            )
        else:
            kalimat.append(
                f"Untuk ukuran risiko, "
                f"{_klausa_ukuran('Semideviation harian', semidev_max, semidev_min, 'kecil')}, sedangkan "
                f"{_klausa_ukuran('VaR 95% satu hari', var_max, var_min, 'kecil')} untuk dana {teks_dana}."
            )

    cr_benchmark = hitung_cumulative_return(st.session_state.benchmark_aligned) * 100
    kalimat.append(
        f"Pada periode yang sama, Cumulative Return {nama_benchmark(config['benchmark'])} sebesar "
        f"{format_angka(cr_benchmark, 2, akhiran='%')}."
    )
    return " ".join(kalimat)


def _histogram_return(nama: str, metrik: dict) -> go.Figure:
    """Histogram return harian satu strategi dengan garis VaR dan garis mean."""
    return_plot = metrik["return_portofolio"]
    var_plot = metrik["var"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=return_plot * 100,
        nbinsx=40,
        marker=dict(color=WARNA_STRATEGI[nama], opacity=0.75),
        name="Return Harian",
        hovertemplate="Return Harian (%): %{x}<br>Frekuensi: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(
        x=var_plot["q_alpha"] * 100,
        line=dict(color=WARNA_VAR, width=2, dash="dash"),
        annotation_text=f"Return ke-{var_plot['posisi_k']} = {format_angka(var_plot['q_alpha'] * 100, 2, akhiran='%')}",
        annotation_position="top left",
        annotation_font=dict(size=11, color=WARNA_VAR),
    )
    fig_hist.add_vline(
        x=np.mean(return_plot) * 100,
        line=dict(color=WARNA_BENCHMARK, width=1.5, dash="dot"),
        annotation_text=f"Mean = {format_angka(np.mean(return_plot) * 100, 2, akhiran='%')}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=WARNA_BENCHMARK),
    )
    fig_hist.update_layout(
        template="plotly_white",
        separators=PEMISAH_PLOTLY,
        height=380,
        xaxis_title="Return Harian (%)",
        yaxis_title="Frekuensi",
        showlegend=False,
    )
    return fig_hist


def render_hasil() -> None:
    """Halaman Hasil & Perbandingan: tabel perbandingan, narasi, Cumulative Return, dan distribusi return."""
    hairline_header("Hasil & Perbandingan")
    require_analisis()

    metrik = st.session_state.metrik
    benchmark = st.session_state.benchmark_aligned
    metrik_max, metrik_min = metrik[STRATEGI_MAX], metrik[STRATEGI_MIN]

    info_box(
        "Kedua strategi dioptimasi menggunakan matriks risiko yang sama, yaitu "
        "matriks Semivariance-Semicovariance dari data yang sama. Perbedaan hanya "
        "terletak pada fungsi tujuan masing-masing.",
        kind="info",
    )

    for nama in WARNA_STRATEGI:
        if metrik[nama]["var"]["pesan"]:
            info_box(f"{nama}: {metrik[nama]['var']['pesan']}", kind="warning")

    section_title("Tabel Perbandingan")
    st.table(_tabel_perbandingan(metrik_max, metrik_min))

    section_title("Narasi Hasil")

    st.markdown(
        f"""
        <div style="border: {BORDER_HAIRLINE};
                    border-radius: 8px; padding: 20px 24px;
                    font-size: 0.88rem; line-height: 1.75;">
            {_narasi_hasil(metrik_max, metrik_min)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Cumulative Return")

    idx = st.session_state.log_return.index
    tanggal_teks = [format_tanggal(t) for t in idx]
    seri = {nama: metrik[nama]["return_portofolio"] for nama in WARNA_STRATEGI}
    seri["Benchmark"] = benchmark

    fig = go.Figure()
    for nama, nilai in seri.items():
        kumulatif = np.exp(pd.Series(nilai, index=idx).cumsum()) - 1
        fig.add_trace(go.Scatter(
            x=kumulatif.index,
            y=kumulatif * 100,
            mode="lines",
            name=nama,
            line=dict(
                color=WARNA_STRATEGI.get(nama, "#475569"),
                width=2 if nama in WARNA_STRATEGI else 1.5,
                dash=None if nama in WARNA_STRATEGI else "dash",
            ),
            customdata=tanggal_teks,
            hovertemplate="<b>%{fullData.name}</b><br>Tanggal: %{customdata}<br>Cumulative Return: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        separators=PEMISAH_PLOTLY,
        height=420,
        xaxis_title="",
        yaxis_title="Cumulative Return (%)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0),
    )
    atur_sumbu_tanggal(fig, idx)
    st.plotly_chart(fig, width="stretch")

    section_title("Distribusi Return & VaR 95%")

    for kolom, nama in zip(st.columns(2), WARNA_STRATEGI):
        with kolom:
            st.markdown(f"**{nama}**")
            st.plotly_chart(_histogram_return(nama, metrik[nama]), width="stretch")

    var_contoh = metrik_max["var"]
    st.caption(
        f"VaR dihitung untuk dana awal {format_angka(st.session_state.config['dana_awal'], 0, awalan='Rp')} "
        f"dengan k = {var_contoh['posisi_k']} dari {var_contoh['n_obs']} return."
    )


def angka_latex(nilai: float, desimal: int) -> str:
    """Angka format Indonesia untuk st.latex (koma desimal ditulis {,})."""
    return format_angka(nilai, desimal).replace(",", "{,}")


def _latex_matriks(nilai: np.ndarray, desimal: int, skala: int | None = None) -> str:
    """Matriks atau vektor kolom dalam bmatrix LaTeX dengan faktor skala 10^skala di depan."""
    nilai = np.atleast_2d(np.asarray(nilai, dtype=float))
    if nilai.shape[0] == 1:
        nilai = nilai.T
    pembagi = 10.0 ** skala if skala is not None else 1.0
    baris = [" & ".join(angka_latex(v / pembagi, desimal) for v in b) for b in nilai]
    awalan = rf"10^{{{skala}}} \times " if skala is not None else ""
    return awalan + r"\begin{bmatrix}" + r" \\ ".join(baris) + r"\end{bmatrix}"


def _tampilkan_matriks(simbol: str, nilai: np.ndarray, label: list[str], desimal: int, skala: int | None = None) -> None:
    """Matriks atau vektor sebagai bmatrix; sebagai tabel bila jumlah saham lebih dari MAKS_SAHAM_BMATRIX."""
    if len(label) > MAKS_SAHAM_BMATRIX:
        nilai_2d = np.atleast_2d(np.asarray(nilai, dtype=float))
        if nilai_2d.shape[0] == 1:
            tabel = pd.DataFrame({"Kode Saham": label, "Nilai": nilai_2d[0]})
        else:
            tabel = pd.DataFrame(nilai_2d, columns=label).rename_axis("Kode Saham").reset_index()
            tabel["Kode Saham"] = label
        kolom_angka = [k for k in tabel.columns if k != "Kode Saham"]
        st.dataframe(
            format_tabel(tabel, {k: desimal + (-skala if skala else 0) for k in kolom_angka}),
            hide_index=True, width="stretch", height=tinggi_tabel(len(tabel)),
        )
        return
    st.latex(rf"{simbol} = {_latex_matriks(nilai, desimal, skala)}")
    st.caption(f"Urutan baris: {', '.join(label)}.")


def _tab_return() -> None:
    """Tab Return: rumus log return, contoh substitusi, dan lima baris pertama log return."""
    st.latex(r"R_{i,t}=\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)")

    data_saham = st.session_state.data_saham
    log_return = st.session_state.log_return
    benchmark = st.session_state.config["benchmark"]
    saham_contoh = st.session_state.saham_terpilih[0]
    harga_lalu, harga_kini = data_saham[saham_contoh].iloc[0], data_saham[saham_contoh].iloc[1]

    st.markdown(
        f"Contoh {saham_contoh}, {format_tanggal_panjang(str(data_saham.index[1].date()))} "
        f"(harga hari sebelumnya {format_tanggal_panjang(str(data_saham.index[0].date()))}):"
    )
    st.latex(
        rf"R_{{t}}=\ln\left(\frac{{{angka_latex(harga_kini, 4)}}}{{{angka_latex(harga_lalu, 4)}}}\right)"
        rf"={angka_latex(log_return[saham_contoh].iloc[0], 6)}"
    )

    lima_baris = log_return[st.session_state.saham_terpilih].head(5) * 100
    lima_baris[label_benchmark(benchmark)] = st.session_state.log_return_benchmark.head(5).values * 100
    lima_baris.index = [format_tanggal(t) for t in lima_baris.index]
    tabel = lima_baris.rename_axis("Tanggal").reset_index()
    st.markdown("**Log return (%), lima hari pertama**")
    st.dataframe(
        format_tabel(tabel, {k: 4 for k in tabel.columns if k != "Tanggal"}),
        hide_index=True, width="stretch",
    )

    semua = log_return.copy()
    semua[benchmark] = st.session_state.log_return_benchmark.loc[log_return.index]
    semua.index = semua.index.strftime("%Y-%m-%d")
    tombol_unduh_csv(semua.rename_axis("Tanggal").reset_index(), "Unduh log return (CSV)", "log_return.csv")


def _tab_seleksi() -> None:
    """Tab Seleksi: rumus Spearman dan matriks korelasi saham yang lolos tahap 1."""
    st.latex(
        r"\rho_{ij}=\frac{\sum_{t=1}^{T}\left(X_t-\bar{X}\right)\left(Y_t-\bar{Y}\right)}"
        r"{\sqrt{\sum_{t=1}^{T}\left(X_t-\bar{X}\right)^2\sum_{t=1}^{T}\left(Y_t-\bar{Y}\right)^2}}"
    )
    st.caption(
        "X_t dan Y_t adalah peringkat return saham ke-i dan ke-j pada waktu ke-t; "
        "nilai yang kembar diberi peringkat rata-rata."
    )

    spearman_matrix = st.session_state.hasil_seleksi["spearman_matrix"]
    if spearman_matrix is None:
        empty_state("Matriks Spearman tidak dihitung karena kurang dari 2 saham lolos tahap 1.")
        return

    label = spearman_matrix.columns.tolist()
    _tampilkan_matriks(r"\boldsymbol{\rho}", spearman_matrix.values, label, 2)
    tombol_unduh_csv(
        spearman_matrix.rename_axis("Kode Saham").reset_index(),
        "Unduh matriks Spearman (CSV)", "matriks_spearman.csv",
    )


def _tab_matriks_risiko() -> None:
    """Tab Matriks Risiko: mu, B-bar, matriks Semivariance-Semicovariance, dan nilai eigen."""
    st.latex(r"E(R_i)=\frac{\sum_{t=1}^{T}R_{i,t}}{T} \qquad \bar{B}=\frac{\sum_{t=1}^{T}B_t}{T}")
    st.latex(r"S_i^2=\frac{\sum_{t=1}^{T}\left[\min\left(R_{i,t}-B_t,0\right)\right]^2}{T}")
    st.latex(r"S_{ij}=\frac{\sum_{t=1}^{T}\min\left(R_{i,t}-B_t,0\right)\min\left(R_{j,t}-B_t,0\right)}{T}")

    saham_terpilih = st.session_state.saham_terpilih
    mu = st.session_state.mu_terpilih
    matriks_sv = st.session_state.matriks_sv
    rata_benchmark = float(np.mean(st.session_state.benchmark_aligned))
    nilai_eigen = np.linalg.eigvalsh(matriks_sv)

    _tampilkan_matriks(r"\boldsymbol{\mu}", mu, saham_terpilih, 3, -2)
    st.latex(rf"\bar{{B}}={angka_latex(rata_benchmark, 6)}")
    _tampilkan_matriks(r"\boldsymbol{\Sigma}_{sv}", matriks_sv, saham_terpilih, 3, -4)
    st.latex(rf"\boldsymbol{{\lambda}} = {_latex_matriks(nilai_eigen, 4, -4)}")
    st.caption("Nilai eigen diurutkan dari terkecil.")
    if np.all(nilai_eigen > 0):
        st.markdown("Semua nilai eigen positif, sehingga matriks definit positif.")
    else:
        st.markdown("Tidak semua nilai eigen positif, sehingga matriks tidak definit positif.")

    tabel = pd.DataFrame(matriks_sv, columns=saham_terpilih)
    tabel.insert(0, "Kode Saham", saham_terpilih)
    tabel.insert(1, "mu", mu)
    tabel.insert(2, "B_bar", rata_benchmark)
    tabel["Nilai Eigen (urut naik)"] = nilai_eigen
    tombol_unduh_csv(tabel, "Unduh mu, B̄, matriks, dan nilai eigen (CSV)", "matriks_risiko.csv")


def _tab_optimasi() -> None:
    """Tab Optimasi: fungsi tujuan dan kendala kedua strategi beserta vektor bobotnya."""
    st.latex(
        r"\begin{aligned}\max_{\mathbf{w}}\quad & \frac{\mathbf{w}^\top\boldsymbol{\mu}-\bar{B}}"
        r"{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}}}\\"
        r"\text{terhadap}\quad & \sum_{i=1}^{I}w_i=1,\quad 0\le w_i\le1\end{aligned}"
    )
    st.latex(
        r"\begin{aligned}\min_{\mathbf{w}}\quad & \mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}\\"
        r"\text{terhadap}\quad & \sum_{i=1}^{I}w_i=1,\quad 0\le w_i\le1\end{aligned}"
    )

    saham_terpilih = st.session_state.saham_terpilih
    bobot_max = st.session_state.bobot_max_sortino
    bobot_min = st.session_state.bobot_min_sv

    kolom_max, kolom_min = st.columns(2)
    with kolom_max:
        st.markdown(f"**{STRATEGI_MAX}**")
        _tampilkan_matriks(r"\mathbf{w}", bobot_max, saham_terpilih, 4)
    with kolom_min:
        st.markdown(f"**{STRATEGI_MIN}**")
        _tampilkan_matriks(r"\mathbf{w}", bobot_min, saham_terpilih, 4)

    st.markdown(
        f"Kedua masalah diselesaikan dengan SLSQP dari {JUMLAH_TITIK_AWAL} titik awal acak "
        f"(benih 0 sampai {JUMLAH_TITIK_AWAL - 1}), maxiter 1000, dan ftol 10⁻⁹. "
        "Bobot yang dipakai berasal dari titik awal dengan nilai fungsi tujuan terbaik."
    )

    tabel = pd.DataFrame({"Kode Saham": saham_terpilih, STRATEGI_MAX: bobot_max, STRATEGI_MIN: bobot_min})
    tombol_unduh_csv(tabel, "Unduh vektor bobot (CSV)", "bobot_optimal.csv")


def _substitusi_evaluasi(nama: str, metrik: dict, bobot: np.ndarray, k: int) -> None:
    """Substitusi Sortino Ratio, Cumulative Return, dan VaR satu strategi serta tabel k return terkecil."""
    sortino = metrik["sortino"]
    var = metrik["var"]
    return_p = metrik["return_portofolio"]
    rata_benchmark = float(np.mean(st.session_state.benchmark_aligned))
    expected_return_p = float(bobot @ st.session_state.mu_terpilih)
    total_log_return = float(np.sum(return_p))
    teks_dana = format_angka(st.session_state.config["dana_awal"], 0).replace(",", "{,}")

    section_title(nama)
    st.latex(
        rf"Sortino_P=\frac{{{angka_latex(expected_return_p, 6)}-({angka_latex(rata_benchmark, 6)})}}"
        rf"{{\sqrt{{{angka_latex(sortino['semideviation'] ** 2, 8)}}}}}"
        rf"=\frac{{{angka_latex(sortino['excess_return'], 6)}}}{{{angka_latex(sortino['semideviation'], 6)}}}"
        rf"={angka_latex(sortino['sortino_harian'], 4)}"
    )
    st.latex(
        rf"R_{{cum}}=\exp\left({angka_latex(total_log_return, 6)}\right)-1"
        rf"={angka_latex(metrik['cumulative_return'], 4)}"
    )

    urutan = np.argsort(return_p, kind="stable")[:k]
    tanggal = st.session_state.log_return.index
    tabel = pd.DataFrame({
        "Urutan": np.arange(1, k + 1),
        "Tanggal": [format_tanggal(tanggal[i]) for i in urutan],
        "Return (%)": [format_angka(return_p[i] * 100, 4) for i in urutan],
    })
    st.table(
        tabel.style.apply(
            lambda baris: ["font-weight: bold" if baris["Urutan"] == k else "" for _ in baris], axis=1
        ).hide(axis="index")
    )

    st.latex(
        rf"VaR={angka_latex(-var['q_alpha'], 8)}\times{teks_dana}\times\sqrt{{{var['holding_period']}}}"
        rf"=\text{{Rp}}{format_angka(var['var_rupiah'], 0).replace(',', '{,}')}"
    )


def _tab_evaluasi() -> None:
    """Tab Evaluasi: rumus Sortino Ratio, Cumulative Return, dan VaR dengan substitusi per strategi."""
    st.latex(
        r"Sortino_P=\frac{\mathbf{w}^\top\boldsymbol{\mu}-\bar{B}}{\sqrt{\mathbf{w}^\top\boldsymbol{\Sigma}_{sv}\mathbf{w}}}"
        r"\qquad R_{cum}=\exp\left(\sum_{t=1}^{T}R_{P,t}\right)-1"
    )
    st.latex(r"k=\lceil\alpha T\rceil \qquad VaR=V_0\times\left(-R_{P,(k)}\right)\times\sqrt{t}")

    metrik = st.session_state.metrik
    var = metrik[STRATEGI_MAX]["var"]
    k = var["posisi_k"]
    st.latex(rf"k=\lceil {angka_latex(ALPHA_VAR, 2)}\times{var['n_obs']}\rceil={k}")

    bobot_strategi = {STRATEGI_MAX: st.session_state.bobot_max_sortino, STRATEGI_MIN: st.session_state.bobot_min_sv}
    for nama in WARNA_STRATEGI:
        _substitusi_evaluasi(nama, metrik[nama], bobot_strategi[nama], k)

    tanggal = st.session_state.log_return.index.strftime("%Y-%m-%d")
    tabel = pd.DataFrame({"Tanggal": tanggal} | {nama: metrik[nama]["return_portofolio"] for nama in WARNA_STRATEGI})
    tombol_unduh_csv(tabel, "Unduh return portofolio harian (CSV)", "return_portofolio.csv")


def render_detail() -> None:
    """Halaman Detail Perhitungan: rumus dan angka antara setiap tahap untuk dicocokkan secara manual."""
    hairline_header("Detail Perhitungan")
    require_analisis()

    tabs = st.tabs(["Return", "Seleksi", "Matriks Risiko", "Optimasi", "Evaluasi"])
    for tab, isi in zip(tabs, (_tab_return, _tab_seleksi, _tab_matriks_risiko, _tab_optimasi, _tab_evaluasi)):
        with tab:
            isi()


def main() -> None:
    """Inisialisasi state, tampilkan sidebar, lalu buka halaman yang dipilih."""
    init_state()
    page, wadah_status = render_sidebar()

    if page == "Beranda":
        render_home()
    elif page == "Data":
        render_data(wadah_status)
    elif page == "Seleksi Saham":
        render_seleksi()
    elif page == "Optimasi Portofolio":
        render_optimasi()
    elif page == "Hasil & Perbandingan":
        render_hasil()
    elif page == "Detail Perhitungan":
        render_detail()


if __name__ == "__main__":
    main()
