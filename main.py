"""
Portofolio Optimizer — aplikasi Streamlit untuk optimasi portofolio.

Aplikasi untuk analisis dan optimasi portofolio saham dengan pendekatan
Mean-Semivariance. Menyediakan dua strategi optimasi berbasis SLSQP:
memaksimalkan Sortino Ratio dan meminimalkan Semivariance.

Bagian dari skripsi S1 Statistika Universitas Diponegoro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from functions import (
    DAFTAR_SAHAM_DEFAULT,
    BENCHMARK_DEFAULT,
    RISK_FREE_TAHUNAN_DEFAULT,
    START_DATE_DEFAULT,
    END_DATE_DEFAULT,
    THRESHOLD_KORELASI_DEFAULT,
    MAX_SAHAM_DEFAULT,
    DANA_AWAL_VAR_DEFAULT,
    HARI_BURSA_SETAHUN,
    MIN_OBSERVASI,
    load_data_saham,
    load_data_benchmark,
    hitung_log_return,
    hitung_expected_return,
    seleksi_saham_dua_tahap,
    hitung_matriks_semivariance,
    optimasi_slsqp_max_sortino,
    optimasi_slsqp_min_semivariance,
    hitung_sortino,
    hitung_cumulative_return,
    hitung_var_historical,
    bandingkan_strategi,
    deteksi_ketidakcocokan_benchmark,
)


st.set_page_config(
    page_title="Portofolio Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# Palet warna garis untuk Plotly. Sengaja pakai hex karena plotly_white
# punya tema sendiri yang tidak mengikuti dark mode Streamlit.
PALETTE_LINE = [
    "#1E3A8A", "#D97706", "#475569", "#047857", "#B91C1C",
    "#0369A1", "#7C3AED", "#DB2777", "#0891B2", "#65A30D",
]

PALETTE_PIE = [
    "#1E3A8A", "#D97706", "#047857", "#B91C1C", "#0369A1",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#475569",
    "#0F1E54", "#92400E",
]

# Bobot di bawah ambang ini dianggap nol. Mencegah pie chart menampilkan
# notasi ilmiah (misal 2.49e-15%) untuk saham dengan bobot numerik nol.
BOBOT_MINIMAL_TAMPIL = 1e-6

# Border tipis untuk hairline dan card. Memakai rgba mid-gray agar tetap
# terlihat di light mode (abu di atas putih) maupun dark mode (abu di atas hitam).
BORDER_HAIRLINE = "1px solid rgba(128, 128, 128, 0.2)"
BORDER_DASHED = "1px dashed rgba(128, 128, 128, 0.25)"


def init_state():
    """
    Inisialisasi semua key session_state dengan nilai default.
    Dipanggil sekali di awal main() agar konsisten di seluruh halaman.
    """
    defaults = {
        "data_saham": None,
        "data_benchmark": None,
        "log_return": None,
        "log_return_benchmark": None,
        "expected_return": None,
        "hasil_seleksi": None,
        "saham_terpilih": None,
        "returns_terpilih": None,
        "benchmark_aligned": None,
        "mu_terpilih": None,
        "matriks_sv": None,
        "bobot_max_sortino": None,
        "bobot_min_sv": None,
        "sortino_max": None,
        "perbandingan": None,
        "config": {
            "tickers": DAFTAR_SAHAM_DEFAULT,
            "benchmark": BENCHMARK_DEFAULT,
            "risk_free_pct": RISK_FREE_TAHUNAN_DEFAULT * 100,
            "start_date": START_DATE_DEFAULT,
            "end_date": END_DATE_DEFAULT,
            "threshold": THRESHOLD_KORELASI_DEFAULT,
            "max_saham": MAX_SAHAM_DEFAULT,
            "dana_awal": DANA_AWAL_VAR_DEFAULT,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def require_data():
    """Guard: pastikan data sudah dimuat sebelum lanjut render halaman."""
    if st.session_state.data_saham is None:
        st.info("Silakan muat data terlebih dahulu di halaman **Input Data**.")
        st.stop()


def require_selection():
    """Guard: pastikan seleksi saham sudah dijalankan."""
    require_data()
    if not st.session_state.saham_terpilih:
        st.info(
            "Seleksi saham belum dijalankan atau tidak menghasilkan saham. "
            "Silakan periksa halaman **Input Data**."
        )
        st.stop()


def require_optimization():
    """Guard: pastikan optimasi sudah dijalankan sebelum halaman evaluasi."""
    require_selection()
    if st.session_state.bobot_max_sortino is None:
        st.info(
            "Optimasi belum dijalankan. Silakan buka halaman **Optimasi Portofolio** "
            "dan klik **Jalankan Optimasi**."
        )
        st.stop()


# Helper UI

def hairline_header(title, metadata=""):
    """
    Header halaman dengan garis horizontal di atas dan bawah.
    Warna teks inherit dari theme Streamlit; border pakai rgba mid-gray
    agar tetap terlihat di light maupun dark mode.
    """
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
                {metadata}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
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


def metric_card(label, value, accent="#1E3A8A", sublabel=None):
    """
    Kartu metrik untuk menampilkan satu nilai KPI.

    Memakai custom HTML agar tampilan identik dengan versi lama (border
    tipis dengan accent bar di kiri). Border pakai rgba mid-gray agar
    tetap terlihat di dark mode. Warna aksen tetap hex karena distintif
    di kedua tema.
    """
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


def info_box(text, kind="info"):
    """
    Kotak informasi dengan border kiri berwarna sesuai jenisnya.

    Background memakai rgba semi-transparan agar blend dengan warna
    background tema. Di light mode menghasilkan tint lembut di atas
    putih; di dark mode menghasilkan tint gelap di atas hitam.
    """
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


def empty_state(message):
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


def _pie_bobot(saham_list, bobot, warna_series):
    """
    Bangun pie chart bobot portofolio.

    Bobot di bawah BOBOT_MINIMAL_TAMPIL difilter agar tidak muncul sebagai
    notasi ilmiah (misal 2.49e-15%) di label. Saham tersebut tetap tampil
    di tabel bobot dengan bobot 0.
    """
    mask = bobot > BOBOT_MINIMAL_TAMPIL
    labels = [s for s, m in zip(saham_list, mask) if m]
    values = bobot[mask] * 100
    colors = [warna_series[i % len(warna_series)] for i in range(len(labels))]

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
        height=380,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# Sidebar

def render_sidebar():
    """
    Sidebar dengan navigasi dan status data.
    Icons memakai Material Symbols native Streamlit (prefix :material/nama:).
    Return nama halaman bersih untuk routing.
    """
    with st.sidebar:
        # Judul aplikasi
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

        # Navigasi utama dengan icon Material Symbols di setiap opsi
        page_raw = st.radio(
            "Navigasi",
            options=[
                ":material/home: Home",
                ":material/upload: Input Data",
                ":material/analytics: Eksplorasi Data",
                ":material/filter_alt: Seleksi Saham",
                ":material/tune: Optimasi Portofolio",
                ":material/assessment: Evaluasi Kinerja",
                ":material/compare_arrows: Perbandingan Strategi",
            ],
            label_visibility="collapsed",
        )

        # Strip prefix icon (:material/xxx: Nama) untuk dapat nama halaman
        page = page_raw.split(" ", 1)[-1] if " " in page_raw else page_raw

        st.divider()

        # Status data yang sudah dimuat
        st.markdown(
            """
            <div style="font-size: 0.65rem; font-weight: 600;
                        letter-spacing: 0.1em; text-transform: uppercase;
                        opacity: 0.6; margin-bottom: 10px;">
                Status Data
            </div>
            """,
            unsafe_allow_html=True,
        )

        threshold = st.session_state.config["threshold"]

        if st.session_state.data_saham is not None:
            n_saham = st.session_state.data_saham.shape[1]
            n_hari = st.session_state.data_saham.shape[0]
            bench = st.session_state.config["benchmark"]
            rf = st.session_state.config["risk_free_pct"]
            n_terpilih = len(st.session_state.saham_terpilih or [])

            st.markdown(
                f"""
                <div style="border: {BORDER_HAIRLINE};
                            border-radius: 6px; padding: 12px 14px;
                            font-family: monospace;
                            font-size: 0.72rem; line-height: 1.7;">
                    <div style="font-weight: 500;">
                        {n_saham} saham · {n_hari} hari
                    </div>
                    <div style="opacity: 0.8;">Benchmark: {bench}</div>
                    <div style="opacity: 0.8;">Risk-free: {rf:.2f}% p.a.</div>
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
                    Belum ada data dimuat
                </div>
                """,
                unsafe_allow_html=True,
            )

    return page


def _sidebar_metadata():
    """Metadata singkat yang muncul di kanan atas setiap halaman."""
    if st.session_state.data_saham is None:
        return "Belum ada data"

    n_saham = st.session_state.data_saham.shape[1]
    n_hari = st.session_state.data_saham.shape[0]
    bench = st.session_state.config["benchmark"]

    return f"{n_saham} saham · {n_hari} hari · {bench}"


# Halaman Home

def render_home():
    """Halaman Home: penjelasan aplikasi, alur analisis, dan konfigurasi default."""
    hairline_header("Portofolio Optimizer", "By: Ridho Akbar Fadhilah")

    # Judul dan deskripsi singkat
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

        - **Max Sortino Ratio** — memaksimalkan risk-adjusted return dengan
          menekan risiko downside terhadap benchmark.
        - **Min Semivariance** — meminimalkan risiko downside tanpa memperhatikan
          expected return secara langsung.

        Kedua strategi menggunakan matriks risiko yang sama, sehingga perbandingan
        kinerja bersifat langsung dan setara.
        """
    )

    section_title("Alur Analisis")

    # Enam kartu langkah analisis
    steps = [
        ("01", "Input Data", "Unduh harga historis dari Yahoo Finance dan hitung log return harian."),
        ("02", "Eksplorasi Data", "Statistik deskriptif, distribusi return, dan matriks korelasi antar aset."),
        ("03", "Seleksi Saham", "Filter E(R) > 0, lalu seleksi berdasarkan Spearman threshold."),
        ("04", "Optimasi", "SLSQP max Sortino Ratio dan SLSQP min Semivariance."),
        ("05", "Evaluasi", "Sortino harian & tahunan, Cumulative Return, VaR 95% Historical."),
        ("06", "Perbandingan", "Uji empiris: apakah max Sortino unggul dari min Semivariance?"),
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

    # Tabel konfigurasi default
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
                <td style="padding: 10px 0; opacity: 0.7;">Risk-free</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    BI Rate {RISK_FREE_TAHUNAN_DEFAULT * 100:.2f}% p.a.
                </td>
            </tr>
            <tr style="border-bottom: {BORDER_HAIRLINE};">
                <td style="padding: 10px 0; opacity: 0.7;">Threshold Spearman</td>
                <td style="padding: 10px 0; font-family: monospace;">
                    {THRESHOLD_KORELASI_DEFAULT}
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


# Halaman Input Data

def _parse_tickers(text):
    """
    Parse input ticker dari textarea.
    Menerima pemisah enter atau koma; return list ticker unik uppercase.
    """
    tickers = []
    for line in text.replace(",", "\n").split("\n"):
        t = line.strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def _jalankan_pipeline(tickers, benchmark, risk_free_pct, start_date, end_date,
                       threshold, max_saham, dana_awal):
    """
    Pipeline utama saat user klik "Muat Data & Jalankan Seleksi":
    1. Unduh data saham dan benchmark dari Yahoo Finance
    2. Selaraskan tanggal (intersection)
    3. Hitung log return
    4. Jalankan seleksi dua tahap
    5. Simpan hasil ke session_state
    """
    if len(tickers) < 2:
        st.error("Minimal 2 ticker diperlukan untuk membentuk portofolio.")
        return

    try:
        with st.spinner(f"Mengunduh data {len(tickers)} ticker dari Yahoo Finance..."):
            data_saham = load_data_saham(tickers, start_date, end_date)
            data_benchmark = load_data_benchmark(benchmark, start_date, end_date)
    except (ConnectionError, ValueError) as e:
        st.error(str(e))
        return

    # Hanya ambil tanggal yang ada di kedua dataset
    idx_bersama = data_saham.index.intersection(data_benchmark.index)
    data_saham = data_saham.loc[idx_bersama]
    data_benchmark = data_benchmark.loc[idx_bersama]

    if len(idx_bersama) < MIN_OBSERVASI:
        st.error(
            f"Hanya {len(idx_bersama)} hari bursa yang beririsan antara saham dan benchmark. "
            f"Minimal {MIN_OBSERVASI} hari diperlukan."
        )
        return

    # Perhitungan return dan seleksi
    log_return = hitung_log_return(data_saham)
    log_return_benchmark = hitung_log_return(data_benchmark)
    expected_return = hitung_expected_return(log_return)

    hasil_seleksi = seleksi_saham_dua_tahap(
        log_return, expected_return,
        threshold=threshold, max_saham=max_saham,
    )

    saham_terpilih = hasil_seleksi["saham_terpilih"]

    # Peringatan jika seleksi menghasilkan sedikit saham
    if len(saham_terpilih) < 3:
        st.warning(
            f"Hanya {len(saham_terpilih)} saham lolos seleksi. "
            f"Portofolio terlalu terkonsentrasi. "
            f"Naikkan threshold atau perpanjang periode."
        )
        if hasil_seleksi.get("pesan"):
            st.info(hasil_seleksi["pesan"])

    # Simpan hasil ke session_state
    st.session_state.data_saham = data_saham
    st.session_state.data_benchmark = data_benchmark
    st.session_state.log_return = log_return
    st.session_state.log_return_benchmark = log_return_benchmark
    st.session_state.expected_return = expected_return
    st.session_state.hasil_seleksi = hasil_seleksi
    st.session_state.saham_terpilih = saham_terpilih

    # Update config agar konsisten di sidebar
    st.session_state.config.update({
        "tickers": tickers,
        "benchmark": benchmark,
        "risk_free_pct": risk_free_pct,
        "start_date": start_date,
        "end_date": end_date,
        "threshold": threshold,
        "max_saham": max_saham,
        "dana_awal": dana_awal,
    })

    # Reset hasil hilir karena data berubah
    for key in ["matriks_sv", "returns_terpilih", "benchmark_aligned",
                "mu_terpilih", "bobot_max_sortino", "bobot_min_sv",
                "sortino_max", "perbandingan"]:
        st.session_state[key] = None

    info_box(
        f"Data dimuat: <strong>{data_saham.shape[1]} saham</strong> · "
        f"<strong>{data_saham.shape[0]} hari bursa</strong>. "
        f"Seleksi menghasilkan <strong>{len(saham_terpilih)} saham</strong>.",
        kind="success",
    )


def render_input_data():
    """Halaman input: ticker, benchmark, periode, dan parameter seleksi."""
    hairline_header("Input Data", _sidebar_metadata())

    if st.session_state.data_saham is not None:
        info_box(
            f"Data sudah dimuat: <strong>{st.session_state.data_saham.shape[1]} saham</strong> · "
            f"<strong>{st.session_state.data_saham.shape[0]} hari bursa</strong>. "
            f"Ubah parameter dan klik Muat Ulang untuk memperbarui.",
            kind="info",
        )

    info_box(
        "Default disetel untuk BISNIS-27 dengan benchmark IHSG dan risk-free BI Rate. "
        "Untuk aset lain, sesuaikan benchmark dan risk-free di kolom sebelah.",
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
        risk_free_input = st.number_input(
            "Risk-Free Rate (% p.a.)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state.config["risk_free_pct"]),
            step=0.25,
            help="BI Rate untuk saham ID, Fed Rate untuk saham US, 0 untuk kripto",
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

    # Peringatan otomatis jika ticker dan benchmark tidak cocok
    tickers_parsed = _parse_tickers(tickers_input)
    warning = deteksi_ketidakcocokan_benchmark(tickers_parsed, benchmark_input)
    if warning:
        info_box(warning, kind="warning")

    st.markdown("")
    run = st.button(
        "Muat Data & Jalankan Seleksi",
        type="primary",
        width="stretch",
        icon=":material/download:",
    )

    if run:
        _jalankan_pipeline(
            tickers=tickers_parsed,
            benchmark=benchmark_input.strip().upper(),
            risk_free_pct=risk_free_input,
            start_date=str(start_input),
            end_date=str(end_input),
            threshold=threshold_input,
            max_saham=max_saham_input,
            dana_awal=dana_awal_input,
        )


# Halaman Eksplorasi Data

def render_eksplorasi():
    """
    Halaman eksplorasi: statistik deskriptif return, plot harga ternormalisasi
    Base 100, dan heatmap korelasi return.
    """
    require_data()
    hairline_header("Eksplorasi Data", _sidebar_metadata())

    log_return = st.session_state.log_return
    data_saham = st.session_state.data_saham

    section_title("Statistik Deskriptif Return Harian")

    # Statistik per saham; nilai dalam persen agar mudah dibaca
    stat_desc = pd.DataFrame({
        "Mean": log_return.mean(),
        "Std Dev": log_return.std(),
        "Min": log_return.min(),
        "Max": log_return.max(),
        "Skewness": log_return.skew(),
        "Kurtosis": log_return.kurtosis(),
    })
    stat_desc = stat_desc.reset_index()
    stat_desc.columns = ["Ticker"] + list(stat_desc.columns[1:])

    for col in ["Mean", "Std Dev", "Min", "Max"]:
        stat_desc[col] = stat_desc[col] * 100

    stat_desc = stat_desc.round(4)
    st.dataframe(stat_desc, hide_index=True, width="stretch")

    section_title("Pergerakan Harga Ternormalisasi (Base 100)")

    st.caption(
        "Base 100 memungkinkan perbandingan performa antar aset dengan skala harga berbeda. "
        "Semua aset mulai dari nilai 100 pada hari pertama periode. Nilai 150 berarti naik 50% "
        "dari hari pertama; nilai 80 berarti turun 20%."
    )

    # Normalisasi harga ke Base 100
    normalized = data_saham / data_saham.iloc[0] * 100

    fig = go.Figure()
    for i, ticker in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized[ticker],
            mode="lines",
            name=ticker,
            line=dict(width=1.5, color=PALETTE_LINE[i % len(PALETTE_LINE)]),
            hovertemplate="<b>%{fullData.name}</b><br>Tanggal: %{x}<br>Nilai: %{y:.2f}<extra></extra>",
        ))

    # hovermode="closest" agar hanya satu garis yang di-highlight saat hover.
    # Legend ditaruh di samping kanan agar tidak menutupi plot.
    fig.update_layout(
        template="plotly_white",
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
    st.plotly_chart(fig, width="stretch")

    section_title("Heatmap Korelasi Return")

    corr_matrix = log_return.corr()

    # Warna diverging: biru negatif, putih nol, merah positif
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[
            [0.0, "#1E3A8A"],
            [0.5, "#FFFFFF"],
            [1.0, "#B91C1C"],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="Korelasi(%{x}, %{y}) = %{z:.3f}<extra></extra>",
    ))

    fig_heat.update_layout(
        template="plotly_white",
        height=max(400, 20 * len(corr_matrix.columns)),
        xaxis=dict(tickfont=dict(size=10), side="bottom"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    st.plotly_chart(fig_heat, width="stretch")


# Halaman Seleksi Saham

def render_seleksi():
    """
    Halaman seleksi saham: menampilkan hasil dua tahap seleksi
    beserta heatmap korelasi Spearman.
    """
    require_selection()
    hairline_header("Seleksi Saham", _sidebar_metadata())

    hasil = st.session_state.hasil_seleksi
    saham_terpilih = st.session_state.saham_terpilih
    threshold = st.session_state.config["threshold"]

    info_box(
        f"Seleksi dua tahap: filter E(R) > 0, lalu korelasi Spearman &lt; "
        f"<strong>{threshold}</strong>. Terpilih <strong>{len(saham_terpilih)} saham</strong>.",
        kind="info",
    )

    section_title("Tahap 1 — Expected Return Positif")

    # Tabel saham yang lolos filter E(R) > 0
    if hasil["saham_positif"]:
        expected_return = st.session_state.expected_return
        er_positif = pd.DataFrame({
            "Ticker": hasil["saham_positif"],
            "Expected Return (harian)": [
                f"{expected_return[t]:.6f}" for t in hasil["saham_positif"]
            ],
            "Expected Return (tahunan)": [
                f"{(np.exp(expected_return[t] * HARI_BURSA_SETAHUN) - 1) * 100:.2f}%"
                for t in hasil["saham_positif"]
            ],
        })
        st.dataframe(er_positif, hide_index=True, width="stretch")
    else:
        empty_state("Tidak ada saham dengan expected return positif.")

    section_title(f"Tahap 2 — Korelasi Spearman < {threshold}")

    if hasil["pasangan_df"] is not None:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**10 pasangan dengan korelasi terendah:**")
            top_pairs = hasil["pasangan_df"].head(10).copy()
            top_pairs["Korelasi"] = top_pairs["Korelasi"].round(4)
            st.dataframe(top_pairs, hide_index=True, width="stretch")

        with col_b:
            if hasil["pasangan_terpilih"] is not None:
                n_pasangan = len(hasil["pasangan_terpilih"])
                st.markdown(
                    f"**{n_pasangan} pasangan memenuhi threshold < {threshold}**"
                )
                if n_pasangan > 0:
                    pairs_threshold = hasil["pasangan_terpilih"].copy()
                    pairs_threshold["Korelasi"] = pairs_threshold["Korelasi"].round(4)
                    st.dataframe(pairs_threshold.head(20), hide_index=True,
                                 width="stretch")

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
            text=np.round(spearman_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="Spearman(%{x}, %{y}) = %{z:.3f}<extra></extra>",
        ))

        fig_heat.update_layout(
            template="plotly_white",
            height=max(400, 22 * len(spearman_matrix.columns)),
            xaxis=dict(tickfont=dict(size=10), side="bottom"),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, width="stretch")

    section_title("Saham Terpilih")

    if saham_terpilih:
        expected_return = st.session_state.expected_return
        df_terpilih = pd.DataFrame({
            "Ticker": saham_terpilih,
            "Expected Return (tahunan)": [
                f"{(np.exp(expected_return[t] * HARI_BURSA_SETAHUN) - 1) * 100:.2f}%"
                for t in saham_terpilih
            ],
        })
        st.dataframe(df_terpilih, hide_index=True, width="stretch")
    else:
        empty_state("Tidak ada saham yang lolos seleksi.")


# Halaman Optimasi Portofolio

def _jalankan_optimasi():
    """
    Jalankan dua optimasi SLSQP:
    1. Max Sortino Ratio
    2. Min Semivariance
    Simpan hasil bobot dan Sortino optimal ke session_state.
    """
    mu = st.session_state.mu_terpilih
    sigma_sv = st.session_state.matriks_sv
    rf_annual = st.session_state.config["risk_free_pct"] / 100
    rf_daily = rf_annual / HARI_BURSA_SETAHUN

    with st.spinner("Mengoptimasi portofolio dengan SLSQP..."):
        bobot_max_sortino, sortino_max = optimasi_slsqp_max_sortino(
            mu, sigma_sv, rf_daily, n_starts=10
        )
        bobot_min_sv = optimasi_slsqp_min_semivariance(
            sigma_sv, n_starts=10
        )

    st.session_state.bobot_max_sortino = bobot_max_sortino
    st.session_state.bobot_min_sv = bobot_min_sv
    st.session_state.sortino_max = sortino_max


def render_optimasi():
    """
    Halaman optimasi: tombol jalankan, lalu tampilkan hasil kedua strategi
    beserta pie chart dan tabel perbandingan bobot.
    """
    require_selection()
    hairline_header("Optimasi Portofolio", _sidebar_metadata())

    saham_terpilih = st.session_state.saham_terpilih
    log_return = st.session_state.log_return
    log_return_benchmark = st.session_state.log_return_benchmark

    # Hitung matriks semivariance sekali saja; simpan di session_state
    if st.session_state.matriks_sv is None:
        returns_terpilih = log_return[saham_terpilih].values
        benchmark_aligned = log_return_benchmark.loc[log_return.index].values

        matriks_sv = hitung_matriks_semivariance(returns_terpilih, benchmark_aligned)
        mu_terpilih = log_return[saham_terpilih].mean().values

        st.session_state.returns_terpilih = returns_terpilih
        st.session_state.benchmark_aligned = benchmark_aligned
        st.session_state.matriks_sv = matriks_sv
        st.session_state.mu_terpilih = mu_terpilih

    info_box(
        f"Optimasi akan dijalankan untuk <strong>{len(saham_terpilih)} saham</strong> "
        f"dengan matriks semivariance berukuran {len(saham_terpilih)}×{len(saham_terpilih)}. "
        f"Dua strategi: max Sortino dan min Semivariance (keduanya long-only).",
        kind="info",
    )

    if st.button(
        "Jalankan Optimasi",
        type="primary",
        icon=":material/play_arrow:",
    ):
        _jalankan_optimasi()
        st.success("Optimasi selesai.")

    # Tampilkan hasil kalau sudah ada
    if st.session_state.bobot_max_sortino is not None:
        _render_hasil_optimasi()


def _render_hasil_optimasi():
    """
    Render hasil kedua strategi: tabel bobot + pie chart, tabel perbandingan,
    lalu metric Sortino untuk masing-masing.
    """
    saham_terpilih = st.session_state.saham_terpilih
    bobot_max = st.session_state.bobot_max_sortino
    bobot_min = st.session_state.bobot_min_sv
    rf_annual = st.session_state.config["risk_free_pct"] / 100

    returns = st.session_state.returns_terpilih
    benchmark = st.session_state.benchmark_aligned

    section_title("Portofolio Optimal — SLSQP Max Sortino")

    col_left, col_right = st.columns(2)

    with col_left:
        df_bobot_max = pd.DataFrame({
            "Ticker": saham_terpilih,
            "Bobot (%)": np.round(bobot_max * 100, 2),
        }).sort_values("Bobot (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_bobot_max, hide_index=True, width="stretch")

        st.caption(
            f"Total bobot: {bobot_max.sum():.6f} · Bobot minimum: {bobot_max.min():.6f}"
        )

    with col_right:
        fig_pie_max = _pie_bobot(saham_terpilih, bobot_max, PALETTE_PIE)
        if fig_pie_max is not None:
            st.plotly_chart(fig_pie_max, width="stretch")
        else:
            st.info("Tidak ada bobot signifikan untuk ditampilkan.")

    section_title("Portofolio Optimal — SLSQP Min Semivariance")

    col_left, col_right = st.columns(2)

    with col_left:
        df_bobot_min = pd.DataFrame({
            "Ticker": saham_terpilih,
            "Bobot (%)": np.round(bobot_min * 100, 2),
        }).sort_values("Bobot (%)", ascending=False).reset_index(drop=True)
        st.dataframe(df_bobot_min, hide_index=True, width="stretch")

        st.caption(
            f"Total bobot: {bobot_min.sum():.6f} · Bobot minimum: {bobot_min.min():.6f}"
        )

    with col_right:
        fig_pie_min = _pie_bobot(saham_terpilih, bobot_min, PALETTE_PIE)
        if fig_pie_min is not None:
            st.plotly_chart(fig_pie_min, width="stretch")
        else:
            st.info("Tidak ada bobot signifikan untuk ditampilkan.")

    section_title("Perbandingan Bobot")

    df_compare = pd.DataFrame({
        "Ticker": saham_terpilih,
        "Max Sortino (%)": np.round(bobot_max * 100, 2),
        "Min Semivariance (%)": np.round(bobot_min * 100, 2),
    })
    st.dataframe(df_compare, hide_index=True, width="stretch")

    # Hitung Sortino aktual dari return yang dihasilkan masing-masing strategi
    return_max = returns @ bobot_max
    return_min = returns @ bobot_min

    sortino_max_actual = hitung_sortino(return_max, rf_annual, benchmark)
    sortino_min_actual = hitung_sortino(return_min, rf_annual, benchmark)

    section_title("Sortino Ratio — Perbandingan")

    col1, col2 = st.columns(2)
    with col1:
        metric_card(
            "Sortino Tahunan · Max Sortino",
            f"{sortino_max_actual['sortino_tahunan']:.4f}",
            accent="#1E3A8A",
            sublabel=f"Sortino harian: {sortino_max_actual['sortino_harian']:.4f}",
        )
    with col2:
        metric_card(
            "Sortino Tahunan · Min Semivariance",
            f"{sortino_min_actual['sortino_tahunan']:.4f}",
            accent="#D97706",
            sublabel=f"Sortino harian: {sortino_min_actual['sortino_harian']:.4f}",
        )


# Halaman Evaluasi Kinerja

def render_evaluasi():
    """
    Halaman evaluasi kinerja: metric cards untuk kedua strategi,
    plot cumulative return, dan histogram return + VaR.
    """
    require_optimization()
    hairline_header("Evaluasi Kinerja", _sidebar_metadata())

    returns = st.session_state.returns_terpilih
    benchmark = st.session_state.benchmark_aligned
    bobot_max = st.session_state.bobot_max_sortino
    bobot_min = st.session_state.bobot_min_sv
    rf_annual = st.session_state.config["risk_free_pct"] / 100
    dana_awal = st.session_state.config["dana_awal"]

    # Hitung metrik untuk kedua strategi
    return_max = returns @ bobot_max
    return_min = returns @ bobot_min

    sortino_max = hitung_sortino(return_max, rf_annual, benchmark)
    sortino_min = hitung_sortino(return_min, rf_annual, benchmark)
    cumret_max = hitung_cumulative_return(return_max)
    cumret_min = hitung_cumulative_return(return_min)
    var_max = hitung_var_historical(return_max, alpha=0.05, dana_awal=dana_awal)
    var_min = hitung_var_historical(return_min, alpha=0.05, dana_awal=dana_awal)

    section_title("Metrik Kinerja — Max Sortino")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Return Tahunan",
            f"{sortino_max['return_tahunan'] * 100:.2f}%",
            accent="#1E3A8A",
        )
    with col2:
        metric_card(
            "Sortino Tahunan",
            f"{sortino_max['sortino_tahunan']:.4f}",
            accent="#1E3A8A",
        )
    with col3:
        metric_card(
            "Cumulative Return",
            f"{cumret_max * 100:.2f}%",
            accent="#1E3A8A",
        )
    with col4:
        metric_card(
            "VaR 95% (1 hari)",
            f"Rp {var_max['var_rupiah']:,.0f}",
            accent="#1E3A8A",
            sublabel=f"Setara {var_max['var_persen']:.2f}% dari dana",
        )

    section_title("Metrik Kinerja — Min Semivariance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Return Tahunan",
            f"{sortino_min['return_tahunan'] * 100:.2f}%",
            accent="#D97706",
        )
    with col2:
        metric_card(
            "Sortino Tahunan",
            f"{sortino_min['sortino_tahunan']:.4f}",
            accent="#D97706",
        )
    with col3:
        metric_card(
            "Cumulative Return",
            f"{cumret_min * 100:.2f}%",
            accent="#D97706",
        )
    with col4:
        metric_card(
            "VaR 95% (1 hari)",
            f"Rp {var_min['var_rupiah']:,.0f}",
            accent="#D97706",
            sublabel=f"Setara {var_min['var_persen']:.2f}% dari dana",
        )

    # Peringatan kalau VaR tidak bisa dihitung
    if var_max.get("pesan"):
        info_box(var_max["pesan"], kind="warning")

    section_title("Cumulative Return")

    idx = st.session_state.log_return.index[:len(return_max)]

    # Cumulative return dari log return
    cum_max = np.exp(pd.Series(return_max, index=idx).cumsum()) - 1
    cum_min = np.exp(pd.Series(return_min, index=idx).cumsum()) - 1
    cum_bench = np.exp(pd.Series(benchmark, index=idx).cumsum()) - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_max.index,
        y=cum_max * 100,
        mode="lines",
        name="Max Sortino",
        line=dict(color="#1E3A8A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cum_min.index,
        y=cum_min * 100,
        mode="lines",
        name="Min Semivariance",
        line=dict(color="#D97706", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cum_bench.index,
        y=cum_bench * 100,
        mode="lines",
        name="Benchmark",
        line=dict(color="#475569", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="",
        yaxis_title="Cumulative Return (%)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")

    section_title("Distribusi Return & VaR 95%")

    # Dropdown untuk memilih strategi yang mau dilihat distribusinya
    pilihan = st.selectbox(
        "Pilih strategi untuk distribusi return:",
        options=["Max Sortino", "Min Semivariance"],
        key="evaluasi_strategi",
    )

    if pilihan == "Max Sortino":
        return_plot = return_max
        var_plot = var_max
        warna = "#1E3A8A"
    else:
        return_plot = return_min
        var_plot = var_min
        warna = "#D97706"

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=return_plot * 100,
        nbinsx=40,
        marker=dict(color=warna, opacity=0.75),
        name="Return Harian",
    ))

    # Garis vertikal VaR dan mean sebagai referensi
    fig_hist.add_vline(
        x=var_plot["q_alpha"] * 100,
        line=dict(color="#92400E", width=2, dash="dash"),
        annotation_text=f"VaR 95% = {var_plot['q_alpha'] * 100:.2f}%",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#92400E"),
    )
    fig_hist.add_vline(
        x=np.mean(return_plot) * 100,
        line=dict(color="#475569", width=1.5, dash="dot"),
        annotation_text=f"Mean = {np.mean(return_plot) * 100:.2f}%",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#475569"),
    )

    fig_hist.update_layout(
        template="plotly_white",
        height=380,
        xaxis_title="Return Harian (%)",
        yaxis_title="Frekuensi",
        showlegend=False,
    )
    st.plotly_chart(fig_hist, width="stretch")


# Halaman Perbandingan Strategi

def render_perbandingan():
    """
    Halaman perbandingan strategi: tabel metrik, plot cumulative return,
    dan narasi otomatis hasil perbandingan.
    """
    require_optimization()
    hairline_header("Perbandingan Strategi", _sidebar_metadata())

    returns = st.session_state.returns_terpilih
    benchmark = st.session_state.benchmark_aligned
    bobot_max = st.session_state.bobot_max_sortino
    bobot_min = st.session_state.bobot_min_sv
    rf_annual = st.session_state.config["risk_free_pct"] / 100

    # Cache hasil perbandingan agar tidak hitung ulang setiap rerun
    if st.session_state.perbandingan is None:
        perbandingan = bandingkan_strategi(
            returns, bobot_max, bobot_min, rf_annual, benchmark,
        )
        st.session_state.perbandingan = perbandingan

    info_box(
        "Kedua strategi dioptimasi menggunakan matriks risiko yang sama, yaitu "
        "matriks Semivariance-Semicovariance dari data yang sama. Perbedaan hanya "
        "terletak pada fungsi tujuan masing-masing.",
        kind="info",
    )

    section_title("Tabel Perbandingan")

    st.dataframe(
        st.session_state.perbandingan,
        hide_index=True,
        width="stretch",
    )

    section_title("Cumulative Return")

    return_max = returns @ bobot_max
    return_min = returns @ bobot_min

    idx = st.session_state.log_return.index[:len(return_max)]
    cum_max = np.exp(pd.Series(return_max, index=idx).cumsum()) - 1
    cum_min = np.exp(pd.Series(return_min, index=idx).cumsum()) - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_max.index,
        y=cum_max * 100,
        mode="lines",
        name="Max Sortino",
        line=dict(color="#1E3A8A", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cum_min.index,
        y=cum_min * 100,
        mode="lines",
        name="Min Semivariance",
        line=dict(color="#D97706", width=2),
    ))

    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="",
        yaxis_title="Cumulative Return (%)",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")

    section_title("Narasi Hasil")

    # Kumpulkan nilai metrik untuk narasi otomatis
    sortino_max_val = hitung_sortino(return_max, rf_annual, benchmark)
    sortino_min_val = hitung_sortino(return_min, rf_annual, benchmark)
    var_max_val = hitung_var_historical(return_max)
    var_min_val = hitung_var_historical(return_min)
    cumret_max_val = hitung_cumulative_return(return_max)
    cumret_min_val = hitung_cumulative_return(return_min)

    parts = []

    # Narasi 1: perbandingan Sortino
    if sortino_max_val["sortino_tahunan"] > sortino_min_val["sortino_tahunan"]:
        parts.append(
            f"<strong>SLSQP max Sortino unggul dalam risk-adjusted return</strong> "
            f"(Sortino tahunan {sortino_max_val['sortino_tahunan']:.4f} vs "
            f"{sortino_min_val['sortino_tahunan']:.4f})."
        )
    else:
        parts.append(
            f"SLSQP min Semivariance menghasilkan Sortino tahunan lebih tinggi "
            f"({sortino_min_val['sortino_tahunan']:.4f} vs "
            f"{sortino_max_val['sortino_tahunan']:.4f}), meskipun tujuannya "
            f"bukan memaksimalkan Sortino."
        )

    # Narasi 2: perbandingan cumulative return
    if cumret_max_val > cumret_min_val:
        parts.append(
            f"Cumulative return max Sortino lebih tinggi "
            f"({cumret_max_val * 100:.2f}% vs {cumret_min_val * 100:.2f}%)."
        )
    else:
        parts.append(
            f"Cumulative return min Semivariance lebih tinggi "
            f"({cumret_min_val * 100:.2f}% vs {cumret_max_val * 100:.2f}%)."
        )

    # Narasi 3: perbandingan VaR
    if var_max_val["var_rupiah"] < var_min_val["var_rupiah"]:
        parts.append(
            f"Risiko downside max Sortino lebih kecil "
            f"(VaR Rp {var_max_val['var_rupiah']:,.0f} vs "
            f"Rp {var_min_val['var_rupiah']:,.0f})."
        )
    else:
        parts.append(
            f"Risiko downside min Semivariance lebih kecil "
            f"(VaR Rp {var_min_val['var_rupiah']:,.0f} vs "
            f"Rp {var_max_val['var_rupiah']:,.0f})."
        )

    # Render narasi dalam box dengan border adaptif
    st.markdown(
        f"""
        <div style="border: {BORDER_HAIRLINE};
                    border-radius: 8px; padding: 20px 24px;
                    font-size: 0.88rem; line-height: 1.75;">
            {' '.join(parts)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Entry point: init state, render sidebar, route ke halaman yang dipilih."""
    init_state()
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Input Data":
        render_input_data()
    elif page == "Eksplorasi Data":
        render_eksplorasi()
    elif page == "Seleksi Saham":
        render_seleksi()
    elif page == "Optimasi Portofolio":
        render_optimasi()
    elif page == "Evaluasi Kinerja":
        render_evaluasi()
    elif page == "Perbandingan Strategi":
        render_perbandingan()


if __name__ == "__main__":
    main()