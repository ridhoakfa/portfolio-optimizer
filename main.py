"""Aplikasi web Streamlit optimasi portofolio Mean-Semivariance dengan SLSQP (skripsi S1 Statistika Undip)."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from functions import (
    DAFTAR_SAHAM_DEFAULT,
    BENCHMARK_DEFAULT,
    START_DATE_DEFAULT,
    END_DATE_DEFAULT,
    THRESHOLD_KORELASI_DEFAULT,
    MAX_SAHAM_DEFAULT,
    DANA_AWAL_VAR_DEFAULT,
    MIN_OBSERVASI,
    load_data_saham,
    load_data_benchmark,
    hitung_log_return,
    hitung_expected_return,
    seleksi_saham_dua_tahap,
    hitung_matriks_semivariance,
    optimasi_slsqp_max_sortino,
    optimasi_slsqp_min_semivariance,
    hitung_metrik_strategi,
    tabel_perbandingan,
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

WARNA_STRATEGI = {"Max Sortino": "#1E3A8A", "Min Semivariance": "#D97706"}

BOBOT_MINIMAL_TAMPIL = 1e-6

BORDER_HAIRLINE = "1px solid rgba(128, 128, 128, 0.2)"
BORDER_DASHED = "1px dashed rgba(128, 128, 128, 0.25)"


def init_state() -> None:
    """Isi nilai awal session_state sekali per sesi."""
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
        "metrik": None,
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def require_data() -> None:
    """Guard: pastikan data sudah dimuat sebelum lanjut render halaman."""
    if st.session_state.data_saham is None:
        st.info("Silakan muat data terlebih dahulu di halaman **Input Data**.")
        st.stop()


def require_selection() -> None:
    """Guard: pastikan seleksi saham sudah dijalankan."""
    require_data()
    if not st.session_state.saham_terpilih:
        st.info(
            "Seleksi saham belum dijalankan atau tidak menghasilkan saham. "
            "Silakan periksa halaman **Input Data**."
        )
        st.stop()


def require_optimization() -> None:
    """Guard: pastikan optimasi sudah dijalankan sebelum halaman evaluasi."""
    require_selection()
    if st.session_state.bobot_max_sortino is None:
        st.info(
            "Optimasi belum dijalankan. Silakan buka halaman **Optimasi Portofolio** "
            "dan klik **Jalankan Optimasi**."
        )
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
                {metadata}
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


def _pie_bobot(saham_list: list[str], bobot: np.ndarray, warna_series: list[str]) -> go.Figure | None:
    """Pie chart bobot; bobot di bawah BOBOT_MINIMAL_TAMPIL tidak digambar."""
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


def render_sidebar() -> str:
    """Sidebar navigasi dan status data; mengembalikan nama halaman terpilih."""
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

        page = page_raw.split(" ", 1)[-1] if " " in page_raw else page_raw

        st.divider()

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


def _sidebar_metadata() -> str:
    """Metadata singkat yang muncul di kanan atas setiap halaman."""
    if st.session_state.data_saham is None:
        return "Belum ada data"

    n_saham = st.session_state.data_saham.shape[1]
    n_hari = st.session_state.data_saham.shape[0]
    bench = st.session_state.config["benchmark"]

    return f"{n_saham} saham · {n_hari} hari · {bench}"


def render_home() -> None:
    """Halaman Home: penjelasan aplikasi, alur analisis, dan konfigurasi default."""
    hairline_header("Portofolio Optimizer", "By: Ridho Akbar Fadhilah")

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

    steps = [
        ("01", "Input Data", "Unduh harga historis dari Yahoo Finance dan hitung log return harian."),
        ("02", "Eksplorasi Data", "Statistik deskriptif return harian dan pergerakan harga ternormalisasi."),
        ("03", "Seleksi Saham", "Filter E(R) > 0, lalu seleksi berdasarkan Spearman threshold."),
        ("04", "Optimasi", "SLSQP max Sortino Ratio dan SLSQP min Semivariance."),
        ("05", "Evaluasi", "Sortino Ratio harian, Cumulative Return, dan VaR 95% Simulasi Historis."),
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


def _parse_tickers(text: str) -> list[str]:
    """Pisahkan input ticker (baris baru atau koma) menjadi daftar unik huruf kapital."""
    tickers = []
    for line in text.replace(",", "\n").split("\n"):
        t = line.strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def _jalankan_pipeline(tickers: list[str], benchmark: str, start_date: str, end_date: str,
                       threshold: float, max_saham: int, dana_awal: int) -> None:
    """Unduh data, selaraskan tanggal, hitung log return, dan jalankan seleksi dua tahap."""
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

    idx_bersama = data_saham.index.intersection(data_benchmark.index)
    data_saham = data_saham.loc[idx_bersama]
    data_benchmark = data_benchmark.loc[idx_bersama]

    if len(idx_bersama) < MIN_OBSERVASI:
        st.error(
            f"Hanya {len(idx_bersama)} hari bursa yang beririsan antara saham dan benchmark. "
            f"Minimal {MIN_OBSERVASI} hari diperlukan."
        )
        return

    log_return = hitung_log_return(data_saham)
    log_return_benchmark = hitung_log_return(data_benchmark)
    expected_return = hitung_expected_return(log_return)

    hasil_seleksi = seleksi_saham_dua_tahap(
        log_return, expected_return,
        threshold=threshold, max_saham=max_saham,
    )

    saham_terpilih = hasil_seleksi["saham_terpilih"]

    if len(saham_terpilih) < 3:
        st.warning(
            f"Hanya {len(saham_terpilih)} saham lolos seleksi. "
            f"Portofolio terlalu terkonsentrasi. "
            f"Naikkan threshold atau perpanjang periode."
        )
        if hasil_seleksi.get("pesan"):
            st.info(hasil_seleksi["pesan"])

    st.session_state.data_saham = data_saham
    st.session_state.data_benchmark = data_benchmark
    st.session_state.log_return = log_return
    st.session_state.log_return_benchmark = log_return_benchmark
    st.session_state.expected_return = expected_return
    st.session_state.hasil_seleksi = hasil_seleksi
    st.session_state.saham_terpilih = saham_terpilih

    st.session_state.config.update({
        "tickers": tickers,
        "benchmark": benchmark,
        "start_date": start_date,
        "end_date": end_date,
        "threshold": threshold,
        "max_saham": max_saham,
        "dana_awal": dana_awal,
    })

    for key in ["matriks_sv", "returns_terpilih", "benchmark_aligned",
                "mu_terpilih", "bobot_max_sortino", "bobot_min_sv", "metrik"]:
        st.session_state[key] = None
    st.session_state.solver_gagal = []

    info_box(
        f"Data dimuat: <strong>{data_saham.shape[1]} saham</strong> · "
        f"<strong>{data_saham.shape[0]} hari bursa</strong>. "
        f"Seleksi menghasilkan <strong>{len(saham_terpilih)} saham</strong>.",
        kind="success",
    )


def render_input_data() -> None:
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
        "Default disetel untuk BISNIS-27 dengan benchmark IHSG. "
        "Untuk aset lain, sesuaikan benchmark di kolom sebelah.",
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
        "Muat Data & Jalankan Seleksi",
        type="primary",
        width="stretch",
        icon=":material/download:",
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


def render_eksplorasi() -> None:
    """Halaman Eksplorasi Data: statistik deskriptif return dan harga ternormalisasi."""
    require_data()
    hairline_header("Eksplorasi Data", _sidebar_metadata())

    log_return = st.session_state.log_return
    data_saham = st.session_state.data_saham

    section_title("Statistik Deskriptif Return Harian")

    stat_desc = pd.DataFrame({
        "Mean": log_return.mean(),
        "Std Dev": log_return.std(),
        "Min": log_return.min(),
        "Max": log_return.max(),
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


def render_seleksi() -> None:
    """Halaman Seleksi Saham: hasil tahap 1, tahap 2, dan heatmap Spearman."""
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

    if hasil["saham_positif"]:
        expected_return = st.session_state.expected_return
        er_positif = pd.DataFrame({
            "Ticker": hasil["saham_positif"],
            "Expected Return (harian)": [
                f"{expected_return[t]:.6f}" for t in hasil["saham_positif"]
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
            "Expected Return (harian)": [
                f"{expected_return[t]:.6f}" for t in saham_terpilih
            ],
        })
        st.dataframe(df_terpilih, hide_index=True, width="stretch")
    else:
        empty_state("Tidak ada saham yang lolos seleksi.")


def _jalankan_optimasi() -> None:
    """Jalankan kedua strategi SLSQP lalu simpan bobot dan metrik evaluasinya."""
    mu = st.session_state.mu_terpilih
    sigma_sv = st.session_state.matriks_sv
    benchmark = st.session_state.benchmark_aligned

    with st.spinner("Mengoptimasi portofolio dengan SLSQP..."):
        bobot_max, _, berhasil_max = optimasi_slsqp_max_sortino(mu, sigma_sv, float(np.mean(benchmark)))
        bobot_min, berhasil_min = optimasi_slsqp_min_semivariance(sigma_sv)

    st.session_state.bobot_max_sortino = bobot_max
    st.session_state.bobot_min_sv = bobot_min
    st.session_state.solver_gagal = [
        nama for nama, berhasil in (("Max Sortino", berhasil_max), ("Min Semivariance", berhasil_min))
        if not berhasil
    ]

    dana_awal = st.session_state.config["dana_awal"]
    returns = st.session_state.returns_terpilih
    st.session_state.metrik = {
        nama: hitung_metrik_strategi(returns, bobot, mu, sigma_sv, benchmark, dana_awal)
        for nama, bobot in (("Max Sortino", bobot_max), ("Min Semivariance", bobot_min))
    }


def render_optimasi() -> None:
    """Halaman Optimasi Portofolio: hitung matriks risiko lalu jalankan kedua strategi."""
    require_selection()
    hairline_header("Optimasi Portofolio", _sidebar_metadata())

    saham_terpilih = st.session_state.saham_terpilih
    log_return = st.session_state.log_return
    log_return_benchmark = st.session_state.log_return_benchmark

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

    if st.session_state.bobot_max_sortino is not None:
        _render_hasil_optimasi()


def _render_hasil_optimasi() -> None:
    """Tabel dan pie chart bobot kedua strategi."""
    saham_terpilih = st.session_state.saham_terpilih
    bobot_max = st.session_state.bobot_max_sortino
    bobot_min = st.session_state.bobot_min_sv


    if st.session_state.solver_gagal:
        st.warning(
            "SLSQP tidak konvergen dari semua titik awal untuk strategi "
            f"{', '.join(st.session_state.solver_gagal)}. Bobot diisi sama rata "
            "dan bukan hasil optimasi; periksa data sebelum memakai hasil ini."
        )

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


def _kartu_metrik(metrik: dict, warna: str) -> None:
    """Tiga kartu metrik evaluasi untuk satu strategi."""
    sortino, var = metrik["sortino"], metrik["var"]
    kolom = st.columns(3)
    with kolom[0]:
        metric_card("Sortino Ratio Harian", f"{sortino['sortino_harian']:.4f}", accent=warna)
    with kolom[1]:
        metric_card("Cumulative Return", f"{metrik['cumulative_return'] * 100:.2f}%", accent=warna)
    with kolom[2]:
        metric_card(
            "VaR 95% (1 hari)",
            f"Rp {var['var_rupiah']:,.0f}",
            accent=warna,
            sublabel=f"Setara {var['var_persen']:.2f}% dari dana · k = {var['posisi_k']}",
        )


def render_evaluasi() -> None:
    """Halaman Evaluasi Kinerja: metrik kedua strategi, grafik Cumulative Return, dan distribusi return dengan VaR."""
    require_optimization()
    hairline_header("Evaluasi Kinerja", _sidebar_metadata())

    metrik = st.session_state.metrik
    benchmark = st.session_state.benchmark_aligned

    for nama, warna in WARNA_STRATEGI.items():
        section_title(f"Metrik Kinerja — {nama}")
        _kartu_metrik(metrik[nama], warna)
        if metrik[nama]["var"]["pesan"]:
            info_box(metrik[nama]["var"]["pesan"], kind="warning")

    section_title("Cumulative Return")

    idx = st.session_state.log_return.index
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

    pilihan = st.selectbox(
        "Pilih strategi untuk distribusi return:",
        options=list(WARNA_STRATEGI),
        key="evaluasi_strategi",
    )
    return_plot = metrik[pilihan]["return_portofolio"]
    var_plot = metrik[pilihan]["var"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=return_plot * 100,
        nbinsx=40,
        marker=dict(color=WARNA_STRATEGI[pilihan], opacity=0.75),
        name="Return Harian",
    ))
    fig_hist.add_vline(
        x=var_plot["q_alpha"] * 100,
        line=dict(color="#92400E", width=2, dash="dash"),
        annotation_text=f"Return ke-{var_plot['posisi_k']} = {var_plot['q_alpha'] * 100:.2f}%",
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


def _kalimat_unggul(label: str, nilai_max: float, nilai_min: float, format_nilai, lebih_kecil_lebih_baik: bool = False) -> str:
    """Satu kalimat narasi yang menyebut strategi dengan nilai lebih baik pada satu metrik."""
    max_unggul = nilai_max < nilai_min if lebih_kecil_lebih_baik else nilai_max > nilai_min
    unggul, kalah = ("Max Sortino", "Min Semivariance") if max_unggul else ("Min Semivariance", "Max Sortino")
    nilai_unggul, nilai_kalah = (nilai_max, nilai_min) if max_unggul else (nilai_min, nilai_max)
    return f"{label} {unggul} lebih baik ({format_nilai(nilai_unggul)} vs {format_nilai(nilai_kalah)} pada {kalah})."


def render_perbandingan() -> None:
    """Halaman Perbandingan Strategi: tabel metrik dan narasi hasil."""
    require_optimization()
    hairline_header("Perbandingan Strategi", _sidebar_metadata())

    metrik_max = st.session_state.metrik["Max Sortino"]
    metrik_min = st.session_state.metrik["Min Semivariance"]

    info_box(
        "Kedua strategi dioptimasi menggunakan matriks risiko yang sama, yaitu "
        "matriks Semivariance-Semicovariance dari data yang sama. Perbedaan hanya "
        "terletak pada fungsi tujuan masing-masing.",
        kind="info",
    )

    section_title("Tabel Perbandingan")
    st.dataframe(tabel_perbandingan(metrik_max, metrik_min), hide_index=True, width="stretch")

    section_title("Narasi Hasil")

    kalimat = [
        _kalimat_unggul(
            "Sortino Ratio harian",
            metrik_max["sortino"]["sortino_harian"], metrik_min["sortino"]["sortino_harian"],
            lambda x: f"{x:.4f}",
        ),
        _kalimat_unggul(
            "Cumulative Return",
            metrik_max["cumulative_return"], metrik_min["cumulative_return"],
            lambda x: f"{x * 100:.2f}%",
        ),
        _kalimat_unggul(
            "VaR 95%",
            metrik_max["var"]["var_rupiah"], metrik_min["var"]["var_rupiah"],
            lambda x: f"Rp {x:,.0f}",
            lebih_kecil_lebih_baik=True,
        ),
    ]

    st.markdown(
        f"""
        <div style="border: {BORDER_HAIRLINE};
                    border-radius: 8px; padding: 20px 24px;
                    font-size: 0.88rem; line-height: 1.75;">
            {' '.join(kalimat)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Inisialisasi state, tampilkan sidebar, lalu buka halaman yang dipilih."""
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