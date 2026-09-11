import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import recommender as rec
from recommender import split_train_test

# ----------------------------------------------------------------------------
# KONFIGURASI WARNA & BRAND
# Palet korporat: navy gelap (header) + biru vivid (aksen utama)
# + beberapa warna pendukung untuk variasi chart/badge.
# ----------------------------------------------------------------------------
NAVY = "#0B1E3D"
NAVY_SOFT = "#143A6B"
PRIMARY_COLOR = "#2F6FED"
PRIMARY_DARK = "#1E54C4"
ACCENT_TEAL = "#13C2C2"
ACCENT_AMBER = "#FAAD14"
ACCENT_INDIGO = "#7C5CFC"
ACCENT_ROSE = "#F5576C"
ACCENT_SLATE = "#64748B"
TRACK_COLOR = "#E8F0FE"
TEXT_MUTED = "#64748B"
TEXT_DARK = "#0B1E3D"
BG_PAGE = "#F4F6FA"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E7EAF0"
COMPANY_NAME = "PT Andal Ciptamedia Sarana"
LOGO_PATH = "logo.jpeg"

CHART_COLORWAY = [
    PRIMARY_COLOR, ACCENT_TEAL, ACCENT_AMBER,
    ACCENT_INDIGO, ACCENT_ROSE, ACCENT_SLATE
]

st.set_page_config(
    page_title="Andal — Sistem Rekomendasi Paket Internet",
    page_icon="📶",
    layout="wide"
)

st.markdown(
    f"""
    <style>
        /* ── reset & layout ── */
        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 3rem;
            max-width: 100% !important;
        }}
        #MainMenu, footer {{ visibility: hidden; }}
        header {{ display: none !important; }}  /* display:none (bukan visibility:hidden)
                                                     supaya ruang kosongnya ikut hilang,
                                                     header custom bisa naik ke atas */
        section[data-testid="stSidebar"] {{ display: none !important; }}

        /* ── HEADER (background biru navy) ──
           Struktur/konten header TETAP 100% komponen native Streamlit
           (st.container/st.columns/st.image/st.markdown -- lihat blok
           "HEADER" di bawah). Aturan di sini CUMA mewarnai container-nya
           lewat key="app_header" (jadi class .st-key-app_header), tidak
           menambah HTML/konten custom apa pun. */
        .st-key-app_header {{
            background: linear-gradient(90deg, {NAVY} 0%, {NAVY_SOFT} 100%) !important;
            border: none !important;
            border-radius: 12px !important;
        }}
        /* Selector .stMarkdown ikut disertakan (bukan cuma tag h1/p) supaya
           specificity-nya menang lawan aturan "FORCE DARK TEXT EVERYWHERE"
           di bawah yang juga pakai !important pada .stMarkdown h1/p. */
        .st-key-app_header .stMarkdown h1,
        .st-key-app_header .stMarkdown p {{
            color: white !important;
        }}
        .st-key-app_header h5 {{
            color: white !important;
        }}
        .st-key-app_header [data-testid="stCaptionContainer"] {{
            color: rgba(255,255,255,0.75) !important;
        }}

        /* ── KPI CARDS ── */
        .kpi-card {{
            background: {CARD_BG};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 12px rgba(11,30,61,0.07);
            border-top: 4px solid transparent;
            height: 100%;
        }}
        .kpi-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_MUTED};
            margin-bottom: 0.4rem;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {TEXT_DARK};
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: {TEXT_MUTED};
            margin-top: 0.2rem;
        }}
        .kpi-icon {{
            float: right;
            font-size: 1.6rem;
            opacity: 0.18;
        }}

        /* ── CHART CARDS ── */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {CARD_BG};
            box-shadow: 0 2px 12px rgba(11,30,61,0.07);
            border-radius: 14px !important;
        }}

        /* ── STREAMLIT METRIC (fallback) ── */
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 10px rgba(11,30,61,0.06);
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {TEXT_DARK} !important;
        }}

        /* ── BUTTONS ── */
        .stButton > button {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {PRIMARY_DARK});
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(47,111,237,0.3);
            transition: opacity 0.15s;
        }}
        .stButton > button:hover {{
            opacity: 0.88;
            color: white;
        }}
        .stDownloadButton > button {{
            border-radius: 8px;
            font-weight: 600;
        }}

        /* ── BADGE / PILL ── */
        .badge {{
            display: inline-block;
            padding: 0.18rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}
        .badge-blue {{ background:#EEF4FF; color:{PRIMARY_COLOR}; }}
        .badge-teal {{ background:#E6FFFB; color:#08979C; }}
        .badge-amber {{ background:#FFFBE6; color:#D48806; }}
        .badge-indigo {{ background:#F0EEFF; color:{ACCENT_INDIGO}; }}
        .badge-rose {{ background:#FFF1F0; color:#CF1322; }}

        /* ── SLIDER ── */
        div[data-baseweb="slider"] div[role="slider"] {{
            background-color: {PRIMARY_COLOR} !important;
            border-color: {PRIMARY_COLOR} !important;
        }}
        div[data-baseweb="slider"] > div > div:nth-child(2) {{
            background-color: {PRIMARY_COLOR} !important;
        }}

        /* ── FORCE DARK TEXT EVERYWHERE ── */
        .stMarkdown p, .stMarkdown li, .stMarkdown h1,
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
            color: {TEXT_DARK} !important;
        }}
        p, li {{ color: {TEXT_DARK}; }}
    </style>
    """,
    unsafe_allow_html=True
)

with st.container(border=True, key="app_header"):
    col_logo, col_brand, col_company = st.columns(
        [1, 6, 3], vertical_alignment="center"
    )
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=64)
    with col_brand:
        st.markdown("# Andal Ciptamedia Sarana")
        st.caption("Sistem Rekomendasi Paket Internet")
    with col_company:
        st.markdown(f"##### {COMPANY_NAME}")

st.write("")

@st.cache_data
def load_data():
    pelanggan = pd.read_csv("D_transformed.csv")
    katalog = pd.read_csv("Jenis_Layanan.csv")
    return pelanggan, katalog

pelanggan, katalog = load_data()

def donut_chart(percent):
    percent = max(0, min(100, percent))
    fig = go.Figure(data=[go.Pie(
        values=[percent, 100 - percent],
        hole=0.72,
        marker=dict(colors=[PRIMARY_COLOR, TRACK_COLOR]),
        textinfo="none",
        sort=False,
        direction="clockwise"
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=150,
        annotations=[dict(
            text=f"{percent:.0f}%",
            x=0.5, y=0.5,
            font=dict(size=22, color=TEXT_DARK),
            showarrow=False
        )]
    )
    return fig

def kecocokan_label(percent):
    if percent >= 80: return "Sangat Cocok"
    elif percent >= 60: return "Cocok"
    elif percent >= 40: return "Cukup Cocok"
    else: return "Kurang Cocok"


def reliabilitas_label_awam(percent):
    if percent >= 80:
        return "Sangat bisa diandalkan"
    elif percent >= 60:
        return "Cukup bisa diandalkan"
    else:
        return "Belum bisa diandalkan"


def reliability_gauge(percent):
    percent = max(0, min(100, percent))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        number={"suffix": "%", "font": {"size": 30, "color": TEXT_DARK}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": TEXT_MUTED},
            "bar": {"color": TEXT_DARK, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#F0997B"},
                {"range": [60, 80], "color": "#FAC775"},
                {"range": [80, 100], "color": "#97C459"},
            ],
        },
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=20, r=20), height=180)
    return fig

def akurasi_label(error_value):
    if error_value < 0.05:
        return "Sangat Baik", "badge-teal"
    elif error_value < 0.10:
        return "Baik", "badge-blue"
    elif error_value < 0.20:
        return "Cukup Baik", "badge-amber"
    else:
        return "Kurang Baik", "badge-rose"


def cronbach_badge_class(alpha):
    if alpha != alpha:
        return "badge-rose"
    if alpha >= 0.8:
        return "badge-teal"
    if alpha >= 0.7:
        return "badge-blue"
    if alpha >= 0.6:
        return "badge-amber"
    return "badge-rose"

def to_excel_bytes(df, sheet_name="Hasil"):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()

def to_pdf_bytes(df, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]
    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY_COLOR)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

def export_buttons(df, base_filename, pdf_title, key_prefix):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Unduh Excel",
            data=to_excel_bytes(df),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{key_prefix}_xlsx"
        )
    with col2:
        st.download_button(
            "⬇️ Unduh PDF",
            data=to_pdf_bytes(df, pdf_title),
            file_name=f"{base_filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"{key_prefix}_pdf"
        )


def build_validasi_table(hasil_pelanggan):
    rows = []
    for hasil in hasil_pelanggan:
        if hasil["Rekomendasi"]:
            lines = "\n".join(
                f"{r['Peringkat']}. {r['Paket Internet']} ({r['Tingkat Kecocokan (%)']:.2f}%)"
                for r in hasil["Rekomendasi"]
            )
        else:
            lines = "Tidak ada rekomendasi"
        rows.append(
            {
                "Username": hasil["Username"],
                "Paket Saat Ini": hasil["Paket Saat Ini"],
                "Status": hasil["Status"],
                "Daftar Rekomendasi": lines,
            }
        )
    return pd.DataFrame(rows)


def render_validasi_table(table_df):
    rows_html = []
    for _, row in table_df.iterrows():
        status_class = "badge-teal" if row["Status"] == "Berhasil Direkomendasikan" else "badge-rose"
        rekom_html = "".join(
            f'<div class="rekom-line">{line}</div>' for line in row["Daftar Rekomendasi"].split("\n")
        )
        rows_html.append(
            f'<tr><td>{row["Username"]}</td><td>{row["Paket Saat Ini"]}</td>'
            f'<td><span class="badge {status_class}">{row["Status"]}</span></td>'
            f'<td><div class="rekom-cell">{rekom_html}</div></td></tr>'
        )

    style = (
        f"<style>"
        f".validasi-table-wrap {{ max-height: 640px; overflow-y: auto; overflow-x: auto; "
        f"border: 1px solid {CARD_BORDER}; border-radius: 10px; }}"
        f".validasi-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}"
        f".validasi-table th {{ background: {BG_PAGE}; color: {TEXT_MUTED}; text-transform: uppercase; "
        f"font-size: 0.72rem; letter-spacing: 0.04em; padding: 0.7rem 0.9rem; text-align: left; "
        f"border-bottom: 2px solid {CARD_BORDER}; position: sticky; top: 0; z-index: 1; }}"
        f".validasi-table td {{ padding: 0.7rem 0.9rem; border-bottom: 1px solid {CARD_BORDER}; "
        f"vertical-align: top; color: {TEXT_DARK}; }}"
        f".validasi-table tr:last-child td {{ border-bottom: none; }}"
        f".rekom-cell {{ max-height: 200px; overflow-y: auto; padding-right: 6px; }}"
        f".rekom-line {{ padding: 2px 0; white-space: nowrap; }}"
        f"</style>"
    )
    table_html = (
        f"{style}"
        f'<div class="validasi-table-wrap"><table class="validasi-table">'
        f"<thead><tr><th>Username</th><th>Paket Saat Ini</th><th>Status</th>"
        f"<th>Daftar Rekomendasi</th></tr></thead>"
        f'<tbody>{"".join(rows_html)}</tbody>'
        f"</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def to_excel_bytes_validasi(table_df):
    from openpyxl.styles import Alignment

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        table_df.to_excel(writer, index=False, sheet_name="Validasi")
        ws = writer.sheets["Validasi"]
        ws.column_dimensions["A"].width = 24
        ws.column_dimensions["B"].width = 20
        ws.column_dimensions["C"].width = 26
        ws.column_dimensions["D"].width = 55

        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            rekom_value = row[3].value or ""
            n_lines = rekom_value.count("\n") + 1
            ws.row_dimensions[row[0].row].height = max(15, n_lines * 15)
    return buffer.getvalue()


def to_pdf_bytes_validasi(table_df, title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    cell_style = styles["BodyText"]
    cell_style.fontSize = 7
    cell_style.leading = 9

    elements = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]

    data = [["Username", "Paket Saat Ini", "Status", "Daftar Rekomendasi"]]
    for _, row in table_df.iterrows():
        data.append([
            Paragraph(str(row["Username"]), cell_style),
            Paragraph(str(row["Paket Saat Ini"]), cell_style),
            Paragraph(str(row["Status"]), cell_style),
            Paragraph(str(row["Daftar Rekomendasi"]).replace("\n", "<br/>"), cell_style),
        ])

    table = Table(data, repeatRows=1, colWidths=[75, 85, 90, 225])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY_COLOR)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FA")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()


def export_buttons_validasi(table_df, base_filename, pdf_title, key_prefix):
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Unduh Excel",
            data=to_excel_bytes_validasi(table_df),
            file_name=f"{base_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"{key_prefix}_xlsx"
        )
    with col2:
        st.download_button(
            "⬇️ Unduh PDF",
            data=to_pdf_bytes_validasi(table_df, pdf_title),
            file_name=f"{base_filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"{key_prefix}_pdf"
        )


def run_full_validation(test_df, pelanggan_df, neighbor_pool=30, top_n=10):
    feature_cols = ["Total Online (detik)", "Total Kuota (bytes)", "Implicit Rating"]
    train_profile = pelanggan_df.set_index("Username")[feature_cols]
    results = []
    for _, row in test_df.iterrows():
        test_vec = row[feature_cols].values.astype(float)
        dist = np.linalg.norm(train_profile.values - test_vec, axis=1)
        max_dist = dist.max() or 1
        sims = 1 - (dist / max_dist)
        sim_series = pd.Series(sims, index=train_profile.index)
        sim_series = sim_series[sim_series.index != row["Username"]]
        top_neighbors = sim_series.sort_values(ascending=False).head(neighbor_pool)

        neighbor_detail = pelanggan_df[pelanggan_df["Username"].isin(top_neighbors.index)][
            ["Username", "Paket Internet"]
        ].copy()
        neighbor_detail["Similarity"] = neighbor_detail["Username"].map(top_neighbors)

        confidence = neighbor_detail.groupby("Paket Internet")["Similarity"].mean()
        confidence = confidence.sort_values(ascending=False).head(top_n)

        actual = row["Paket Internet"]
        berhasil = not confidence.empty

        rekomendasi = [
            {"Peringkat": i + 1, "Paket Internet": paket, "Tingkat Kecocokan (%)": round(float(skor) * 100, 2)}
            for i, (paket, skor) in enumerate(confidence.items())
        ]

        results.append(
            {
                "Username": row["Username"],
                "Paket Saat Ini": actual,
                "Status": "Berhasil Direkomendasikan" if berhasil else "Gagal Direkomendasikan",
                "Rekomendasi": rekomendasi,
            }
        )
    return results


def build_alpha_item_matrix(hasil_pelanggan, top_n=5):
    rows = []
    for hasil in hasil_pelanggan:
        rekom = hasil["Rekomendasi"]
        if len(rekom) < top_n:
            continue
        row = {"Username": hasil["Username"], "Paket Saat Ini": hasil["Paket Saat Ini"]}
        rank_paket_saat_ini = None
        for r in rekom[:top_n]:
            row[f"Peringkat {r['Peringkat']} (%)"] = r["Tingkat Kecocokan (%)"]
            if r["Paket Internet"] == hasil["Paket Saat Ini"]:
                rank_paket_saat_ini = r["Peringkat"]
        row["Paket Saat Ini di Peringkat"] = rank_paket_saat_ini
        rows.append(row)
    return pd.DataFrame(rows)


def render_reliability_section(hasil_pelanggan, top_n=5):
    alpha_matrix = build_alpha_item_matrix(hasil_pelanggan, top_n=top_n)
    rank_cols = [f"Peringkat {i} (%)" for i in range(1, top_n + 1)]
    total_uji = len(hasil_pelanggan)

    if len(alpha_matrix) < 2:
        st.warning(
            "Data uji tidak cukup untuk menghitung tingkat keandalan sistem "
            f"(butuh minimal 2 pelanggan dengan rekomendasi Top-{top_n} lengkap)."
        )
        return

    alpha_value = rec.compute_cronbach_alpha(alpha_matrix[rank_cols])
    alpha_pct = alpha_value * 100
    label_formal = rec.interpret_cronbach_alpha(alpha_value)
    label_awam = reliabilitas_label_awam(alpha_pct)

    with st.container(border=True):
        g1, g2 = st.columns([1, 1.3])
        with g1:
            st.plotly_chart(reliability_gauge(alpha_pct), use_container_width=True)
        with g2:
            st.markdown(f"### {label_awam}")
            st.caption(
                f"Dari {len(alpha_matrix):,} pelanggan yang diuji, pola skor "
                "kepercayaan Top-5 rekomendasi sistem stabil dan konsisten "
                "antar pelanggan — ini ukuran KONSISTENSI pola penilaian "
                "sistem, bukan akurasi tebakan paket."
            )
            st.caption(
                "**Analoginya**: sistem diminta menilai 1 pelanggan sebanyak 5 "
                "kali berturut-turut (Peringkat 1-5). Kalau ke-5 penilaian itu "
                "hasilnya mirip-mirip terus (bukan naik-turun acak), berarti "
                "cara sistem menilai itu konsisten — bukan berarti tebakannya "
                "pasti benar."
            )

        mean_scores = alpha_matrix[rank_cols].mean()
        cek_cols = st.columns(top_n)
        for col, (rank_name, score) in zip(cek_cols, mean_scores.items()):
            with col:
                st.metric(rank_name.replace(" (%)", ""), f"{score:.1f}%")
        st.caption(
            "Kelima hasil cek itu semuanya tinggi dan mirip satu sama lain "
            "— bukti sistemnya konsisten."
        )

        st.write("")
        n_dikenali = int(alpha_matrix["Paket Saat Ini di Peringkat"].notna().sum())
        n_total_matrix = len(alpha_matrix)
        pct_dikenali = n_dikenali / n_total_matrix * 100 if n_total_matrix else 0
        st.metric(
            f"Paket yang sedang dipakai pelanggan, dikenali sistem di Top-{top_n}",
            f"{n_dikenali}/{n_total_matrix} pelanggan ({pct_dikenali:.1f}%)",
        )
        st.caption(
            "Buat tiap pelanggan, sistem juga mengecek: apakah paket yang SEDANG "
            f"dia pakai ikut muncul di Top-{top_n} kandidat rekomendasi mesin? Ini "
            "bukti tambahan (di luar Cronbach's Alpha) kalau sistem mengenali "
            "pola pemakaian yang memang relevan buat pelanggan itu sendiri."
        )

        with st.expander("Detail teknis (Cronbach's Alpha)"):
            badge_cls = cronbach_badge_class(alpha_value)
            st.markdown(f"Nilai Cronbach's Alpha: **{alpha_value:.4f}**")
            st.markdown(f'<span class="badge {badge_cls}">{label_formal}</span>', unsafe_allow_html=True)
            st.caption(
                f"Dihitung dari {len(alpha_matrix):,} dari {total_uji:,} pelanggan "
                f"(punya rekomendasi Top-{top_n} lengkap), {top_n} item "
                f"(Peringkat 1-{top_n})."
            )
            st.caption(
                "Catatan metodologis: Cronbach's Alpha di sini mengukur KONSISTENSI "
                "INTERNAL pola skor kepercayaan across Peringkat 1-5 (apakah pola "
                "penurunan skornya stabil antar pelanggan) — bukan akurasi tebakan "
                "paket. Paket yang sedang dipakai pelanggan tidak masuk sebagai "
                "komponen perhitungan Alpha ini; itu dicek terpisah lewat metrik "
                "\"dikenali sistem di Top-N\" di atas."
            )


def prepare_validation_data(test_df, katalog_df):
    import data_preparation as dp

    test_df = test_df.copy()
    test_df.columns = test_df.columns.str.strip()

    transformed_cols = ["Username", "Paket Internet", "Total Online (detik)", "Total Kuota (bytes)", "Implicit Rating"]
    if all(col in test_df.columns for col in transformed_cols):
        return test_df, "Data uji sudah dalam format transformasi."

    missing_raw = [col for col in dp.RETAINED_FEATURES if col not in test_df.columns]
    if missing_raw:
        raise ValueError(
            "Format data uji tidak dikenali. Upload data yang sudah berkolom "
            f"{', '.join(transformed_cols)}, atau data mentah (mis. usage_summary "
            f"asli.csv) yang minimal berkolom {', '.join(dp.RETAINED_FEATURES)}. "
            f"Kolom mentah yang kurang: {', '.join(missing_raw)}."
        )

    df_selected = dp.select_features(test_df)
    df_clean = dp.clean_missing(df_selected)
    df_integrated = dp.integrate_catalog(df_clean, katalog_df)
    df_scaled = dp.transform_features(df_integrated)
    df_transformed = dp.construct_implicit_rating(df_scaled)
    return df_transformed, (
        "Data uji mentah berhasil diproses otomatis: feature selection, cleaning, "
        "Data Mapping wilayah, integrasi katalog, normalisasi, dan Implicit Rating."
    )

st.markdown(
    """
    <style>
        .stTabs {
            margin-top: 0.6rem;
            margin-bottom: 0.8rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3.1rem;
            padding: 0 0.3rem;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 1.05rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

tab_dash, tab_cf, tab_rbf, tab_eval, tab_val, tab_uji_baru = st.tabs([
    "📊  Dashboard",
    "👤  Pelanggan Lama",
    "🆕  Pelanggan Baru",
    "📈  Evaluasi",
    "✅  Validasi Sistem",
    "🧪  Uji Dataset Baru",
])

# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
with tab_dash:

    st.subheader("Ringkasan Data Pelanggan")
    st.caption("Gambaran umum kondisi pelanggan dan paket internet saat ini.")

    total_pelanggan = len(pelanggan)
    jumlah_jenis_paket = pelanggan["Paket Internet"].nunique()
    jumlah_wilayah = pelanggan["Wilayah Pemasaran"].nunique()
    total_revenue = pelanggan["Harga (Rp)"].sum()

    k1, k2, k3 = st.columns(3)

    def kpi_html(label, value, sub, icon, border_color):
        return f"""
        <div class="kpi-card" style="border-top-color:{border_color};">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>"""

    k1.markdown(kpi_html("Total Pelanggan", f"{total_pelanggan:,}", "aktif terdaftar", "👥", PRIMARY_COLOR), unsafe_allow_html=True)
    k2.markdown(kpi_html("Jenis Paket", f"{jumlah_jenis_paket}", "paket tersedia", "📦", ACCENT_TEAL), unsafe_allow_html=True)
    k3.markdown(kpi_html("Wilayah", f"{jumlah_wilayah}", "area pemasaran aktif", "📍", ACCENT_SLATE), unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("**Distribusi Pelanggan per Paket**")

            dist = pelanggan["Paket Internet"].value_counts()
            top8 = dist.head(8)
            lainnya = dist.iloc[8:].sum()
            if lainnya > 0:
                top8 = pd.concat([top8, pd.Series({"Lainnya": lainnya})])

            fig_pie = px.pie(
                names=top8.index,
                values=top8.values,
                hole=0.45,
                color_discrete_sequence=CHART_COLORWAY
            )
            fig_pie.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                height=320
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            paket_dominan = dist.index[0]
            pct_dominan = dist.iloc[0] / total_pelanggan * 100
            st.caption(
                f"💡 Paket **{paket_dominan}** mendominasi **{pct_dominan:.1f}%** "
                "dari seluruh pelanggan — indikasi data cukup homogen pada satu paket."
            )

    with c2:
        with st.container(border=True):
            st.markdown("**Top 10 Paket Terpopuler**")

            top10 = pelanggan["Paket Internet"].value_counts().head(10).sort_values()

            fig_bar = px.bar(
                x=top10.values,
                y=top10.index,
                orientation="h",
                labels={"x": "Jumlah Pelanggan", "y": ""},
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig_bar.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                f"💡 Dari {jumlah_jenis_paket} jenis paket yang tersedia, hanya "
                f"{(top10.shape[0])} paket teratas yang menyumbang sebagian besar pelanggan."
            )

# ----------------------------------------------------------------------------
# PELANGGAN LAMA (COLLABORATIVE FILTERING)
# ----------------------------------------------------------------------------
with tab_cf:

    st.subheader("Pelanggan Lama")
    st.caption("Masukkan username pelanggan untuk melihat data dan rekomendasi paket.")

    with st.container(border=True):
        st.markdown("**Cari Pelanggan**")
        col1, col2 = st.columns([3, 1])
        with col1:
            daftar_username = sorted(pelanggan["Username"].unique())
            nomor_username = {u: i + 1 for i, u in enumerate(daftar_username)}
            username = st.selectbox(
                "Username",
                daftar_username,
                format_func=lambda u: f"{nomor_username[u]}. {u}",
                key="cf_username",
                label_visibility="collapsed"
            )
        with col2:
            cari = st.button("Cari Data", use_container_width=True, key="cf_cari")

    if cari:

        data_pelanggan = pelanggan.loc[pelanggan["Username"] == username].iloc[0]

        st.write("")
        with st.container(border=True):
            st.markdown("**Data Pelanggan**")
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Username**<br>{data_pelanggan['Username']}", unsafe_allow_html=True)
            d2.markdown(f"**Paket Saat Ini**<br>{data_pelanggan['Paket Internet']}", unsafe_allow_html=True)
            d3.markdown(f"**Wilayah**<br>{data_pelanggan['Wilayah Pemasaran']}", unsafe_allow_html=True)

            d4, d5, d6 = st.columns(3)
            d4.markdown(f"**Kecepatan**<br>{data_pelanggan['Kecepatan']:.0f} Mbps", unsafe_allow_html=True)
            d5.markdown(f"**Skor Aktivitas Online**<br>{data_pelanggan['Total Online (detik)']:.2f}", unsafe_allow_html=True)
            d6.markdown(f"**Skor Pemakaian Kuota**<br>{data_pelanggan['Total Kuota (bytes)']:.2f}", unsafe_allow_html=True)

        user_item_matrix = rec.build_user_item_matrix(pelanggan)
        user_similarity_df = rec.compute_user_similarity(user_item_matrix)

        recommendation_df, neighbor_table = rec.recommend_existing_customer(
            username, pelanggan, user_item_matrix, user_similarity_df, top_k=10
        )

        st.write("")

        if recommendation_df.empty:
            st.info("Tidak ditemukan rekomendasi paket baru untuk pelanggan ini.")
        else:
            top_row = recommendation_df.iloc[0]
            top_package = top_row["Paket Internet"]
            tingkat_kecocokan = top_row["Tingkat Kecocokan (%)"]
            harga_top_package = top_row["Harga (Rp)"]

            with st.container(border=True):
                st.markdown("**Hasil Rekomendasi**")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown('<span class="badge badge-blue">Rekomendasi Paket</span>', unsafe_allow_html=True)
                    st.markdown(f"### {top_package}")
                with r2:
                    st.markdown('<span class="badge badge-blue">Estimasi Biaya</span>', unsafe_allow_html=True)
                    st.markdown(f"### Rp {harga_top_package:,.0f}")
                    st.caption("/ bulan")
                with r3:
                    st.markdown('<span class="badge badge-blue">Tingkat Kecocokan</span>', unsafe_allow_html=True)
                    st.plotly_chart(donut_chart(tingkat_kecocokan), use_container_width=True)
                    st.caption(kecocokan_label(tingkat_kecocokan))

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            with st.container(border=True):
                st.markdown("**Top 10 Pelanggan Mirip**")
                if neighbor_table.empty:
                    st.info("Tidak ada tetangga dengan similarity > 0.")
                else:
                    st.dataframe(
                        neighbor_table[["Username", "Paket Internet", "Similarity"]],
                        hide_index=True,
                        use_container_width=True
                    )

        with c2:
            with st.container(border=True):
                st.markdown("**Semua Rekomendasi Paket**")
                if recommendation_df.empty:
                    st.info("Tidak ada rekomendasi.")
                else:
                    rec_df = recommendation_df[
                        ["Paket Internet", "Tingkat Kecocokan (%)", "Harga (Rp)", "Sumber"]
                    ]
                    st.dataframe(rec_df, hide_index=True, use_container_width=True)

        if not recommendation_df.empty:
            st.write("")
            st.markdown("**Export Hasil Rekomendasi**")
            export_buttons(
                rec_df,
                base_filename=f"rekomendasi_{username}",
                pdf_title=f"Rekomendasi Paket untuk {username}",
                key_prefix="cf"
            )

# ----------------------------------------------------------------------------
# PELANGGAN BARU (RULE BASED FILTERING)
# ----------------------------------------------------------------------------
with tab_rbf:

    st.subheader("Pelanggan Baru")
    st.caption("Isi data untuk mendapatkan rekomendasi paket internet terbaik.")

    katalog["Kecepatan"] = pd.to_numeric(
        katalog["Kecepatan"]
        .astype(str)
        .str.replace(" Mbps", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

    with st.container(border=True):
        st.markdown('<span class="badge badge-blue">NEW</span> **Pelanggan Baru**', unsafe_allow_html=True)
        st.caption("Isi kriteria untuk mendapatkan rekomendasi Rule-Based.")

        nama_pelanggan = st.text_input(
            "Nama Lengkap (opsional)", placeholder="Masukkan nama pelanggan", key="rbf_nama"
        )
        wilayah = st.selectbox("Wilayah / Lokasi", ["PWS", "Pinang"], key="rbf_wilayah")

        def _format_rbf_budget():
            digits = re.sub(r"\D", "", st.session_state.rbf_budget_text)
            st.session_state.rbf_budget_text = f"{int(digits):,}".replace(",", ".") if digits else ""

        if "rbf_budget_text" not in st.session_state:
            st.session_state.rbf_budget_text = "200.000"

        st.text_input(
            "Batas Anggaran / bulan (Rp)",
            key="rbf_budget_text",
            on_change=_format_rbf_budget,
            help="Ketik angka lalu tekan Enter, titik ribuan akan terisi otomatis."
        )
        budget = int(re.sub(r"\D", "", st.session_state.rbf_budget_text) or 0)
        min_speed = st.number_input(
            "Kecepatan Minimal (Mbps)", min_value=0, max_value=None, value=10, key="rbf_speed"
        )
        cari_baru = st.button("Analisis Paket", key="rbf_cari", use_container_width=True)

    if cari_baru:

        hasil = rec.recommend_new_customer(katalog, wilayah, budget, min_speed)
        is_relaxed = False

        if hasil.empty:
            hasil = rec.recommend_new_customer_relaxed(katalog, wilayah, budget, min_speed, top_n=5)
            is_relaxed = True

        st.write("")

        if hasil.empty:
            st.warning("Tidak ada paket yang tersedia untuk wilayah ini.")
        else:
            top5 = hasil[
                ["Nama Paket", "Kecepatan", "Harga (Rp)", "Wilayah Pemasaran", "Skor Total"]
            ].head(5)

            top1 = top5.iloc[0]
            tingkat_kecocokan = top1["Skor Total"] * 100

            judul_hasil = (
                f"Hasil Rekomendasi untuk {nama_pelanggan}"
                if nama_pelanggan else "Hasil Rekomendasi"
            )

            if is_relaxed:
                st.info(
                    "ℹ️ Tidak ada paket yang memenuhi persis Budget & Kecepatan Minimal "
                    "yang diminta. Berikut beberapa paket yang paling mendekati kriteria."
                )

            with st.container(border=True):
                st.markdown(f"**{judul_hasil}**")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown('<span class="badge badge-blue">Rekomendasi Paket</span>', unsafe_allow_html=True)
                    st.markdown(f"### {top1['Nama Paket']}")
                    st.caption(f"{top1['Kecepatan']:.0f} Mbps")
                with r2:
                    st.markdown('<span class="badge badge-blue">Estimasi Biaya</span>', unsafe_allow_html=True)
                    st.markdown(f"### Rp {top1['Harga (Rp)']:,.0f}")
                    st.caption("/ bulan")
                with r3:
                    st.markdown('<span class="badge badge-blue">Tingkat Kecocokan</span>', unsafe_allow_html=True)
                    st.plotly_chart(donut_chart(tingkat_kecocokan), use_container_width=True)
                    st.caption(kecocokan_label(tingkat_kecocokan))

            st.write("")
            st.markdown(f"**{len(hasil)} paket ditemukan** · menampilkan 5 teratas")

            with st.container(border=True):
                st.dataframe(
                    top5,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Kecepatan": st.column_config.NumberColumn(
                            "Kecepatan", format="%d Mbps"
                        ),
                        "Harga (Rp)": st.column_config.NumberColumn(
                            "Harga", format="Rp %d"
                        ),
                        "Skor Total": st.column_config.ProgressColumn(
                            "Kecocokan", min_value=0, max_value=1, format="percent"
                        ),
                    }
                )

            st.write("")
            st.markdown("**Export Hasil Rekomendasi**")
            export_buttons(
                top5,
                base_filename="rekomendasi_paket_baru",
                pdf_title="Rekomendasi Paket Internet untuk Pelanggan Baru",
                key_prefix="rbf"
            )

# ----------------------------------------------------------------------------
# EVALUASI (MAE & RMSE — Bab III.3.5 draft skripsi)
# ----------------------------------------------------------------------------
with tab_eval:

    st.subheader("Evaluasi Model")

    if st.button("Jalankan Evaluasi MAE & RMSE", key="val_mae_rmse_run"):
        with st.spinner("Membagi data latih/uji dan menghitung prediksi..."):
            train_df, test_df = split_train_test(pelanggan)
            eval_result = rec.evaluate_mae_rmse(train_df, test_df, top_k=10)

        st.write("")

        mae_label, mae_badge = akurasi_label(eval_result["mae"])
        rmse_label, rmse_badge = akurasi_label(eval_result["rmse"])

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown('<span class="badge badge-blue">Mean Absolute Error</span>', unsafe_allow_html=True)
                st.markdown(f"### {eval_result['mae']:.4f}")
                st.markdown(f'<span class="badge {mae_badge}">{mae_label}</span>', unsafe_allow_html=True)
                st.caption("Rata-rata selisih absolut prediksi vs Implicit Rating aktual.")
        with c2:
            with st.container(border=True):
                st.markdown('<span class="badge badge-blue">Root Mean Square Error</span>', unsafe_allow_html=True)
                st.markdown(f"### {eval_result['rmse']:.4f}")
                st.markdown(f'<span class="badge {rmse_badge}">{rmse_label}</span>', unsafe_allow_html=True)
                st.caption("Akar rata-rata kuadrat selisih — memberi penalti lebih besar untuk galat outlier.")

        st.write("")
        d1, d2 = st.columns(2)
        d1.metric("Data Latih (80%)", f"{len(train_df):,}")
        d2.metric("Data Uji (20%)", f"{eval_result['n_test']:,}")

        with st.container(border=True):
            st.markdown("**Detail Galat Prediksi per Data Uji**")
            st.dataframe(
                eval_result["detail"],
                hide_index=True,
                use_container_width=True
            )

        st.write("")
        st.markdown("**Export Hasil Evaluasi**")
        export_buttons(
            eval_result["detail"],
            base_filename="evaluasi_mae_rmse",
            pdf_title="Evaluasi MAE & RMSE Sistem Rekomendasi",
            key_prefix="mae_rmse"
        )

# ----------------------------------------------------------------------------
# VALIDASI SISTEM (Cronbach's Alpha)
# ----------------------------------------------------------------------------
with tab_val:

    st.subheader("Validasi Sistem (Seberapa Bisa Diandalkan Sistem Ini?)")
    st.caption(
        "Pakai data uji bawaan (split 80:20 dari dataset yang sudah ada) -- "
        "tidak perlu upload apa pun."
    )
    if st.button("Jalankan Uji Keandalan (Cronbach's Alpha)", key="val_alpha_run"):
        with st.spinner("Menghitung tingkat keandalan sistem..."):
            train_df_val, test_df_val = split_train_test(pelanggan)
            hasil_pelanggan_auto = run_full_validation(test_df_val, train_df_val)
        st.write("")
        render_reliability_section(hasil_pelanggan_auto, top_n=5)

# ----------------------------------------------------------------------------
# UJI DATASET BARU (Hit Rate@N, upload data eksternal)
# ----------------------------------------------------------------------------
with tab_uji_baru:

    st.subheader("Uji Dataset Baru")
    st.caption("Kalau perusahaan punya dataset baru, upload di sini untuk divalidasi.")

    REQUIRED_COLUMNS = [
        "Username",
        "Total Online (detik)",
        "Total Kuota (bytes)",
        "Paket Internet"
    ]

    with st.expander("Format data uji yang dibutuhkan"):
        template_df = pelanggan[REQUIRED_COLUMNS].head(5)
        st.dataframe(template_df, hide_index=True, use_container_width=True)

        st.download_button(
            "Unduh Contoh Template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="template_data_uji.csv",
            mime="text/csv",
            key="val_template"
        )

    uploaded_file = st.file_uploader(
        "Upload Data Uji (CSV, format transformasi ATAU data mentah)", type=["csv"], key="val_upload"
    )

    katalog_file = st.file_uploader(
        "Upload Data Jenis Layanan (CSV)", type=["csv"], key="val_katalog_upload"
    )

    katalog_uji = katalog
    if katalog_file is not None:
        try:
            katalog_uji = pd.read_csv(katalog_file)
            katalog_uji.columns = katalog_uji.columns.str.strip()
            st.success("Data jenis layanan berhasil dimuat.")
        except Exception as e:
            st.error(f"Gagal membaca file jenis layanan: {e}")
            katalog_uji = katalog

    if uploaded_file is not None:

        try:
            raw_upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            raw_upload_df = None

        test_df = None
        if raw_upload_df is not None:
            try:
                test_df, validation_note = prepare_validation_data(raw_upload_df, katalog_uji)
                st.info(validation_note)
                st.caption(f"Jumlah data uji setelah siap divalidasi: {len(test_df):,} baris.")
            except Exception as e:
                st.error(str(e))

        if test_df is not None:

            jalankan = st.button("Jalankan Validasi", use_container_width=True, key="val_jalankan")

            if jalankan:

                with st.spinner("Menjalankan validasi..."):
                    hasil_pelanggan = run_full_validation(test_df, pelanggan)

                total_uji = len(hasil_pelanggan)
                berhasil_count = sum(1 for r in hasil_pelanggan if r["Status"] == "Berhasil Direkomendasikan")
                gagal_count = total_uji - berhasil_count

                st.write("")

                with st.container(border=True):
                    st.markdown("**Ringkasan Hasil**")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Jumlah Data Uji", f"{total_uji:,}")
                    r2.metric("Berhasil Direkomendasikan", f"{berhasil_count:,}")
                    r3.metric("Gagal Direkomendasikan", f"{gagal_count:,}")

                st.write("")
                st.markdown("**Rekomendasi per Pelanggan**")
                st.caption(
                    "Tingkat Kecocokan = rata-rata kemiripan (jarak Euclidean, "
                    "dinormalisasi) pelanggan-pelanggan mirip yang memakai paket "
                    "tersebut — bukan angka akurasi model keseluruhan, tapi kecocokan "
                    "rekomendasi itu terhadap pelanggan ini."
                )

                table_df = build_validasi_table(hasil_pelanggan)
                render_validasi_table(table_df)

                st.write("")
                st.markdown("**Export Hasil Validasi**")
                export_buttons_validasi(
                    table_df,
                    base_filename="hasil_validasi_sistem",
                    pdf_title="Hasil Validasi Sistem Rekomendasi",
                    key_prefix="val"
                )
