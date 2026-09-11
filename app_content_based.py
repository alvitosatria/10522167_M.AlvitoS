import pandas as pd
import plotly.express as px
import streamlit as st

import content_based as cb
import recommender as rec
from recommender import split_train_test

st.set_page_config(
    page_title="Content-Based Filtering — Perbandingan Metode",
    page_icon="📶",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    pelanggan = pd.read_csv("D_transformed.csv")
    katalog = pd.read_csv("Jenis_Layanan.csv")
    katalog.columns = katalog.columns.str.strip()
    katalog["Kecepatan"] = (
        katalog["Kecepatan"].astype(str).str.replace(" Mbps", "", regex=False).str.strip().astype(float)
    )
    katalog["Harga (Rp)"] = pd.to_numeric(katalog["Harga (Rp)"], errors="coerce")

    item_profile = cb.build_item_profile(katalog)
    item_similarity = cb.compute_item_similarity(item_profile)

    user_item_matrix = rec.build_user_item_matrix(pelanggan)
    user_similarity = rec.compute_user_similarity(user_item_matrix)

    return pelanggan, katalog, item_similarity, user_item_matrix, user_similarity


pelanggan, katalog, item_similarity, user_item_matrix, user_similarity = load_data()


@st.cache_data(show_spinner="Menghitung MAE & RMSE kedua metode...")
def run_mae_rmse_comparison():
    train_df, test_df = split_train_test(pelanggan)
    eval_cf = rec.evaluate_mae_rmse(train_df, test_df, top_k=10)
    eval_cb = cb.evaluate_mae_rmse(train_df, test_df, item_similarity)
    return train_df, test_df, eval_cf, eval_cb


st.title("📶 Perbandingan Metode: User-Based CF vs Content-Based Filtering")
st.caption(
    "Sesuai arahan pembimbing: membandingkan AKURASI dan HASIL REKOMENDASI antara "
    "User-Based Collaborative Filtering (metode utama di app.py, sesuai draft "
    "skripsi Bab III.3.4) dengan Content-Based Filtering (algoritma pembanding, "
    "kemiripan atribut paket)."
)

tab_rekom, tab_eval, tab_info = st.tabs(
    ["👤  Rekomendasi", "📊  Evaluasi", "ℹ️  Penjelasan"]
)

# ----------------------------------------------------------------------------
# REKOMENDASI
# ----------------------------------------------------------------------------
with tab_rekom:
    st.subheader("Perbandingan Hasil Rekomendasi")
    st.caption(
        "Pilih pelanggan untuk melihat rekomendasi dari KEDUA metode secara "
        "berdampingan, untuk pelanggan dan kriteria yang sama persis."
    )

    username = st.selectbox(
        "Username", sorted(pelanggan["Username"].unique()), key="cb_username"
    )
    cari = st.button("Cari Rekomendasi", key="cb_cari")

    if cari:
        data_pelanggan = pelanggan.loc[pelanggan["Username"] == username].iloc[0]
        current_package = data_pelanggan["Paket Internet"]

        st.write("")
        with st.container(border=True):
            st.markdown("**Data Pelanggan**")
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Username**<br>{username}", unsafe_allow_html=True)
            d2.markdown(f"**Paket Saat Ini**<br>{current_package}", unsafe_allow_html=True)
            d3.markdown(f"**Wilayah**<br>{data_pelanggan['Wilayah Pemasaran']}", unsafe_allow_html=True)

        st.write("")

        recs_cf, neighbors_cf = rec.recommend_existing_customer(
            username, pelanggan, user_item_matrix, user_similarity, top_k=10
        )
        recs_cb, _ = cb.recommend_content_based(username, pelanggan, item_similarity, top_n=5)

        col_cf, col_cb = st.columns(2)

        with col_cf:
            with st.container(border=True):
                st.markdown("**🔄 User-Based Collaborative Filtering**")
                if recs_cf.empty:
                    st.info("Tidak ada rekomendasi.")
                else:
                    top1_cf = recs_cf.iloc[0]
                    st.metric("Rekomendasi Utama", top1_cf["Paket Internet"])
                    m1, m2 = st.columns(2)
                    m1.metric("Estimasi Biaya", f"Rp {top1_cf['Harga (Rp)']:,.0f}")
                    m2.metric("Tingkat Kecocokan", f"{top1_cf['Tingkat Kecocokan (%)']:.1f}%")
                    if top1_cf["Sumber"] != "Collaborative Filtering":
                        st.caption("ℹ️ CF tidak punya tetangga lintas-paket — hasil dari fallback Hybrid.")
                    st.markdown("Semua rekomendasi:")
                    st.dataframe(
                        recs_cf[["Paket Internet", "Tingkat Kecocokan (%)", "Harga (Rp)"]],
                        hide_index=True, use_container_width=True,
                    )

        with col_cb:
            with st.container(border=True):
                st.markdown("**🧩 Content-Based Filtering**")
                if recs_cb.empty:
                    st.info("Tidak ada rekomendasi.")
                else:
                    top1_cb = recs_cb.iloc[0]
                    st.metric("Rekomendasi Utama", top1_cb["Paket Internet"])
                    m1, m2 = st.columns(2)
                    m1.metric("Estimasi Biaya", f"Rp {top1_cb['Harga (Rp)']:,.0f}")
                    m2.metric("Tingkat Kecocokan", f"{top1_cb['Tingkat Kecocokan (%)']:.1f}%")
                    st.markdown("Semua rekomendasi:")
                    st.dataframe(
                        recs_cb[["Paket Internet", "Tingkat Kecocokan (%)", "Harga (Rp)"]],
                        hide_index=True, use_container_width=True,
                    )

        st.write("")
        rekom_cf_set = set(recs_cf["Paket Internet"]) if not recs_cf.empty else set()
        rekom_cb_set = set(recs_cb["Paket Internet"]) if not recs_cb.empty else set()
        overlap = rekom_cf_set & rekom_cb_set
        if rekom_cf_set or rekom_cb_set:
            st.caption(
                f"💡 Dari rekomendasi kedua metode, {len(overlap)} paket muncul di "
                f"KEDUANYA: {', '.join(overlap) if overlap else '(tidak ada yang sama)'}."
            )

# ----------------------------------------------------------------------------
# PERBANDINGAN METODE
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluasi & Perbandingan Metode")

    st.markdown("#### 📈 MAE & RMSE (Perbandingan Langsung)")
    st.caption(
        "Split data latih:uji 80:20 (reproducible, sama persis dgn app.py). CF "
        "memprediksi rating lewat Weighted Sum dari tetangga pelanggan mirip "
        "(sesuai Bab III.3.5). Content-Based memprediksi rating dari rata-rata "
        "rating tiap paket di data latih, dibobot kemiripan ATRIBUT paket "
        "(bukan kemiripan pelanggan) — kedua metode diuji di data uji yang "
        "identik, sehingga MAE/RMSE-nya bisa dibandingkan langsung apple-to-apple."
    )

    train_df, test_df, eval_cf, eval_cb = run_mae_rmse_comparison()

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**🔄 Collaborative Filtering**")
            m1, m2 = st.columns(2)
            m1.metric("MAE", f"{eval_cf['mae']:.4f}")
            m2.metric("RMSE", f"{eval_cf['rmse']:.4f}")
    with c2:
        with st.container(border=True):
            st.markdown("**🧩 Content-Based Filtering**")
            m1, m2 = st.columns(2)
            m1.metric("MAE", f"{eval_cb['mae']:.4f}")
            m2.metric("RMSE", f"{eval_cb['rmse']:.4f}")

    st.write("")
    st.caption(f"Data Latih: {len(train_df):,} baris · Data Uji: {eval_cf['n_test']:,} baris (identik untuk kedua metode)")

    compare_df = pd.DataFrame(
        {
            "Metrik": ["MAE", "RMSE"],
            "Collaborative Filtering": [eval_cf["mae"], eval_cf["rmse"]],
            "Content-Based Filtering": [eval_cb["mae"], eval_cb["rmse"]],
        }
    )
    with st.container(border=True):
        st.markdown("**Tabel Perbandingan**")
        display_df = compare_df.copy()
        display_df["Collaborative Filtering"] = display_df["Collaborative Filtering"].apply(lambda x: f"{x:.4f}")
        display_df["Content-Based Filtering"] = display_df["Content-Based Filtering"].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        melted = compare_df.melt(id_vars="Metrik", var_name="Metode", value_name="Nilai")
        fig = px.bar(
            melted, x="Metrik", y="Nilai", color="Metode",
            barmode="group", text="Nilai",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detail Galat Prediksi — Collaborative Filtering"):
        st.dataframe(eval_cf["detail"], hide_index=True, use_container_width=True)
    with st.expander("Detail Galat Prediksi — Content-Based Filtering"):
        st.dataframe(eval_cb["detail"], hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------
# PENJELASAN
# ----------------------------------------------------------------------------
with tab_info:
    st.subheader("Penjelasan Metode")

    with st.container(border=True):
        st.markdown("### 🔄 Content-Based Filtering (metode di app ini)")
        st.markdown(
            "Merekomendasikan paket berdasar **kemiripan atribut** (Kecepatan, Harga) "
            "terhadap paket yang sedang dipakai pelanggan:\n\n"
            "1. Tiap paket direpresentasikan sebagai vektor `[Kecepatan, Harga]` yang "
            "dinormalisasi (Min-Max).\n"
            "2. Kemiripan antar paket dihitung dari **jarak Euclidean** antar vektor "
            "tersebut (bukan Cosine Similarity — untuk atribut yang saling berkorelasi "
            "seperti kecepatan & harga, Cosine Similarity mengukur kesamaan rasio, "
            "bukan kedekatan nilai, sehingga bisa salah merekomendasikan paket yang "
            "jauh berbeda spesifikasinya).\n"
            "3. Paket lain yang jaraknya paling dekat ke paket pelanggan saat ini, dan "
            "berada di wilayah yang sama, dijadikan rekomendasi.\n\n"
            "**Keunggulan dibanding Collaborative Filtering**: tidak butuh data pelanggan "
            "lain sama sekali (murni dari katalog), sehingga tidak terpengaruh masalah "
            "sparsitas data (pelanggan hanya tercatat memakai 1 paket) yang membuat CF "
            "murni sering tidak punya tetangga lintas-paket."
        )

    with st.container(border=True):
        st.markdown("### ✅ Evaluasi MAE & RMSE (Perbandingan Langsung)")
        st.markdown(
            "**Collaborative Filtering** (lihat app.py) memprediksi Implicit Rating lewat "
            "Weighted Sum dari tetangga pelanggan yang mirip pola pemakaiannya, sesuai "
            "Bab III.3.5 draft skripsi.\n\n"
            "**Content-Based Filtering** tidak punya konsep 'tetangga pelanggan', jadi "
            "prediksi rating-nya dibentuk berbeda: rata-rata Implicit Rating tiap paket "
            "di data latih, dibobot kemiripan ATRIBUT (Kecepatan, Harga) ke paket target "
            "— murni dari katalog, tidak melibatkan perilaku pelanggan lain sama sekali.\n\n"
            "Karena keduanya sama-sama menghasilkan angka prediksi Implicit Rating, dan "
            "diuji pada **split data latih:uji 80:20 yang identik**, MAE & RMSE kedua "
            "metode bisa dibandingkan langsung apple-to-apple — itu yang ditampilkan di "
            "tab **Evaluasi**."
        )

    st.info(
        "💡 Jalankan evaluasinya di tab **Evaluasi** — dihitung langsung (bukan angka "
        "statis) supaya selalu konsisten dengan data terbaru."
    )
