from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def _print_step(nomor: str, judul: str) -> None:
    print("-" * 70)
    print(f"{nomor}  {judul}")
    print("-" * 70)


# ----------------------------------------------------------------------------
# 3.3.4.1  Pembagian Data (Data Splitting)
# ----------------------------------------------------------------------------

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(df, test_size=test_size, random_state=random_state)


# ----------------------------------------------------------------------------
# 3.3.4.2  Pembentukan User-Item Matrix
# ----------------------------------------------------------------------------

def build_user_item_matrix(df: pd.DataFrame, user_col: str = "Username") -> pd.DataFrame:
    return df.pivot_table(
        index=user_col,
        columns="Paket Internet",
        values="Implicit Rating",
        aggfunc="mean",
        fill_value=0,
    )


# ----------------------------------------------------------------------------
# 3.3.4.3  Perhitungan Jarak Kemiripan (Cosine Similarity)
# ----------------------------------------------------------------------------

def compute_user_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    sim = cosine_similarity(matrix.values)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


# ----------------------------------------------------------------------------
# 3.3.4.4  Prediksi Skor (Weighted Sum)
# ----------------------------------------------------------------------------

def predict_scores(
    username: str,
    matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.Series:
    if username not in similarity_df.index:
        return pd.Series(dtype=float)

    neighbors = (
        similarity_df[username]
        .drop(index=username, errors="ignore")
        .sort_values(ascending=False)
        .head(top_k)
    )
    neighbors = neighbors[neighbors > 0]
    if neighbors.empty:
        return pd.Series(dtype=float)

    neighbor_ratings = matrix.loc[neighbors.index]
    weights = neighbors.values.reshape(-1, 1)

    weighted_sum = (neighbor_ratings.values * weights).sum(axis=0)
    weight_total = np.abs(weights).sum()
    predicted = weighted_sum / weight_total if weight_total > 0 else np.zeros_like(weighted_sum)

    predicted_series = pd.Series(predicted, index=matrix.columns)

    used_by_neighbor = (neighbor_ratings > 0).any(axis=0)
    return predicted_series[used_by_neighbor]


# ----------------------------------------------------------------------------
# 3.3.4.5  Pembangkitan Rekomendasi (Top-N Recommendation)
# ----------------------------------------------------------------------------

def recommend_existing_customer(
    username: str,
    df: pd.DataFrame,
    matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_rows = df[df["Username"] == username]
    if customer_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    owned_packages = set(customer_rows["Paket Internet"].unique())
    customer_wilayah = customer_rows["Wilayah Pemasaran"].iloc[0]

    predicted = predict_scores(username, matrix, similarity_df, top_k=top_k)

    package_info = (
        df.drop_duplicates(subset=["Paket Internet"])
        .set_index("Paket Internet")[["Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]]
    )

    rows = []
    for paket, skor in predicted.items():
        if paket in owned_packages:
            continue
        info = package_info.loc[paket] if paket in package_info.index else None
        wilayah_paket = info["Wilayah Pemasaran"] if info is not None else None
        if wilayah_paket != customer_wilayah:
            continue
        rows.append(
            {
                "Paket Internet": paket,
                "Skor Prediksi": skor,
                "Tingkat Kecocokan (%)": round(float(np.clip(skor, 0, 1)) * 100, 2),
                "Wilayah Pemasaran": wilayah_paket,
                "Kecepatan": info["Kecepatan"] if info is not None else None,
                "Harga (Rp)": info["Harga (Rp)"] if info is not None else None,
                "Sumber": "Collaborative Filtering",
            }
        )

    rec_cols = [
        "Paket Internet", "Skor Prediksi", "Tingkat Kecocokan (%)",
        "Wilayah Pemasaran", "Kecepatan", "Harga (Rp)", "Sumber",
    ]
    if rows:
        recommendations = pd.DataFrame(rows).sort_values("Skor Prediksi", ascending=False).reset_index(drop=True)
    else:
        first_owned = list(owned_packages)[0] if owned_packages else None
        current_price = package_info.loc[first_owned, "Harga (Rp)"] if first_owned in package_info.index else 0
        current_speed = package_info.loc[first_owned, "Kecepatan"] if first_owned in package_info.index else 0

        same_region = (
            df.drop_duplicates(subset=["Paket Internet"])[
                ["Paket Internet", "Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]
            ]
            .query("`Wilayah Pemasaran` == @customer_wilayah")
        )
        same_region = same_region[~same_region["Paket Internet"].isin(owned_packages)].copy()

        if same_region.empty:
            recommendations = pd.DataFrame(columns=rec_cols)
        else:
            harga_range = same_region["Harga (Rp)"].max() - same_region["Harga (Rp)"].min() or 1
            speed_range = same_region["Kecepatan"].max() - same_region["Kecepatan"].min() or 1
            same_region["Skor Total"] = 1 - (
                0.5 * (same_region["Harga (Rp)"] - current_price).abs() / harga_range
                + 0.5 * (same_region["Kecepatan"] - current_speed).abs() / speed_range
            )
            same_region = same_region.sort_values("Skor Total", ascending=False)

            recommendations = pd.DataFrame(
                {
                    "Paket Internet": same_region["Paket Internet"],
                    "Skor Prediksi": same_region["Skor Total"].clip(lower=0),
                    "Tingkat Kecocokan (%)": (same_region["Skor Total"].clip(lower=0) * 100).round(2),
                    "Wilayah Pemasaran": same_region["Wilayah Pemasaran"],
                    "Kecepatan": same_region["Kecepatan"],
                    "Harga (Rp)": same_region["Harga (Rp)"],
                    "Sumber": "Fallback Hybrid (Kedekatan Profil Paket — CF tidak punya tetangga lintas-paket)",
                }
            ).reset_index(drop=True)

    neighbors = (
        similarity_df[username]
        .drop(index=username, errors="ignore")
        .sort_values(ascending=False)
        .head(top_k)
    )
    neighbor_table = (
        df[df["Username"].isin(neighbors.index)][["Username", "Paket Internet", "Wilayah Pemasaran"]]
        .drop_duplicates()
        .merge(neighbors.rename("Similarity"), left_on="Username", right_index=True)
        .sort_values("Similarity", ascending=False)
        .reset_index(drop=True)
    )
    return recommendations, neighbor_table


# ----------------------------------------------------------------------------
# 3.3.4.6  Penanganan Cold Start Problem (Rule-Based Filtering)
# ----------------------------------------------------------------------------

def recommend_new_customer(
    catalog: pd.DataFrame,
    wilayah: str,
    budget: float,
    min_speed: float,
) -> pd.DataFrame:
    filtered = catalog[
        catalog["Wilayah Pemasaran"].astype(str).str.contains(wilayah, case=False, na=False)
    ].copy()
    filtered = filtered[(filtered["Harga (Rp)"] <= budget) & (filtered["Kecepatan"] >= min_speed)]

    if filtered.empty:
        return filtered

    filtered["Skor Budget"] = 1 - (filtered["Harga (Rp)"] - budget).abs() / max(float(budget), 1)
    filtered["Skor Budget"] = filtered["Skor Budget"].clip(lower=0)
    filtered["Skor Kecepatan"] = filtered["Kecepatan"] / filtered["Kecepatan"].max()
    filtered["Skor Total"] = 0.6 * filtered["Skor Budget"] + 0.4 * filtered["Skor Kecepatan"]
    return filtered.sort_values("Skor Total", ascending=False)


def recommend_new_customer_relaxed(
    catalog: pd.DataFrame,
    wilayah: str,
    budget: float,
    min_speed: float,
    top_n: int = 5,
) -> pd.DataFrame:
    same_region = catalog[
        catalog["Wilayah Pemasaran"].astype(str).str.contains(wilayah, case=False, na=False)
    ].copy()

    if same_region.empty:
        return same_region

    over_budget = (same_region["Harga (Rp)"] - budget).clip(lower=0)
    harga_range = same_region["Harga (Rp)"].max() - budget
    harga_range = harga_range if harga_range > 0 else 1
    same_region["Skor Budget"] = (1 - over_budget / harga_range).clip(lower=0)
    same_region.loc[same_region["Harga (Rp)"] <= budget, "Skor Budget"] = 1.0

    under_speed = (min_speed - same_region["Kecepatan"]).clip(lower=0)
    speed_range = min_speed - same_region["Kecepatan"].min()
    speed_range = speed_range if speed_range > 0 else 1
    same_region["Skor Kecepatan"] = (1 - under_speed / speed_range).clip(lower=0)
    same_region.loc[same_region["Kecepatan"] >= min_speed, "Skor Kecepatan"] = 1.0

    same_region["Skor Total"] = 0.6 * same_region["Skor Budget"] + 0.4 * same_region["Skor Kecepatan"]

    return same_region.sort_values("Skor Total", ascending=False).head(top_n)


# ----------------------------------------------------------------------------
# 3.3.5  Evaluasi — Mean Absolute Error (MAE) & Root Mean Square Error (RMSE)
# ----------------------------------------------------------------------------

def evaluate_mae_rmse(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int = 10) -> dict:
    detail = _predict_test_scores(train_df, test_df, top_k=top_k)
    detail["Galat Absolut"] = detail["Galat"].abs()
    errors = detail["Galat"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(errors))) if len(errors) else float("nan")
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "n_test": len(test_df),
        "detail": detail.drop(columns=["Galat"]),
    }


def _predict_test_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_matrix = build_user_item_matrix(full_df)
    full_similarity = compute_user_similarity(full_matrix)
    overall_mean = full_df["Implicit Rating"].mean()

    rows = []
    for _, row in test_df.iterrows():
        username = row["Username"]
        paket = row["Paket Internet"]
        actual = row["Implicit Rating"]

        if username in full_matrix.index:
            predicted_all = predict_scores(username, full_matrix, full_similarity, top_k=top_k)
            if paket in predicted_all.index:
                predicted = predicted_all[paket]
            else:
                predicted = overall_mean
        else:
            predicted = overall_mean

        rows.append(
            {
                "Username": username,
                "Paket Internet": paket,
                "Implicit Rating Aktual": actual,
                "Implicit Rating Prediksi": predicted,
                "Galat": predicted - actual,
            }
        )

    return pd.DataFrame(rows)


def evaluate_mae(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int = 10) -> dict:
    detail = _predict_test_scores(train_df, test_df, top_k=top_k)
    detail["Galat Absolut"] = detail["Galat"].abs()

    mae = float(detail["Galat Absolut"].mean()) if len(detail) else float("nan")

    return {
        "mae": mae,
        "n_test": len(test_df),
        "detail": detail.drop(columns=["Galat"]),
    }


def evaluate_rmse(train_df: pd.DataFrame, test_df: pd.DataFrame, top_k: int = 10) -> dict:
    detail = _predict_test_scores(train_df, test_df, top_k=top_k)
    detail["Kuadrat Galat"] = detail["Galat"] ** 2

    rmse = float(np.sqrt(detail["Kuadrat Galat"].mean())) if len(detail) else float("nan")

    return {
        "rmse": rmse,
        "n_test": len(test_df),
        "detail": detail.drop(columns=["Galat"]),
    }


# ----------------------------------------------------------------------------
# VALIDASI SISTEM TAMBAHAN — Cronbach's Alpha (permintaan pembimbing)
# ----------------------------------------------------------------------------

def evaluate_rank_confidence(
    test_df: pd.DataFrame,
    pelanggan_df: pd.DataFrame,
    top_n: int = 5,
    neighbor_pool: int = 30,
) -> pd.DataFrame:
    feature_cols = ["Total Online (detik)", "Total Kuota (bytes)", "Implicit Rating"]
    train_profile = pelanggan_df.set_index("Username")[feature_cols]

    rows = []
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

        result_row = {"Username": row["Username"], "Paket Saat Ini": row["Paket Internet"]}
        rank_paket_saat_ini = None
        for i, (paket, skor) in enumerate(confidence.items(), start=1):
            result_row[f"Peringkat {i} (%)"] = round(float(skor) * 100, 2)
            if paket == row["Paket Internet"]:
                rank_paket_saat_ini = i
        result_row["Paket Saat Ini di Peringkat"] = rank_paket_saat_ini
        rows.append(result_row)

    return pd.DataFrame(rows)


def compute_cronbach_alpha(item_df: pd.DataFrame) -> float:
    k = item_df.shape[1]
    item_variances = item_df.var(axis=0, ddof=1)
    total_scores = item_df.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_variances.sum() / total_variance))


def interpret_cronbach_alpha(alpha: float) -> str:
    if alpha != alpha:
        return "Tidak dapat dihitung"
    if alpha >= 0.9:
        return "Sangat Baik (Excellent)"
    if alpha >= 0.8:
        return "Baik (Good)"
    if alpha >= 0.7:
        return "Dapat Diterima (Acceptable)"
    if alpha >= 0.6:
        return "Diragukan (Questionable)"
    if alpha >= 0.5:
        return "Buruk (Poor)"
    return "Tidak Dapat Diterima (Unacceptable)"


if __name__ == "__main__":
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    csv_path = "D_transformed.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan. Jalankan 'python data_preparation.py' "
              "dulu untuk membuat dataset analitik hasil Data Preparation.")
        raise SystemExit(1)

    df_transformed = pd.read_csv(csv_path)

    _print_step("3.3.4.1", "PEMBAGIAN DATA (DATA SPLITTING 80:20)")
    train_df, test_df = split_train_test(df_transformed)
    print(f"Total data analitik : {len(df_transformed)}")
    print(f"Data latih (80%)    : {len(train_df)}")
    print(f"Data uji (20%)      : {len(test_df)}")
    print()

    _print_step("3.3.4.2", "PEMBENTUKAN USER-ITEM MATRIX")
    matrix = build_user_item_matrix(train_df)
    print(f"Ukuran Matrix : {matrix.shape[0]} pelanggan x {matrix.shape[1]} paket")
    print("\nContoh Matrix (5 pelanggan pertama, 5 kolom pertama):")
    print(matrix.iloc[:5, :5].to_string())
    print()

    _print_step("3.3.4.3", "PERHITUNGAN JARAK KEMIRIPAN (COSINE SIMILARITY)")
    similarity = compute_user_similarity(matrix)
    sample_user = matrix.index[0]
    print(f"Contoh kemiripan pelanggan '{sample_user}' terhadap 5 pelanggan lain:")
    print(similarity.loc[sample_user].sort_values(ascending=False).head(6).to_string())
    print()

    _print_step("3.3.4.4", "PREDIKSI SKOR (WEIGHTED SUM)")
    predicted = predict_scores(sample_user, matrix, similarity, top_k=10)
    print(f"Contoh prediksi skor untuk pelanggan '{sample_user}':")
    if predicted.empty:
        print("(tidak ada kandidat — tidak ada tetangga dengan paket berbeda)")
    else:
        print(predicted.sort_values(ascending=False).head(10).to_string())
    print()

    _print_step("3.3.4.5", "PEMBANGKITAN REKOMENDASI (TOP-N RECOMMENDATION)")
    recommendations, neighbor_table = recommend_existing_customer(
        sample_user, train_df, matrix, similarity, top_k=10
    )
    print(f"Rekomendasi Top-N untuk pelanggan '{sample_user}':")
    print(recommendations.to_string() if not recommendations.empty else "(tidak ada rekomendasi)")
    print()

    _print_step("3.3.4.6", "PENANGANAN COLD START PROBLEM (RULE-BASED FILTERING)")
    catalog_demo = train_df.drop_duplicates(subset=["Paket Internet"])[
        ["Paket Internet", "Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]
    ]
    sample_wilayah = catalog_demo["Wilayah Pemasaran"].iloc[0]
    demo_budget = 150000
    demo_speed = 5
    print(
        f"Contoh calon pelanggan baru: Budget=Rp{demo_budget:,}, "
        f"MinSpeed={demo_speed}Mbps, Wilayah='{sample_wilayah}'"
    )
    new_customer_reco = recommend_new_customer(catalog_demo, sample_wilayah, demo_budget, demo_speed)
    print(new_customer_reco.to_string() if not new_customer_reco.empty else "(tidak ada paket yang lolos filter ketat)")
    print()

    _print_step("3.3.5", "EVALUASI (MEAN ABSOLUTE ERROR & ROOT MEAN SQUARE ERROR)")
    hasil_evaluasi = evaluate_mae_rmse(train_df, test_df)
    print(f"MAE  : {hasil_evaluasi['mae']:.4f}")
    print(f"RMSE : {hasil_evaluasi['rmse']:.4f}")
    print(f"Jumlah data uji : {hasil_evaluasi['n_test']}")
    print("\nContoh Detail Galat Prediksi:")
    print(hasil_evaluasi["detail"].head().to_string())
