from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_item_profile(catalog: pd.DataFrame) -> pd.DataFrame:
    profile = (
        catalog.drop_duplicates(subset=["Nama Paket"])
        .set_index("Nama Paket")[["Kecepatan", "Harga (Rp)"]]
        .copy()
    )
    scaler = MinMaxScaler()
    profile[["Kecepatan", "Harga (Rp)"]] = scaler.fit_transform(profile[["Kecepatan", "Harga (Rp)"]])
    return profile


def compute_item_similarity(item_profile: pd.DataFrame) -> pd.DataFrame:
    values = item_profile.values
    n = len(values)
    dist = np.zeros((n, n))
    for i in range(n):
        dist[i] = np.linalg.norm(values - values[i], axis=1)
    max_dist = dist.max() or 1
    sim = 1 - (dist / max_dist)
    return pd.DataFrame(sim, index=item_profile.index, columns=item_profile.index)


def recommend_content_based(
    username: str,
    pelanggan_df: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, str | None]:
    customer_rows = pelanggan_df[pelanggan_df["Username"] == username]
    if customer_rows.empty:
        return pd.DataFrame(), None

    current_package = customer_rows["Paket Internet"].iloc[0]
    wilayah = customer_rows["Wilayah Pemasaran"].iloc[0]

    if current_package not in item_similarity_df.index:
        return pd.DataFrame(), current_package

    package_info = (
        pelanggan_df.drop_duplicates(subset=["Paket Internet"])
        .set_index("Paket Internet")[["Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]]
    )

    sims = item_similarity_df[current_package].drop(index=current_package, errors="ignore")
    sims = sims.sort_values(ascending=False)

    rows = []
    for paket, skor in sims.items():
        if paket not in package_info.index:
            continue
        info = package_info.loc[paket]
        if info["Wilayah Pemasaran"] != wilayah:
            continue
        rows.append(
            {
                "Paket Internet": paket,
                "Skor Kemiripan": skor,
                "Tingkat Kecocokan (%)": round(float(np.clip(skor, 0, 1)) * 100, 2),
                "Wilayah Pemasaran": info["Wilayah Pemasaran"],
                "Kecepatan": info["Kecepatan"],
                "Harga (Rp)": info["Harga (Rp)"],
            }
        )
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows), current_package


def evaluate_mae_rmse(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_similarity_df: pd.DataFrame,
) -> dict:
    avg_rating_train = train_df.groupby("Paket Internet")["Implicit Rating"].mean()
    train_mean = train_df["Implicit Rating"].mean()

    errors = []
    detail_rows = []
    for _, row in test_df.iterrows():
        paket = row["Paket Internet"]
        actual = row["Implicit Rating"]

        if paket in item_similarity_df.index:
            sims = item_similarity_df.loc[paket, avg_rating_train.index.intersection(item_similarity_df.columns)]
            weights = sims.values
            values = avg_rating_train.loc[sims.index].values
            weight_total = weights.sum()
            predicted = float((weights * values).sum() / weight_total) if weight_total > 0 else train_mean
        else:
            predicted = train_mean

        error = predicted - actual
        errors.append(error)
        detail_rows.append(
            {
                "Username": row["Username"],
                "Paket Internet": paket,
                "Implicit Rating Aktual": actual,
                "Implicit Rating Prediksi": predicted,
                "Galat Absolut": abs(error),
            }
        )

    errors = np.array(errors, dtype=float)
    mae = float(np.mean(np.abs(errors))) if len(errors) else float("nan")
    rmse = float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "n_test": len(test_df),
        "detail": pd.DataFrame(detail_rows),
    }

