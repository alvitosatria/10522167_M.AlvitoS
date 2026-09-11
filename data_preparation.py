from __future__ import annotations

import re

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from recommender import split_train_test


def _print_step(nomor: str, judul: str) -> None:
    print("-" * 70)
    print(f"{nomor}  {judul}")
    print("-" * 70)


# ----------------------------------------------------------------------------
# 3.3.3.1  Feature Selection
# ----------------------------------------------------------------------------

RETAINED_FEATURES = [
    "Username",
    "Paket Internet",
    "Total Online (detik)",
    "Total Kuota (bytes)",
    "Alamat",
]

_RAW_TEXT_COLS = ["Total Waktu Online", "Total Pemakaian Kuota"]


def select_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    return df_raw[RETAINED_FEATURES].copy()


# ----------------------------------------------------------------------------
# 3.3.3.2  Data Cleaning
# ----------------------------------------------------------------------------

def clean_missing(df_selected: pd.DataFrame) -> pd.DataFrame:
    required = ["Username", "Paket Internet", "Total Online (detik)", "Total Kuota (bytes)", "Alamat"]
    cleaned = df_selected.dropna(subset=required).copy()

    aggregated = (
        cleaned.groupby(["Username", "Paket Internet"], as_index=False)
        .agg({"Total Online (detik)": "sum", "Total Kuota (bytes)": "sum", "Alamat": "first"})
    )
    return aggregated


# ----------------------------------------------------------------------------
# 3.3.3.3  Data Parsing (persamaan 6)
# ----------------------------------------------------------------------------

_DURATION_PATTERN = re.compile(
    r"(?:(?P<hari>\d+)\s*hari)?\s*"
    r"(?:(?P<jam>\d+)\s*jam)?\s*"
    r"(?:(?P<menit>\d+)\s*menit)?\s*"
    r"(?:(?P<detik>\d+)\s*detik)?",
    re.IGNORECASE,
)

_QUOTA_PATTERN = re.compile(r"([\d.,]+)\s*(TB|GB|MB|KB|B)", re.IGNORECASE)

_QUOTA_UNIT_BYTES = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


def parse_duration_to_seconds(text) -> float:
    if pd.isna(text):
        return 0.0
    match = _DURATION_PATTERN.search(str(text))
    if not match:
        return 0.0
    hari = int(match.group("hari") or 0)
    jam = int(match.group("jam") or 0)
    menit = int(match.group("menit") or 0)
    detik = int(match.group("detik") or 0)
    return float(hari * 86400 + jam * 3600 + menit * 60 + detik)


def parse_quota_to_bytes(text) -> float:
    if pd.isna(text):
        return 0.0
    match = _QUOTA_PATTERN.search(str(text))
    if not match:
        return 0.0
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).upper()
    return value * _QUOTA_UNIT_BYTES.get(unit, 1)


def verify_parsing(df_raw: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    if not all(col in df_raw.columns for col in _RAW_TEXT_COLS):
        return pd.DataFrame()

    parsed_online = df_raw["Total Waktu Online"].apply(parse_duration_to_seconds)
    parsed_kuota = df_raw["Total Pemakaian Kuota"].apply(parse_quota_to_bytes)

    online_diff = (parsed_online - df_raw["Total Online (detik)"]).abs()
    online_rel = online_diff / df_raw["Total Online (detik)"].replace(0, pd.NA)

    kuota_diff = (parsed_kuota - df_raw["Total Kuota (bytes)"]).abs()
    kuota_rel = kuota_diff / df_raw["Total Kuota (bytes)"].replace(0, pd.NA)

    summary = pd.DataFrame(
        {
            "Username": df_raw["Username"],
            "Online (detik) - parsed": parsed_online,
            "Online (detik) - asli": df_raw["Total Online (detik)"],
            "Selisih relatif Online": online_rel,
            "Kuota (bytes) - parsed": parsed_kuota,
            "Kuota (bytes) - asli": df_raw["Total Kuota (bytes)"],
            "Selisih relatif Kuota": kuota_rel,
        }
    )
    summary["Konsisten"] = (
        summary["Selisih relatif Online"].fillna(0) <= tolerance
    ) & (summary["Selisih relatif Kuota"].fillna(0) <= tolerance)
    return summary


# ----------------------------------------------------------------------------
# 3.3.3.4  Data Integration
#          - Data Mapping (Alamat -> Wilayah)
#          - Inner Join ke Katalog Layanan (persamaan 6/inner join di draft)
# ----------------------------------------------------------------------------

_WILAYAH_1 = "PWS, Sudirman, Tigaraksa"
_WILAYAH_2 = "Pinang, Pabuaran, Katomas, Leuwihalu"

_WILAYAH_1_KEYWORDS = ("pws", "tigaraksa", "margasari", "triraksa", "sudirman")
_WILAYAH_2_KEYWORDS = ("pinang", "pabuaran", "katomas", "leuwihalu")


def map_wilayah(alamat) -> str | None:
    if pd.isna(alamat):
        return None
    text = str(alamat).lower()
    if any(keyword in text for keyword in _WILAYAH_1_KEYWORDS):
        return _WILAYAH_1
    if any(keyword in text for keyword in _WILAYAH_2_KEYWORDS):
        return _WILAYAH_2
    return None


def integrate_catalog(df_clean: pd.DataFrame, katalog: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    df["Wilayah (Mapping Alamat)"] = df["Alamat"].apply(map_wilayah)

    katalog = katalog.copy()
    katalog.columns = katalog.columns.str.strip()
    katalog["Kecepatan"] = (
        katalog["Kecepatan"].astype(str).str.replace(" Mbps", "", regex=False).str.strip().astype(float)
    )
    katalog["Harga (Rp)"] = pd.to_numeric(katalog["Harga (Rp)"], errors="coerce")

    integrated = pd.merge(
        df,
        katalog[["Nama Paket", "Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]],
        left_on="Paket Internet",
        right_on="Nama Paket",
        how="inner",
    )

    n_unmapped = integrated["Wilayah (Mapping Alamat)"].isna().sum()
    n_mismatch = (
        integrated["Wilayah (Mapping Alamat)"].notna()
        & (integrated["Wilayah (Mapping Alamat)"] != integrated["Wilayah Pemasaran"])
    ).sum()
    if n_unmapped or n_mismatch:
        print(
            f"[integrate_catalog] Data Mapping Alamat: {n_unmapped} baris tidak cocok kata kunci "
            f"manapun, {n_mismatch} baris hasil mapping BEDA dengan wilayah resmi paket "
            "(nama tempat ambigu lintas wilayah). Kolom 'Wilayah Pemasaran' (dari katalog) "
            "tetap dipakai sebagai sumber kebenaran untuk filter rekomendasi."
        )

    return integrated


# ----------------------------------------------------------------------------
# 3.3.3.5  Data Transformation (Min-Max Normalization)
# ----------------------------------------------------------------------------

def transform_features(df_integrated: pd.DataFrame) -> pd.DataFrame:
    df = df_integrated.copy()
    scaler = MinMaxScaler()
    df[["Total Online (detik)", "Total Kuota (bytes)"]] = scaler.fit_transform(
        df[["Total Online (detik)", "Total Kuota (bytes)"]]
    )
    return df


# ----------------------------------------------------------------------------
# 3.3.3.6  Pembentukan Rating Implicit (Data Construction)
# ----------------------------------------------------------------------------

def construct_implicit_rating(df_transformed: pd.DataFrame) -> pd.DataFrame:
    df = df_transformed.copy()
    df["Implicit Rating"] = 0.5 * df["Total Online (detik)"] + 0.5 * df["Total Kuota (bytes)"]
    return df


# ----------------------------------------------------------------------------
# Orkestrasi — Output Akhir Data Preparation
# ----------------------------------------------------------------------------

def run_pipeline(usage_csv: str, catalog_csv: str, verbose: bool = False):
    df_raw = pd.read_csv(usage_csv)
    df_raw.columns = df_raw.columns.str.strip()
    katalog = pd.read_csv(catalog_csv)
    katalog.columns = katalog.columns.str.strip()

    if verbose:
        _print_step("0.", "DATA MENTAH (SEBELUM DATA PREPARATION)")
        print(f"Jumlah Baris : {df_raw.shape[0]}")
        print(f"Jumlah Kolom : {df_raw.shape[1]}")
        print("\nDaftar Kolom Dataset:")
        for col in df_raw.columns:
            print("-", col)
        print()

    if verbose:
        _print_step("3.3.3.1", "FEATURE SELECTION")
    df_selected = select_features(df_raw)
    if verbose:
        print(f"Kolom dipertahankan ({len(df_selected.columns)}):")
        for col in df_selected.columns:
            print("-", col)
        print(f"\nJumlah Baris : {df_selected.shape[0]}")
        print(f"Jumlah Kolom : {df_selected.shape[1]}")
        print("\nContoh Data:")
        print(df_selected.head().to_string())
        print()

    if verbose:
        _print_step("3.3.3.2", "DATA CLEANING")
        print("Jumlah Missing Value per Kolom (sebelum cleaning):")
        print(df_selected.isnull().sum().to_string())
        print(f"\nJumlah Data Duplikat (baris identik): {df_selected.duplicated().sum()}")
    df_clean = clean_missing(df_selected)
    if verbose:
        print(f"\nJumlah Baris sebelum cleaning : {df_selected.shape[0]}")
        print(f"Jumlah Baris setelah cleaning : {df_clean.shape[0]}")
        print("(baris berkurang karena missing value AND duplikat "
              "Username+Paket Internet diagregasi jadi satu baris)")
        print()

    if verbose:
        _print_step("3.3.3.3", "DATA PARSING (PEMBUKTIAN PERSAMAAN 6)")
        verify_df = verify_parsing(df_raw)
        if verify_df.empty:
            print("Kolom teks mentah (Total Waktu Online / Total Pemakaian Kuota) "
                  "tidak ditemukan di sumber data ini, verifikasi dilewati.")
        else:
            n_konsisten = int(verify_df["Konsisten"].sum())
            print(f"Baris konsisten (parser teks vs kolom numerik) : {n_konsisten}/{len(verify_df)}")
            print("\nContoh Hasil Parsing:")
            print(verify_df.head().to_string())
        print()

    if verbose:
        _print_step("3.3.3.4", "DATA INTEGRATION (DATA MAPPING + INNER JOIN)")
    df_integrated = integrate_catalog(df_clean, katalog)
    if verbose:
        print(f"Jumlah Baris sebelum integrasi : {df_clean.shape[0]}")
        print(f"Jumlah Baris setelah integrasi : {df_integrated.shape[0]}")
        print("\nContoh Data Setelah Integrasi:")
        print(
            df_integrated[
                ["Username", "Paket Internet", "Wilayah (Mapping Alamat)", "Wilayah Pemasaran", "Kecepatan", "Harga (Rp)"]
            ].head().to_string()
        )
        print()

    if verbose:
        _print_step("3.3.3.5", "DATA TRANSFORMATION (MIN-MAX NORMALIZATION)")
        print("Sebelum normalisasi:")
        print(df_integrated[["Total Online (detik)", "Total Kuota (bytes)"]].describe().loc[["min", "max"]].to_string())
    df_scaled = transform_features(df_integrated)
    if verbose:
        print("\nSetelah normalisasi (skala 0-1):")
        print(df_scaled[["Total Online (detik)", "Total Kuota (bytes)"]].describe().loc[["min", "max"]].to_string())
        print()

    if verbose:
        _print_step("3.3.3.6", "PEMBENTUKAN RATING IMPLICIT (DATA CONSTRUCTION)")
    df_transformed = construct_implicit_rating(df_scaled)
    if verbose:
        print("Contoh Implicit Rating:")
        print(
            df_transformed[
                ["Username", "Paket Internet", "Total Online (detik)", "Total Kuota (bytes)", "Implicit Rating"]
            ].head().to_string()
        )
        print()

    final_cols = [
        "Username",
        "Paket Internet",
        "Total Online (detik)",
        "Total Kuota (bytes)",
        "Nama Paket",
        "Wilayah Pemasaran",
        "Wilayah (Mapping Alamat)",
        "Kecepatan",
        "Harga (Rp)",
        "Implicit Rating",
    ]
    df_transformed = df_transformed[final_cols]

    if verbose:
        _print_step("OUTPUT", "OUTPUT AKHIR DATA PREPARATION")
        print(f"Jumlah Baris : {df_transformed.shape[0]}")
        print(f"Jumlah Kolom : {df_transformed.shape[1]}")
        print("\nContoh Dataset Analitik Final:")
        print(df_transformed.head().to_string())
        print()

    train_df, test_df = split_train_test(df_transformed)
    if verbose:
        _print_step("3.3.4.1", "PEMBAGIAN DATA (DATA SPLITTING 80:20)")
        print(f"Total data analitik : {len(df_transformed)}")
        print(f"Data latih (80%)    : {len(train_df)}")
        print(f"Data uji (20%)      : {len(test_df)}")
        print()

    return df_transformed, train_df, test_df


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    df_transformed, train_df, test_df = run_pipeline(
        "usage_summary asli.csv", "Jenis_Layanan.csv", verbose=True
    )
    df_transformed.to_csv("D_transformed.csv", index=False)
    print("D_transformed.csv berhasil diregenerate.")
