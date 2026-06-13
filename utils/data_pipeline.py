"""
Data pipeline for the Disaster Resource Forecasting system.

Handles:
  - Loading raw CSV datasets
  - Cleaning & imputing missing values
  - Feature engineering (lags, rolling averages, seasonal indicators, risk scores)
  - Merging climate, demographic, and resource data into a single training-ready DataFrame
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RESOURCE_COLS = [
    "food_kits", "medical_kits", "ors_packets",
    "drinking_water_litres", "tarpaulins",
]

FEATURE_COLS_NUMERIC = [
    "rainfall_mm", "temperature_c", "humidity_pct",
    "flood_severity", "river_water_level_m",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_disaster_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load and parse historical disaster / climate CSV."""
    path = path or SAMPLE_DIR / "historical_disaster_data.csv"
    logger.info("Loading disaster data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def load_demographics(path: Optional[Path] = None) -> pd.DataFrame:
    """Load district-level demographic CSV."""
    path = path or SAMPLE_DIR / "demographics.csv"
    logger.info("Loading demographics from %s", path)
    df = pd.read_csv(path)
    logger.info("  Loaded %d districts", len(df))
    return df


def load_resource_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load historical resource allocation CSV."""
    path = path or SAMPLE_DIR / "historical_resources.csv"
    logger.info("Loading resource data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("  Loaded %d rows", len(df))
    return df


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
      - Numeric columns → forward-fill within each district, then median fill.
      - Categorical columns → fill with 'Unknown'.
    """
    logger.info("Cleaning data – handling missing values")
    df = df.copy()

    # Forward fill within each district group for time-series continuity
    if "district" in df.columns and "date" in df.columns:
        df.sort_values(["district", "date"], inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df.groupby("district")[numeric_cols].transform(
            lambda s: s.ffill().bfill()
        )

    # Remaining NaNs → column median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info("  Filled %d NaNs in '%s' with median %.2f",
                        df[col].isna().sum(), col, median_val)

    # Categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna("Unknown", inplace=True)

    return df


# ── Feature Engineering ─────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the 'date' column."""
    logger.info("Adding temporal features")
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 3, 7, 14],
) -> pd.DataFrame:
    """Create lag features for specified columns within each district."""
    logger.info("Adding lag features (lags=%s) for %d columns", lags, len(columns))
    df = df.copy()
    df.sort_values(["district", "date"], inplace=True)

    for col in columns:
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df.groupby("district")[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = [3, 7, 14, 30],
) -> pd.DataFrame:
    """Create rolling mean and std features within each district."""
    logger.info("Adding rolling features (windows=%s)", windows)
    df = df.copy()
    df.sort_values(["district", "date"], inplace=True)

    for col in columns:
        for w in windows:
            roll = df.groupby("district")[col].transform(
                lambda s: s.rolling(window=w, min_periods=1).mean()
            )
            df[f"{col}_roll_mean_{w}"] = roll

            roll_std = df.groupby("district")[col].transform(
                lambda s: s.rolling(window=w, min_periods=1).std()
            )
            df[f"{col}_roll_std_{w}"] = roll_std
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite flood risk score (0–100) combining:
      - Normalised rainfall
      - Flood severity
      - River water level
      - Monsoon flag
    """
    logger.info("Computing composite risk score")
    df = df.copy()

    rain_q99 = max(df["rainfall_mm"].quantile(0.99), 1)
    rain_norm = df["rainfall_mm"] / rain_q99
    sev_norm = df["flood_severity"] / 5.0
    water_q99 = max(df["river_water_level_m"].quantile(0.99), 1)
    water_norm = df["river_water_level_m"] / water_q99
    monsoon = df.get("is_monsoon", 0)

    df["risk_score"] = (
        0.35 * rain_norm + 0.30 * sev_norm + 0.25 * water_norm + 0.10 * monsoon
    ).clip(0, 1) * 100
    df["risk_score"] = df["risk_score"].round(2)

    return df


# ── Merge all datasets ──────────────────────────────────────────────────────

def merge_datasets(
    disaster_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge climate, resource, and demographic data on district (and date)."""
    logger.info("Merging datasets")
    merged = disaster_df.merge(resource_df, on=["date", "district"], how="left")
    merged = merged.merge(demo_df, on="district", how="left")
    logger.info("  Merged shape: %s", merged.shape)
    return merged


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_pipeline(save: bool = True) -> pd.DataFrame:
    """
    Execute the complete data pipeline:
      1. Load raw CSVs
      2. Clean
      3. Merge
      4. Feature engineering
      5. Optionally save processed output
    """
    logger.info("=" * 60)
    logger.info("Starting full data pipeline")
    logger.info("=" * 60)

    # 1 – Load
    disaster_df = load_disaster_data()
    resource_df = load_resource_data()
    demo_df = load_demographics()

    # 2 – Clean
    disaster_df = clean_data(disaster_df)
    resource_df = clean_data(resource_df)

    # 3 – Merge
    df = merge_datasets(disaster_df, resource_df, demo_df)

    # 4 – Feature engineering
    df = add_temporal_features(df)
    df = add_lag_features(df, FEATURE_COLS_NUMERIC)
    df = add_rolling_features(df, FEATURE_COLS_NUMERIC)
    df = add_risk_score(df)

    # Fill any remaining NaNs introduced by lags / rolling
    df = clean_data(df)

    # 5 – Save
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "features.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved processed features -> %s  (%d rows)", out_path, len(df))

    logger.info("Pipeline complete.")
    return df


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
