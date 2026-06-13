"""
Test suite for the data pipeline.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_pipeline import (
    clean_data,
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_risk_score,
    RESOURCE_COLS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    districts = ["Patna", "Darbhanga"]

    rows = []
    for d in districts:
        for date in dates:
            rows.append({
                "date": date,
                "district": d,
                "rainfall_mm": np.random.uniform(0, 300),
                "temperature_c": np.random.uniform(15, 40),
                "humidity_pct": np.random.uniform(30, 95),
                "flood_severity": np.random.randint(0, 6),
                "river_water_level_m": np.random.uniform(1, 8),
            })
    return pd.DataFrame(rows)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_clean_data_fills_nans(sample_df):
    """clean_data should fill NaN values."""
    # Introduce NaNs
    df = sample_df.copy()
    df.loc[5, "rainfall_mm"] = np.nan
    df.loc[10, "temperature_c"] = np.nan

    cleaned = clean_data(df)
    assert cleaned["rainfall_mm"].isna().sum() == 0
    assert cleaned["temperature_c"].isna().sum() == 0


def test_temporal_features(sample_df):
    """add_temporal_features should add month, year, etc."""
    df = add_temporal_features(sample_df)
    assert "year" in df.columns
    assert "month" in df.columns
    assert "is_monsoon" in df.columns
    assert "sin_month" in df.columns
    assert "cos_month" in df.columns
    assert df["year"].iloc[0] == 2023


def test_lag_features(sample_df):
    """add_lag_features should create lagged columns."""
    df = add_lag_features(sample_df, ["rainfall_mm"], lags=[1, 3])
    assert "rainfall_mm_lag_1" in df.columns
    assert "rainfall_mm_lag_3" in df.columns


def test_rolling_features(sample_df):
    """add_rolling_features should create rolling mean/std columns."""
    df = add_rolling_features(sample_df, ["rainfall_mm"], windows=[3, 7])
    assert "rainfall_mm_roll_mean_3" in df.columns
    assert "rainfall_mm_roll_std_7" in df.columns


def test_risk_score(sample_df):
    """add_risk_score should compute a 0–100 risk score."""
    df = add_temporal_features(sample_df)
    df = add_risk_score(df)
    assert "risk_score" in df.columns
    assert df["risk_score"].min() >= 0
    assert df["risk_score"].max() <= 100


def test_pipeline_output_shape(sample_df):
    """Full feature pipeline should produce more columns than input."""
    df = add_temporal_features(sample_df)
    df = add_lag_features(df, ["rainfall_mm", "flood_severity"], lags=[1, 3])
    df = add_rolling_features(df, ["rainfall_mm"], windows=[3])
    df = add_risk_score(df)
    df = clean_data(df)

    assert len(df.columns) > len(sample_df.columns)
    assert df.isna().sum().sum() == 0
