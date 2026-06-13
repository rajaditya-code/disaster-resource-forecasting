"""
Prediction service – loads trained models and generates resource forecasts.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.data_pipeline import (
    load_disaster_data, load_demographics, load_resource_data,
    RESOURCE_COLS, SAMPLE_DIR,
)

logger = get_logger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

# ── Singleton model cache ───────────────────────────────────────────────────

_models: dict = {}
_feature_names: list[str] = []
_demographics: pd.DataFrame | None = None
_latest_data: pd.DataFrame | None = None


def _load_models():
    """Load all trained models from disk into memory (once)."""
    global _models, _feature_names, _demographics, _latest_data

    if _models:
        return

    logger.info("Loading model artifacts from %s", ARTIFACTS_DIR)
    for target in RESOURCE_COLS:
        model_path = ARTIFACTS_DIR / f"{target}_model.joblib"
        if model_path.exists():
            _models[target] = joblib.load(model_path)
            logger.info("  Loaded model: %s", target)
        else:
            logger.warning("  Model file not found: %s", model_path)

    # Feature names
    fn_path = ARTIFACTS_DIR / "feature_names.json"
    if fn_path.exists():
        with open(fn_path) as f:
            _feature_names = json.load(f)
        logger.info("  Loaded %d feature names", len(_feature_names))

    # Demographics cache
    _demographics = load_demographics()

    # Latest data for each district (used for lag-based features)
    try:
        processed_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
        if processed_path.exists():
            _latest_data = pd.read_csv(processed_path, parse_dates=["date"])
            logger.info("  Loaded processed features (%d rows)", len(_latest_data))
    except Exception as e:
        logger.warning("  Could not load processed features: %s", e)

    logger.info("Model loading complete – %d models ready", len(_models))


def get_models() -> dict:
    _load_models()
    return _models


def get_feature_names() -> list[str]:
    _load_models()
    return _feature_names


def get_demographics() -> pd.DataFrame:
    _load_models()
    return _demographics


# ── Prediction logic ────────────────────────────────────────────────────────

def _build_feature_vector(
    district: str,
    forecast_horizon_days: int,
    rainfall_mm: float,
    temperature_c: float,
    humidity_pct: float,
    flood_severity: int,
    river_water_level_m: float,
) -> pd.DataFrame:
    """
    Build a feature vector matching the trained model's expected columns.

    Uses the latest historical data for the district to populate lag/rolling
    features, then overrides with the provided climate inputs.
    """
    _load_models()

    feature_dict: dict = {}

    # ── Primary climate features ──
    feature_dict["rainfall_mm"] = rainfall_mm
    feature_dict["temperature_c"] = temperature_c
    feature_dict["humidity_pct"] = humidity_pct
    feature_dict["flood_severity"] = flood_severity
    feature_dict["river_water_level_m"] = river_water_level_m

    # ── Temporal features ──
    import datetime as dt
    target_date = dt.date.today() + dt.timedelta(days=forecast_horizon_days)
    feature_dict["year"] = target_date.year
    feature_dict["month"] = target_date.month
    feature_dict["day_of_year"] = target_date.timetuple().tm_yday
    feature_dict["week_of_year"] = target_date.isocalendar()[1]
    feature_dict["is_monsoon"] = int(target_date.month in (6, 7, 8, 9))
    feature_dict["sin_month"] = np.sin(2 * np.pi * target_date.month / 12)
    feature_dict["cos_month"] = np.cos(2 * np.pi * target_date.month / 12)

    # ── Demographic features ──
    if _demographics is not None:
        row = _demographics[_demographics["district"] == district]
        if not row.empty:
            for col in ["population", "vulnerable_population", "area_sqkm",
                         "population_density", "literacy_rate", "hospitals",
                         "flood_shelters", "latitude", "longitude"]:
                if col in row.columns:
                    feature_dict[col] = row.iloc[0][col]

    # ── Risk score (simplified inline computation) ──
    rain_norm = min(rainfall_mm / 400, 1.0)
    sev_norm = flood_severity / 5.0
    water_norm = min(river_water_level_m / 8.0, 1.0)
    monsoon_flag = feature_dict.get("is_monsoon", 0)
    feature_dict["risk_score"] = round(
        (0.35 * rain_norm + 0.30 * sev_norm + 0.25 * water_norm + 0.10 * monsoon_flag) * 100,
        2,
    )

    # ── Lag & rolling features from historical data ──
    if _latest_data is not None:
        dist_data = _latest_data[_latest_data["district"] == district].sort_values("date")
        if not dist_data.empty:
            latest_row = dist_data.iloc[-1]
            for col_name in _feature_names:
                if col_name not in feature_dict and col_name in latest_row.index:
                    val = latest_row[col_name]
                    feature_dict[col_name] = val if pd.notna(val) else 0.0

    # ── Ensure all expected features are present ──
    for col_name in _feature_names:
        if col_name not in feature_dict:
            feature_dict[col_name] = 0.0

    # Build single-row DataFrame in the exact column order
    X = pd.DataFrame([feature_dict])[_feature_names]
    return X


def predict_resources(
    district: str,
    forecast_horizon_days: int = 7,
    rainfall_mm: float = 0.0,
    temperature_c: float = 30.0,
    humidity_pct: float = 60.0,
    flood_severity: int = 0,
    river_water_level_m: float = 3.0,
) -> dict:
    """
    Generate resource predictions for a district.

    Returns dict with predictions, risk info, confidence intervals,
    and a human-readable recommendation.
    """
    _load_models()

    X = _build_feature_vector(
        district, forecast_horizon_days, rainfall_mm,
        temperature_c, humidity_pct, flood_severity, river_water_level_m,
    )

    predictions: dict = {}
    confidence: dict = {}

    for target, model in _models.items():
        pred = float(model.predict(X)[0])
        pred = max(0, pred)

        # Scale by forecast horizon (model trained on daily data)
        scaled = pred * forecast_horizon_days
        predictions[target] = int(round(scaled))

        # Simple confidence interval: ±15%
        lower = int(round(scaled * 0.85))
        upper = int(round(scaled * 1.15))
        confidence[target] = {"lower": lower, "upper": upper}

    # Risk
    risk_score = float(X["risk_score"].iloc[0])
    if risk_score >= 70:
        risk_level = "Critical"
    elif risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 30:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Recommendation
    recommendation = _generate_recommendation(
        district, risk_level, predictions, forecast_horizon_days,
    )

    return {
        "district": district,
        "forecast_horizon_days": forecast_horizon_days,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "predictions": predictions,
        "confidence_interval": confidence,
        "recommendation": recommendation,
    }


def _generate_recommendation(
    district: str,
    risk_level: str,
    predictions: dict,
    horizon: int,
) -> str:
    """Generate a human-readable recommendation string."""
    if risk_level in ("Critical", "High"):
        urgency = "Immediate action required"
    elif risk_level == "Moderate":
        urgency = "Proactive measures recommended"
    else:
        urgency = "Maintain standard readiness"

    parts = [
        f"{district} district has {risk_level.lower()} flood risk. {urgency}.",
        f"Pre-position {predictions.get('food_kits', 0):,} food kits, "
        f"{predictions.get('medical_kits', 0):,} medical kits, "
        f"and {predictions.get('ors_packets', 0):,} ORS packets "
        f"within the next {horizon} days.",
    ]

    if predictions.get("drinking_water_litres", 0) > 10_000:
        parts.append(
            f"Ensure {predictions['drinking_water_litres']:,} litres of drinking water "
            "are available at distribution centres."
        )
    if predictions.get("tarpaulins", 0) > 500:
        parts.append(
            f"Deploy {predictions['tarpaulins']:,} tarpaulins to temporary shelters."
        )

    return " ".join(parts)


def get_all_district_risks() -> list[dict]:
    """Compute current risk scores for all districts (used by dashboard overview)."""
    _load_models()
    if _latest_data is None:
        return []

    results = []
    districts = _latest_data["district"].unique()
    for district in districts:
        dist_data = _latest_data[_latest_data["district"] == district].sort_values("date")
        if dist_data.empty:
            continue
        latest = dist_data.iloc[-1]
        risk = float(latest.get("risk_score", 0))

        if risk >= 70:
            level = "Critical"
        elif risk >= 50:
            level = "High"
        elif risk >= 30:
            level = "Moderate"
        else:
            level = "Low"

        demo_row = None
        if _demographics is not None:
            matches = _demographics[_demographics["district"] == district]
            if not matches.empty:
                demo_row = matches.iloc[0]

        results.append({
            "district": district,
            "risk_score": round(risk, 2),
            "risk_level": level,
            "latitude": float(demo_row["latitude"]) if demo_row is not None else 0,
            "longitude": float(demo_row["longitude"]) if demo_row is not None else 0,
            "population": int(demo_row["population"]) if demo_row is not None else 0,
        })

    return sorted(results, key=lambda r: r["risk_score"], reverse=True)
