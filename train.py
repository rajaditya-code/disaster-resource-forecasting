"""
Model training script for the Disaster Resource Forecasting system.

Trains one LightGBM regressor per resource target, evaluates with MAE/RMSE/MAPE,
and saves model artifacts + metrics to model_artifacts/.
"""

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger
from utils.data_pipeline import run_pipeline, RESOURCE_COLS

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns to exclude from model features
DROP_COLS = [
    "date", "district",
    # Targets are excluded individually per model
] + RESOURCE_COLS

# LightGBM hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 30,
    "random_state": 42,
}


# ── Metrics ──────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (handles zero actuals gracefully)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, and MAPE for a prediction."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


# ── Training ─────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, target: str):
    """Split a processed DataFrame into X and y, dropping non-feature columns."""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    # Keep only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target].values
    return X, y


def train_single_model(
    df: pd.DataFrame,
    target: str,
) -> dict:
    """
    Train a LightGBM model for a single resource target.

    Uses the most recent 20% of data as holdout to respect temporal ordering.

    Returns:
        Dict with model, metrics, feature importance, and feature names.
    """
    logger.info("-" * 50)
    logger.info("Training model for target: %s", target)

    X, y = prepare_features(df, target)
    feature_names = X.columns.tolist()

    # Temporal split – last 20% as test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("  Train size: %d, Test size: %d", len(X_train), len(X_test))

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # resources can't be negative

    metrics = evaluate(y_test, y_pred)
    logger.info("  MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%",
                metrics["mae"], metrics["rmse"], metrics["mape"])

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_names": feature_names,
    }


def train_all_models(df: pd.DataFrame) -> dict:
    """Train models for every resource target and save artefacts."""
    all_metrics = {}
    all_importances = {}

    for target in RESOURCE_COLS:
        result = train_single_model(df, target)

        # Save model
        model_path = ARTIFACTS_DIR / f"{target}_model.joblib"
        joblib.dump(result["model"], model_path)
        logger.info("  Saved model -> %s", model_path)

        all_metrics[target] = result["metrics"]
        all_importances[target] = result["feature_importance"]

    # Save feature names (same for all models)
    feature_names_path = ARTIFACTS_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(result["feature_names"], f, indent=2)

    # Save consolidated metrics
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics -> %s", metrics_path)

    # Save feature importances
    imp_path = ARTIFACTS_DIR / "feature_importances.json"
    with open(imp_path, "w") as f:
        json.dump(all_importances, f, indent=2)
    logger.info("Saved feature importances -> %s", imp_path)

    return all_metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Run data pipeline first (generates processed features.csv)
    df = run_pipeline(save=True)

    # Train
    metrics = train_all_models(df)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for target, m in metrics.items():
        logger.info("  %-25s  MAE=%8.2f  RMSE=%8.2f  MAPE=%6.2f%%",
                     target, m["mae"], m["rmse"], m["mape"])
    logger.info("=" * 60)
    logger.info("All models saved to %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
