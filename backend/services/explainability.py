"""
Explainability service – SHAP values and feature importance retrieval.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"


def get_feature_importances(target: str | None = None) -> dict:
    """
    Return stored feature importances.

    Args:
        target: Specific resource target (e.g., 'food_kits').
                If None, returns importances for all targets.

    Returns:
        Dict mapping target → {feature: importance_score}.
    """
    imp_path = ARTIFACTS_DIR / "feature_importances.json"
    if not imp_path.exists():
        logger.warning("Feature importances file not found at %s", imp_path)
        return {}

    with open(imp_path) as f:
        all_imp = json.load(f)

    if target:
        return {target: all_imp.get(target, {})}
    return all_imp


def get_top_features(target: str, top_n: int = 15) -> list[dict]:
    """
    Get the top-N most important features for a given target.

    Returns list of {feature, importance} dicts sorted descending.
    """
    importances = get_feature_importances(target)
    if target not in importances:
        return []

    imp = importances[target]
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": k, "importance": v} for k, v in sorted_imp]


def get_metrics() -> dict:
    """Load saved evaluation metrics."""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        logger.warning("Metrics file not found at %s", metrics_path)
        return {}

    with open(metrics_path) as f:
        return json.load(f)


def compute_shap_summary(model, X_sample) -> dict:
    """
    Compute SHAP values for a sample of data.

    Returns dict with mean absolute SHAP values per feature.
    Note: This is computationally expensive; use sparingly.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_names = X_sample.columns.tolist()
        return dict(zip(feature_names, mean_abs.tolist()))
    except ImportError:
        logger.warning("SHAP not installed – skipping SHAP explanation")
        return {}
    except Exception as e:
        logger.error("SHAP computation failed: %s", e)
        return {}
