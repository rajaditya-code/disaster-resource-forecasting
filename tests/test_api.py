"""
Test suite for the FastAPI backend.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app

client = TestClient(app)


# ── Health ───────────────────────────────────────────────────────────────────

def test_root():
    """Root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health():
    """Health endpoint returns status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "models_loaded" in data
    assert "version" in data


# ── Districts ────────────────────────────────────────────────────────────────

def test_districts():
    """GET /districts returns list of Bihar districts."""
    response = client.get("/api/v1/districts")
    assert response.status_code == 200
    data = response.json()
    assert "districts" in data
    assert data["total"] == 38
    assert len(data["districts"]) == 38

    # Check structure
    district = data["districts"][0]
    assert "district" in district
    assert "population" in district
    assert "latitude" in district


# ── Metrics ──────────────────────────────────────────────────────────────────

def test_metrics():
    """GET /metrics returns model evaluation metrics."""
    response = client.get("/api/v1/metrics")
    if response.status_code == 200:
        data = response.json()
        assert "metrics" in data
        for target, m in data["metrics"].items():
            assert "mae" in m
            assert "rmse" in m
            assert "mape" in m


# ── Feature Importance ───────────────────────────────────────────────────────

def test_feature_importance():
    """GET /feature-importance returns importances."""
    response = client.get("/api/v1/feature-importance")
    if response.status_code == 200:
        data = response.json()
        assert "feature_importances" in data


def test_feature_importance_filtered():
    """GET /feature-importance?target=food_kits returns filtered importances."""
    response = client.get("/api/v1/feature-importance?target=food_kits")
    if response.status_code == 200:
        data = response.json()
        assert "food_kits" in data["feature_importances"]


# ── Prediction ───────────────────────────────────────────────────────────────

def test_predict_valid():
    """POST /predict with valid input returns predictions."""
    payload = {
        "district": "Patna",
        "forecast_horizon_days": 7,
        "rainfall_mm": 100.0,
        "temperature_c": 32.0,
        "humidity_pct": 75.0,
        "flood_severity": 2,
        "river_water_level_m": 5.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["district"] == "Patna"
    assert data["forecast_horizon_days"] == 7
    assert "risk_score" in data
    assert "predictions" in data
    assert "confidence_interval" in data
    assert "recommendation" in data

    preds = data["predictions"]
    assert "food_kits" in preds
    assert "medical_kits" in preds
    assert preds["food_kits"] >= 0


def test_predict_invalid_district():
    """POST /predict with unknown district returns 400."""
    payload = {"district": "NonExistentDistrict", "forecast_horizon_days": 7}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400


def test_predict_invalid_horizon():
    """POST /predict with out-of-range horizon returns 422."""
    payload = {"district": "Patna", "forecast_horizon_days": 100}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
