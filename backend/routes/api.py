"""
FastAPI route definitions for the Disaster Resource Forecasting API.
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    ResourcePrediction,
    HealthResponse,
)
from backend.services.prediction import (
    predict_resources,
    get_demographics,
    get_all_district_risks,
    get_models,
)
from backend.services.explainability import (
    get_feature_importances,
    get_metrics,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── District list (loaded once from demographics CSV) ────────────────────────

BIHAR_DISTRICTS = [
    "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur",
    "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj",
    "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj",
    "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur",
    "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa",
    "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan",
    "Supaul", "Vaishali", "West Champaran",
]


# ── POST /predict ────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(req: PredictionRequest):
    """Generate resource demand predictions for a district."""
    if req.district not in BIHAR_DISTRICTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown district '{req.district}'. Use GET /districts for valid names.",
        )

    try:
        result = predict_resources(
            district=req.district,
            forecast_horizon_days=req.forecast_horizon_days,
            rainfall_mm=req.rainfall_mm,
            temperature_c=req.temperature_c,
            humidity_pct=req.humidity_pct,
            flood_severity=req.flood_severity,
            river_water_level_m=req.river_water_level_m,
        )
        return PredictionResponse(
            district=result["district"],
            forecast_horizon_days=result["forecast_horizon_days"],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            predictions=ResourcePrediction(**result["predictions"]),
            confidence_interval=result["confidence_interval"],
            recommendation=result["recommendation"],
        )
    except Exception as e:
        logger.error("Prediction failed for %s: %s", req.district, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /districts ───────────────────────────────────────────────────────────

@router.get("/districts")
async def list_districts():
    """Return all Bihar districts with demographic info and current risk scores."""
    try:
        demo = get_demographics()
        risks = {r["district"]: r for r in get_all_district_risks()}

        districts = []
        for _, row in demo.iterrows():
            name = row["district"]
            risk_info = risks.get(name, {})
            districts.append({
                "district": name,
                "population": int(row["population"]),
                "vulnerable_population": int(row["vulnerable_population"]),
                "area_sqkm": int(row["area_sqkm"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "current_risk_score": risk_info.get("risk_score"),
                "risk_level": risk_info.get("risk_level", "Unknown"),
            })

        return {"districts": districts, "total": len(districts)}
    except Exception as e:
        logger.error("Error fetching districts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /metrics ─────────────────────────────────────────────────────────────

@router.get("/metrics")
async def model_metrics():
    """Return evaluation metrics (MAE, RMSE, MAPE) for all trained models."""
    metrics = get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")
    return {"metrics": metrics}


# ── GET /feature-importance ──────────────────────────────────────────────────

@router.get("/feature-importance")
async def feature_importance(target: str = None):
    """
    Return feature importances.

    Query params:
        target: optional resource name (e.g., 'food_kits') to filter.
    """
    importances = get_feature_importances(target)
    if not importances:
        raise HTTPException(status_code=404, detail="No feature importances found.")
    return {"feature_importances": importances}


# ── GET /health ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check – confirms models are loaded."""
    models = get_models()
    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=len(models),
        version="1.0.0",
    )
