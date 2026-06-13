"""
Pydantic schemas for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """POST /predict request body."""
    district: str = Field(..., description="Name of the Bihar district")
    forecast_horizon_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days to forecast (1–30)",
    )
    rainfall_mm: float = Field(default=0.0, ge=0, description="Expected rainfall in mm")
    temperature_c: float = Field(default=30.0, description="Expected temperature °C")
    humidity_pct: float = Field(default=60.0, ge=0, le=100, description="Expected humidity %")
    flood_severity: int = Field(default=0, ge=0, le=5, description="Expected flood severity 0–5")
    river_water_level_m: float = Field(default=3.0, ge=0, description="Expected river water level in metres")


class ResourcePrediction(BaseModel):
    """Predicted resource requirements."""
    food_kits: int
    medical_kits: int
    ors_packets: int
    drinking_water_litres: int
    tarpaulins: int


class PredictionResponse(BaseModel):
    """POST /predict response body."""
    district: str
    forecast_horizon_days: int
    risk_score: float
    risk_level: str
    predictions: ResourcePrediction
    confidence_interval: dict
    recommendation: str


class DistrictInfo(BaseModel):
    """District summary returned by GET /districts."""
    district: str
    population: int
    vulnerable_population: int
    area_sqkm: int
    latitude: float
    longitude: float
    current_risk_score: Optional[float] = None


class MetricsResponse(BaseModel):
    """Model evaluation metrics."""
    target: str
    mae: float
    rmse: float
    mape: float


class FeatureImportanceResponse(BaseModel):
    """Feature importance for a given target."""
    target: str
    importances: dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    version: str
