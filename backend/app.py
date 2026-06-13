"""
FastAPI application entry point for the Disaster Resource Forecasting API.

Usage:
    uvicorn backend.app:app --reload --port 8000
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes.api import router
from utils.logger import get_logger

logger = get_logger(__name__)

# ── App Configuration ────────────────────────────────────────────────────────

app = FastAPI(
    title="Disaster Resource Allocation Forecasting API",
    description=(
        "Predict disaster relief resource requirements for Bihar districts "
        "7–30 days in advance using LightGBM models trained on historical "
        "climatic, demographic, and disaster data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit / any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1", tags=["Forecasting"])


# ── Root endpoint ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "Disaster Resource Allocation Forecasting API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ── Startup event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("API starting up - pre-loading models ...")
    from backend.services.prediction import get_models
    models = get_models()
    logger.info("   Loaded %d models", len(models))
    logger.info("API ready.")
