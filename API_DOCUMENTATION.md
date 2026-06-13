# API Reference
## Disaster Resource Allocation Forecasting API v1.0

Base URL: `http://localhost:8000/api/v1`

Interactive docs: `http://localhost:8000/docs` (Swagger UI) | `http://localhost:8000/redoc` (ReDoc)

---

## Authentication

No authentication is required for the current version. All endpoints are publicly accessible. In a production deployment, you'd want to add API key authentication via a middleware or OAuth2.

---

## Endpoints

### POST /predict

Generate resource demand predictions for a specific district.

This is the core endpoint. You give it a district name, a forecast horizon, and expected weather conditions, and it returns predicted quantities for all five resource types along with a risk assessment and a plain-English recommendation.

**Request Body** (JSON):

```json
{
    "district": "Patna",
    "forecast_horizon_days": 7,
    "rainfall_mm": 150.0,
    "temperature_c": 32.0,
    "humidity_pct": 80.0,
    "flood_severity": 3,
    "river_water_level_m": 6.5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `district` | string | Yes | — | Name of a Bihar district (must match exactly, e.g. "East Champaran") |
| `forecast_horizon_days` | integer | No | 7 | How many days ahead to forecast (1-30) |
| `rainfall_mm` | float | No | 0.0 | Expected cumulative rainfall in millimetres |
| `temperature_c` | float | No | 30.0 | Expected temperature in Celsius |
| `humidity_pct` | float | No | 60.0 | Expected humidity percentage (0-100) |
| `flood_severity` | integer | No | 0 | Expected flood severity on a 0-5 scale |
| `river_water_level_m` | float | No | 3.0 | Expected river water level in metres |

**Response** (200 OK):

```json
{
    "district": "Patna",
    "forecast_horizon_days": 7,
    "risk_score": 62.5,
    "risk_level": "High",
    "predictions": {
        "food_kits": 4500,
        "medical_kits": 2000,
        "ors_packets": 12000,
        "drinking_water_litres": 35000,
        "tarpaulins": 1200
    },
    "confidence_interval": {
        "food_kits": {"lower": 3825, "upper": 5175},
        "medical_kits": {"lower": 1700, "upper": 2300},
        "ors_packets": {"lower": 10200, "upper": 13800},
        "drinking_water_litres": {"lower": 29750, "upper": 40250},
        "tarpaulins": {"lower": 1020, "upper": 1380}
    },
    "recommendation": "Patna district has high flood risk. Proactive measures recommended. Pre-position 4,500 food kits, 2,000 medical kits, and 12,000 ORS packets within the next 7 days."
}
```

**Error Responses:**

| Status | Reason |
|--------|--------|
| 400 | Unknown district name. Use GET /districts to see valid options. |
| 422 | Validation error (e.g. forecast_horizon_days > 30). |
| 500 | Internal model error. Check server logs. |

---

### GET /districts

Returns all 38 Bihar districts with demographic data and current risk scores.

**Query Parameters:** None

**Response** (200 OK):

```json
{
    "districts": [
        {
            "district": "Darbhanga",
            "population": 3937385,
            "vulnerable_population": 1181215,
            "area_sqkm": 2279,
            "latitude": 26.17,
            "longitude": 85.9,
            "current_risk_score": 72.3,
            "risk_level": "Critical"
        }
    ],
    "total": 38
}
```

This is useful for populating dropdown menus, rendering maps, or getting a quick overview of which districts are currently at elevated risk.

---

### GET /metrics

Returns model evaluation metrics from the most recent training run.

**Query Parameters:** None

**Response** (200 OK):

```json
{
    "metrics": {
        "food_kits": {
            "mae": 112.62,
            "rmse": 138.89,
            "mape": 113.48
        },
        "medical_kits": {
            "mae": 44.40,
            "rmse": 54.39,
            "mape": 105.87
        },
        "ors_packets": {
            "mae": 233.54,
            "rmse": 288.28,
            "mape": 114.22
        },
        "drinking_water_litres": {
            "mae": 394.88,
            "rmse": 486.97,
            "mape": 126.60
        },
        "tarpaulins": {
            "mae": 30.29,
            "rmse": 37.29,
            "mape": 96.45
        }
    }
}
```

**A note on MAPE:** The percentage errors look large because the dataset includes many non-monsoon days with near-zero resource demand. Small absolute errors on small actual values produce inflated MAPE. During monsoon events, the models perform materially better.

---

### GET /feature-importance

Returns feature importance scores from the trained LightGBM models.

**Query Parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `target` | string | No | Filter to a specific resource (e.g. `food_kits`). Omit to get all. |

**Response** (200 OK):

```json
{
    "feature_importances": {
        "food_kits": {
            "rainfall_mm_roll_mean_7": 245,
            "flood_severity_lag_1": 198,
            "risk_score": 187,
            "rainfall_mm": 156,
            "humidity_pct_roll_mean_14": 134
        }
    }
}
```

The importance values are LightGBM's split-based importance scores (how many times each feature was used for a split across all trees). Higher numbers mean the feature was more useful for making predictions.

---

### GET /health

Quick health check to confirm the API is running and models are loaded.

**Query Parameters:** None

**Response** (200 OK):

```json
{
    "status": "healthy",
    "models_loaded": 5,
    "version": "1.0.0"
}
```

| Status | Meaning |
|--------|---------|
| `healthy` | All 5 models loaded and ready. |
| `degraded` | Some or no models loaded. Predictions may fail. Run training pipeline. |

---

## Error Format

All error responses follow this structure:

```json
{
    "detail": "Human-readable error message explaining what went wrong."
}
```

---

## Rate Limits

No rate limits are enforced in the current version. If deploying publicly, consider adding rate limiting via a reverse proxy (nginx, Cloudflare) or FastAPI middleware.

---

## Example: cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get all districts
curl http://localhost:8000/api/v1/districts

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Darbhanga",
    "forecast_horizon_days": 14,
    "rainfall_mm": 200,
    "flood_severity": 4
  }'

# Get feature importance for food kits
curl "http://localhost:8000/api/v1/feature-importance?target=food_kits"
```

## Example: Python (requests)

```python
import requests

# Predict for Muzaffarpur, 7-day horizon
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "district": "Muzaffarpur",
        "forecast_horizon_days": 7,
        "rainfall_mm": 180,
        "temperature_c": 31,
        "humidity_pct": 85,
        "flood_severity": 3,
        "river_water_level_m": 6.0,
    }
)

data = response.json()
print(f"Risk Level: {data['risk_level']}")
print(f"Food Kits Needed: {data['predictions']['food_kits']:,}")
print(f"Recommendation: {data['recommendation']}")
```
