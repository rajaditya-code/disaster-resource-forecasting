# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for geopandas / shapely
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgdal-dev libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Generate data & train models (build-time) ───────────────────────────────
RUN python utils/generate_data.py && python train.py

# ── Expose ports ─────────────────────────────────────────────────────────────
# FastAPI: 8000, Streamlit: 8501
EXPOSE 8000 8501

# ── Default command: run both services ───────────────────────────────────────
CMD ["sh", "-c", \
     "uvicorn backend.app:app --host 0.0.0.0 --port 8000 & \
      streamlit run dashboard/Home.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
