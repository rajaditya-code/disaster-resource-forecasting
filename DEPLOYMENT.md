# Deployment Guide
## Getting This System Running in Production

This guide covers three deployment options: Docker (easiest), Render + Streamlit Cloud (free tier friendly), and plain server deployment.

---

## Option 1: Docker (Recommended for Self-Hosting)

The simplest way to get everything running. One container, both services.

### Prerequisites
- Docker installed on your machine or server.

### Steps

```bash
# 1. Build the image (this generates data and trains models during build)
docker build -t disaster-forecast .

# 2. Run it
docker run -d -p 8000:8000 -p 8501:8501 --name forecast disaster-forecast

# 3. Check it's working
curl http://localhost:8000/api/v1/health
# Open http://localhost:8501 in your browser
```

The Dockerfile handles everything — installs dependencies, generates synthetic data, trains the models, and starts both FastAPI and Streamlit. If you have your own real data, mount it as a volume:

```bash
docker run -d -p 8000:8000 -p 8501:8501 \
  -v /path/to/your/data:/app/data/sample \
  disaster-forecast
```

### Updating with Real Data

1. Replace the CSV files in `data/sample/` with your actual data (same column names).
2. Rebuild the image: `docker build -t disaster-forecast .`
3. Restart: `docker stop forecast && docker rm forecast && docker run ...`

---

## Option 2: Render (Backend) + Streamlit Cloud (Dashboard)

Good if you want free hosting and don't mind the services being separate.

### Deploy the Backend on Render

1. Push your code to a GitHub repository.
2. Go to [render.com](https://render.com) and create a new Web Service.
3. Connect your GitHub repo.
4. Render will detect the `render.yaml` file and configure itself automatically.
5. The build command installs dependencies, generates data, and trains models.
6. Your API will be available at `https://your-app-name.onrender.com`.

**Important:** Render's free tier has limited memory (512 MB). The training step should work fine, but if you have very large datasets, consider training locally and committing the model artifacts directly.

### Deploy the Dashboard on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click "New app" and point it to your GitHub repo.
3. Set the main file to `dashboard/Home.py`.
4. Set Python version to 3.11.
5. In the app settings, add an environment variable:
   ```
   API_BASE_URL=https://your-app-name.onrender.com
   ```

**Note:** You'll need to update the dashboard code to make API calls to the backend instead of importing services directly. For the current version, the dashboard imports Python modules directly, which means both services need access to the same codebase.

---

## Option 3: Plain Server (VPS, EC2, etc.)

If you have a Linux server with SSH access.

### Setup

```bash
# 1. Clone your repo
git clone https://github.com/your-username/disaster-resource-forecasting.git
cd disaster-resource-forecasting

# 2. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train models
python utils/generate_data.py
python train.py

# 5. Start the backend (in background)
nohup uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# 6. Start the dashboard (in background)
nohup streamlit run dashboard/Home.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true &
```

### Using systemd (for auto-restart)

Create `/etc/systemd/system/forecast-api.service`:

```ini
[Unit]
Description=Disaster Forecast API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/disaster-resource-forecasting
ExecStart=/home/ubuntu/disaster-resource-forecasting/venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/forecast-dashboard.service`:

```ini
[Unit]
Description=Disaster Forecast Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/disaster-resource-forecasting
ExecStart=/home/ubuntu/disaster-resource-forecasting/venv/bin/streamlit run dashboard/Home.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable forecast-api forecast-dashboard
sudo systemctl start forecast-api forecast-dashboard
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | Backend bind address |
| `API_PORT` | 8000 | Backend port |
| `API_DEBUG` | false | Enable FastAPI debug mode |
| `STREAMLIT_PORT` | 8501 | Dashboard port |
| `API_BASE_URL` | http://localhost:8000 | Backend URL (used by dashboard if calling API) |
| `MODEL_DIR` | model_artifacts | Path to saved models |
| `DATA_DIR` | data | Path to data directory |

---

## Retraining with New Data

After each monsoon season (or whenever you get updated data):

```bash
# 1. Replace CSVs in data/sample/ with updated files
# 2. Re-run the pipeline
python utils/generate_data.py   # skip this if using real data
python train.py

# 3. Restart the backend to pick up new models
# (if using systemd)
sudo systemctl restart forecast-api
```

The training script will overwrite the existing model artifacts. The backend loads models at startup, so a restart is needed to pick up the new versions.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No models found" error | Run `python train.py` first |
| Dashboard shows "Error loading data" | Make sure the backend is running on port 8000 |
| Port already in use | Change the port in the start command or kill the existing process |
| Unicode errors on Windows | We've fixed the known ones, but if you see more, set `PYTHONIOENCODING=utf-8` |
| Out of memory during training | Reduce the date range in `generate_data.py` or use a machine with more RAM |

---

*Part of the AI-Powered Disaster Resource Allocation Forecasting System.*
