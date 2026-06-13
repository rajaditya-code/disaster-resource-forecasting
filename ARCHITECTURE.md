# Architecture Overview
## AI-Powered Disaster Resource Allocation Forecasting System

This document describes how the system is put together, what each piece does, and how data flows from raw CSVs all the way to the predictions a relief coordinator sees on their screen.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph Data Layer
        A1["historical_disaster_data.csv<br/>Daily rainfall, temperature,<br/>flood severity for 38 districts"]
        A2["demographics.csv<br/>Population, area, coordinates<br/>for each district"]
        A3["historical_resources.csv<br/>Daily resource allocation<br/>records"]
    end

    subgraph Processing Layer
        B1["Data Pipeline<br/>(utils/data_pipeline.py)<br/>Cleaning, imputation,<br/>feature engineering"]
    end

    subgraph Feature Engineering
        C1["Temporal Features<br/>Month, season, day-of-year,<br/>cyclical encoding"]
        C2["Lag Features<br/>1, 3, 7, 14-day lags<br/>for climate variables"]
        C3["Rolling Statistics<br/>3, 7, 14, 30-day<br/>means and std devs"]
        C4["Risk Score<br/>Composite 0-100 score<br/>from rainfall + severity +<br/>water level + season"]
    end

    subgraph ML Layer
        D1["LightGBM Model<br/>Food Kits"]
        D2["LightGBM Model<br/>Medical Kits"]
        D3["LightGBM Model<br/>ORS Packets"]
        D4["LightGBM Model<br/>Drinking Water"]
        D5["LightGBM Model<br/>Tarpaulins"]
    end

    subgraph API Layer
        E1["FastAPI Backend<br/>(backend/app.py)<br/>REST API with 5 endpoints"]
    end

    subgraph Presentation Layer
        F1["Streamlit Dashboard<br/>5 interactive pages"]
        F2["Folium Maps<br/>Geospatial visualization"]
        F3["Plotly Charts<br/>Interactive analytics"]
    end

    A1 & A2 & A3 --> B1
    B1 --> C1 & C2 & C3 & C4
    C1 & C2 & C3 & C4 --> D1 & D2 & D3 & D4 & D5
    D1 & D2 & D3 & D4 & D5 --> E1
    E1 --> F1
    F1 --> F2 & F3
```

---

## Data Flow

The system processes data in three distinct phases:

### Phase 1: Batch Pipeline (runs once, or periodically)

```mermaid
flowchart LR
    A[Raw CSVs] --> B[Clean & Impute]
    B --> C[Merge on district + date]
    C --> D[Add temporal features]
    D --> E[Add lag features]
    E --> F[Add rolling stats]
    F --> G[Compute risk scores]
    G --> H[features.csv]
    H --> I[Train 5 LightGBM models]
    I --> J[Save .joblib + metrics.json]
```

### Phase 2: Real-time Prediction (on each API request)

```mermaid
flowchart LR
    A[User Input<br/>district + weather + horizon] --> B[Build feature vector<br/>from latest historical data]
    B --> C[Load cached models]
    C --> D[Run 5 predictions]
    D --> E[Scale by horizon]
    E --> F[Compute confidence intervals]
    F --> G[Generate recommendation text]
    G --> H[Return JSON response]
```

### Phase 3: Visualization (Streamlit dashboard)

```mermaid
flowchart LR
    A[User opens dashboard] --> B[Load processed data + model artifacts]
    B --> C{Which page?}
    C --> D[Overview: risk table + gauges]
    C --> E[Risk Map: Folium + heatmap]
    C --> F[Forecasting: input form + predictions]
    C --> G[Analytics: trends + feature importance]
    C --> H[Recommendations: action cards + CSV export]
```

---

## Component Details

### Data Pipeline (`utils/data_pipeline.py`)

This is where the heavy lifting happens. The pipeline takes three raw CSV files and produces a single feature-rich dataset with 80+ columns. Here's what it does:

- **Cleaning**: Forward-fills missing values within each district's time series, then fills any remaining gaps with column medians. This handles the reality that some districts have patchy reporting.
- **Temporal features**: Extracts month, day-of-year, week number, and adds a binary monsoon flag. Also creates cyclical sin/cos encodings of the month so the model understands that December and January are close together.
- **Lag features**: For each climate variable (rainfall, temperature, humidity, flood severity, river level), creates 1-day, 3-day, 7-day, and 14-day lagged versions. This lets the model see "what happened last week" when predicting tomorrow.
- **Rolling features**: Computes 3, 7, 14, and 30-day rolling means and standard deviations. These capture trends and volatility.
- **Risk score**: A weighted composite of normalised rainfall (35%), flood severity (30%), river water level (25%), and monsoon flag (10%), scaled to 0-100.

### ML Models (`train.py`)

We train five independent LightGBM regressors, one per resource type. Why separate models instead of one multi-output model? Because each resource has different demand patterns — tarpaulin demand spikes sharply with severity while ORS demand correlates more with humidity and temperature.

Training uses a temporal split (80% train, 20% test) rather than random k-fold, because this is time-series data and random splits would leak future information into training.

### FastAPI Backend (`backend/`)

The API is intentionally simple. Five endpoints, no authentication, CORS wide open. The prediction service caches loaded models in memory so the first request is slow (model loading) but subsequent requests are fast.

The prediction flow:
1. Receive district + weather inputs
2. Look up the district's latest historical data for lag/rolling features
3. Combine with demographic data
4. Build a single-row feature vector matching the model's expected columns
5. Run all five models
6. Multiply daily predictions by forecast horizon
7. Add 15% confidence intervals
8. Generate a recommendation string
9. Return everything as JSON

### Streamlit Dashboard (`dashboard/`)

Built as a multi-page Streamlit app. Each page loads data independently (no shared state between pages except session state). The design uses a dark theme with gradient accents to look professional without being distracting.

Folium maps use CartoDB dark_matter tiles with circle markers sized by risk score and colored by risk level. A heatmap layer shows risk concentration across the state.

---

## Technology Choices and Why

| Choice | Reasoning |
|--------|-----------|
| **LightGBM** over XGBoost or Prophet | Faster training on tabular data, handles missing values natively, good with lag features. Prophet is better for pure time-series but our data is tabular with many features. |
| **FastAPI** over Flask or Django | Automatic OpenAPI docs, async support, Pydantic validation built in. For an API this simple, Flask would also work fine. |
| **Streamlit** over Dash or custom React | Fastest path to a working dashboard. The target users are NGO coordinators, not software engineers — Streamlit's simplicity is a feature. |
| **Folium** over Plotly maps | Better support for tile providers and marker customisation. Plotly maps are nice but Folium gives us more control over the CartoDB dark tiles and heatmap overlays. |
| **CSV files** over a database | The challenge spec requires no database. CSVs keep things simple and portable. For production scale, you'd want PostgreSQL. |

---

## Deployment Architecture

```mermaid
graph LR
    subgraph Production
        A[Render<br/>FastAPI Backend] --> B[Model Artifacts<br/>joblib files]
        C[Streamlit Cloud<br/>Dashboard] --> A
    end

    subgraph Development
        D[Local Machine<br/>uvicorn + streamlit] --> E[data/ folder<br/>CSV files]
    end

    subgraph Docker
        F[Single Container<br/>Both services<br/>Ports 8000 + 8501]
    end
```

---

*This architecture document is part of the AI-Powered Disaster Resource Allocation Forecasting System.*
