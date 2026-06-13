"""
🏠 Home – Disaster Resource Allocation Forecasting Dashboard

Main entry point for the Streamlit multi-page app.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Disaster Resource Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(67,97,238,0.15), rgba(114,9,183,0.15));
        border: 1px solid rgba(67,97,238,0.3);
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #b0b0b0;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #4361ee, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Cards */
    .stExpander {
        border: 1px solid rgba(67,97,238,0.2);
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4361ee, #7209b7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(67,97,238,0.4);
    }

    /* Divider */
    hr {
        border-color: rgba(67,97,238,0.2);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌊 Disaster Forecast")
    st.markdown("---")
    st.markdown("""
    **AI-Powered Resource Allocation**

    Predict disaster relief requirements
    for Bihar districts 7–30 days in advance.

    Navigate using the pages below.
    """)
    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>v1.0.0 • Built with Streamlit & LightGBM</small>",
        unsafe_allow_html=True,
    )

# ── Main content ─────────────────────────────────────────────────────────────

st.markdown("# 🌊 AI-Powered Disaster Resource Forecasting")
st.markdown("### Predicting Bihar's Relief Needs Before Disaster Strikes")

st.markdown("---")

# Hero section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(67,97,238,0.2), rgba(114,9,183,0.1));
        border: 1px solid rgba(67,97,238,0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    ">
        <h2 style="font-size: 2.5rem; margin: 0;">38</h2>
        <p style="color: #b0b0b0; margin: 0;">Districts Monitored</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(247,37,133,0.2), rgba(114,9,183,0.1));
        border: 1px solid rgba(247,37,133,0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    ">
        <h2 style="font-size: 2.5rem; margin: 0;">5</h2>
        <p style="color: #b0b0b0; margin: 0;">Resource Types Predicted</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(6,214,160,0.2), rgba(17,138,178,0.1));
        border: 1px solid rgba(6,214,160,0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    ">
        <h2 style="font-size: 2.5rem; margin: 0;">30</h2>
        <p style="color: #b0b0b0; margin: 0;">Day Forecast Horizon</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Feature cards
st.markdown("### 🧭 Dashboard Navigation")
st.markdown("")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    with st.expander("📊 **Overview** — System Status & Risk Summary", expanded=True):
        st.markdown("""
        Get a bird's-eye view of all monitored districts, high-risk areas,
        and aggregate resource demand forecasts.
        """)

    with st.expander("🗺️ **Risk Map** — Interactive Bihar Map", expanded=True):
        st.markdown("""
        Explore an interactive Folium map with district risk visualisation,
        heatmap overlays, and population-weighted risk indicators.
        """)

with feat_col2:
    with st.expander("🔮 **Forecasting** — Predict Resource Demand", expanded=True):
        st.markdown("""
        Select a district and forecast horizon to generate AI-powered
        resource predictions with confidence intervals.
        """)

    with st.expander("📈 **Analytics** — Model Performance & Insights", expanded=True):
        st.markdown("""
        View historical trends, feature importance charts, SHAP explanations,
        and model evaluation metrics (MAE, RMSE, MAPE).
        """)

st.markdown("")
with st.expander("💡 **Recommendations** — Actionable Insights", expanded=True):
    st.markdown("""
    Get human-readable, district-specific recommendations for
    pre-positioning relief resources based on forecasted risk levels.
    """)

st.markdown("---")

# Architecture
st.markdown("### 🏗️ System Architecture")
st.markdown("""
```
CSV Data → Data Pipeline → Feature Engineering → LightGBM Models → FastAPI Backend → Streamlit Dashboard
                                                                          ↓
                                                              Resource Predictions
                                                              Risk Assessment
                                                              Recommendations
```
""")

st.markdown(
    "<div style='text-align:center; color:#666; padding:20px;'>"
    "Built with ❤️ for Bihar Disaster Management"
    "</div>",
    unsafe_allow_html=True,
)
