"""
🗺️ Risk Map – Interactive Bihar map with district risk visualization.
"""

import sys
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.prediction import get_all_district_risks
from dashboard.components.maps import create_bihar_risk_map

st.set_page_config(page_title="Risk Map", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1 {
        background: linear-gradient(90deg, #4361ee, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🗺️ Bihar Risk Map")
st.markdown("Interactive geospatial visualization of flood risk across all 38 districts.")
st.markdown("---")

# ── Load risk data ──
try:
    district_risks = get_all_district_risks()
except Exception as e:
    st.error(f"Error loading risk data: {e}")
    st.info("Make sure you've run `python train.py` to generate model artifacts and processed data.")
    st.stop()

if not district_risks:
    st.warning("No district risk data available.")
    st.stop()

# ── Filter controls ──
with st.sidebar:
    st.markdown("### 🎛️ Map Controls")
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["Critical", "High", "Moderate", "Low"],
        default=["Critical", "High", "Moderate", "Low"],
    )

    min_score = st.slider("Minimum Risk Score", 0, 100, 0)

# Apply filters
filtered_risks = [
    d for d in district_risks
    if d["risk_level"] in risk_filter and d["risk_score"] >= min_score
]

# ── Stats bar ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 Critical", sum(1 for d in filtered_risks if d["risk_level"] == "Critical"))
col2.metric("🟡 High", sum(1 for d in filtered_risks if d["risk_level"] == "High"))
col3.metric("🟢 Moderate", sum(1 for d in filtered_risks if d["risk_level"] == "Moderate"))
col4.metric("🔵 Low", sum(1 for d in filtered_risks if d["risk_level"] == "Low"))

st.markdown("---")

# ── Render map ──
st.markdown("### 📍 District Risk Visualization")
st.markdown(
    "_Click on markers for detailed information. "
    "Heatmap intensity reflects risk concentration._"
)

bihar_map = create_bihar_risk_map(filtered_risks)
st_folium(bihar_map, width=None, height=600, returned_objects=[])

# ── Risk legend (inline) ──
st.markdown("---")
st.markdown("### 📋 Legend")
legend_cols = st.columns(4)
with legend_cols[0]:
    st.markdown("🔵 **Low** (0–30): Standard preparedness")
with legend_cols[1]:
    st.markdown("🟢 **Moderate** (30–50): Enhanced monitoring")
with legend_cols[2]:
    st.markdown("🟡 **High** (50–70): Alert & pre-position")
with legend_cols[3]:
    st.markdown("🔴 **Critical** (70–100): Immediate action")
