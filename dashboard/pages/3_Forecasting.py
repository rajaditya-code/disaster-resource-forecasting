"""
🔮 Forecasting – District-level resource demand prediction.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.prediction import predict_resources, get_demographics
from dashboard.components.charts import (
    resource_bar_chart,
    confidence_interval_chart,
    risk_gauge,
)
from dashboard.components.maps import create_district_detail_map
from streamlit_folium import st_folium

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")

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
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(67,97,238,0.15), rgba(114,9,183,0.15));
        border: 1px solid rgba(67,97,238,0.3);
        border-radius: 12px;
        padding: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4361ee, #7209b7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔮 Resource Demand Forecasting")
st.markdown("Generate AI-powered predictions for disaster relief resource requirements.")
st.markdown("---")

# ── Load demographics for district list ──
try:
    demo = get_demographics()
    districts = sorted(demo["district"].unique().tolist())
except Exception:
    districts = [
        "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur",
        "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj",
        "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj",
        "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur",
        "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa",
        "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan",
        "Supaul", "Vaishali", "West Champaran",
    ]
    demo = None

# ── Input form ──
with st.sidebar:
    st.markdown("### 🎯 Forecast Parameters")

    selected_district = st.selectbox("📍 Select District", districts, index=districts.index("Patna"))
    forecast_horizon = st.slider("📅 Forecast Horizon (days)", 1, 30, 7)

    st.markdown("---")
    st.markdown("### 🌤️ Expected Conditions")

    rainfall = st.slider("🌧️ Rainfall (mm)", 0.0, 500.0, 50.0, step=5.0)
    temperature = st.slider("🌡️ Temperature (°C)", 10.0, 48.0, 30.0, step=0.5)
    humidity = st.slider("💧 Humidity (%)", 20.0, 100.0, 65.0, step=1.0)
    flood_severity = st.slider("🌊 Flood Severity (0–5)", 0, 5, 1)
    water_level = st.slider("📏 River Water Level (m)", 0.0, 12.0, 3.5, step=0.1)

    st.markdown("---")
    predict_btn = st.button("🚀 Generate Forecast", use_container_width=True)

# ── Predictions ──
if predict_btn:
    with st.spinner("Running LightGBM models..."):
        try:
            result = predict_resources(
                district=selected_district,
                forecast_horizon_days=forecast_horizon,
                rainfall_mm=rainfall,
                temperature_c=temperature,
                humidity_pct=humidity,
                flood_severity=flood_severity,
                river_water_level_m=water_level,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Results header ──
    st.markdown(f"### 📋 Forecast for **{selected_district}** ({forecast_horizon}-day horizon)")

    # Risk & key stats
    r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
    r_col1.metric("🔮 Risk Score", f"{result['risk_score']:.1f}")
    r_col2.metric("⚡ Risk Level", result["risk_level"])
    r_col3.metric("🍱 Food Kits", f"{result['predictions']['food_kits']:,}")
    r_col4.metric("💊 Medical Kits", f"{result['predictions']['medical_kits']:,}")
    r_col5.metric("💧 ORS Packets", f"{result['predictions']['ors_packets']:,}")

    st.markdown("---")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.plotly_chart(
            resource_bar_chart(result["predictions"], f"Predicted Resources – {selected_district}"),
            use_container_width=True,
        )

    with chart_col2:
        st.plotly_chart(
            confidence_interval_chart(
                result["predictions"],
                result["confidence_interval"],
                "Confidence Intervals (±15%)",
            ),
            use_container_width=True,
        )

    # Risk gauge + map
    gauge_col, map_col = st.columns([1, 2])

    with gauge_col:
        st.plotly_chart(risk_gauge(result["risk_score"]), use_container_width=True)

    with map_col:
        if demo is not None:
            dist_row = demo[demo["district"] == selected_district]
            if not dist_row.empty:
                lat = float(dist_row.iloc[0]["latitude"])
                lon = float(dist_row.iloc[0]["longitude"])
                detail_map = create_district_detail_map(lat, lon, selected_district, result["risk_score"])
                st_folium(detail_map, width=None, height=350, returned_objects=[])

    # Recommendation
    st.markdown("---")
    st.markdown("### 💡 Recommendation")
    rec_style = (
        "background: linear-gradient(135deg, rgba(239,71,111,0.15), rgba(255,209,102,0.1));"
        "border: 1px solid rgba(239,71,111,0.3); border-radius: 12px; padding: 20px;"
        if result["risk_level"] in ("Critical", "High")
        else "background: linear-gradient(135deg, rgba(6,214,160,0.15), rgba(17,138,178,0.1));"
        "border: 1px solid rgba(6,214,160,0.3); border-radius: 12px; padding: 20px;"
    )
    st.markdown(
        f'<div style="{rec_style}">{result["recommendation"]}</div>',
        unsafe_allow_html=True,
    )

    # Detailed table
    st.markdown("---")
    st.markdown("### 📊 Detailed Predictions")

    import pandas as pd
    detail_rows = []
    for resource, qty in result["predictions"].items():
        ci = result["confidence_interval"][resource]
        detail_rows.append({
            "Resource": resource.replace("_", " ").title(),
            "Predicted": f"{qty:,}",
            "Lower Bound": f"{ci['lower']:,}",
            "Upper Bound": f"{ci['upper']:,}",
        })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

else:
    # Default state
    st.info("👈 Configure forecast parameters in the sidebar and click **Generate Forecast**.")

    st.markdown("""
    ### How it works

    1. **Select a district** from Bihar's 38 districts
    2. **Set the forecast horizon** (1–30 days)
    3. **Input expected weather conditions** (rainfall, temperature, humidity)
    4. **Set disaster severity** based on current situation
    5. Click **Generate Forecast** to get AI-powered predictions

    The system uses LightGBM models trained on 8 years of historical data
    to predict resource requirements with confidence intervals.
    """)
