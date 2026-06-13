"""
💡 Recommendations – AI-generated actionable recommendations for all districts.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.prediction import predict_resources, get_all_district_risks, get_demographics

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")

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

st.markdown("# 💡 Resource Allocation Recommendations")
st.markdown("AI-generated, district-specific recommendations for pre-positioning disaster relief resources.")
st.markdown("---")

# ── Controls ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 1, 30, 7)
    risk_threshold = st.selectbox(
        "Show recommendations for",
        ["All Districts", "Critical & High Risk Only", "Critical Only"],
    )
    st.markdown("---")
    generate_btn = st.button("🔄 Generate Recommendations", use_container_width=True)

# ── Generate recommendations ──
if generate_btn or "recommendations" not in st.session_state:
    with st.spinner("Generating recommendations for all districts..."):
        try:
            district_risks = get_all_district_risks()

            # Filter by risk threshold
            if risk_threshold == "Critical & High Risk Only":
                district_risks = [d for d in district_risks if d["risk_level"] in ("Critical", "High")]
            elif risk_threshold == "Critical Only":
                district_risks = [d for d in district_risks if d["risk_level"] == "Critical"]

            recommendations = []
            for dist in district_risks:
                try:
                    result = predict_resources(
                        district=dist["district"],
                        forecast_horizon_days=forecast_days,
                        flood_severity=max(0, int(dist["risk_score"] / 25)),
                        rainfall_mm=max(0, dist["risk_score"] * 3),
                    )
                    recommendations.append(result)
                except Exception:
                    continue

            st.session_state["recommendations"] = recommendations
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.stop()

recommendations = st.session_state.get("recommendations", [])

if not recommendations:
    st.info("No recommendations to display. Click **Generate Recommendations** in the sidebar.")
    st.stop()

# ── Summary stats ──
total = len(recommendations)
critical_count = sum(1 for r in recommendations if r["risk_level"] == "Critical")
high_count = sum(1 for r in recommendations if r["risk_level"] == "High")
total_food = sum(r["predictions"]["food_kits"] for r in recommendations)
total_medical = sum(r["predictions"]["medical_kits"] for r in recommendations)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📋 Districts", total)
col2.metric("🔴 Critical", critical_count)
col3.metric("🟡 High", high_count)
col4.metric("🍱 Total Food Kits", f"{total_food:,}")
col5.metric("💊 Total Medical Kits", f"{total_medical:,}")

st.markdown("---")

# ── Recommendation cards ──
st.markdown(f"### 📑 District Recommendations ({forecast_days}-day forecast)")

for i, rec in enumerate(recommendations):
    risk_level = rec["risk_level"]
    risk_score = rec["risk_score"]
    district = rec["district"]
    preds = rec["predictions"]

    # Dynamic styling based on risk
    if risk_level == "Critical":
        gradient = "linear-gradient(135deg, rgba(239,71,111,0.2), rgba(208,0,0,0.1))"
        border_color = "rgba(239,71,111,0.5)"
        icon = "🔴"
    elif risk_level == "High":
        gradient = "linear-gradient(135deg, rgba(255,209,102,0.2), rgba(255,150,50,0.1))"
        border_color = "rgba(255,209,102,0.5)"
        icon = "🟡"
    elif risk_level == "Moderate":
        gradient = "linear-gradient(135deg, rgba(6,214,160,0.15), rgba(6,214,160,0.05))"
        border_color = "rgba(6,214,160,0.4)"
        icon = "🟢"
    else:
        gradient = "linear-gradient(135deg, rgba(17,138,178,0.15), rgba(17,138,178,0.05))"
        border_color = "rgba(17,138,178,0.4)"
        icon = "🔵"

    with st.expander(f"{icon} **{district}** — Risk: {risk_score:.1f} ({risk_level})", expanded=(risk_level in ("Critical", "High"))):
        # Recommendation text
        st.markdown(f"""
        <div style="
            background: {gradient};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            line-height: 1.6;
        ">
            {rec['recommendation']}
        </div>
        """, unsafe_allow_html=True)

        # Resource breakdown
        res_cols = st.columns(5)
        res_cols[0].metric("🍱 Food Kits", f"{preds['food_kits']:,}")
        res_cols[1].metric("💊 Medical Kits", f"{preds['medical_kits']:,}")
        res_cols[2].metric("💧 ORS Packets", f"{preds['ors_packets']:,}")
        res_cols[3].metric("🚰 Water (L)", f"{preds['drinking_water_litres']:,}")
        res_cols[4].metric("🏕️ Tarpaulins", f"{preds['tarpaulins']:,}")

# ── Export ──
st.markdown("---")
st.markdown("### 📥 Export Recommendations")

export_data = []
for rec in recommendations:
    export_data.append({
        "District": rec["district"],
        "Risk Score": rec["risk_score"],
        "Risk Level": rec["risk_level"],
        "Food Kits": rec["predictions"]["food_kits"],
        "Medical Kits": rec["predictions"]["medical_kits"],
        "ORS Packets": rec["predictions"]["ors_packets"],
        "Drinking Water (L)": rec["predictions"]["drinking_water_litres"],
        "Tarpaulins": rec["predictions"]["tarpaulins"],
        "Recommendation": rec["recommendation"],
    })

export_df = pd.DataFrame(export_data)
csv = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download as CSV",
    data=csv,
    file_name=f"disaster_recommendations_{forecast_days}d.csv",
    mime="text/csv",
)
