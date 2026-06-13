"""
📊 Overview – System status, high-risk districts, and forecast summary.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.prediction import get_all_district_risks, get_demographics
from backend.services.explainability import get_metrics
from dashboard.components.charts import risk_gauge, metrics_table

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# ── Custom CSS (same theme) ──
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 System Overview")
st.markdown("Real-time status of disaster monitoring across Bihar districts.")
st.markdown("---")

# ── Load data ──
try:
    district_risks = get_all_district_risks()
    demographics = get_demographics()
    model_metrics = get_metrics()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you've run `python train.py` to generate model artifacts.")
    st.stop()

# ── Key metrics ──
total_districts = len(district_risks)
critical_districts = sum(1 for d in district_risks if d["risk_level"] == "Critical")
high_districts = sum(1 for d in district_risks if d["risk_level"] == "High")
avg_risk = sum(d["risk_score"] for d in district_risks) / max(total_districts, 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏘️ Total Districts", total_districts)
col2.metric("🔴 Critical Risk", critical_districts)
col3.metric("🟡 High Risk", high_districts)
col4.metric("📊 Avg Risk Score", f"{avg_risk:.1f}")

st.markdown("---")

# ── Risk distribution ──
st.markdown("### 🎯 District Risk Distribution")

risk_col1, risk_col2 = st.columns([2, 1])

with risk_col1:
    risk_df = pd.DataFrame(district_risks)
    if not risk_df.empty:
        risk_df = risk_df.sort_values("risk_score", ascending=False)

        # Color-coded table
        def color_risk(val):
            if val == "Critical":
                return "color: #ef476f; font-weight: bold;"
            elif val == "High":
                return "color: #ffd166; font-weight: bold;"
            elif val == "Moderate":
                return "color: #06d6a0; font-weight: bold;"
            return "color: #118ab2;"

        display_df = risk_df[["district", "risk_score", "risk_level", "population"]].copy()
        display_df.columns = ["District", "Risk Score", "Risk Level", "Population"]
        display_df["Population"] = display_df["Population"].apply(lambda x: f"{x:,}")
        display_df["Risk Score"] = display_df["Risk Score"].apply(lambda x: f"{x:.1f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            hide_index=True,
        )

with risk_col2:
    st.plotly_chart(risk_gauge(avg_risk), use_container_width=True)

    # Risk level counts
    st.markdown("#### Risk Breakdown")
    for level, emoji, color in [
        ("Critical", "🔴", "#ef476f"),
        ("High", "🟡", "#ffd166"),
        ("Moderate", "🟢", "#06d6a0"),
        ("Low", "🔵", "#118ab2"),
    ]:
        count = sum(1 for d in district_risks if d["risk_level"] == level)
        st.markdown(
            f"<span style='color:{color}; font-size:1.1rem;'>"
            f"{emoji} **{level}**: {count} districts</span>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Model performance summary ──
st.markdown("### 🤖 Model Performance Summary")

if model_metrics:
    st.dataframe(
        metrics_table(model_metrics),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No model metrics available. Run training pipeline first.")

# ── Top at-risk districts ──
st.markdown("---")
st.markdown("### ⚠️ Top 5 At-Risk Districts")

if district_risks:
    top5 = district_risks[:5]
    cols = st.columns(5)
    for i, dist in enumerate(top5):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,
                    rgba(239,71,111,{0.3 - i*0.04}),
                    rgba(114,9,183,0.1));
                border: 1px solid rgba(239,71,111,{0.5 - i*0.08});
                border-radius: 12px;
                padding: 16px;
                text-align: center;
            ">
                <h3 style="margin:0; font-size:1.1rem;">{dist['district']}</h3>
                <h2 style="margin:4px 0; color:#ef476f;">{dist['risk_score']:.1f}</h2>
                <p style="color:#b0b0b0; margin:0; font-size:0.85rem;">
                    {dist['risk_level']}
                </p>
            </div>
            """, unsafe_allow_html=True)
