"""
📈 Analytics – Historical trends, feature importance, and model performance.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.explainability import get_feature_importances, get_metrics
from dashboard.components.charts import (
    feature_importance_chart,
    metrics_table,
    trend_line_chart,
)
from utils.data_pipeline import RESOURCE_COLS

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

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

st.markdown("# 📈 Analytics & Model Insights")
st.markdown("Explore historical trends, feature importances, and model performance.")
st.markdown("---")

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["📊 Historical Trends", "🧠 Feature Importance", "🎯 Model Performance"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Historical Trends
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Historical Resource & Climate Trends")

    # Load processed data
    processed_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    if processed_path.exists():
        df = pd.read_csv(processed_path, parse_dates=["date"])
        districts = sorted(df["district"].unique())

        sel_district = st.selectbox("Select District", districts, index=districts.index("Patna") if "Patna" in districts else 0, key="trend_dist")
        dist_df = df[df["district"] == sel_district].sort_values("date")

        # Aggregate to monthly for cleaner trends
        monthly = dist_df.set_index("date").resample("ME").mean(numeric_only=True).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                trend_line_chart(monthly, "rainfall_mm", "🌧️ Monthly Rainfall (mm)"),
                use_container_width=True,
            )
            st.plotly_chart(
                trend_line_chart(monthly, "flood_severity", "🌊 Monthly Avg Flood Severity"),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                trend_line_chart(monthly, "risk_score", "⚠️ Monthly Avg Risk Score"),
                use_container_width=True,
            )
            if "food_kits" in monthly.columns:
                st.plotly_chart(
                    trend_line_chart(monthly, "food_kits", "🍱 Monthly Avg Food Kits"),
                    use_container_width=True,
                )

        # Resource correlation heatmap
        st.markdown("### 🔗 Resource Correlation Matrix")
        available_resource_cols = [c for c in RESOURCE_COLS if c in dist_df.columns]
        corr_cols = ["rainfall_mm", "flood_severity", "risk_score"] + available_resource_cols
        corr_matrix = dist_df[corr_cols].corr()

        import plotly.express as px
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            title="Correlation Heatmap",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Processed data not found. Run `python train.py` first.")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🧠 Feature Importance Analysis")

    all_importances = get_feature_importances()

    if all_importances:
        target_options = list(all_importances.keys())
        selected_target = st.selectbox(
            "Select Resource Target",
            target_options,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        if selected_target in all_importances:
            imp = all_importances[selected_target]
            st.plotly_chart(
                feature_importance_chart(imp, top_n=20),
                use_container_width=True,
            )

            # Feature importance table
            st.markdown("#### Top 20 Features")
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
            imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
            imp_df.index = range(1, len(imp_df) + 1)
            st.dataframe(imp_df, use_container_width=True)
    else:
        st.warning("No feature importances available. Train models first.")

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🎯 Model Evaluation Metrics")

    model_metrics = get_metrics()

    if model_metrics:
        # Summary metrics
        st.dataframe(
            metrics_table(model_metrics),
            use_container_width=True,
            hide_index=True,
        )

        # Visual comparison
        st.markdown("### 📊 Metric Comparison Across Models")

        import plotly.graph_objects as go

        targets = list(model_metrics.keys())
        mae_vals = [model_metrics[t]["mae"] for t in targets]
        rmse_vals = [model_metrics[t]["rmse"] for t in targets]
        mape_vals = [model_metrics[t]["mape"] for t in targets]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MAE", x=[t.replace("_", " ").title() for t in targets],
            y=mae_vals, marker_color="#4361ee",
        ))
        fig.add_trace(go.Bar(
            name="RMSE", x=[t.replace("_", " ").title() for t in targets],
            y=rmse_vals, marker_color="#f72585",
        ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            title=dict(text="MAE vs RMSE by Resource", font=dict(size=18, color="#e0e0e0")),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # MAPE chart
        fig2 = go.Figure(go.Bar(
            x=[t.replace("_", " ").title() for t in targets],
            y=mape_vals,
            marker_color="#7209b7",
            text=[f"{v:.1f}%" for v in mape_vals],
            textposition="outside",
        ))
        fig2.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            title=dict(text="MAPE (%) by Resource", font=dict(size=18, color="#e0e0e0")),
            yaxis_title="MAPE (%)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No model metrics found. Run training pipeline first.")
