"""
Reusable chart components for the Streamlit dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def resource_bar_chart(predictions: dict, title: str = "Predicted Resource Requirements") -> go.Figure:
    """Horizontal bar chart of predicted resource quantities."""
    labels = {
        "food_kits": "🍱 Food Kits",
        "medical_kits": "💊 Medical Kits",
        "ors_packets": "💧 ORS Packets",
        "drinking_water_litres": "🚰 Drinking Water (L)",
        "tarpaulins": "🏕️ Tarpaulins",
    }

    names = [labels.get(k, k) for k in predictions.keys()]
    values = list(predictions.values())

    colors = ["#4361ee", "#3a0ca3", "#7209b7", "#f72585", "#4cc9f0"]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors[:len(names)],
        text=[f"{v:,}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e0e0e0")),
        xaxis_title="Quantity",
        yaxis_title="",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def confidence_interval_chart(
    predictions: dict,
    confidence: dict,
    title: str = "Predictions with Confidence Intervals",
) -> go.Figure:
    """Bar chart with error bars showing confidence intervals."""
    labels = list(predictions.keys())
    values = list(predictions.values())
    lower = [confidence[k]["lower"] for k in labels]
    upper = [confidence[k]["upper"] for k in labels]
    errors = [(v - lo, hi - v) for v, lo, hi in zip(values, lower, upper)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[e[1] for e in errors],
            arrayminus=[e[0] for e in errors],
            color="#f72585",
        ),
        marker_color="#4361ee",
        text=[f"{v:,}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e0e0e0")),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def feature_importance_chart(importances: dict, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing top-N feature importances."""
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [item[0] for item in reversed(sorted_imp)]
    scores = [item[1] for item in reversed(sorted_imp)]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale="Viridis",
        ),
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importances", font=dict(size=18, color="#e0e0e0")),
        xaxis_title="Importance Score",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def metrics_table(metrics: dict) -> pd.DataFrame:
    """Format model metrics as a styled DataFrame."""
    rows = []
    for target, m in metrics.items():
        rows.append({
            "Resource": target.replace("_", " ").title(),
            "MAE": f"{m['mae']:,.2f}",
            "RMSE": f"{m['rmse']:,.2f}",
            "MAPE (%)": f"{m['mape']:.2f}",
        })
    return pd.DataFrame(rows)


def risk_gauge(risk_score: float) -> go.Figure:
    """Semi-circular gauge chart for risk score."""
    if risk_score >= 70:
        color = "#ef476f"
        level = "Critical"
    elif risk_score >= 50:
        color = "#ffd166"
        level = "High"
    elif risk_score >= 30:
        color = "#06d6a0"
        level = "Moderate"
    else:
        color = "#118ab2"
        level = "Low"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title=dict(text=f"Risk Level: {level}", font=dict(size=20, color="#e0e0e0")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#555"),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="rgba(17,138,178,0.2)"),
                dict(range=[30, 50], color="rgba(6,214,160,0.2)"),
                dict(range=[50, 70], color="rgba(255,209,102,0.2)"),
                dict(range=[70, 100], color="rgba(239,71,111,0.2)"),
            ],
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    )
    return fig


def trend_line_chart(df: pd.DataFrame, y_col: str, title: str = "") -> go.Figure:
    """Time series line chart for a resource or climate variable."""
    fig = px.line(
        df, x="date", y=y_col,
        title=title,
        template="plotly_dark",
        color_discrete_sequence=["#4361ee"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        font=dict(family="Inter, sans-serif"),
        title_font=dict(size=18, color="#e0e0e0"),
    )
    fig.update_traces(line_width=2)
    return fig
