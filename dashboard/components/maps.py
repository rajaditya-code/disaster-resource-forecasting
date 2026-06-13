"""
Reusable map components for the Streamlit dashboard.

Uses Folium for interactive maps and GeoPandas for geospatial data handling.
"""

import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap


def create_bihar_risk_map(
    district_risks: list[dict],
    zoom_start: int = 7,
    center: tuple = (25.6, 85.5),
) -> folium.Map:
    """
    Create an interactive Folium map of Bihar showing district risk levels.

    Args:
        district_risks: List of dicts with keys:
            district, risk_score, risk_level, latitude, longitude, population
        zoom_start: Initial zoom level.
        center: Map center coordinates.

    Returns:
        folium.Map object.
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # ── Risk color mapping ──
    def risk_color(score: float) -> str:
        if score >= 70:
            return "#ef476f"  # Critical - red
        elif score >= 50:
            return "#ffd166"  # High - amber
        elif score >= 30:
            return "#06d6a0"  # Moderate - green
        else:
            return "#118ab2"  # Low - blue

    def risk_radius(score: float) -> float:
        return max(8, min(25, score / 100 * 30))

    # ── Add district markers ──
    for dist in district_risks:
        lat = dist.get("latitude", 0)
        lon = dist.get("longitude", 0)
        score = dist.get("risk_score", 0)
        level = dist.get("risk_level", "Unknown")
        population = dist.get("population", 0)
        name = dist.get("district", "Unknown")

        popup_html = f"""
        <div style="font-family: Inter, sans-serif; min-width: 200px;">
            <h4 style="margin:0; color: #333;">{name}</h4>
            <hr style="margin: 4px 0;">
            <b>Risk Score:</b> {score:.1f}/100<br>
            <b>Risk Level:</b> <span style="color:{risk_color(score)};
                font-weight:bold;">{level}</span><br>
            <b>Population:</b> {population:,}<br>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=risk_radius(score),
            color=risk_color(score),
            fill=True,
            fill_color=risk_color(score),
            fill_opacity=0.65,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{name}: {score:.1f}",
        ).add_to(m)

    # ── Add heatmap layer ──
    heat_data = [
        [d["latitude"], d["longitude"], d.get("risk_score", 0) / 100]
        for d in district_risks
        if d.get("latitude") and d.get("longitude")
    ]
    if heat_data:
        HeatMap(
            heat_data,
            radius=30,
            blur=20,
            max_zoom=10,
            gradient={
                "0.2": "#118ab2",
                "0.4": "#06d6a0",
                "0.6": "#ffd166",
                "0.8": "#ef476f",
                "1.0": "#d00000",
            },
        ).add_to(m)

    # ── Legend ──
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(30,30,30,0.9); padding: 12px 16px;
        border-radius: 8px; font-family: Inter, sans-serif;
        color: #e0e0e0; font-size: 13px; border: 1px solid #444;
    ">
        <b>Risk Levels</b><br>
        <span style="color:#118ab2;">●</span> Low (0–30)<br>
        <span style="color:#06d6a0;">●</span> Moderate (30–50)<br>
        <span style="color:#ffd166;">●</span> High (50–70)<br>
        <span style="color:#ef476f;">●</span> Critical (70–100)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_district_detail_map(
    lat: float,
    lon: float,
    district: str,
    risk_score: float,
) -> folium.Map:
    """Create a zoomed-in map for a single district."""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    def risk_color(score: float) -> str:
        if score >= 70:
            return "#ef476f"
        elif score >= 50:
            return "#ffd166"
        elif score >= 30:
            return "#06d6a0"
        return "#118ab2"

    folium.CircleMarker(
        location=[lat, lon],
        radius=20,
        color=risk_color(risk_score),
        fill=True,
        fill_color=risk_color(risk_score),
        fill_opacity=0.6,
        weight=3,
        tooltip=f"{district}: {risk_score:.1f}",
    ).add_to(m)

    folium.Marker(
        location=[lat, lon],
        popup=f"<b>{district}</b><br>Risk: {risk_score:.1f}",
        icon=folium.Icon(color="red" if risk_score > 50 else "blue", icon="info-sign"),
    ).add_to(m)

    return m
