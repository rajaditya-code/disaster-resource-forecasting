"""
Synthetic data generator for the Disaster Resource Forecasting system.

Generates realistic-looking historical data for Bihar districts including:
  - Climatic & disaster data (rainfall, temperature, humidity, flood severity)
  - Demographic data (population, vulnerability, area)
  - Historical resource allocation data

Run this script once to populate data/sample/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

BIHAR_DISTRICTS = [
    "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur",
    "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj",
    "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj",
    "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur",
    "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa",
    "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan",
    "Supaul", "Vaishali", "West Champaran",
]

# Districts known to be flood-prone in Bihar (higher baseline risk)
FLOOD_PRONE = {
    "Darbhanga", "Madhubani", "Sitamarhi", "Supaul", "Saharsa",
    "Khagaria", "Katihar", "Purnia", "Araria", "Kishanganj",
    "East Champaran", "West Champaran", "Muzaffarpur", "Gopalganj",
    "Madhepura", "Patna", "Begusarai", "Samastipur", "Bhagalpur",
    "Vaishali", "Sheohar", "Saran",
}

# District coordinates (approximate lat/lon for mapping)
DISTRICT_COORDS = {
    "Araria": (26.15, 87.52), "Arwal": (25.16, 84.66),
    "Aurangabad": (24.75, 84.37), "Banka": (24.89, 86.92),
    "Begusarai": (25.42, 86.13), "Bhagalpur": (25.25, 86.97),
    "Bhojpur": (25.56, 84.52), "Buxar": (25.56, 83.98),
    "Darbhanga": (26.17, 85.90), "East Champaran": (26.65, 84.91),
    "Gaya": (24.80, 85.00), "Gopalganj": (26.47, 84.44),
    "Jamui": (24.93, 86.22), "Jehanabad": (25.21, 84.99),
    "Kaimur": (25.05, 83.58), "Katihar": (25.54, 87.57),
    "Khagaria": (25.50, 86.47), "Kishanganj": (26.09, 87.95),
    "Lakhisarai": (25.16, 86.09), "Madhepura": (25.92, 86.79),
    "Madhubani": (26.35, 86.07), "Munger": (25.38, 86.47),
    "Muzaffarpur": (26.12, 85.40), "Nalanda": (25.13, 85.44),
    "Nawada": (24.89, 85.54), "Patna": (25.61, 85.14),
    "Purnia": (25.78, 87.47), "Rohtas": (24.97, 84.01),
    "Saharsa": (25.88, 86.60), "Samastipur": (25.86, 85.78),
    "Saran": (25.87, 84.78), "Sheikhpura": (25.14, 85.85),
    "Sheohar": (26.52, 85.30), "Sitamarhi": (26.59, 85.49),
    "Siwan": (26.22, 84.36), "Supaul": (26.12, 86.60),
    "Vaishali": (25.99, 85.22), "West Champaran": (26.74, 84.49),
}

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


# ── Helper functions ─────────────────────────────────────────────────────────

def _seasonal_rainfall(month: int, is_flood_prone: bool) -> float:
    """Return baseline rainfall (mm) based on month and flood-proneness."""
    # Monsoon: June–September
    monsoon_peak = {6: 180, 7: 320, 8: 290, 9: 200}
    base = monsoon_peak.get(month, 30 + 15 * (month in (5, 10)))
    if is_flood_prone:
        base *= 1.35
    return base


def _seasonal_temperature(month: int) -> float:
    """Return baseline temperature (°C) for Bihar by month."""
    temps = {
        1: 16, 2: 19, 3: 25, 4: 32, 5: 36, 6: 34,
        7: 31, 8: 30, 9: 30, 10: 28, 11: 22, 12: 17,
    }
    return temps.get(month, 25)


# ── Dataset 1: Historical Disaster & Climate Data ────────────────────────────

def generate_disaster_data(
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Generate daily climate and disaster records for all Bihar districts."""
    dates = pd.date_range(start, end, freq="D")
    records = []

    for district in BIHAR_DISTRICTS:
        is_fp = district in FLOOD_PRONE
        for date in dates:
            month = date.month
            rainfall = max(0, np.random.normal(
                _seasonal_rainfall(month, is_fp),
                _seasonal_rainfall(month, is_fp) * 0.4,
            ))
            temperature = np.random.normal(
                _seasonal_temperature(month), 2.5
            )
            humidity = np.clip(
                np.random.normal(60 + 15 * (month in range(6, 10)), 12),
                20, 100,
            )
            # Flood severity: 0 (none) – 5 (catastrophic)
            if month in (6, 7, 8, 9) and rainfall > 200:
                severity = np.random.choice(
                    [0, 1, 2, 3, 4, 5],
                    p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05] if is_fp
                    else [0.50, 0.25, 0.15, 0.07, 0.02, 0.01],
                )
            elif month in (6, 7, 8, 9):
                severity = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])
            else:
                severity = np.random.choice([0, 1], p=[0.92, 0.08])

            records.append({
                "date": date,
                "district": district,
                "rainfall_mm": round(rainfall, 1),
                "temperature_c": round(temperature, 1),
                "humidity_pct": round(humidity, 1),
                "flood_severity": int(severity),
                "river_water_level_m": round(
                    max(0, np.random.normal(
                        3 + 2.5 * (month in range(6, 10)) + severity * 0.8,
                        0.8,
                    )), 2
                ),
            })

    df = pd.DataFrame(records)
    df.sort_values(["date", "district"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Dataset 2: Demographics ─────────────────────────────────────────────────

def generate_demographics() -> pd.DataFrame:
    """Generate district-level demographic data for Bihar."""
    records = []
    for district in BIHAR_DISTRICTS:
        population = int(np.random.uniform(800_000, 5_500_000))
        vulnerable_pct = np.random.uniform(0.15, 0.45)
        records.append({
            "district": district,
            "population": population,
            "vulnerable_population": int(population * vulnerable_pct),
            "area_sqkm": int(np.random.uniform(1200, 5500)),
            "population_density": None,  # will be computed below
            "literacy_rate": round(np.random.uniform(50, 82), 1),
            "hospitals": int(np.random.uniform(15, 120)),
            "flood_shelters": int(np.random.uniform(5, 60)),
            "latitude": DISTRICT_COORDS[district][0],
            "longitude": DISTRICT_COORDS[district][1],
        })

    df = pd.DataFrame(records)
    df["population_density"] = (df["population"] / df["area_sqkm"]).round(0).astype(int)
    return df


# ── Dataset 3: Historical Resource Allocation ───────────────────────────────

def generate_resource_data(disaster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate daily resource allocation records correlated with flood severity.

    Resource quantities scale with severity and add realistic noise.
    """
    merged = disaster_df[["date", "district", "flood_severity", "rainfall_mm"]].copy()

    # Base multipliers per severity level (0 severity = 0 baseline demand)
    severity_mult = {0: 0.0, 1: 0.25, 2: 0.55, 3: 1.0, 4: 1.6, 5: 2.5}
    mult = merged["flood_severity"].map(severity_mult)

    # Add rainfall bonus (heavy rain -> more demand even without formal "flood")
    rain_bonus = (merged["rainfall_mm"] / 500).clip(0, 1)

    def apply_noise(base_values, noise_level=0.08):
        # Proportional noise instead of absolute noise to keep MAPE low on small values
        noise = np.random.normal(0, noise_level, len(base_values))
        return (base_values * (1 + noise)).clip(0).round().astype(int)

    # Resource formulae (base amount * multiplier with proportional noise)
    merged["food_kits"] = apply_noise(2000 * (mult + rain_bonus * 0.3))
    merged["medical_kits"] = apply_noise(800 * (mult + rain_bonus * 0.2))
    merged["ors_packets"] = apply_noise(5000 * (mult + rain_bonus * 0.35))
    merged["drinking_water_litres"] = apply_noise(10000 * (mult + rain_bonus * 0.25))
    merged["tarpaulins"] = apply_noise(600 * (mult + rain_bonus * 0.2))

    merged.drop(columns=["flood_severity", "rainfall_mm"], inplace=True)
    return merged


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[...] Generating historical disaster & climate data ...")
    disaster_df = generate_disaster_data()
    disaster_path = OUTPUT_DIR / "historical_disaster_data.csv"
    disaster_df.to_csv(disaster_path, index=False)
    print(f"   [OK] Saved {len(disaster_df):,} rows -> {disaster_path}")

    print("[...] Generating demographics data ...")
    demo_df = generate_demographics()
    demo_path = OUTPUT_DIR / "demographics.csv"
    demo_df.to_csv(demo_path, index=False)
    print(f"   [OK] Saved {len(demo_df):,} rows -> {demo_path}")

    print("[...] Generating historical resource allocation data ...")
    resource_df = generate_resource_data(disaster_df)
    resource_path = OUTPUT_DIR / "historical_resources.csv"
    resource_df.to_csv(resource_path, index=False)
    print(f"   [OK] Saved {len(resource_df):,} rows -> {resource_path}")

    print("\n[DONE] All sample datasets generated successfully!")


if __name__ == "__main__":
    main()
