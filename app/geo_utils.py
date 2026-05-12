"""
app/geo_utils.py
================
Geospatial helper functions — Africa Copperbelt edition.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT           = Path(__file__).resolve().parent.parent
MODELS_DIR     = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Prospectivity score for a single lat/lon ─────────────────────────────────

def score_point(
    lat: float,
    lon: float,
    predictions_df: pd.DataFrame,
    k: int = 4,
) -> dict[str, Any]:
    """
    Find the k nearest grid cells and return IDW-interpolated prospectivity
    score + feature values.
    """
    df = predictions_df.copy()
    df["_d"] = np.hypot(df.lon - lon, df.lat - lat)
    nearest = df.nsmallest(k, "_d")
    top     = nearest.iloc[0]

    dists   = nearest["_d"].values.clip(1e-6)
    weights = 1.0 / dists ** 2
    score   = float((weights * nearest["prospectivity_score"].values).sum() / weights.sum())

    feat_cols = [
        c for c in df.columns
        if c not in ("lon", "lat", "prospectivity_score", "risk_tier", "_d")
    ]
    features = {
        col: float((weights * nearest[col].values).sum() / weights.sum())
        for col in feat_cols
        if col in nearest.columns
    }

    if score >= 0.70:
        tier = "Very High"
    elif score >= 0.50:
        tier = "High"
    elif score >= 0.30:
        tier = "Moderate"
    else:
        tier = "Low"

    return {
        "lat":    lat,
        "lon":    lon,
        "score":  round(score, 4),
        "risk_tier": tier,
        "features": {k: round(v, 4) for k, v in features.items()},
        "nearest_grid_distance_km": round(float(top["_d"]) * 111.0, 2),
    }


# ── Feature labels ────────────────────────────────────────────────────────────

FEATURE_LABELS: dict[str, tuple[str, str, str]] = {
    # key: (display_name, unit, high_direction_note)
    "elevation_m":     ("Elevation",          "m",       "higher elevation"),
    "slope_deg":       ("Slope",              "deg",     "steeper terrain"),
    "log_cu_ppm":      ("Copper",             "log ppm", "elevated Cu signal"),
    "log_co_ppm":      ("Cobalt",             "log ppm", "elevated Co signal"),      # DRC Katanga
    "log_ni_ppm":      ("Nickel",             "log ppm", "elevated Ni signal"),      # Botswana
    "log_au_ppb":      ("Gold",               "log ppb", "elevated Au signal"),
    "fe_pct":          ("Iron",               "%",       "elevated Fe"),
    "log_pb_ppm":      ("Lead",               "log ppm", "elevated Pb"),
    "log_zn_ppm":      ("Zinc",               "log ppm", "elevated Zn"),
    "log_mo_ppm":      ("Molybdenum",         "log ppm", "elevated Mo"),
    "log_as_ppm":      ("Arsenic",            "log ppm", "elevated As"),
    "dist_fault_km":   ("Dist. to Fault",     "km",      "closer to fault"),
    "dist_deposit_km": ("Dist. to Deposit",   "km",      "closer to deposit"),
}


def humanise_features(raw_features: dict) -> list[dict]:
    out = []
    for key, val in raw_features.items():
        if key in FEATURE_LABELS:
            label, unit, direction = FEATURE_LABELS[key]
            out.append({
                "key":            key,
                "label":          label,
                "value":          val,
                "unit":           unit,
                "direction_note": direction,
            })
    return out


# ── Colour ramp ───────────────────────────────────────────────────────────────

SCORE_COLOURS = [
    (0.00, "#1a1a2e"),
    (0.25, "#16213e"),
    (0.40, "#0f3460"),
    (0.55, "#533483"),
    (0.65, "#e94560"),
    (0.75, "#f5a623"),
    (0.85, "#f7e018"),
    (1.00, "#ffffff"),
]


def score_to_hex(score: float) -> str:
    score = max(0.0, min(1.0, score))
    for i in range(len(SCORE_COLOURS) - 1):
        lo_s, lo_c = SCORE_COLOURS[i]
        hi_s, hi_c = SCORE_COLOURS[i + 1]
        if lo_s <= score <= hi_s + 1e-9:
            t = (score - lo_s) / (hi_s - lo_s + 1e-9)
            return _lerp_hex(lo_c, hi_c, t)
    return SCORE_COLOURS[-1][1]


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return f"#{int(r1+(r2-r1)*t):02x}{int(g1+(g2-g1)*t):02x}{int(b1+(b2-b1)*t):02x}"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def score_to_rgba(score: float, alpha: int = 180) -> list[int]:
    r, g, b = _hex_to_rgb(score_to_hex(score))
    return [r, g, b, alpha]


# ── Study-area bounds ─────────────────────────────────────────────────────────

# Central / Southern Africa Copperbelt study area
STUDY_BBOX = dict(minlon=15.0, minlat=-28.0, maxlon=38.0, maxlat=0.0)


def within_africa_study_area(lat: float, lon: float) -> bool:
    """Return True if coordinates fall within the Africa Copperbelt study bbox."""
    b = STUDY_BBOX
    return b["minlon"] <= lon <= b["maxlon"] and b["minlat"] <= lat <= b["maxlat"]


# Map centre: Zambia Copperbelt (Kitwe / Ndola area)
REGION_CENTER = {"lat": -13.0, "lon": 28.5}
