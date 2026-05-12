"""
scripts/01_download_data.py
===========================
Phase 1 – Data Acquisition  (Africa Copperbelt Edition)

Downloads / generates:
  1. USGS MRDS global mineral deposits (Cu / Co / Ni, Africa bbox)
       → data/raw/mrds_africa_copper.geojson
  2. GEM Global Active Faults (Africa subset)
       → data/raw/faults_africa.geojson
  3. Soil geochemistry – Africa Copperbelt calibrated dataset
       → data/raw/geochem_africa.csv

Region  : Central / Southern Africa — Zambia Copperbelt, DRC Katanga,
          Botswana, Zimbabwe, Mozambique, Namibia  (15 °E–38 °E, 28 °S–0 °)
Metals  : Copper · Cobalt · Nickel   (KoBold Metals critical-mineral focus)

Data sources
------------
Deposits  : USGS MRDS WFS  (global endpoint; Africa bbox filter)
            https://mrdata.usgs.gov/services/wfs/mrds
Faults    : GEM Global Active Faults v2023
            https://github.com/GEMScienceTools/gem-global-active-faults
Geochem   : Calibrated synthetic dataset.  Background statistics drawn from:
              • Tembo et al. (2009) — Zambia Copperbelt soil geochemistry
              • Cailteux et al. (2005) — Katanga Supergroup Cu-Co systems
              • USGS Professional Paper 1802, Chapter Africa
            No free global API provides Africa soil geochemistry at NGDB
            resolution; synthetic generation is documented and reproducible.
Elevation : srtm.py (SRTM global 90 m DEM) — fetched in Phase 2.

All sources fall back to geologically calibrated synthetic data when
live endpoints are unreachable, so the pipeline always completes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Spatial extent: Central / Southern Africa Copperbelt ────────────────────
BBOX = dict(minlon=15.0, minlat=-28.0, maxlon=38.0, maxlat=0.0)

# Known Cu / Co / Ni districts — bias synthetic fallback placement
# Sources: USGS MRDS, SNL Metals & Mining, BGS World Mineral Statistics 2023
MINERAL_DISTRICTS: list[tuple] = [
    # (center_lon, center_lat, radius_deg, district_name, primary_commodity)

    # ── Zambia Copperbelt ────────────────────────────────────────────────────
    (27.85, -12.37, 0.50, "Chililabombwe-Mingomba",  "Cu"),     # KoBold Mingomba
    (28.17, -12.53, 0.60, "Nchanga-Chingola",         "Cu"),
    (28.63, -12.56, 0.50, "Mufulira",                 "Cu"),
    (28.22, -12.82, 0.50, "Nkana-Kitwe",              "Cu-Co"),
    (28.87, -12.27, 0.40, "Ndola",                    "Cu"),
    (28.97, -11.47, 0.40, "Luanshya",                 "Cu"),
    (27.46, -12.17, 0.50, "Kansanshi",                "Cu-Au"),
    (29.36, -14.45, 0.40, "Konkola",                  "Cu"),
    (28.50, -13.50, 0.35, "Nampundwe",                "Cu"),

    # ── DRC Katanga Copperbelt ───────────────────────────────────────────────
    (25.47, -10.73, 0.70, "Kolwezi",                  "Cu-Co"),
    (26.97, -11.67, 0.50, "Lubumbashi",               "Cu-Co"),
    (26.73, -10.93, 0.50, "Likasi",                   "Cu-Co"),
    (26.11, -10.60, 0.60, "Tenke-Fungurume",          "Cu-Co"),
    (27.25, -10.85, 0.40, "Kambove",                  "Cu-Co"),
    (26.60, -9.75,  0.40, "Kinsenda-Musoshi",         "Cu"),
    (25.90, -10.30, 0.45, "Dikulushi",                "Cu-Ag"),
    (27.10, -12.20, 0.35, "Boss-Mining",              "Cu-Co"),
    (26.30, -11.20, 0.40, "Ruashi",                   "Cu-Co"),

    # ── Botswana ─────────────────────────────────────────────────────────────
    (27.83, -22.00, 0.50, "Selebi-Phikwe",            "Cu-Ni"),
    (24.65, -21.55, 0.40, "Boseto-Phoenix",            "Cu"),
    (26.00, -20.00, 0.35, "Khoemacau",                "Cu-Ag"),

    # ── Namibia ──────────────────────────────────────────────────────────────
    (17.70, -19.50, 0.50, "Tsumeb",                   "Cu-Pb-Zn"),
    (15.60, -22.20, 0.40, "Matchless Belt",           "Cu"),
    (17.00, -18.50, 0.35, "Otjihase",                 "Cu"),

    # ── Zimbabwe ─────────────────────────────────────────────────────────────
    (30.00, -19.50, 0.40, "Empress",                  "Cu"),
    (29.40, -18.00, 0.35, "Alaska Mine",              "Cu-Co"),

    # ── Mozambique ───────────────────────────────────────────────────────────
    (35.20, -18.00, 0.40, "Tete Province",            "Cu"),
    (36.60, -16.20, 0.30, "Montepuez Area",           "Cu"),
]

TIMEOUT = 25
GEM_TIMEOUT = 120   # GEM GeoJSON is ~40 MB


# ═══════════════════════════════════════════════════════════════════════════════
# 1. USGS MRDS — Cu / Co / Ni mineral deposits (global endpoint, Africa bbox)
# ═══════════════════════════════════════════════════════════════════════════════

MRDS_WFS = (
    "https://mrdata.usgs.gov/services/wfs/mrds"
    "?service=WFS&version=1.1.0&request=GetFeature"
    "&typeName=mrds"
    "&BBOX={minlon},{minlat},{maxlon},{maxlat},EPSG:4326"
    "&outputFormat=application/json"
    "&count=5000"
)


def _fetch_mrds_wfs(bbox: dict) -> gpd.GeoDataFrame | None:
    url = MRDS_WFS.format(**bbox)
    log.info("Requesting USGS MRDS (Africa bbox) …")
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        fc = r.json()
        if not fc.get("features"):
            return None
        gdf = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
        log.info("  MRDS returned %d raw features", len(gdf))
        return gdf
    except Exception as exc:
        log.warning("  MRDS WFS unavailable: %s", exc)
        return None


def _filter_critical_minerals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Retain Cu / Co / Ni deposits."""
    pattern = r"\b(CU|COPPER|CO|COBALT|NI|NICKEL)\b"
    mask = pd.Series(False, index=gdf.index)
    for col in ("commod1", "commod2", "commod3", "commod_list", "commodities"):
        if col in gdf.columns:
            mask |= (
                gdf[col].astype(str).str.upper().str.contains(pattern, na=False, regex=True)
            )
    result = gdf[mask].copy()
    log.info("  Cu/Co/Ni deposits after commodity filter: %d", len(result))
    return result


def _synthetic_deposits(n: int = 420) -> gpd.GeoDataFrame:
    """
    Synthetic Cu/Co/Ni deposits biased toward known African mineral districts.
    75 % within district clusters, 25 % random background.
    """
    log.info("Generating %d synthetic African mineral deposits …", n)
    rng = np.random.default_rng(42)
    lons, lats, names, commodities = [], [], [], []

    n_clustered = int(n * 0.75)
    n_random = n - n_clustered
    per_district = max(1, n_clustered // len(MINERAL_DISTRICTS))

    for clon, clat, rad, dname, commod in MINERAL_DISTRICTS:
        for _ in range(per_district):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, rad)
            lons.append(clon + r * np.cos(angle))
            lats.append(clat + r * np.sin(angle))
            names.append(dname)
            commodities.append(commod)

    lons += rng.uniform(BBOX["minlon"], BBOX["maxlon"], n_random).tolist()
    lats += rng.uniform(BBOX["minlat"], BBOX["maxlat"], n_random).tolist()
    names += ["background"] * n_random
    commodities += ["Cu"] * n_random

    gdf = gpd.GeoDataFrame(
        {
            "dep_id":   [f"AFR-{i:04d}" for i in range(len(lons))],
            "name":     [f"Synthetic deposit {i}" for i in range(len(lons))],
            "district": names,
            "commod1":  commodities,
            "source":   "synthetic_africa",
        },
        geometry=[Point(lo, la) for lo, la in zip(lons, lats)],
        crs="EPSG:4326",
    )
    gdf = gdf.cx[BBOX["minlon"]:BBOX["maxlon"], BBOX["minlat"]:BBOX["maxlat"]]
    return gdf.reset_index(drop=True)


def download_deposits(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "mrds_africa_copper.geojson"
    if out.exists() and not force:
        log.info("Deposits cache hit → %s", out)
        return gpd.read_file(out)

    gdf = _fetch_mrds_wfs(BBOX)
    if gdf is not None and len(gdf) > 10:
        gdf = _filter_critical_minerals(gdf)
    if gdf is None or len(gdf) < 10:
        log.info("Falling back to synthetic Africa deposits.")
        gdf = _synthetic_deposits(420)

    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved deposits → %s  (%d rows)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Soil geochemistry — Africa Copperbelt calibrated dataset
# ═══════════════════════════════════════════════════════════════════════════════
#
# No free globally-accessible API provides Africa soil geochemistry at NGDB
# resolution.  Calibrated synthetic data with published background statistics:
#
#   Element  Background   Anomaly threshold   Reference
#   Cu       50–150 ppm   ≥ 500 ppm           Tembo et al. (2009)
#   Co       30–80 ppm    ≥ 100 ppm           Cailteux et al. (2005)
#   Ni       25–75 ppm    ≥ 100 ppm           Maier et al. (2013)
#   Zn       80–200 ppm   ≥ 300 ppm           USGS Prof. Paper 1802
#   Fe       3–7 %        —                   Lateritic tropical soils
#   Au       0.3–2 ppb    —                   Non-porphyry SHSC systems
#   As       5–25 ppm     —

AFRICA_GEOCHEM_BG: dict[str, tuple[float, float]] = {
    # (log-normal median, log-sigma)
    "cu_ppm":  (80,   0.65),
    "co_ppm":  (45,   0.70),   # cobalt — hallmark of DRC Katanga systems
    "ni_ppm":  (40,   0.65),   # nickel — Botswana Cu-Ni focus
    "au_ppb":  (0.8,  0.80),
    "fe_pct":  (4.5,  0.35),
    "pb_ppm":  (25,   0.60),
    "zn_ppm":  (100,  0.55),
    "mo_ppm":  (0.6,  0.65),   # Mo low — SHSC, not porphyry
    "as_ppm":  (12,   0.75),
}

CLIP_LIMITS: dict[str, tuple[float, float]] = {
    "cu_ppm":  (5,    50_000),
    "co_ppm":  (2,    15_000),
    "ni_ppm":  (2,     8_000),
    "au_ppb":  (0.1,   3_000),
    "fe_pct":  (0.5,      25),
    "pb_ppm":  (2,     3_000),
    "zn_ppm":  (5,    10_000),
    "mo_ppm":  (0.1,     500),
    "as_ppm":  (1,     2_000),
}


def _synthetic_geochem(n: int = 4000) -> pd.DataFrame:
    log.info("Generating %d Africa Copperbelt geochemistry samples …", n)
    rng = np.random.default_rng(123)

    lons = rng.uniform(BBOX["minlon"], BBOX["maxlon"], n)
    lats = rng.uniform(BBOX["minlat"], BBOX["maxlat"], n)

    arrays: dict[str, np.ndarray] = {}
    for elem, (med, sigma) in AFRICA_GEOCHEM_BG.items():
        arrays[elem] = rng.lognormal(mean=np.log(med), sigma=sigma, size=n)

    for clon, clat, rad, _, commod in MINERAL_DISTRICTS:
        dist = np.hypot(lons - clon, lats - clat)
        mask = dist < rad * 2.0
        if not mask.any():
            continue
        proximity = np.maximum(0.0, 1.0 - dist[mask] / (rad * 2.0))

        arrays["cu_ppm"][mask] *= np.exp(3.5 * proximity)
        arrays["co_ppm"][mask] *= np.exp(3.0 * proximity if "Co" in commod else 1.2 * proximity)
        arrays["ni_ppm"][mask] *= np.exp(2.5 * proximity if "Ni" in commod else 0.8 * proximity)
        arrays["au_ppb"][mask] *= np.exp(1.5 * proximity)
        arrays["zn_ppm"][mask] *= np.exp(2.0 * proximity)
        arrays["as_ppm"][mask] *= np.exp(1.5 * proximity)

    df = pd.DataFrame({"lon": lons, "lat": lats, "source": "synthetic_copperbelt"})
    for elem, (lo, hi) in CLIP_LIMITS.items():
        df[elem] = arrays[elem].clip(lo, hi)
    return df


def download_geochem(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "geochem_africa.csv"
    if out.exists() and not force:
        log.info("Geochem cache hit → %s", out)
        return pd.read_csv(out)

    df = _synthetic_geochem(4000)
    df.to_csv(out, index=False)
    log.info("Saved geochem → %s  (%d rows)", out.name, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GEM Global Active Faults — Africa subset
# ═══════════════════════════════════════════════════════════════════════════════

GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults"
    "/master/geojson/gem_active_faults_harmonized.geojson"
)


def _fetch_gem_faults(bbox: dict) -> gpd.GeoDataFrame | None:
    """Download GEM Global Active Faults and clip to Africa study bbox."""
    log.info("Downloading GEM Global Active Faults (~40 MB, may take ~60 s) …")
    try:
        r = requests.get(GEM_FAULTS_URL, timeout=GEM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
        log.info("  GEM: %d global fault segments downloaded", len(gdf))

        from shapely.geometry import box as shapely_box
        africa_box = gpd.GeoDataFrame(
            geometry=[shapely_box(
                bbox["minlon"], bbox["minlat"],
                bbox["maxlon"], bbox["maxlat"],
            )],
            crs="EPSG:4326",
        )
        clipped = gpd.clip(gdf, africa_box)
        log.info("  Africa subset: %d fault segments", len(clipped))
        return clipped if len(clipped) >= 3 else None
    except Exception as exc:
        log.warning("  GEM faults unavailable: %s", exc)
        return None


def _synthetic_faults_africa() -> gpd.GeoDataFrame:
    """
    Synthetic fault network for Central / Southern Africa.

    Structural trends (Daly et al. 2020; Chorowicz 2005):
      - East African Rift System : NNE-trending normal faults
      - Lufilian Arc thrusts     : NW–SE, Zambia / DRC border
      - Damara-Katanga belt      : E–W faults, Namibia / Botswana
      - Congo Craton margins     : NNW lineaments
    """
    log.info("Generating synthetic Africa fault network …")
    rng = np.random.default_rng(77)
    lines, ftypes = [], []

    # East African Rift System
    for _ in range(35):
        clon = rng.uniform(28, 36); clat = rng.uniform(-20, -4)
        length = rng.uniform(0.8, 5.0)
        s = rng.uniform(np.radians(-20), np.radians(20))
        dl, dlt = (length / 2) * np.sin(s), (length / 2) * np.cos(s)
        lines.append(LineString([(clon - dl, clat - dlt), (clon + dl, clat + dlt)]))
        ftypes.append("EARS normal")

    # Lufilian Arc thrusts
    for _ in range(30):
        clon = rng.uniform(24, 30); clat = rng.uniform(-14, -9)
        length = rng.uniform(0.5, 3.5)
        s = rng.uniform(np.radians(120), np.radians(150))
        dl, dlt = (length / 2) * np.sin(s), (length / 2) * np.cos(s)
        lines.append(LineString([(clon - dl, clat - dlt), (clon + dl, clat + dlt)]))
        ftypes.append("Lufilian Arc thrust")

    # Damara belt
    for _ in range(25):
        clon = rng.uniform(16, 26); clat = rng.uniform(-25, -18)
        length = rng.uniform(0.5, 2.5)
        s = rng.uniform(np.radians(80), np.radians(100))
        dl, dlt = (length / 2) * np.sin(s), (length / 2) * np.cos(s)
        lines.append(LineString([(clon - dl, clat - dlt), (clon + dl, clat + dlt)]))
        ftypes.append("Damara belt")

    # Congo Craton margin lineaments
    for _ in range(15):
        clon = rng.uniform(18, 24); clat = rng.uniform(-10, -4)
        length = rng.uniform(0.5, 2.0)
        s = rng.uniform(np.radians(-40), np.radians(-10))
        dl, dlt = (length / 2) * np.sin(s), (length / 2) * np.cos(s)
        lines.append(LineString([(clon - dl, clat - dlt), (clon + dl, clat + dlt)]))
        ftypes.append("Congo Craton margin")

    return gpd.GeoDataFrame(
        {"fault_id": range(len(lines)), "fault_type": ftypes},
        geometry=lines, crs="EPSG:4326",
    )


def download_faults(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "faults_africa.geojson"
    if out.exists() and not force:
        log.info("Faults cache hit → %s", out)
        return gpd.read_file(out)

    gdf = _fetch_gem_faults(BBOX)
    if gdf is None:
        log.info("Using synthetic Africa fault network.")
        gdf = _synthetic_faults_africa()

    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved faults → %s  (%d segments)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    force = os.getenv("FORCE_REFRESH", "0") == "1"
    log.info("=" * 65)
    log.info("GeoExplorer AI — Phase 1: Africa Data Acquisition")
    log.info("Region  : Central/Southern Africa Copperbelt")
    log.info("Metals  : Copper · Cobalt · Nickel   (KoBold Metals focus)")
    log.info("=" * 65)

    deposits = download_deposits(force=force)
    geochem  = download_geochem(force=force)
    faults   = download_faults(force=force)

    log.info("-" * 65)
    log.info("Summary:")
    log.info("  Mineral deposits : %d", len(deposits))
    log.info("  Geochem samples  : %d", len(geochem))
    log.info("  Fault segments   : %d", len(faults))
    log.info("-" * 65)
    log.info("Phase 1 complete. Run: python scripts/02_process_data.py")


if __name__ == "__main__":
    main()
