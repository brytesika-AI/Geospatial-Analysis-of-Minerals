"""
scripts/01_download_data.py
===========================
Phase 1 – Data Acquisition

Downloads:
  1. USGS MRDS copper deposits (WFS endpoint)  → data/raw/mrds_copper_az_nv.geojson
  2. USGS NGDB soil geochemistry               → data/raw/geochem_az_nv.csv
  3. USGS Quaternary Fault lines               → data/raw/faults_az_nv.geojson

Falls back to realistic synthetic data (based on published AZ/NV mineralisation stats)
if any source is unreachable, so the pipeline always completes.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, Point, mapping
import geopandas as gpd

# ── project root ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── spatial extent: Arizona + Nevada ────────────────────────────────────────────
BBOX = dict(minlon=-120.0, minlat=31.0, maxlon=-109.0, maxlat=42.0)

# Known copper-productive sub-regions (used to bias synthetic deposit placement)
COPPER_DISTRICTS = [
    # (center_lon, center_lat, radius_deg, name)
    (-110.7, 31.6, 0.8, "Bisbee–Cochise"),
    (-110.8, 32.7, 0.6, "Globe–Miami"),
    (-112.5, 33.4, 0.5, "Bagdad"),
    (-111.3, 33.4, 0.4, "Ray"),
    (-111.0, 33.9, 0.5, "Superior"),
    (-109.8, 33.4, 0.4, "Morenci–Clifton"),
    (-114.9, 36.2, 0.5, "Searchlight–NV"),
    (-116.5, 40.8, 0.4, "Battle Mountain–NV"),
    (-117.1, 38.5, 0.5, "Tonopah–NV"),
    (-115.4, 36.0, 0.4, "El Dorado–NV"),
    (-118.6, 38.2, 0.5, "Mineral County–NV"),
    (-114.2, 38.7, 0.4, "White Pine–NV"),
]

TIMEOUT = 15  # seconds per HTTP request


# ═══════════════════════════════════════════════════════════════════════════════
# 1. USGS MRDS — copper mineral deposits
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
    """Attempt to download MRDS deposits via WFS."""
    url = MRDS_WFS.format(**bbox)
    log.info("Requesting MRDS WFS …")
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        fc = r.json()
        gdf = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
        log.info("  WFS returned %d features", len(gdf))
        return gdf
    except Exception as exc:
        log.warning("  MRDS WFS unavailable: %s", exc)
        return None


def _filter_copper(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only records that list copper as a primary or secondary commodity."""
    cu_mask = pd.Series(False, index=gdf.index)
    for col in ("commod1", "commod2", "commod3", "commod_list", "commodities"):
        if col in gdf.columns:
            cu_mask |= gdf[col].astype(str).str.upper().str.contains(r"\bCU\b|COPPER", na=False)
    filtered = gdf[cu_mask].copy()
    log.info("  Copper deposits after filter: %d", len(filtered))
    return filtered


def _synthetic_deposits(n: int = 320) -> gpd.GeoDataFrame:
    """
    Generate realistic synthetic copper deposits for AZ/NV.

    Strategy:
      - 75% placed near known productive districts (spatially biased)
      - 25% placed randomly within the bounding box (for background variety)
    """
    log.info("Generating %d synthetic copper deposits …", n)
    rng = np.random.default_rng(42)
    lons, lats, names = [], [], []

    n_clustered = int(n * 0.75)
    n_random = n - n_clustered

    # Clustered around known districts
    per_district = n_clustered // len(COPPER_DISTRICTS)
    for clon, clat, rad, dname in COPPER_DISTRICTS:
        for _ in range(per_district):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, rad)
            lons.append(clon + r * np.cos(angle))
            lats.append(clat + r * np.sin(angle))
            names.append(dname)

    # Random background
    lons += rng.uniform(BBOX["minlon"], BBOX["maxlon"], n_random).tolist()
    lats += rng.uniform(BBOX["minlat"], BBOX["maxlat"], n_random).tolist()
    names += ["background"] * n_random

    gdf = gpd.GeoDataFrame(
        {
            "dep_id": [f"SYN-{i:04d}" for i in range(len(lons))],
            "name": [f"Synthetic deposit {i}" for i in range(len(lons))],
            "district": names,
            "commod1": "CU",
            "source": "synthetic",
        },
        geometry=[Point(lo, la) for lo, la in zip(lons, lats)],
        crs="EPSG:4326",
    )
    # Clip to bbox
    gdf = gdf.cx[BBOX["minlon"]:BBOX["maxlon"], BBOX["minlat"]:BBOX["maxlat"]]
    return gdf.reset_index(drop=True)


def download_deposits(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "mrds_copper_az_nv.geojson"
    if out.exists() and not force:
        log.info("Deposits cache hit → %s", out)
        return gpd.read_file(out)

    gdf = _fetch_mrds_wfs(BBOX)
    if gdf is not None and len(gdf) > 10:
        gdf = _filter_copper(gdf)
    if gdf is None or len(gdf) < 10:
        gdf = _synthetic_deposits(320)

    # Ensure we have lon/lat columns for easy use later
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved deposits → %s  (%d rows)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 2. USGS NGDB — soil geochemistry
# ═══════════════════════════════════════════════════════════════════════════════

# USGS NGDB sediment state-level query (public REST)
NGDB_REST = (
    "https://mrdata.usgs.gov/geochem/ngdb/select.php"
    "?state={state}&type=sed&format=csv"
)

# Elements we care about (ppm unless noted)
GEOCHEM_COLS = {
    "cu": "cu_ppm",    # copper
    "au": "au_ppb",    # gold (ppb)
    "fe": "fe_pct",    # iron (%)
    "pb": "pb_ppm",    # lead
    "zn": "zn_ppm",    # zinc
    "mo": "mo_ppm",    # molybdenum
    "as": "as_ppm",    # arsenic
}


def _fetch_ngdb_state(state: str) -> pd.DataFrame | None:
    url = NGDB_REST.format(state=state)
    log.info("  Requesting NGDB/%s …", state)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content), low_memory=False)
        log.info("    %d rows", len(df))
        return df
    except Exception as exc:
        log.warning("    NGDB/%s unavailable: %s", state, exc)
        return None


def _synthetic_geochem(n: int = 3000) -> pd.DataFrame:
    """
    Synthetic soil geochemistry with AZ/NV-realistic statistics.

    Background values derived from published USGS median soil concentrations
    for the Basin-and-Range province.  Anomalies are injected near the known
    copper districts to make the model task learnable.
    """
    log.info("Generating %d synthetic geochemistry samples …", n)
    rng = np.random.default_rng(123)

    lons = rng.uniform(BBOX["minlon"], BBOX["maxlon"], n)
    lats = rng.uniform(BBOX["minlat"], BBOX["maxlat"], n)

    # Background geochemistry (log-normal distributions, basin-range medians)
    cu_bg = rng.lognormal(mean=np.log(18), sigma=0.7, size=n)    # ~18 ppm median
    au_bg = rng.lognormal(mean=np.log(1.2), sigma=0.8, size=n)   # ~1.2 ppb median
    fe_bg = rng.lognormal(mean=np.log(2.8), sigma=0.4, size=n)   # ~2.8 % median
    pb_bg = rng.lognormal(mean=np.log(14), sigma=0.6, size=n)
    zn_bg = rng.lognormal(mean=np.log(45), sigma=0.5, size=n)
    mo_bg = rng.lognormal(mean=np.log(0.8), sigma=0.7, size=n)
    as_bg = rng.lognormal(mean=np.log(5), sigma=0.8, size=n)

    # Inject anomalies near copper districts
    for clon, clat, rad, _ in COPPER_DISTRICTS:
        dist = np.hypot(lons - clon, lats - clat)
        mask = dist < rad * 1.5
        multiplier = np.exp(2.0 * np.maximum(0, 1 - dist[mask] / (rad * 1.5)))
        cu_bg[mask] *= multiplier
        au_bg[mask] *= multiplier ** 0.5
        mo_bg[mask] *= multiplier ** 0.7

    df = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "cu_ppm": cu_bg.clip(1, 10000),
            "au_ppb": au_bg.clip(0.1, 5000),
            "fe_pct": fe_bg.clip(0.1, 20),
            "pb_ppm": pb_bg.clip(1, 2000),
            "zn_ppm": zn_bg.clip(1, 3000),
            "mo_ppm": mo_bg.clip(0.1, 500),
            "as_ppm": as_bg.clip(0.5, 1000),
            "source": "synthetic",
        }
    )
    return df


def _normalise_geochem(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw NGDB columns to our standard names."""
    rename = {}
    for raw, std in GEOCHEM_COLS.items():
        for col in df.columns:
            if col.lower().startswith(raw):
                rename[col] = std
                break
    df = df.rename(columns=rename)

    # Ensure lat/lon present
    for latcol in ("latitude", "lat_dd", "lat"):
        if latcol in df.columns:
            df = df.rename(columns={latcol: "lat"})
            break
    for loncol in ("longitude", "lon_dd", "lon"):
        if loncol in df.columns:
            df = df.rename(columns={loncol: "lon"})
            break

    keep = ["lat", "lon"] + [c for c in GEOCHEM_COLS.values() if c in df.columns]
    df = df[keep].dropna(subset=["lat", "lon"])

    # Clip to AZ/NV bbox
    df = df[
        (df.lon >= BBOX["minlon"]) & (df.lon <= BBOX["maxlon"]) &
        (df.lat >= BBOX["minlat"]) & (df.lat <= BBOX["maxlat"])
    ].copy()
    return df.reset_index(drop=True)


def download_geochem(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "geochem_az_nv.csv"
    if out.exists() and not force:
        log.info("Geochem cache hit → %s", out)
        return pd.read_csv(out)

    log.info("Downloading USGS NGDB geochemistry …")
    frames = []
    for state in ("AZ", "NV"):
        df = _fetch_ngdb_state(state)
        if df is not None:
            frames.append(df)
        time.sleep(0.5)   # be polite

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = _normalise_geochem(df)
        if len(df) >= 100:
            log.info("Real NGDB data: %d samples", len(df))
            df["source"] = "usgs_ngdb"
            df.to_csv(out, index=False)
            return df

    # Fallback
    df = _synthetic_geochem(3000)
    df.to_csv(out, index=False)
    log.info("Saved geochem → %s  (%d rows)", out.name, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. USGS Quaternary Faults
# ═══════════════════════════════════════════════════════════════════════════════

FAULT_WFS = (
    "https://earthquake.usgs.gov/arcgis/rest/services/eq/HazardsSummary_US/MapServer/4"
    "/query?where=1%3D1&outFields=*&geometry="
    "{minlon}%2C{minlat}%2C{maxlon}%2C{maxlat}"
    "&geometryType=esriGeometryEnvelope&inSR=4326&outSR=4326&f=geojson"
)


def _synthetic_faults() -> gpd.GeoDataFrame:
    """Generate synthetic fault lines for AZ/NV based on Basin-and-Range structure."""
    log.info("Generating synthetic fault network …")
    rng = np.random.default_rng(77)
    lines = []

    # NW-trending Basin-and-Range normal faults
    for _ in range(40):
        clon = rng.uniform(BBOX["minlon"], BBOX["maxlon"])
        clat = rng.uniform(BBOX["minlat"], BBOX["maxlat"])
        length = rng.uniform(0.5, 3.0)
        strike_rad = rng.uniform(np.radians(-30), np.radians(30))  # ~N-S ± 30°
        dlon = length / 2 * np.sin(strike_rad)
        dlat = length / 2 * np.cos(strike_rad)
        lines.append(
            LineString([(clon - dlon, clat - dlat), (clon + dlon, clat + dlat)])
        )

    gdf = gpd.GeoDataFrame(
        {"fault_id": range(len(lines)), "fault_type": "Basin-and-Range normal"},
        geometry=lines,
        crs="EPSG:4326",
    )
    return gdf


def download_faults(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "faults_az_nv.geojson"
    if out.exists() and not force:
        log.info("Faults cache hit → %s", out)
        return gpd.read_file(out)

    log.info("Downloading USGS fault data …")
    url = FAULT_WFS.format(**BBOX)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        if len(gdf) >= 5:
            log.info("  Real fault data: %d features", len(gdf))
            gdf.to_file(out, driver="GeoJSON")
            return gdf
    except Exception as exc:
        log.warning("  Fault API unavailable: %s", exc)

    gdf = _synthetic_faults()
    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved faults → %s  (%d features)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    force = os.getenv("FORCE_REFRESH", "0") == "1"
    log.info("=" * 60)
    log.info("GeoExplorer AI — Phase 1: Data Acquisition")
    log.info("Region: Arizona + Nevada  |  Commodity: Copper")
    log.info("=" * 60)

    deposits = download_deposits(force=force)
    geochem = download_geochem(force=force)
    faults = download_faults(force=force)

    log.info("-" * 60)
    log.info("Summary:")
    log.info("  Copper deposits : %d", len(deposits))
    log.info("  Geochem samples : %d", len(geochem))
    log.info("  Fault segments  : %d", len(faults))
    log.info("-" * 60)
    log.info("Phase 1 data ready. Run: python scripts/02_process_data.py")


if __name__ == "__main__":
    main()
