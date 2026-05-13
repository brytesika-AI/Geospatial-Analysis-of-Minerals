"""
scripts/01_download_data.py
===========================
Phase 1 – Data Acquisition  (Sub-Saharan Africa Edition)

Downloads real data from:
  1. USGS MRDS WFS  — Cu/Co/Ni + green metals (Li/Mn/REE/Graphite/V)
  2. OpenStreetMap Overpass API — real mine/quarry locations in Africa
  3. GEM Global Active Faults — real fault network (Africa clip)
  4. Natural Earth — cities (pop > 100k), railways, seaports, country boundaries
  5. Oil & Gas — known Sub-Saharan basins (OSM + curated list)
  6. Soil geochemistry — calibrated synthetic (no free Africa geochem API)

Region : Sub-Saharan Africa  (15 °N – 35 °S, 18 °W – 53 °E)
         ML model: Central/Southern Africa Copperbelt (0 °N – 35 °S, 15 °E – 38 °E)
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
from shapely.geometry import LineString, Point, box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ML model bbox — Central/Southern Africa Copperbelt
BBOX = dict(minlon=15.0, minlat=-35.0, maxlon=38.0, maxlat=0.0)

# Broader sub-Saharan Africa context bbox
SSA_BBOX = dict(minlon=-18.0, minlat=-35.0, maxlon=53.0, maxlat=18.0)

TIMEOUT     = 30
GEM_TIMEOUT = 120
OSM_TIMEOUT = 180

# ── Known mineral districts (Cu/Co/Ni + green metals) ─────────────────────────

MINERAL_DISTRICTS: list[tuple] = [
    # ── Zambia Copperbelt ────────────────────────────────────────────────────────
    (27.85, -12.37, 0.50, "Chililabombwe-Mingomba",  "Cu",     "SHSC"),
    (28.17, -12.53, 0.60, "Nchanga-Chingola",         "Cu",     "SHSC"),
    (28.63, -12.56, 0.50, "Mufulira",                 "Cu",     "SHSC"),
    (28.22, -12.82, 0.50, "Nkana-Kitwe",              "Cu-Co",  "SHSC"),
    (28.87, -12.27, 0.40, "Ndola",                    "Cu",     "SHSC"),
    (28.97, -11.47, 0.40, "Luanshya",                 "Cu",     "SHSC"),
    (27.46, -12.17, 0.50, "Kansanshi",                "Cu-Au",  "SHSC"),
    (29.36, -14.45, 0.40, "Konkola",                  "Cu",     "SHSC"),

    # ── DRC Katanga Copperbelt ────────────────────────────────────────────────
    (25.47, -10.73, 0.70, "Kolwezi",                  "Cu-Co",  "SHSC"),
    (26.97, -11.67, 0.50, "Lubumbashi",               "Cu-Co",  "SHSC"),
    (26.73, -10.93, 0.50, "Likasi",                   "Cu-Co",  "SHSC"),
    (26.11, -10.60, 0.60, "Tenke-Fungurume",          "Cu-Co",  "SHSC"),
    (27.25, -10.85, 0.40, "Kambove",                  "Cu-Co",  "SHSC"),
    (26.60,  -9.75, 0.40, "Kinsenda-Musoshi",         "Cu",     "SHSC"),
    (27.10, -12.20, 0.35, "Boss-Mining",              "Cu-Co",  "SHSC"),
    (26.30, -11.20, 0.40, "Ruashi",                   "Cu-Co",  "SHSC"),

    # ── DRC Green Metals ─────────────────────────────────────────────────────
    (27.40,  -7.30, 0.55, "Manono Lithium",           "Li",     "Pegmatite"),
    (27.20, -10.50, 0.45, "DRC Coltan Belt",          "Ta-Nb",  "Pegmatite"),

    # ── Botswana ─────────────────────────────────────────────────────────────
    (27.83, -22.00, 0.50, "Selebi-Phikwe",            "Cu-Ni",  "Magmatic"),
    (24.65, -21.55, 0.40, "Boseto-Phoenix",            "Cu",     "SHSC"),
    (26.00, -20.00, 0.35, "Khoemacau",                "Cu-Ag",  "SHSC"),

    # ── Namibia ──────────────────────────────────────────────────────────────
    (17.70, -19.50, 0.50, "Tsumeb",                   "Cu-Pb-Zn","VMS"),
    (15.60, -22.20, 0.40, "Matchless Belt",           "Cu",     "SHSC"),
    (21.90, -15.80, 0.45, "Karibib Lithium",          "Li",     "Pegmatite"),

    # ── Zimbabwe ─────────────────────────────────────────────────────────────
    (30.00, -19.50, 0.40, "Empress",                  "Cu",     "SHSC"),
    (31.90, -20.10, 0.45, "Bikita Lithium",           "Li",     "Pegmatite"),
    (30.50, -17.30, 0.40, "Trojan Nickel",            "Ni-Cu",  "Magmatic"),

    # ── South Africa ─────────────────────────────────────────────────────────
    (31.13, -23.93, 0.50, "Phalaborwa-Palabora",      "Cu",     "Alkaline"),
    (18.85, -29.12, 0.55, "Aggeneys-Black Mountain",  "Cu-Zn",  "SEDEX"),
    (17.80, -29.57, 0.50, "O'Kiep-Namaqualand",       "Cu",     "SHSC"),
    (29.00, -24.80, 0.65, "Bushveld East Ni-Cu",      "Ni-Cu",  "Magmatic"),
    (27.20, -24.30, 0.55, "Waterberg-Platreef",       "Ni-Cu",  "Magmatic"),
    (27.80, -25.80, 0.50, "Bushveld West Ni-Cu",      "Ni-Cu",  "Magmatic"),
    (20.50, -28.50, 0.45, "Kalahari Copper Belt SA",  "Cu",     "SHSC"),
    (23.00, -27.60, 0.55, "Kalahari Manganese",       "Mn",     "Sedimentary"),
    (19.50, -30.50, 0.40, "Copperton-Areachap",       "Cu-Zn",  "VMS"),

    # ── Mozambique ──────────────────────────────────────────────────────────
    (38.60, -13.20, 0.50, "Balama Graphite",          "Graphite","Sedimentary"),
    (35.20, -18.00, 0.40, "Tete Province",            "Cu",     "SHSC"),

    # ── Tanzania ─────────────────────────────────────────────────────────────
    (36.70,  -8.70, 0.50, "Mahenge Graphite",         "Graphite","Sedimentary"),
    (32.80,  -8.00, 0.45, "Ngualla REE",              "REE",    "Carbonatite"),
    (31.00, -10.00, 0.40, "Kabanga Nickel",           "Ni-Cu",  "Magmatic"),

    # ── Kenya ────────────────────────────────────────────────────────────────
    (39.40,  -4.20, 0.40, "Mrima Hill REE",           "REE",    "Carbonatite"),

    # ── Malawi ───────────────────────────────────────────────────────────────
    (33.80, -10.00, 0.40, "Songwe Hill REE",          "REE",    "Carbonatite"),

    # ── Ghana (West Africa) ──────────────────────────────────────────────────
    (-0.80,   5.20, 0.45, "Ewoyaa Lithium",           "Li",     "Pegmatite"),

    # ── Gabon ─────────────────────────────────────────────────────────────────
    (13.90,  -1.50, 0.45, "Moanda Manganese",         "Mn",     "Sedimentary"),
]

# Known Sub-Saharan Africa oil & gas fields
OIL_GAS_FIELDS: list[dict] = [
    {"name": "Cabinda/Block 0",        "lon": 12.20, "lat":  -5.70, "type": "Oil",     "country": "Angola"},
    {"name": "Block 15 Offshore",      "lon": 11.50, "lat":  -8.50, "type": "Oil",     "country": "Angola"},
    {"name": "Block 17 (Girassol)",    "lon": 10.80, "lat":  -9.20, "type": "Oil",     "country": "Angola"},
    {"name": "Rovuma Basin LNG",       "lon": 40.70, "lat": -11.00, "type": "Gas",     "country": "Mozambique"},
    {"name": "Coral FLNG",             "lon": 41.20, "lat": -12.50, "type": "Gas",     "country": "Mozambique"},
    {"name": "Lokichar Basin",         "lon": 35.70, "lat":  -1.50, "type": "Oil",     "country": "Kenya"},
    {"name": "Lake Albert (Kingfisher)","lon": 31.20, "lat":   1.00, "type": "Oil",    "country": "Uganda"},
    {"name": "Lake Albert (Tilenga)",  "lon": 31.50, "lat":   2.00, "type": "Oil",     "country": "Uganda"},
    {"name": "Unity Field",            "lon": 29.80, "lat":   9.50, "type": "Oil",     "country": "South Sudan"},
    {"name": "Heglig Field",           "lon": 29.00, "lat":  10.00, "type": "Oil",     "country": "South Sudan"},
    {"name": "Jubilee Field",          "lon":  -1.50, "lat":  5.50, "type": "Oil",     "country": "Ghana"},
    {"name": "TEN Field",              "lon":  -2.00, "lat":  5.00, "type": "Oil",     "country": "Ghana"},
    {"name": "Muanda Basin",           "lon": 12.60, "lat":  -5.80, "type": "Oil",     "country": "DRC"},
    {"name": "Ras Sidr / Suez",        "lon": 33.00, "lat":  29.00, "type": "Oil",     "country": "Egypt"},
    {"name": "Orange Basin (Offshore)","lon": 17.00, "lat": -29.50, "type": "Gas",     "country": "South Africa"},
    {"name": "Block 2C Namibia",       "lon": 12.50, "lat": -28.00, "type": "Gas",     "country": "Namibia"},
    {"name": "Port-Gentil Offshore",   "lon":  8.80, "lat":  -0.70, "type": "Oil",     "country": "Gabon"},
    {"name": "Mvimba",                 "lon": 11.90, "lat":  -4.00, "type": "Oil",     "country": "Congo"},
    {"name": "Rufiji Delta Gas",       "lon": 39.40, "lat":  -7.90, "type": "Gas",     "country": "Tanzania"},
    {"name": "Mnazi Bay Gas",          "lon": 40.40, "lat": -10.30, "type": "Gas",     "country": "Tanzania"},
    {"name": "Lake Tanganyika",        "lon": 29.50, "lat":  -6.00, "type": "Oil-prospect","country": "Tanzania/DRC"},
    {"name": "Karoo Basin (shale)",    "lon": 24.00, "lat": -31.00, "type": "Gas-prospect","country": "South Africa"},
]

# ── Natural Earth URLs ─────────────────────────────────────────────────────────

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
NE_CITIES_URL     = f"{NE_BASE}/ne_10m_populated_places_simple.geojson"
NE_RAILWAYS_URL   = f"{NE_BASE}/ne_10m_railroads.geojson"
NE_PORTS_URL      = f"{NE_BASE}/ne_10m_ports.geojson"
NE_COUNTRIES_URL  = f"{NE_BASE}/ne_50m_admin_0_countries.geojson"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. USGS MRDS — mineral deposits (fixed WFS bbox axis order)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_mrds_wfs(bbox: dict) -> gpd.GeoDataFrame | None:
    """
    WFS 1.0.0 uses lon/lat (X/Y) order → BBOX=minlon,minlat,maxlon,maxlat.
    WFS 1.1.0 uses lat/lon (Y/X) order for EPSG:4326 → BBOX=minlat,minlon,maxlat,maxlon.
    Previous code used v1.1.0 with lon/lat order (BUG). Fixed to v1.0.0.
    """
    # Try WFS 1.0.0 first (lon,lat order — most reliable)
    for version, url_template in [
        ("1.0.0",
         "https://mrdata.usgs.gov/services/wfs/mrds"
         "?service=WFS&version=1.0.0&request=GetFeature&typeName=mrds"
         "&bbox={minlon},{minlat},{maxlon},{maxlat}"
         "&outputFormat=application/json&maxFeatures=5000"),
        ("1.1.0",
         "https://mrdata.usgs.gov/services/wfs/mrds"
         "?service=WFS&version=1.1.0&request=GetFeature&typeName=mrds"
         "&BBOX={minlat},{minlon},{maxlat},{maxlon},EPSG:4326"
         "&outputFormat=application/json&count=5000"),
    ]:
        url = url_template.format(**bbox)
        log.info("Requesting USGS MRDS WFS %s …", version)
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            text = r.text.strip()
            if not text or text[0] != "{":
                log.warning("  WFS %s: empty response", version)
                continue
            fc = r.json()
            if not fc.get("features"):
                log.warning("  WFS %s: no features", version)
                continue
            gdf = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
            log.info("  MRDS WFS %s: %d raw features", version, len(gdf))
            return gdf
        except Exception as exc:
            log.warning("  MRDS WFS %s failed: %s", version, exc)
    return None


def _filter_all_minerals(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Retain Cu/Co/Ni + green metals (Li/Mn/REE/Graphite/V/PGM)."""
    pattern = (
        r"\b(CU|COPPER|CO|COBALT|NI|NICKEL|LI|LITHIUM|MN|MANGANESE|"
        r"REE|RARE.EARTH|GRAPHITE|CARBON|VANADIUM|V|PGM|PGE|PLATINUM|"
        r"CHROME|CR|TITANIUM|TI|ILMENITE)\b"
    )
    mask = pd.Series(False, index=gdf.index)
    for col in ("commod1","commod2","commod3","commod_list","commodities","ore"):
        if col in gdf.columns:
            mask |= gdf[col].astype(str).str.upper().str.contains(
                pattern, na=False, regex=True
            )
    result = gdf[mask].copy()
    log.info("  All critical minerals after filter: %d", len(result))
    return result


def _osm_mines(bbox: dict) -> gpd.GeoDataFrame | None:
    """Real mine/quarry locations from OpenStreetMap Overpass API."""
    s, w, n, e = bbox["minlat"], bbox["minlon"], bbox["maxlat"], bbox["maxlon"]
    query = f"""
[out:json][timeout:{OSM_TIMEOUT}][bbox:{s},{w},{n},{e}];
(
  node[man_made=mine][name];
  node[landuse=quarry][name];
  way[man_made=mine][name];
  way[landuse=quarry][name];
);
out center tags;
"""
    log.info("Querying OpenStreetMap Overpass for mines in Africa …")
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query}, timeout=OSM_TIMEOUT,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
        if not elements:
            log.warning("  OSM Overpass: no mine features returned")
            return None

        rows = []
        for el in elements:
            tags = el.get("tags", {})
            if el["type"] == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                lat, lon = center.get("lat"), center.get("lon")
            if lat is None or lon is None:
                continue
            rows.append({
                "name":    tags.get("name", "Mine"),
                "commod1": tags.get("mineral", tags.get("resource", "Mine")),
                "district": tags.get("name", "OSM mine"),
                "source":  "osm",
                "geometry": Point(lon, lat),
            })
        if not rows:
            return None
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        log.info("  OSM Overpass: %d mine/quarry features", len(gdf))
        return gdf
    except Exception as exc:
        log.warning("  OSM Overpass failed: %s", exc)
        return None


def _synthetic_deposits_all(n: int = 500) -> gpd.GeoDataFrame:
    """Synthetic deposits for all minerals biased toward known districts."""
    log.info("Generating %d synthetic mineral deposits …", n)
    rng = np.random.default_rng(42)
    lons, lats, names, commodities, styles = [], [], [], [], []

    n_clustered = int(n * 0.80)
    per_district = max(1, n_clustered // len(MINERAL_DISTRICTS))

    for clon, clat, rad, dname, commod, style in MINERAL_DISTRICTS:
        for _ in range(per_district):
            angle = rng.uniform(0, 2 * np.pi)
            r     = rng.uniform(0, rad)
            lons.append(clon + r * np.cos(angle))
            lats.append(clat + r * np.sin(angle))
            names.append(dname)
            commodities.append(commod)
            styles.append(style)

    n_random = n - len(lons)
    if n_random > 0:
        lons     += rng.uniform(BBOX["minlon"], BBOX["maxlon"], n_random).tolist()
        lats     += rng.uniform(BBOX["minlat"], BBOX["maxlat"], n_random).tolist()
        names    += ["background"] * n_random
        commodities += ["Cu"] * n_random
        styles   += ["SHSC"] * n_random

    gdf = gpd.GeoDataFrame(
        {"dep_id":  [f"AFR-{i:04d}" for i in range(len(lons))],
         "name":    names,
         "district": names,
         "commod1": commodities,
         "style":   styles,
         "source":  "synthetic"},
        geometry=[Point(lo, la) for lo, la in zip(lons, lats)],
        crs="EPSG:4326",
    )
    return gdf.reset_index(drop=True)


def download_deposits(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "mrds_africa_copper.geojson"
    if out.exists() and not force:
        log.info("Deposits cache hit → %s", out)
        return gpd.read_file(out)

    # 1. Try MRDS (fixed bbox)
    gdf = _fetch_mrds_wfs(BBOX)
    if gdf is not None and len(gdf) > 10:
        gdf = _filter_all_minerals(gdf)
        gdf["source"] = "mrds"
        if "commod1" not in gdf.columns:
            gdf["commod1"] = gdf.get("ore", "Cu")

    # 2. Try OSM Overpass (broader SSA bbox)
    osm = _osm_mines(SSA_BBOX)
    if osm is not None and len(osm) > 5:
        if gdf is not None and len(gdf) > 5:
            gdf = pd.concat([gdf, osm], ignore_index=True)
            log.info("  Merged MRDS + OSM: %d total", len(gdf))
        else:
            gdf = osm
            log.info("  Using OSM only: %d mines", len(gdf))

    # 3. Synthetic fallback
    if gdf is None or len(gdf) < 20:
        log.info("Falling back to synthetic deposits.")
        gdf = _synthetic_deposits_all(500)
    else:
        # Supplement with synthetic for known districts not in MRDS/OSM
        syn = _synthetic_deposits_all(200)
        gdf = pd.concat([gdf, syn], ignore_index=True)
        log.info("  Combined real + synthetic: %d", len(gdf))

    if "lon" not in gdf.columns:
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
    if "style" not in gdf.columns:
        gdf["style"] = "Unknown"

    # Ensure it's a GeoDataFrame with geometry
    if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.lon, gdf.lat), crs="EPSG:4326")

    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved deposits → %s  (%d rows)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Soil geochemistry — calibrated synthetic (no free Africa API)
# ═══════════════════════════════════════════════════════════════════════════════

AFRICA_GEOCHEM_BG = {
    "cu_ppm":  (80,   0.65),
    "co_ppm":  (45,   0.70),
    "ni_ppm":  (40,   0.65),
    "au_ppb":  (0.8,  0.80),
    "fe_pct":  (4.5,  0.35),
    "pb_ppm":  (25,   0.60),
    "zn_ppm":  (100,  0.55),
    "mo_ppm":  (0.6,  0.65),
    "as_ppm":  (12,   0.75),
}

CLIP_LIMITS = {
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


def _synthetic_geochem(n: int = 4500) -> pd.DataFrame:
    log.info("Generating %d Copperbelt geochemistry samples …", n)
    rng = np.random.default_rng(123)
    lons = rng.uniform(BBOX["minlon"], BBOX["maxlon"], n)
    lats = rng.uniform(BBOX["minlat"], BBOX["maxlat"], n)
    arrays: dict[str, np.ndarray] = {}
    for elem, (med, sigma) in AFRICA_GEOCHEM_BG.items():
        arrays[elem] = rng.lognormal(mean=np.log(med), sigma=sigma, size=n)

    for clon, clat, rad, _, commod, *_ in MINERAL_DISTRICTS:
        if clat < BBOX["minlat"] or clat > BBOX["maxlat"]:
            continue
        dist = np.hypot(lons - clon, lats - clat)
        mask = dist < rad * 2.0
        if not mask.any():
            continue
        prox = np.maximum(0.0, 1.0 - dist[mask] / (rad * 2.0))
        arrays["cu_ppm"][mask] *= np.exp(3.5 * prox)
        arrays["co_ppm"][mask] *= np.exp(3.0 * prox if "Co" in commod else 1.2 * prox)
        arrays["ni_ppm"][mask] *= np.exp(2.5 * prox if "Ni" in commod else 0.8 * prox)
        arrays["au_ppb"][mask] *= np.exp(1.5 * prox)
        arrays["zn_ppm"][mask] *= np.exp(2.0 * prox)
        arrays["as_ppm"][mask] *= np.exp(1.5 * prox)

    df = pd.DataFrame({"lon": lons, "lat": lats, "source": "synthetic_copperbelt"})
    for elem, (lo, hi) in CLIP_LIMITS.items():
        df[elem] = arrays[elem].clip(lo, hi)
    return df


def download_geochem(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "geochem_africa.csv"
    if out.exists() and not force:
        log.info("Geochem cache hit → %s", out)
        return pd.read_csv(out)
    df = _synthetic_geochem(4500)
    df.to_csv(out, index=False)
    log.info("Saved geochem → %s  (%d rows)", out.name, len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GEM Global Active Faults
# ═══════════════════════════════════════════════════════════════════════════════

GEM_FAULTS_URL = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults"
    "/master/geojson/gem_active_faults_harmonized.geojson"
)


def _synthetic_faults_africa() -> gpd.GeoDataFrame:
    rng = np.random.default_rng(77)
    lines, ftypes = [], []
    for _ in range(35):
        clon = rng.uniform(28, 36); clat = rng.uniform(-20, -4)
        length = rng.uniform(0.8, 5.0)
        s = rng.uniform(np.radians(-20), np.radians(20))
        dl, dlt = (length/2)*np.sin(s), (length/2)*np.cos(s)
        lines.append(LineString([(clon-dl, clat-dlt), (clon+dl, clat+dlt)]))
        ftypes.append("EARS normal")
    for _ in range(30):
        clon = rng.uniform(24, 30); clat = rng.uniform(-14, -9)
        length = rng.uniform(0.5, 3.5)
        s = rng.uniform(np.radians(120), np.radians(150))
        dl, dlt = (length/2)*np.sin(s), (length/2)*np.cos(s)
        lines.append(LineString([(clon-dl, clat-dlt), (clon+dl, clat+dlt)]))
        ftypes.append("Lufilian Arc thrust")
    for _ in range(25):
        clon = rng.uniform(16, 26); clat = rng.uniform(-25, -18)
        length = rng.uniform(0.5, 2.5)
        s = rng.uniform(np.radians(80), np.radians(100))
        dl, dlt = (length/2)*np.sin(s), (length/2)*np.cos(s)
        lines.append(LineString([(clon-dl, clat-dlt), (clon+dl, clat+dlt)]))
        ftypes.append("Damara belt")
    return gpd.GeoDataFrame(
        {"fault_id": range(len(lines)), "fault_type": ftypes},
        geometry=lines, crs="EPSG:4326",
    )


def download_faults(force: bool = False) -> gpd.GeoDataFrame:
    out = DATA_RAW / "faults_africa.geojson"
    if out.exists() and not force:
        log.info("Faults cache hit → %s", out)
        return gpd.read_file(out)

    log.info("Downloading GEM Global Active Faults (~40 MB) …")
    try:
        r = requests.get(GEM_FAULTS_URL, timeout=GEM_TIMEOUT)
        r.raise_for_status()
        gdf_all = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        africa_box = gpd.GeoDataFrame(
            geometry=[box(BBOX["minlon"], BBOX["minlat"], BBOX["maxlon"], BBOX["maxlat"])],
            crs="EPSG:4326",
        )
        gdf = gpd.clip(gdf_all, africa_box)
        log.info("  GEM Africa subset: %d fault segments", len(gdf))
        if len(gdf) < 3:
            raise ValueError("Too few features clipped")
    except Exception as exc:
        log.warning("  GEM failed: %s — using synthetic faults", exc)
        gdf = _synthetic_faults_africa()

    gdf.to_file(out, driver="GeoJSON")
    log.info("Saved faults → %s  (%d segments)", out.name, len(gdf))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Natural Earth — cities, railways, ports, country boundaries
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_ne(url: str, label: str, timeout: int = 60) -> dict | None:
    log.info("Downloading Natural Earth %s …", label)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("  Natural Earth %s failed: %s", label, exc)
        return None


def _in_ssa(lat: float, lon: float) -> bool:
    b = SSA_BBOX
    return b["minlon"] <= lon <= b["maxlon"] and b["minlat"] <= lat <= b["maxlat"]


def download_cities(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "cities_africa.csv"
    if out.exists() and not force:
        log.info("Cities cache hit → %s", out)
        return pd.read_csv(out)

    fc = _fetch_ne(NE_CITIES_URL, "cities", timeout=90)
    rows = []
    if fc:
        for feat in fc.get("features", []):
            p = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            lon, lat = coords[0], coords[1]
            if lon is None or lat is None or not _in_ssa(lat, lon):
                continue
            pop = p.get("pop_max", p.get("pop_min", 0)) or 0
            if pop < 50_000:
                continue
            rows.append({
                "name":    p.get("name", ""),
                "country": p.get("adm0name", p.get("sov0name", "")),
                "pop":     int(pop),
                "lon":     lon,
                "lat":     lat,
                "type":    "city" if pop >= 500_000 else "town",
            })
        log.info("  Cities in Sub-Saharan Africa (pop > 50k): %d", len(rows))

    if not rows:
        # Curated fallback: major SSA capitals and mining hubs
        rows = [
            {"name": "Johannesburg", "country": "South Africa", "pop": 4500000, "lon": 28.04, "lat": -26.20, "type": "city"},
            {"name": "Lagos",        "country": "Nigeria",      "pop": 14800000,"lon":  3.39, "lat":   6.45, "type": "city"},
            {"name": "Nairobi",      "country": "Kenya",        "pop": 4300000, "lon": 36.82, "lat":  -1.29, "type": "city"},
            {"name": "Kinshasa",     "country": "DRC",          "pop": 14300000,"lon": 15.31, "lat":  -4.32, "type": "city"},
            {"name": "Dar es Salaam","country": "Tanzania",     "pop": 4400000, "lon": 39.27, "lat":  -6.80, "type": "city"},
            {"name": "Luanda",       "country": "Angola",       "pop": 2800000, "lon": 13.23, "lat":  -8.84, "type": "city"},
            {"name": "Lusaka",       "country": "Zambia",       "pop": 2300000, "lon": 28.28, "lat": -15.42, "type": "city"},
            {"name": "Harare",       "country": "Zimbabwe",     "pop": 1500000, "lon": 31.05, "lat": -17.83, "type": "city"},
            {"name": "Cape Town",    "country": "South Africa", "pop": 4600000, "lon": 18.42, "lat": -33.93, "type": "city"},
            {"name": "Maputo",       "country": "Mozambique",   "pop": 1100000, "lon": 32.59, "lat": -25.97, "type": "city"},
            {"name": "Gaborone",     "country": "Botswana",     "pop":  250000, "lon": 25.91, "lat": -24.65, "type": "city"},
            {"name": "Windhoek",     "country": "Namibia",      "pop":  350000, "lon": 17.08, "lat": -22.56, "type": "city"},
            {"name": "Lubumbashi",   "country": "DRC",          "pop": 2200000, "lon": 27.47, "lat": -11.67, "type": "city"},
            {"name": "Kitwe",        "country": "Zambia",       "pop":  600000, "lon": 28.21, "lat": -12.82, "type": "city"},
            {"name": "Ndola",        "country": "Zambia",       "pop":  450000, "lon": 28.64, "lat": -12.97, "type": "city"},
            {"name": "Mombasa",      "country": "Kenya",        "pop": 1100000, "lon": 39.67, "lat":  -4.05, "type": "city"},
            {"name": "Accra",        "country": "Ghana",        "pop": 2500000, "lon":  -0.20, "lat":  5.56, "type": "city"},
            {"name": "Antananarivo", "country": "Madagascar",   "pop": 1400000, "lon": 47.52, "lat": -18.91, "type": "city"},
            {"name": "Kampala",      "country": "Uganda",       "pop": 1650000, "lon": 32.58, "lat":   0.32, "type": "city"},
            {"name": "Douala",       "country": "Cameroon",     "pop": 2800000, "lon":  9.71, "lat":   4.05, "type": "city"},
        ]
        log.info("  Using curated city list: %d cities", len(rows))

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info("Saved cities → %s  (%d rows)", out.name, len(df))
    return df


def download_railways(force: bool = False) -> gpd.GeoDataFrame | None:
    out = DATA_RAW / "railways_africa.geojson"
    if out.exists() and not force:
        log.info("Railways cache hit → %s", out)
        return gpd.read_file(out)

    fc = _fetch_ne(NE_RAILWAYS_URL, "railways", timeout=120)
    if fc:
        try:
            gdf_all = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
            ssa_box = gpd.GeoDataFrame(
                geometry=[box(SSA_BBOX["minlon"], SSA_BBOX["minlat"],
                              SSA_BBOX["maxlon"], SSA_BBOX["maxlat"])],
                crs="EPSG:4326",
            )
            gdf = gpd.clip(gdf_all, ssa_box)
            gdf.to_file(out, driver="GeoJSON")
            log.info("Saved railways → %s  (%d segments)", out.name, len(gdf))
            return gdf
        except Exception as exc:
            log.warning("  Railways clip failed: %s", exc)
    return None


def download_ports(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "ports_africa.csv"
    if out.exists() and not force:
        log.info("Ports cache hit → %s", out)
        return pd.read_csv(out)

    fc = _fetch_ne(NE_PORTS_URL, "ports", timeout=60)
    rows = []
    if fc:
        for feat in fc.get("features", []):
            p = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            lon, lat = coords[0], coords[1]
            if lon is None or lat is None or not _in_ssa(lat, lon):
                continue
            rows.append({
                "name":    p.get("name", p.get("NAME", "")),
                "country": p.get("country", p.get("COUNTRY", "")),
                "lon":     lon, "lat": lat,
            })
        log.info("  Ports in Sub-Saharan Africa: %d", len(rows))

    if not rows:
        rows = [
            {"name": "Durban",       "country": "South Africa", "lon": 30.92, "lat": -29.86},
            {"name": "Cape Town",    "country": "South Africa", "lon": 18.42, "lat": -33.91},
            {"name": "Port Elizabeth","country": "South Africa","lon": 25.57, "lat": -33.96},
            {"name": "Dar es Salaam","country": "Tanzania",     "lon": 39.29, "lat":  -6.84},
            {"name": "Mombasa",      "country": "Kenya",        "lon": 39.67, "lat":  -4.06},
            {"name": "Luanda",       "country": "Angola",       "lon": 13.23, "lat":  -8.84},
            {"name": "Beira",        "country": "Mozambique",   "lon": 34.83, "lat": -19.84},
            {"name": "Walvis Bay",   "country": "Namibia",      "lon": 14.50, "lat": -22.95},
            {"name": "Lobito",       "country": "Angola",       "lon": 13.54, "lat":  -12.35},
            {"name": "Nacala",       "country": "Mozambique",   "lon": 40.68, "lat": -14.54},
            {"name": "Tema",         "country": "Ghana",        "lon":  -0.01, "lat":  5.62},
            {"name": "Abidjan",      "country": "Côte d'Ivoire","lon":  -4.01, "lat":  5.35},
        ]

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info("Saved ports → %s  (%d rows)", out.name, len(df))
    return df


def download_oil_gas(force: bool = False) -> pd.DataFrame:
    out = DATA_RAW / "oil_gas_africa.csv"
    if out.exists() and not force:
        log.info("Oil & gas cache hit → %s", out)
        return pd.read_csv(out)

    log.info("Building Sub-Saharan Africa oil & gas field dataset …")
    df = pd.DataFrame(OIL_GAS_FIELDS)
    df.to_csv(out, index=False)
    log.info("Saved oil & gas → %s  (%d fields)", out.name, len(df))
    return df


def download_country_boundaries(force: bool = False) -> gpd.GeoDataFrame | None:
    out = DATA_RAW / "countries_africa.geojson"
    if out.exists() and not force:
        log.info("Countries cache hit → %s", out)
        return gpd.read_file(out)

    fc = _fetch_ne(NE_COUNTRIES_URL, "country boundaries", timeout=90)
    if fc:
        try:
            gdf_all = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
            ssa_box = gpd.GeoDataFrame(
                geometry=[box(SSA_BBOX["minlon"], SSA_BBOX["minlat"],
                              SSA_BBOX["maxlon"], SSA_BBOX["maxlat"])],
                crs="EPSG:4326",
            )
            # Keep countries that intersect SSA bbox
            gdf = gdf_all[gdf_all.intersects(ssa_box.geometry[0])].copy()
            name_col = next((c for c in ("NAME","ADMIN","name","admin") if c in gdf.columns), None)
            if name_col:
                gdf = gdf.rename(columns={name_col: "country_name"})
            gdf.to_file(out, driver="GeoJSON")
            log.info("Saved country boundaries → %s  (%d countries)", out.name, len(gdf))
            return gdf
        except Exception as exc:
            log.warning("  Country boundaries failed: %s", exc)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    force = os.getenv("FORCE_REFRESH", "0") == "1"
    log.info("=" * 70)
    log.info("GeoExplorer AI — Phase 1: Sub-Saharan Africa Data Acquisition")
    log.info("Model region : Copperbelt (15°E–38°E, 35°S–0°)")
    log.info("Context      : Sub-Saharan Africa (18°W–53°E, 18°N–35°S)")
    log.info("=" * 70)

    deposits  = download_deposits(force=force)
    geochem   = download_geochem(force=force)
    faults    = download_faults(force=force)
    cities    = download_cities(force=force)
    ports     = download_ports(force=force)
    railways  = download_railways(force=force)
    oil_gas   = download_oil_gas(force=force)
    countries = download_country_boundaries(force=force)

    log.info("-" * 70)
    log.info("Summary:")
    log.info("  Mineral deposits : %d  (MRDS/OSM/synthetic)", len(deposits))
    log.info("  Geochem samples  : %d", len(geochem))
    log.info("  Fault segments   : %d  (GEM real data)", len(faults))
    log.info("  Cities           : %d", len(cities))
    log.info("  Seaports         : %d", len(ports))
    log.info("  Oil & gas fields : %d", len(oil_gas))
    if railways is not None:
        log.info("  Railway segments : %d", len(railways))
    if countries is not None:
        log.info("  Countries        : %d", len(countries))
    log.info("-" * 70)
    log.info("Phase 1 complete. Run: python scripts/02_process_data.py")


if __name__ == "__main__":
    main()
