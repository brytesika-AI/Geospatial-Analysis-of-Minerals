"""
scripts/02_process_data.py
==========================
Phase 2 – Feature Engineering  (Africa Copperbelt)

Builds the training dataset for mineral prospectivity modelling:
  1. Loads Cu/Co/Ni deposits (positives) + spatially-stratified negatives
  2. IDW-interpolates soil geochemistry onto sample/grid points
  3. Fetches SRTM elevation via srtm.py (global 90 m DEM)
  4. Computes distance-to-fault and distance-to-nearest-deposit features
  5. Derives slope from elevation neighbourhood
  6. Writes data/processed/training_set.csv
  7. Writes data/processed/prediction_grid.csv  (0.2 ° grid over study area)

Feature additions vs AZ/NV version
------------------------------------
  log_co_ppm  — cobalt (hallmark DRC Katanga Cu-Co systems)
  log_ni_ppm  — nickel (Botswana Cu-Ni focus)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import srtm
    SRTM_AVAILABLE = True
except ImportError:
    SRTM_AVAILABLE = False
    logging.warning("srtm.py not installed — elevation will use terrain proxy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BBOX = dict(minlon=15.0, minlat=-35.0, maxlon=38.0, maxlat=0.0)
SEED = 42
NEG_RATIO = 4
BUFFER_DEG = 0.20       # ~22 km exclusion zone around known deposits
GRID_STEP  = 0.20       # degrees (~22 km); coarser than AZ/NV due to larger area

# Central African Copperbelt geochemical background medians
AFRICA_MEDIANS = dict(
    cu_ppm=80,  co_ppm=45,  ni_ppm=40,  au_ppb=0.8,
    fe_pct=4.5, pb_ppm=25,  zn_ppm=100, mo_ppm=0.6, as_ppm=12,
)

GEOCHEM_COLS = [
    "cu_ppm", "co_ppm", "ni_ppm", "au_ppb",
    "fe_pct", "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm",
]

LOG_TRANSFORM_COLS = [
    "cu_ppm", "co_ppm", "ni_ppm", "au_ppb",
    "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm",
]


# ── Elevation ─────────────────────────────────────────────────────────────────

def _get_elevation_batch(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    if SRTM_AVAILABLE:
        ed = srtm.get_data()
        elevs = np.array(
            [ed.get_elevation(float(la), float(lo)) or 0.0
             for la, lo in zip(lats, lons)],
            dtype=float,
        )
        return np.where(np.isnan(elevs), 0.0, elevs)

    # Terrain proxy: Central African plateau ~1 000–1 400 m,
    # East African Rift flanks ~1 200–2 000 m, Kalahari ~900–1 100 m
    rng = np.random.default_rng(99)
    base = (
        1100
        + 300 * np.sin(np.radians((lats + 15) * 8))
        + 250 * np.cos(np.radians((lons - 28) * 6))
    )
    return (base + rng.normal(0, 200, size=len(lats))).clip(50, 2500)


def _compute_slope(lats: np.ndarray, lons: np.ndarray,
                   elevs: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.column_stack([lons, lats]))
    _, idx = tree.query(np.column_stack([lons, lats]), k=5)
    slopes = np.zeros(len(lats))
    for i in range(len(lats)):
        nbrs = idx[i, 1:]
        dz = np.abs(elevs[nbrs] - elevs[i])
        dl = np.hypot(lons[nbrs] - lons[i], lats[nbrs] - lats[i]) * 111_000
        dl = np.where(dl < 1, 1, dl)
        slopes[i] = np.degrees(np.arctan(dz.max() / dl.min()))
    return slopes


# ── IDW geochemical interpolation ─────────────────────────────────────────────

def _idw_interpolate(
    known_lons: np.ndarray, known_lats: np.ndarray, known_vals: np.ndarray,
    query_lons: np.ndarray, query_lats: np.ndarray,
    k: int = 10, power: float = 2.0,
) -> np.ndarray:
    src  = np.column_stack([known_lons, known_lats])
    tgt  = np.column_stack([query_lons, query_lats])
    tree = cKDTree(src)
    dists, idxs = tree.query(tgt, k=min(k, len(known_lons)))
    dists = np.where(dists == 0, 1e-10, dists)
    weights = 1.0 / (dists ** power)
    return (weights * known_vals[idxs]).sum(axis=1) / weights.sum(axis=1)


# ── Fault proximity ────────────────────────────────────────────────────────────

def _dist_to_faults_km(
    query_lons: np.ndarray, query_lats: np.ndarray,
    fault_gdf: gpd.GeoDataFrame, n_sample: int = 8,
) -> np.ndarray:
    if fault_gdf is None or len(fault_gdf) == 0:
        return np.full(len(query_lons), 100.0)

    pts = []
    for geom in fault_gdf.geometry:
        if geom is None:
            continue
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for p in parts:
            try:
                coords = list(p.coords)
                pts.extend(coords[::max(1, len(coords) // n_sample)])
            except Exception:
                pass

    if not pts:
        return np.full(len(query_lons), 100.0)

    fp   = np.array(pts)
    tree = cKDTree(fp)
    d, _ = tree.query(np.column_stack([query_lons, query_lats]), k=1)
    return d * 111.0


# ═══════════════════════════════════════════════════════════════════════════════
# Training set
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_set(
    deposits: gpd.GeoDataFrame,
    geochem:  pd.DataFrame,
    faults:   gpd.GeoDataFrame,
) -> pd.DataFrame:
    log.info("Building Africa training set …")
    rng = np.random.default_rng(SEED)

    pos = pd.DataFrame({
        "lon":   deposits.geometry.x.values,
        "lat":   deposits.geometry.y.values,
        "label": 1,
    })
    n_pos = len(pos)
    log.info("  Positive samples (deposits): %d", n_pos)

    dep_tree = cKDTree(np.column_stack([pos.lon.values, pos.lat.values]))
    neg_rows: list[tuple] = []
    attempts = 0
    while len(neg_rows) < n_pos * NEG_RATIO and attempts < 300_000:
        lo = rng.uniform(BBOX["minlon"], BBOX["maxlon"])
        la = rng.uniform(BBOX["minlat"], BBOX["maxlat"])
        d, _ = dep_tree.query([[lo, la]])
        if d[0] > BUFFER_DEG:
            neg_rows.append((lo, la, 0))
        attempts += 1

    neg = pd.DataFrame(neg_rows, columns=["lon", "lat", "label"])
    log.info("  Negative samples (background): %d", len(neg))

    df = pd.concat([pos, neg], ignore_index=True)

    log.info("  Fetching elevation (SRTM/proxy) …")
    df["elevation_m"] = _get_elevation_batch(df.lat.values, df.lon.values)
    df["slope_deg"]   = _compute_slope(df.lat.values, df.lon.values, df["elevation_m"].values)

    log.info("  Interpolating geochemistry (IDW) …")
    for col in GEOCHEM_COLS:
        if col in geochem.columns:
            valid = geochem.dropna(subset=["lon", "lat", col])
            df[col] = np.maximum(0.0, _idw_interpolate(
                valid.lon.values, valid.lat.values, valid[col].values,
                df.lon.values,    df.lat.values,
            ))
        else:
            df[col] = AFRICA_MEDIANS.get(col, 1.0)

    log.info("  Computing fault proximity …")
    df["dist_fault_km"] = _dist_to_faults_km(df.lon.values, df.lat.values, faults)

    log.info("  Computing deposit proximity …")
    dep_pts   = np.column_stack([pos.lon.values, pos.lat.values])
    dep_tree2 = cKDTree(dep_pts)
    all_pts   = np.column_stack([df.lon.values, df.lat.values])
    dists, _  = dep_tree2.query(all_pts, k=2)
    df["dist_deposit_km"] = dists[:, 1] * 111.0

    for col in LOG_TRANSFORM_COLS:
        df[f"log_{col}"] = np.log1p(df[col])

    log.info("  Training set shape: %s", df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction grid
# ═══════════════════════════════════════════════════════════════════════════════

def build_prediction_grid(
    geochem:  pd.DataFrame,
    faults:   gpd.GeoDataFrame,
    deposits: gpd.GeoDataFrame,
) -> pd.DataFrame:
    log.info("Building Africa prediction grid (step=%.2f °) …", GRID_STEP)

    lons = np.arange(BBOX["minlon"], BBOX["maxlon"], GRID_STEP)
    lats = np.arange(BBOX["minlat"], BBOX["maxlat"], GRID_STEP)
    glon, glat = np.meshgrid(lons, lats)
    grid = pd.DataFrame({"lon": glon.ravel(), "lat": glat.ravel()})
    log.info("  Grid points: %d", len(grid))

    grid["elevation_m"] = _get_elevation_batch(grid.lat.values, grid.lon.values)
    grid["slope_deg"]   = _compute_slope(
        grid.lat.values, grid.lon.values, grid["elevation_m"].values
    )

    for col in GEOCHEM_COLS:
        if col in geochem.columns:
            valid = geochem.dropna(subset=["lon", "lat", col])
            grid[col] = np.maximum(0.0, _idw_interpolate(
                valid.lon.values, valid.lat.values, valid[col].values,
                grid.lon.values,  grid.lat.values, k=12,
            ))
        else:
            grid[col] = AFRICA_MEDIANS.get(col, 1.0)

    grid["dist_fault_km"] = _dist_to_faults_km(
        grid.lon.values, grid.lat.values, faults
    )

    dep_pts  = np.column_stack([deposits.geometry.x.values, deposits.geometry.y.values])
    dep_tree = cKDTree(dep_pts)
    dists, _ = dep_tree.query(np.column_stack([grid.lon.values, grid.lat.values]), k=1)
    grid["dist_deposit_km"] = dists * 111.0

    for col in LOG_TRANSFORM_COLS:
        grid[f"log_{col}"] = np.log1p(grid[col])

    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 65)
    log.info("GeoExplorer AI — Phase 2: Africa Feature Engineering")
    log.info("=" * 65)

    deposits = gpd.read_file(DATA_RAW / "mrds_africa_copper.geojson")
    geochem  = pd.read_csv(DATA_RAW / "geochem_africa.csv")
    faults   = gpd.read_file(DATA_RAW / "faults_africa.geojson")

    train_df = build_training_set(deposits, geochem, faults)
    train_df.to_csv(DATA_PROCESSED / "training_set.csv", index=False)
    log.info("Saved training_set.csv  (%d rows)", len(train_df))

    grid_df = build_prediction_grid(geochem, faults, deposits)
    grid_df.to_csv(DATA_PROCESSED / "prediction_grid.csv", index=False)
    log.info("Saved prediction_grid.csv  (%d points)", len(grid_df))

    log.info("-" * 65)
    log.info("Positives: %d  |  Negatives: %d  |  Features: %d",
             int((train_df.label == 1).sum()),
             int((train_df.label == 0).sum()),
             train_df.shape[1] - 3)
    log.info("Done. Run: python scripts/03_train_model.py")


if __name__ == "__main__":
    main()
