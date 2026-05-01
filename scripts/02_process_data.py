"""
scripts/02_process_data.py
==========================
Phase 1 – Feature Engineering

Builds the training dataset for mineral prospectivity modelling:

  1. Loads deposits (positives) + creates spatially-stratified negatives
  2. Spatially joins geochemical observations via IDW interpolation
  3. Fetches SRTM elevation for every sample point
  4. Computes distance-to-fault and distance-to-nearest-deposit features
  5. Derives slope / roughness from elevation
  6. Writes  data/processed/training_set.csv
  7. Writes  data/processed/prediction_grid.csv  (0.1° grid over AZ/NV)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder

try:
    import srtm
    SRTM_AVAILABLE = True
except ImportError:
    SRTM_AVAILABLE = False
    logging.warning("srtm.py not installed – elevation will be estimated from DEM proxy")

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

BBOX = dict(minlon=-120.0, minlat=31.0, maxlon=-109.0, maxlat=42.0)
SEED = 42
NEG_RATIO = 4          # negatives per positive
BUFFER_DEG = 0.15      # ~15 km exclusion zone around positives for negatives
ELEV_SIGMA = 300.0     # noise std-dev for synthetic elevation proxy (metres)
GRID_STEP = 0.10       # degrees (≈ 11 km); controls heatmap resolution


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_elevation_batch(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return elevation in metres for arrays of lat/lon."""
    if SRTM_AVAILABLE:
        ed = srtm.get_data()
        elevs = np.array([
            ed.get_elevation(float(la), float(lo)) or 0
            for la, lo in zip(lats, lons)
        ], dtype=float)
        # SRTM returns None for ocean tiles; fill with 0
        return np.where(np.isnan(elevs), 0, elevs)

    # ── Proxy: DEM model for AZ/NV Basin-and-Range province ─────────────────
    #   Basin floors ~400-900 m, ranges peak at 2000-3000 m.
    #   Rough approximation as a function of lat/lon:
    rng = np.random.default_rng(99)
    base = (
        1000
        + 600 * np.sin(np.radians((lats - 34) * 10))
        + 400 * np.cos(np.radians((lons + 114) * 8))
    )
    return (base + rng.normal(0, ELEV_SIGMA, size=len(lats))).clip(200, 3500)


def _compute_slope(lats: np.ndarray, lons: np.ndarray, elevs: np.ndarray) -> np.ndarray:
    """
    Approximate slope (degrees) using finite-difference neighbours.
    For scattered points we use the nearest-neighbour gradient.
    """
    tree = cKDTree(np.column_stack([lons, lats]))
    _, idx = tree.query(np.column_stack([lons, lats]), k=5)  # 5 nearest
    slopes = np.zeros(len(lats))
    for i in range(len(lats)):
        nbrs = idx[i, 1:]  # exclude self
        dz = np.abs(elevs[nbrs] - elevs[i])
        dl = np.hypot(lons[nbrs] - lons[i], lats[nbrs] - lats[i]) * 111_000  # approx metres
        dl = np.where(dl < 1, 1, dl)
        slopes[i] = np.degrees(np.arctan(dz.max() / dl.min()))
    return slopes


def _idw_interpolate(
    known_lons: np.ndarray,
    known_lats: np.ndarray,
    known_vals: np.ndarray,
    query_lons: np.ndarray,
    query_lats: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """
    Inverse-Distance Weighting interpolation.
    Transfers geochemical observations from irregular sample points
    to arbitrary query locations.
    """
    src_pts = np.column_stack([known_lons, known_lats])
    tgt_pts = np.column_stack([query_lons, query_lats])
    tree = cKDTree(src_pts)
    dists, idxs = tree.query(tgt_pts, k=min(k, len(known_lons)))
    # Avoid divide-by-zero at exact matches
    dists = np.where(dists == 0, 1e-10, dists)
    weights = 1.0 / (dists ** power)
    w_sum = weights.sum(axis=1, keepdims=True)
    return (weights * known_vals[idxs]).sum(axis=1) / w_sum.squeeze()


def _dist_to_lines_deg(
    query_lons: np.ndarray,
    query_lats: np.ndarray,
    fault_gdf: gpd.GeoDataFrame,
    n_sample_pts: int = 5,
) -> np.ndarray:
    """
    Approximate min-distance from each query point to the nearest fault segment.
    Returns distance in km (Haversine approximation).
    """
    if fault_gdf is None or len(fault_gdf) == 0:
        return np.full(len(query_lons), 50.0)

    # Sample points along each fault line
    fault_pts = []
    for geom in fault_gdf.geometry:
        if geom is None:
            continue
        if hasattr(geom, "geoms"):   # MultiLineString
            parts = list(geom.geoms)
        else:
            parts = [geom]
        for part in parts:
            try:
                coords = list(part.coords)
                fault_pts.extend(coords[::max(1, len(coords) // n_sample_pts)])
            except Exception:
                pass

    if not fault_pts:
        return np.full(len(query_lons), 50.0)

    fp = np.array(fault_pts)   # (N, 2) → (lon, lat)
    tree = cKDTree(fp)
    qp = np.column_stack([query_lons, query_lats])
    dists_deg, _ = tree.query(qp, k=1)
    # Convert approximate degrees → km (1° ≈ 111 km at these latitudes)
    return dists_deg * 111.0


# ═══════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_set(
    deposits: gpd.GeoDataFrame,
    geochem: pd.DataFrame,
    faults: gpd.GeoDataFrame,
) -> pd.DataFrame:
    log.info("Building training set …")
    rng = np.random.default_rng(SEED)

    # ── Positive samples ─────────────────────────────────────────────────────
    pos = pd.DataFrame({
        "lon": deposits.geometry.x.values,
        "lat": deposits.geometry.y.values,
        "label": 1,
    })
    n_pos = len(pos)
    log.info("  Positive samples (deposits): %d", n_pos)

    # ── Negative samples ─────────────────────────────────────────────────────
    # Spatially stratified random sampling outside a buffer around deposits
    dep_tree = cKDTree(np.column_stack([pos.lon.values, pos.lat.values]))
    neg_rows = []
    attempts = 0
    while len(neg_rows) < n_pos * NEG_RATIO and attempts < 200_000:
        lo = rng.uniform(BBOX["minlon"], BBOX["maxlon"])
        la = rng.uniform(BBOX["minlat"], BBOX["maxlat"])
        d, _ = dep_tree.query([[lo, la]])
        if d[0] > BUFFER_DEG:
            neg_rows.append((lo, la, 0))
        attempts += 1

    neg = pd.DataFrame(neg_rows, columns=["lon", "lat", "label"])
    log.info("  Negative samples (background): %d", len(neg))

    df = pd.concat([pos, neg], ignore_index=True)

    # ── Elevation + Slope ────────────────────────────────────────────────────
    log.info("  Fetching elevation …")
    df["elevation_m"] = _get_elevation_batch(df.lat.values, df.lon.values)
    log.info("  Computing slope …")
    df["slope_deg"] = _compute_slope(df.lat.values, df.lon.values, df["elevation_m"].values)

    # ── Geochemistry (IDW) ───────────────────────────────────────────────────
    log.info("  Interpolating geochemistry …")
    geochem_cols = ["cu_ppm", "au_ppb", "fe_pct", "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm"]
    for col in geochem_cols:
        if col not in geochem.columns:
            # Fill with AZ/NV median if column absent
            medians = dict(cu_ppm=18, au_ppb=1.2, fe_pct=2.8, pb_ppm=14,
                           zn_ppm=45, mo_ppm=0.8, as_ppm=5)
            df[col] = medians.get(col, 1.0)
        else:
            valid = geochem.dropna(subset=["lon", "lat", col])
            df[col] = _idw_interpolate(
                valid.lon.values, valid.lat.values, valid[col].values,
                df.lon.values, df.lat.values,
            )
        df[col] = df[col].clip(lower=0)

    # ── Distance to fault ────────────────────────────────────────────────────
    log.info("  Computing fault proximity …")
    df["dist_fault_km"] = _dist_to_lines_deg(df.lon.values, df.lat.values, faults)

    # ── Distance to nearest deposit ──────────────────────────────────────────
    log.info("  Computing deposit proximity …")
    dep_pts = np.column_stack([pos.lon.values, pos.lat.values])
    dep_tree2 = cKDTree(dep_pts)
    all_pts = np.column_stack([df.lon.values, df.lat.values])
    dists_deg, _ = dep_tree2.query(all_pts, k=2)   # k=2 because pos samples will match themselves
    df["dist_deposit_km"] = dists_deg[:, 1] * 111.0

    # ── Log-transform skewed geochemical features ────────────────────────────
    for col in ["cu_ppm", "au_ppb", "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm"]:
        df[f"log_{col}"] = np.log1p(df[col])

    log.info("  Training set shape: %s", df.shape)
    return df


def build_prediction_grid(
    geochem: pd.DataFrame,
    faults: gpd.GeoDataFrame,
    deposits: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Build the regular grid across AZ/NV used for heatmap predictions."""
    log.info("Building prediction grid (step=%.2f°) …", GRID_STEP)

    lons = np.arange(BBOX["minlon"], BBOX["maxlon"], GRID_STEP)
    lats = np.arange(BBOX["minlat"], BBOX["maxlat"], GRID_STEP)
    glon, glat = np.meshgrid(lons, lats)
    glon = glon.ravel()
    glat = glat.ravel()

    grid = pd.DataFrame({"lon": glon, "lat": glat})

    log.info("  Grid points: %d", len(grid))

    grid["elevation_m"] = _get_elevation_batch(grid.lat.values, grid.lon.values)
    grid["slope_deg"] = _compute_slope(grid.lat.values, grid.lon.values, grid["elevation_m"].values)

    geochem_cols = ["cu_ppm", "au_ppb", "fe_pct", "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm"]
    medians = dict(cu_ppm=18, au_ppb=1.2, fe_pct=2.8, pb_ppm=14,
                   zn_ppm=45, mo_ppm=0.8, as_ppm=5)
    for col in geochem_cols:
        if col in geochem.columns:
            valid = geochem.dropna(subset=["lon", "lat", col])
            grid[col] = np.maximum(0, _idw_interpolate(
                valid.lon.values, valid.lat.values, valid[col].values,
                grid.lon.values, grid.lat.values, k=12,
            ))
        else:
            grid[col] = medians.get(col, 1.0)

    grid["dist_fault_km"] = _dist_to_lines_deg(grid.lon.values, grid.lat.values, faults)

    dep_pts = np.column_stack([deposits.geometry.x.values, deposits.geometry.y.values])
    dep_tree = cKDTree(dep_pts)
    dists_deg, _ = dep_tree.query(np.column_stack([grid.lon.values, grid.lat.values]), k=1)
    grid["dist_deposit_km"] = dists_deg * 111.0

    for col in ["cu_ppm", "au_ppb", "pb_ppm", "zn_ppm", "mo_ppm", "as_ppm"]:
        grid[f"log_{col}"] = np.log1p(grid[col])

    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 60)
    log.info("GeoExplorer AI — Phase 1: Feature Engineering")
    log.info("=" * 60)

    deposits = gpd.read_file(DATA_RAW / "mrds_copper_az_nv.geojson")
    geochem = pd.read_csv(DATA_RAW / "geochem_az_nv.csv")
    faults = gpd.read_file(DATA_RAW / "faults_az_nv.geojson")

    # Training set
    train_df = build_training_set(deposits, geochem, faults)
    out_train = DATA_PROCESSED / "training_set.csv"
    train_df.to_csv(out_train, index=False)
    log.info("Saved training set → %s", out_train.name)

    # Prediction grid
    grid_df = build_prediction_grid(geochem, faults, deposits)
    out_grid = DATA_PROCESSED / "prediction_grid.csv"
    grid_df.to_csv(out_grid, index=False)
    log.info("Saved prediction grid → %s  (%d points)", out_grid.name, len(grid_df))

    # Quick sanity check
    pos_count = (train_df.label == 1).sum()
    neg_count = (train_df.label == 0).sum()
    log.info("-" * 60)
    log.info("Training set:  %d positive  |  %d negative  |  %d features",
             pos_count, neg_count, train_df.shape[1] - 3)  # minus lon, lat, label
    log.info("Prediction grid: %d points", len(grid_df))
    log.info("-" * 60)
    log.info("Done. Run: python scripts/03_train_model.py")


if __name__ == "__main__":
    main()
