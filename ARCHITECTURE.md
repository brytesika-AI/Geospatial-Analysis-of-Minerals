# GeoExplorer AI Africa — Architecture & Technical Design

> Copper · Cobalt · Nickel prospectivity intelligence for the Central/Southern Africa Copperbelt  
> Powered by XGBoost ML, Vercel serverless Python, and Leaflet.js

---

## Table of Contents

1. [Business Context](#business-context)
2. [Data Engineering Pipeline](#data-engineering-pipeline)
3. [Machine Learning Architecture](#machine-learning-architecture)
4. [AI / LLM Interpretation Layer](#ai--llm-interpretation-layer)
5. [API & Deployment Architecture](#api--deployment-architecture)
6. [Mineral Systems Framework](#mineral-systems-framework)
7. [Key Insights & Findings](#key-insights--findings)
8. [Repository Structure](#repository-structure)

---

## Business Context

### Strategic Problem

Greenfield mineral exploration in Sub-Saharan Africa is expensive, data-sparse, and high-risk. A single drill campaign costs \$250k–\$1.5M. Without a systematic prioritisation framework, capital is misallocated on low-probability targets.

**GeoExplorer AI Africa** provides an ML-driven prospectivity map that ranks 20,125 grid cells across the Copperbelt — compressing years of geological desk-study into an interactive intelligence platform.

### Market Relevance

| Context | Detail |
|---|---|
| Region | Central/Southern Africa Copperbelt (DRC, Zambia, Zimbabwe, Tanzania, Angola) |
| Commodities | Copper (Cu), Cobalt (Co), Nickel (Ni) — critical minerals for EV batteries and energy transition |
| Peer Benchmark | KoBold Metals raised \$195M to apply ML to African critical minerals — this stack demonstrates the same capability |
| Model ROC-AUC | 0.889 — strong discrimination between prospective and barren ground |
| Data Foundation | 2,045 labelled training points · 409 MRDS deposits · 13 geoscientific features |

### SHSC Mineral Systems

The geological intelligence model is anchored in the **Source–Pathway–Trap–Modifier** (SHSC) framework — the modern paradigm for understanding where metal concentrations form:

- **Source**: Copper-bearing basement rocks and sedimentary sequences (log Cu ppm)
- **Pathway**: Structural corridors — faults and fold axes that channelled hydrothermal fluids (dist_fault_km)
- **Trap**: Redox-favourable host rocks — carbonaceous shales, oxide/sulphide interfaces (geology_code)
- **Modifier**: Proximity to known deposits (dist_deposit_km), distance to basement contacts, geophysics

---

## Data Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA ENGINEERING PIPELINE                            │
├─────────┬──────────────┬───────────────┬────────────────┬───────────────────┤
│ INGEST  │  TRANSFORM   │   FEATURE ENG │   TRAINING     │   SERVING         │
│         │              │               │                │                   │
│ MRDS    │ Bbox clip    │ log(Cu/Co/Ni) │ XGBoost CV     │ predictions.csv   │
│ GeoJSON │ Null impute  │ dist_fault    │ SHAP values    │ Vercel API        │
│         │ CRS reproject│ dist_deposit  │ GridSearchCV   │ IDW scoring       │
│ Govt    │ Deduplicate  │ geology_code  │ Spatial blocks │ Leaflet heatmap   │
│ surveys │ Outlier clip │ EM/mag proxy  │                │                   │
│         │              │               │                │                   │
│ Aster   │ Grid sample  │ basement_dist │ ROC-AUC: 0.889 │ /api/predictions  │
│ DEM     │ 5km cells    │ topo_relief   │ PR-AUC:  0.785 │ /api/targets      │
│         │              │               │                │ /api/score (IDW)  │
└─────────┴──────────────┴───────────────┴────────────────┴───────────────────┘
```

### Phase 1 — Ingest

| Source | Format | Records | Description |
|---|---|---|---|
| USGS MRDS | GeoJSON | 409 deposits | Known Cu/Co/Ni occurrences in Africa |
| Govt geological surveys | CSV/Shapefile | ~800 samples | Geochemistry assay data |
| ASTER GDEM v3 | GeoTIFF | 30m DEM | Topographic relief and drainage |
| Natural Earth | GeoJSON | 593 railways, 68 ports, 52 countries | Infrastructure context |
| OpenStreetMap | GeoJSON | 323 fault segments | Structural geology proxies |

### Phase 2 — Transform

- **Spatial clipping**: Study bbox `[15°E–38°E, 35°S–0°]` for ML model; broader SSA bbox `[18°W–53°E, 35°S–18°N]` for context
- **Grid generation**: 20,125 cells at 5 km resolution across the Copperbelt
- **Null imputation**: Median fill for geochemical features; nearest-neighbour for distance features
- **Log-transform**: `log1p(Cu_ppm)`, `log1p(Co_ppm)`, `log1p(Ni_ppm)` — compresses 6-order-of-magnitude range

### Phase 3 — Feature Engineering

| Feature | Type | Geological Meaning |
|---|---|---|
| `log_cu_ppm` | Geochemical | Primary Cu enrichment (Source) |
| `log_co_ppm` | Geochemical | Co co-enrichment — SHSC indicator |
| `log_ni_ppm` | Geochemical | Ni co-enrichment — magmatic sulphide indicator |
| `dist_fault_km` | Structural | Proximity to fluid pathways (Pathway) |
| `dist_deposit_km` | Spatial | Proximity to known mineralisation (Trap) |
| `geology_code` | Lithological | Host rock type encoded ordinally |
| `basement_dist_km` | Structural | Distance to basement contact |
| `topo_relief_m` | Topographic | Erosional exposure proxy |
| `em_anomaly` | Geophysical | EM conductor proxy for sulphide |
| `mag_residual_nT` | Geophysical | Magnetic susceptibility — mafic intrusions |
| `drainage_density` | Hydrological | Supergene enrichment potential |
| `soil_ph` | Pedological | Oxidation zone indicator |
| `vegetation_index` | Remote sensing | Hydrothermally-altered vegetation proxy |

### Phase 4 — Model Training

See [Machine Learning Architecture](#machine-learning-architecture).

### Phase 5 — Serving

Pre-computed `data/processed/predictions.csv` (20,125 rows) is the serving store:
- **Full grid**: `/api/predictions` streams lat/lon/score/tier for heatmap rendering
- **Point scoring**: `/api/score?lat=&lon=` applies IDW from 4 nearest grid cells — no XGBoost at query time
- **Targets**: `/api/targets?n=50&minScore=0.5&tier=&country=` enriches top-N with drill programs, costs, Co/Cu ratios

---

## Machine Learning Architecture

### Model Selection

```
Candidates evaluated:
  Random Forest      → ROC-AUC 0.851  (baseline ensemble)
  Gradient Boosting  → ROC-AUC 0.867
  XGBoost            → ROC-AUC 0.889  ✓ selected
  LightGBM           → ROC-AUC 0.881
  Logistic Regression→ ROC-AUC 0.743  (underfit — non-linear geology)
```

**Why XGBoost**: Best AUC, handles mixed feature types natively, supports SHAP explanations, fast inference.

### Spatial Cross-Validation

Standard k-fold CV **inflates AUC** for geospatial problems due to spatial autocorrelation — nearby cells share geology and score similarly regardless of label leakage.

**Solution**: 5-fold geographically-blocked CV using 200×200 km spatial blocks:
- Each fold holds out a geographic region (not random rows)
- Prevents adjacent cells from appearing in both train and validation sets
- Reported AUC = mean of 5 spatial-block folds → conservative, production-representative estimate

### Hyperparameter Optimisation

```python
GridSearchCV(
    XGBClassifier(),
    param_grid={
        "n_estimators":    [200, 400, 600],
        "max_depth":       [4, 6, 8],
        "learning_rate":   [0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.85, 1.0],
        "colsample_bytree":[0.7, 0.85, 1.0],
        "scale_pos_weight":[3, 5, 8],   # class imbalance correction
    },
    scoring="roc_auc",
    cv=spatial_cv_folds,
)
```

### Output

`prospectivity_score` ∈ [0, 1] → bucketed into risk tiers:

| Tier | Score Range | Interpretation | Recommended Programme |
|---|---|---|---|
| Very High | ≥ 0.85 | Drill-ready target | Diamond Core 300–500 m + IP/CSAMT |
| High | 0.70–0.85 | Strong follow-up warranted | RC 150–250 m + IP survey |
| Moderate | 0.50–0.70 | Scout drilling | RC 50–150 m + soil geochemistry |
| Low | < 0.50 | Regional reconnaissance | EM/magnetics + recon mapping |

---

## AI / LLM Interpretation Layer

### Architecture

```
User query
    │
    ▼
Prompt template (Jinja2)
    │  ├── Target metadata (score, tier, co_cu_pct, ni_cu_pct)
    │  ├── Feature importances (SHAP-ranked)
    │  └── Geological context (geology_code, dist_fault_km)
    ▼
LLaMA-3 70B (via Groq API, ~500ms p99)
    │
    ▼
Structured interpretation:
  • Geological rationale (SHSC framework)
  • Co/Cu diagnostic (≥10% → Katanga-type SHSC)
  • Ni/Cu diagnostic (≥20% → Bushveld magmatic sulphide)
  • Recommended drill programme
  • Risk factors & uncertainties
    │
    ▼
Fallback (if Groq unavailable): rule-based template
```

### Prompt Engineering

The LLM prompt is designed to produce **geoscientist-grade** narrative, not generic AI text:

- **Role**: "You are a senior economic geologist advising a junior mining company board"
- **Constraints**: Reference specific SHSC components, name the geological terrane, cite analogues (e.g. "Nkana style SHSC")
- **Output format**: Structured JSON → rendered in the UI target detail panel
- **Hallucination guard**: Score and feature values are injected as hard facts; LLM only interprets, never invents numbers

---

## API & Deployment Architecture

### Vercel Serverless

```
GitHub (brytesika-AI/Geospatial-Analysis-of-Minerals)
    │
    ▼
Vercel Build
    ├── api/*.py  → @vercel/python serverless functions
    └── public/   → static file serving (index.html)

Request flow:
  Browser → Vercel Edge → Python function → JSON response
  Browser → Vercel CDN  → public/index.html (static)
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service health check |
| `/api/metadata` | GET | Model metadata + portfolio tier counts |
| `/api/predictions?minScore=&tier=` | GET | Heatmap grid (lat/lon/score/tier) |
| `/api/score?lat=&lon=` | GET | IDW point score from nearest 4 grid cells |
| `/api/targets?n=&minScore=&tier=&country=` | GET | Enriched priority targets |
| `/api/data?type=deposits\|cities\|oilgas\|ports\|railways` | GET | Context layer data |

### IDW Scoring Algorithm

Point scoring without loading XGBoost — uses 4 nearest pre-computed grid cells:

```python
def idw_score(lat, lon, grid_rows, k=4):
    dists = [(haversine(lat, lon, r["lat"], r["lon"]), r["score"]) for r in grid_rows]
    nearest = sorted(dists)[:k]
    weights = [1/(d**2 + 1e-9) for d, _ in nearest]
    score = sum(w*s for (_, s), w in zip(nearest, weights)) / sum(weights)
    return round(score, 3)
```

### Frontend Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Map | Leaflet.js 1.9 + Leaflet.heat | Interactive heatmap, 6 toggleable layer groups |
| Charts | Chart.js 4 | Tier donut, feature importance bar, score histogram, country bar |
| Styling | Custom CSS (dark navy theme) | Professional dark UI: `--bg:#0c1420`, `--gold:#f59e0b` |
| Data | Vanilla fetch() to Vercel API | No framework overhead, fast cold start |
| Coordinates | JS DMS conversion | Drill program coordinate display |
| Export | CSV data URI | Download targets as CSV |

### CORS & Security

All API endpoints return:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

---

## Mineral Systems Framework

### SHSC Applied to the Copperbelt

```
SOURCE          PATHWAY         TRAP            MODIFIER
────────        ────────        ────────        ────────
Kupferschiefer  Fault zones     Redox           Basement
basement Cu     (dist_fault_km) interfaces      unconformity
                                                (basement_dist)
Evaporite-      Fold axes       Carbonaceous    Proximity to
hosted Cu       structural      shale           known deposits
                corridors       (geology_code)  (dist_deposit_km)

Mafic           Syn-sedimentary Oxide/sulphide  Topographic
intrusions      faults          boundary        relief
(mag_residual)                  (em_anomaly)    (topo_relief_m)
```

### Co/Cu Diagnostic Ratios

| Ratio | Threshold | Interpretation | Analogue |
|---|---|---|---|
| Co/Cu | ≥ 10% | Sediment-hosted stratiform Cu-Co | Nkana, Konkola (Zambia) |
| Co/Cu | ≥ 30% | High-grade Co dominant | Tenke Fungurume (DRC) |
| Ni/Cu | ≥ 20% | Magmatic sulphide component | Munali (Zambia), Kapalagulu |
| Ni/Cu | ≥ 50% | Ni-dominant magmatic system | Kabanga (Tanzania) |

---

## Key Insights & Findings

### Feature Importance (SHAP-ranked)

1. **`dist_deposit_km`** (28%) — Proximity to known mineralisation is the strongest predictor; deposits cluster in mineralised corridors
2. **`log_cu_ppm`** (22%) — Primary geochemical signal; background Cu anomalism predicts SHSC systems
3. **`geology_code`** (18%) — Host lithology controls redox trapping; Roan Group equivalents dominate
4. **`dist_fault_km`** (14%) — Structural control on hydrothermal fluid flow
5. **`log_co_ppm`** (9%) — Co co-enrichment confirms SHSC over porphyry origin
6. **`basement_dist_km`** (6%) — Basement unconformity — classic SHSC trap architecture
7. **Other 7 features** (3%) — Topographic, hydrological, geophysical proxies

### Geological Conclusions

- **Dominant system**: Sediment-Hosted Stratiform Copper (SHSC) — same class as Nkana, Konkola, Lumwana, Tenke Fungurume
- **Structural control**: 73% of Very High targets lie within 15 km of a mapped fault — confirming fluid-pathway control
- **Geochemical halo**: Background Cu anomalism (50–200 ppm) at 5 km scale is predictive even without visible mineralisation
- **Co flag**: 41% of Very High targets have Co/Cu ≥ 10% — cobalt by-product potential improves project economics significantly in the EV metals market

### Portfolio Summary (20,125 grid cells)

| Tier | Count | % of Grid | Recommended Action |
|---|---|---|---|
| Very High | ~340 | 1.7% | Immediate drill planning |
| High | ~980 | 4.9% | Detailed ground follow-up |
| Moderate | ~2,800 | 13.9% | Regional soil/EM programme |
| Low | ~16,005 | 79.5% | No immediate action |

---

## Repository Structure

```
Geospatial-Analysis-of-Minerals/
├── api/                          # Vercel serverless Python functions
│   ├── _utils.py                 # BaseHandler, IDW scoring, CSV readers
│   ├── predictions.py            # GET /api/predictions
│   ├── score.py                  # GET /api/score
│   ├── targets.py                # GET /api/targets
│   ├── data.py                   # GET /api/data
│   ├── metadata.py               # GET /api/metadata
│   └── health.py                 # GET /api/health
├── public/
│   └── index.html                # Professional SPA (Leaflet + Chart.js)
├── app/
│   ├── streamlit_app.py          # Streamlit multi-tab dashboard
│   └── geo_utils.py              # Spatial utilities, country inference
├── data/
│   ├── raw/
│   │   ├── mrds_africa_copper.geojson    # 409 Cu/Co/Ni deposits
│   │   ├── cities_africa.csv             # 518 cities
│   │   ├── oil_gas_africa.csv            # 22 O&G fields
│   │   ├── ports_africa.csv              # 68 ports
│   │   ├── railways_africa.geojson       # 593 railway segments
│   │   ├── faults_africa.geojson         # 323 fault segments
│   │   └── countries_africa.geojson      # 52 countries
│   └── processed/
│       ├── predictions.csv               # 20,125 scored grid cells
│       └── features_engineered.csv       # Training feature matrix
├── models/
│   ├── model_metadata.json               # ROC-AUC, PR-AUC, feature list
│   └── feature_importance.json           # SHAP-ranked importances
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_spatial_analysis.ipynb
├── vercel.json                   # Vercel routing configuration
├── requirements.txt              # Pure Python (no GDAL/geopandas)
├── ARCHITECTURE.md               # This document
└── README.md
```

---

## Deployment

### Vercel (Primary)

1. Push to `main` branch of `github.com/brytesika-AI/Geospatial-Analysis-of-Minerals`
2. Vercel auto-deploys on push:
   - `api/*.py` → serverless Python functions
   - `public/index.html` → static CDN
3. Environment variables (set in Vercel dashboard):
   - `GROQ_API_KEY` — LLaMA-3 interpretations (optional)
   - `OPENAI_API_KEY` — fallback LLM (optional)

### Streamlit Cloud (Secondary)

Live at: `https://geospatial-analysis-of-minerals-[hash].streamlit.app`  
Branch: `main` · Entry: `app/streamlit_app.py`

---

*Built by Bright Sikazwe · GeoExplorer AI Africa v2 · 2025*
