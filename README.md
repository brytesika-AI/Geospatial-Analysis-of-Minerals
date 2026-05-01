# GeoExplorer AI

AI-assisted copper prospectivity screening for Arizona and Nevada.

GeoExplorer AI is a portfolio-grade geospatial ML application inspired by the
science-forward mineral exploration work described by KoBold Metals: turning
exploration into a more repeatable, data-rich decision process. It is not an
official KoBold product and does not imply endorsement.

## What it does

| Area | Capability |
| --- | --- |
| Interactive map | Folium heatmap of copper prospectivity across Arizona and Nevada |
| Site scoring | Latitude/longitude lookup against a precomputed scored grid |
| Geological interpretation | HuggingFace LLM explanation with deterministic fallback |
| Batch scoring | Upload feature CSVs and score rows when the trained model bundle is available |
| Model analytics | Spatial cross-validation comparison, feature importance, and score distribution |
| Edge API | Cloudflare Worker routes for `/score`, `/heatmap`, `/interpret`, and `/health` |

## Current artifact status

The checked-in demo artifacts are intended to make the app runnable in cloud
demo environments. The data pipeline first tries public USGS endpoints and then
falls back to synthetic-but-geologically-shaped data when those endpoints are
unavailable.

Current local artifacts:

- Training samples: 1,590
- Prediction grid cells: 12,100
- Best saved model: Random Forest
- Spatial CV ROC-AUC: 0.903
- Spatial CV PR-AUC: 0.716
- Current raw geochemistry/deposit/fault artifacts may be synthetic fallbacks

## Architecture

```text
scripts/01_download_data.py
    -> downloads USGS MRDS, NGDB, and fault data when available
    -> falls back to synthetic AZ/NV copper exploration data

scripts/02_process_data.py
    -> builds positive/negative samples
    -> interpolates geochemistry with IDW
    -> adds elevation, slope, fault proximity, and deposit proximity
    -> creates training_set.csv and prediction_grid.csv

scripts/03_train_model.py
    -> compares Logistic Regression, Random Forest, and XGBoost
    -> evaluates with spatial cross-validation
    -> saves model bundle, metadata, feature importance, and scored grid

app/streamlit_app.py
    -> maps scores, scores coordinates, scores uploaded feature tables,
       and explains results

cloudflare/worker.js
    -> demo edge API with optional Workers AI interpretation
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

To regenerate all data and model artifacts:

```bash
python scripts/01_download_data.py
python scripts/02_process_data.py
python scripts/03_train_model.py
```

## Batch scoring format

Upload a CSV containing the trained model feature columns:

```text
elevation_m,slope_deg,log_cu_ppm,log_au_ppb,fe_pct,log_pb_ppm,log_zn_ppm,log_mo_ppm,log_as_ppm,dist_fault_km,dist_deposit_km
```

When `models/prospectivity_model.pkl` is present, the app adds:

- `prospectivity_score`
- `risk_tier`

## Cloudflare Worker

The Worker is in `cloudflare/`.

```bash
cd cloudflare
npx wrangler deploy
```

`cloudflare/wrangler.jsonc` deploys with Workers AI enabled and no KV binding
by default. Add a `kv_namespaces` binding named `GEO_KV` when you want
production KV-backed grid lookup. Without KV, the Worker returns deterministic
mock scores for development.

## API example

```bash
curl -X POST https://geo-explorer-ai.bryte-sika.workers.dev/score \
  -H "Content-Type: application/json" \
  -d "{\"lat\":33.45,\"lon\":-110.80,\"features\":{\"cu_ppm\":450},\"interpret\":true}"
```

## Tech stack

- Geospatial: GeoPandas, Shapely, SciPy, SRTM
- ML: scikit-learn, XGBoost, SHAP, joblib
- App: Streamlit, Folium, Plotly
- LLM: HuggingFace Inference API
- Edge: Cloudflare Workers and Workers AI

## Author

Bright Sikazwe  
GitHub: https://github.com/brytesika-AI
