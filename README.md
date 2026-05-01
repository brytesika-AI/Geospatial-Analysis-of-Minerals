# GeoExplorer AI

Role-aligned portfolio project for a KoBold Metals Data Scientist position.

GeoExplorer AI is an AI-assisted mineral exploration decision-support tool for
copper prospectivity screening across Arizona and Nevada. It demonstrates how I
would build data tools with geoscientists: curate physical-system data, create
statistically valid spatial predictions, visualize model evidence, rank
exploration targets, and propose field programs that reduce uncertainty.

The visual direction and product language are inspired by KoBold Metals' public
science-forward positioning around AI, human expertise, and repeatable
exploration. This is not an official KoBold product and does not imply
endorsement.

## Why this matches the role

| KoBold role requirement | Evidence in this project |
| --- | --- |
| Build proprietary exploration tools | Modular pipeline, Streamlit decision app, and Cloudflare scoring API |
| Curate geophysical/geochemical/geologic/geographic data | Data ingestion scripts combine deposits, geochemistry, faults, terrain, and derived spatial features |
| Predict compositional anomalies in the crust | Supervised prospectivity model trained on spatial feature vectors and applied to a 2D AZ/NV grid |
| Evaluate statistically valid predictions | Spatial cross-validation to reduce spatial leakage, with ROC-AUC and PR-AUC comparisons |
| Create rapid visualizations | Interactive heatmap, site scoring, feature breakdown, model analytics, score distribution, and target queue |
| Guide field programs | Target Portfolio ranks locations by score and uncertainty proxy, then recommends next data collection |
| Collaborate with geoscientists | Geological interpretation layer translates model output into exploration language and caveated recommendations |
| Cloud computing resources | Cloudflare Worker deploy with Workers AI binding and HTTP scoring endpoints |
| Software engineering practices | Git, tests, GitHub Actions CI, documented deployment, reproducible scripts, and `.env` handling |
| Ownership of ambiguous scientific tools | End-to-end app from data acquisition through model training, interpretation, deployment, and reviewer-facing docs |

## Product workflow

1. Curate data for mineral systems:
   `scripts/01_download_data.py` collects public-source deposits, geochemistry,
   and faults where available, with synthetic fallback data for reproducible demos.

2. Engineer geoscience features:
   `scripts/02_process_data.py` builds positive/negative samples, interpolates
   geochemistry using IDW, adds terrain features, computes fault proximity, and
   creates a 2D prediction grid.

3. Train and evaluate models:
   `scripts/03_train_model.py` compares Logistic Regression, Random Forest, and
   XGBoost using spatial cross-validation, then saves the best model bundle and
   scored grid.

4. Make exploration decisions:
   `app/streamlit_app.py` maps prospectivity, scores sites, scores uploaded
   feature tables, visualizes model evidence, and ranks a target portfolio.

5. Share through cloud tooling:
   `cloudflare/worker.js` exposes demo scoring and interpretation endpoints for
   partner-facing or field-facing workflows.

## Current model snapshot

- Region: Arizona and Nevada
- Commodity: Copper
- Training rows: 1,590
- Prediction grid cells: 12,100
- Best saved model: Random Forest
- Spatial CV ROC-AUC: 0.903
- Spatial CV PR-AUC: 0.716

The checked-in artifacts are meant to keep the app runnable in cloud demo
environments. The pipeline tries public USGS endpoints first and falls back to
synthetic-but-geologically-shaped data when those sources are unavailable.

## App capabilities

| View | Purpose |
| --- | --- |
| Prospectivity Map | Inspect 2D spatial prediction surfaces and training/source points |
| Target Portfolio | Rank exploration targets and connect uncertainty to field-program design |
| Site Scorer | Score a coordinate and generate a geologist-readable interpretation |
| Batch Scoring | Upload feature tables and apply the trained model bundle |
| Model Analytics | Compare model performance, feature importance, and score distributions |
| Role Fit | Reviewer-facing mapping between KoBold responsibilities and project evidence |
| API | Shows edge scoring contract for cloud deployment |

## 2D now, 3D next

The current app implements a 2D prospectivity screen because the available demo
artifacts are surface-level. The same structure is ready for 3D extension:

- Add depth-indexed geophysical inversion voxels as feature tables.
- Add drillhole intervals or lithology logs as labelled observations.
- Replace the 2D grid with `(x, y, z)` blocks or depth slices.
- Add uncertainty estimates from ensembles, Bayesian models, or spatial
  simulations.
- Visualize section views, depth slices, and target volumes.

## Statistical and geoscience choices

- Spatial cross-validation avoids over-optimistic scores from nearby train/test
  leakage.
- Distance-to-fault and distance-to-deposit features encode structural and
  mineral-system context.
- IDW interpolation transfers sparse geochemistry onto target and grid
  locations.
- The LLM layer is interpretation-only; it does not make the score.
- The target queue uses a simple uncertainty proxy to show the decision logic
  that should eventually be replaced with calibrated uncertainty from ensembles
  or Bayesian/geostatistical models.

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

Run tests:

```bash
pytest -q
```

## Batch scoring format

Upload a CSV containing these trained model feature columns:

```text
elevation_m,slope_deg,log_cu_ppm,log_au_ppb,fe_pct,log_pb_ppm,log_zn_ppm,log_mo_ppm,log_as_ppm,dist_fault_km,dist_deposit_km
```

When `models/prospectivity_model.pkl` is present, the app adds:

- `prospectivity_score`
- `risk_tier`

## Cloudflare Worker

The Worker is live at:

https://geo-explorer-ai.bryte-sika.workers.dev

Deploy from the `cloudflare/` folder:

```bash
npx wrangler deploy
```

`cloudflare/wrangler.jsonc` deploys with Workers AI enabled and no KV binding
by default. Add a `kv_namespaces` binding named `GEO_KV` when you want
production KV-backed grid lookup. Without KV, the Worker returns deterministic
mock scores for development.

API example:

```bash
curl -X POST https://geo-explorer-ai.bryte-sika.workers.dev/score \
  -H "Content-Type: application/json" \
  -d "{\"lat\":33.45,\"lon\":-110.80,\"features\":{\"cu_ppm\":450},\"interpret\":true}"
```

## Tech stack

- Python: pandas, NumPy, SciPy, GeoPandas, Shapely
- ML: scikit-learn, XGBoost, SHAP, joblib
- Visualization: Streamlit, Folium, Plotly
- LLM: HuggingFace Inference API and Cloudflare Workers AI
- Cloud: Cloudflare Workers
- Engineering: Git, pytest, GitHub Actions CI

## Author

Bright Sikazwe  
GitHub: https://github.com/brytesika-AI

