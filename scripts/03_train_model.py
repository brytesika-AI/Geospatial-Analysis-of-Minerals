"""
scripts/03_train_model.py
=========================
Phase 3 – Model Training & Evaluation  (Africa Copperbelt)

Trains three classifiers for Cu/Co/Ni mineral prospectivity:
  • Logistic Regression  (interpretable baseline)
  • Random Forest        (ensemble, handles spatial non-linearities)
  • XGBoost             (gradient-boosted trees)

Feature set includes cobalt (log_co_ppm) and nickel (log_ni_ppm) in addition
to the base geochemical suite — both critical for KoBold Metals' DRC Katanga
and Botswana Cu-Ni target areas.

Evaluation  : spatial k-fold cross-validation (avoids spatial leakage)
Outputs     :
  models/prospectivity_model.pkl
  models/feature_importance.json
  models/model_comparison.json
  models/model_metadata.json
  data/processed/predictions.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("shap not installed — skipping SHAP analysis")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR     = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42

# Africa Copperbelt feature set — adds Co and Ni vs AZ/NV version
FEATURE_COLS = [
    "elevation_m",
    "slope_deg",
    "log_cu_ppm",
    "log_co_ppm",      # cobalt — hallmark DRC Katanga Cu-Co systems
    "log_ni_ppm",      # nickel — Botswana Cu-Ni focus
    "log_au_ppb",
    "fe_pct",
    "log_pb_ppm",
    "log_zn_ppm",
    "log_mo_ppm",
    "log_as_ppm",
    "dist_fault_km",
    "dist_deposit_km",
]

FEATURE_DISPLAY_NAMES = {
    "elevation_m":     "Elevation (m)",
    "slope_deg":       "Slope (°)",
    "log_cu_ppm":      "Copper (log ppm)",
    "log_co_ppm":      "Cobalt (log ppm)",
    "log_ni_ppm":      "Nickel (log ppm)",
    "log_au_ppb":      "Gold (log ppb)",
    "fe_pct":          "Iron (%)",
    "log_pb_ppm":      "Lead (log ppm)",
    "log_zn_ppm":      "Zinc (log ppm)",
    "log_mo_ppm":      "Molybdenum (log ppm)",
    "log_as_ppm":      "Arsenic (log ppm)",
    "dist_fault_km":   "Dist. to Fault (km)",
    "dist_deposit_km": "Dist. to Deposit (km)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Spatial cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def spatial_kfold_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple]:
    """
    Spatial k-fold: divide study area into a geographic grid and hold out
    each block as the test set. Prevents spatial autocorrelation leakage.
    """
    n_side = int(np.ceil(np.sqrt(n_splits)))
    lon_bins = np.linspace(df.lon.min(), df.lon.max(), n_side + 1)
    lat_bins = np.linspace(df.lat.min(), df.lat.max(), n_side + 1)
    df = df.reset_index(drop=True)
    splits = []
    for i in range(n_side):
        for j in range(n_side):
            test_mask = (
                (df.lon >= lon_bins[i]) & (df.lon < lon_bins[i + 1]) &
                (df.lat >= lat_bins[j]) & (df.lat < lat_bins[j + 1])
            )
            test_idx  = df.index[test_mask].values
            train_idx = df.index[~test_mask].values
            if len(test_idx) > 0 and len(train_idx) > 0:
                splits.append((train_idx, test_idx))
            if len(splits) >= n_splits:
                break
        if len(splits) >= n_splits:
            break
    return splits[:n_splits]


# ═══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════════════

def get_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED,
            )),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1, random_state=SEED,
            )),
        ]),
        "xgboost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=4, eval_metric="logloss",
                use_label_encoder=False, random_state=SEED, verbosity=0,
            )),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    name: str, pipeline: Pipeline,
    X: pd.DataFrame, y: np.ndarray,
    splits: list[tuple],
) -> dict[str, Any]:
    log.info("  Evaluating %s …", name)
    aucs, pr_aucs = [], []
    for train_idx, test_idx in splits:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if y_te.sum() == 0 or y_tr.sum() == 0:
            continue
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, proba))
        pr_aucs.append(average_precision_score(y_te, proba))

    return {
        "name":         name,
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std":  float(np.std(aucs)),
        "pr_auc_mean":  float(np.mean(pr_aucs)),
        "pr_auc_std":   float(np.std(pr_aucs)),
        "n_folds":      len(aucs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature importance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, float]:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        raw = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scaler = pipeline.named_steps.get("scaler")
        coef   = np.abs(clf.coef_[0])
        if scaler is not None:
            coef = coef * scaler.scale_
        raw = coef / coef.sum()
    else:
        raw = np.ones(len(X.columns)) / len(X.columns)
    importance = dict(zip(X.columns, raw.tolist()))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def run_shap(pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 300) -> dict:
    if not SHAP_AVAILABLE:
        return {}
    try:
        clf = pipeline.named_steps["clf"]
        X_t = X.copy()
        if "scaler" in pipeline.named_steps:
            X_t = pd.DataFrame(
                pipeline.named_steps["scaler"].transform(X), columns=X.columns
            )
        sample = X_t.sample(min(max_samples, len(X_t)), random_state=SEED)
        if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, sample)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]
        return dict(zip(X.columns, np.abs(sv).mean(axis=0).tolist()))
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importance: dict, title: str, out_path: Path) -> None:
    feats = [FEATURE_DISPLAY_NAMES.get(k, k) for k in importance]
    vals  = list(importance.values())
    pairs = sorted(zip(vals, feats), reverse=True)
    vals_s, feats_s = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feats_s)))
    ax.barh(feats_s, vals_s, color=colours[::-1])
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: list[dict], out_path: Path) -> None:
    names    = [r["name"].replace("_", "\n") for r in results]
    auc_m    = [r["roc_auc_mean"] for r in results]
    auc_std  = [r["roc_auc_std"]  for r in results]
    pr_m     = [r["pr_auc_mean"]  for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, auc_m, 0.35, yerr=auc_std, label="ROC-AUC",
           color="#2196F3", alpha=0.85, capsize=5)
    ax.bar(x + 0.2, pr_m,  0.35, label="PR-AUC",
           color="#4CAF50", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Spatial CV (Africa)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 65)
    log.info("GeoExplorer AI — Phase 3: Model Training (Africa)")
    log.info("=" * 65)

    train_df = pd.read_csv(DATA_PROCESSED / "training_set.csv")
    grid_df  = pd.read_csv(DATA_PROCESSED / "prediction_grid.csv")

    available = [c for c in FEATURE_COLS if c in train_df.columns]
    missing   = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        log.warning("Missing features (skipped): %s", missing)
    FEATURES = available

    X = train_df[FEATURES]
    y = train_df["label"].values
    log.info("Features  : %d", len(FEATURES))
    log.info("Positives : %d  |  Negatives : %d", y.sum(), (y == 0).sum())

    splits = spatial_kfold_splits(train_df, n_splits=5)
    log.info("Spatial CV: %d folds", len(splits))

    models  = get_models()
    results = []
    for name, pipeline in models.items():
        res = evaluate_model(name, pipeline, X, y, splits)
        results.append(res)
        log.info("  %-22s  ROC-AUC=%.3f±%.3f  PR-AUC=%.3f±%.3f",
                 name, res["roc_auc_mean"], res["roc_auc_std"],
                 res["pr_auc_mean"],  res["pr_auc_std"])

    (MODELS_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2))
    plot_model_comparison(results, MODELS_DIR / "model_comparison.png")

    best      = max(results, key=lambda r: r["roc_auc_mean"])
    best_name = best["name"]
    log.info("Best model: %s  (ROC-AUC=%.3f)", best_name, best["roc_auc_mean"])

    best_pipeline = models[best_name]
    best_pipeline.fit(X, y)

    log.info("Calibrating probabilities …")
    calibrated = CalibratedClassifierCV(best_pipeline, method="isotonic", cv=3)
    calibrated.fit(X, y)

    importance      = compute_feature_importance(best_pipeline, X)
    shap_importance = run_shap(best_pipeline, X)

    (MODELS_DIR / "feature_importance.json").write_text(json.dumps({
        "model_importance":      importance,
        "shap_importance":       shap_importance,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
    }, indent=2))
    plot_feature_importance(
        importance,
        f"Feature Importance — {best_name.replace('_', ' ').title()} (Africa)",
        MODELS_DIR / "feature_importance.png",
    )

    log.info("Scoring prediction grid (%d points) …", len(grid_df))
    grid_features = [c for c in FEATURES if c in grid_df.columns]
    X_grid = grid_df[grid_features]
    try:
        proba = calibrated.predict_proba(X_grid)[:, 1]
    except Exception:
        proba = best_pipeline.predict_proba(X_grid)[:, 1]

    grid_df["prospectivity_score"] = proba
    grid_df["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    out_pred = DATA_PROCESSED / "predictions.csv"
    grid_df[[
        "lon", "lat", "prospectivity_score", "risk_tier",
        "elevation_m", "log_cu_ppm", "log_co_ppm", "log_ni_ppm",
        "dist_fault_km", "dist_deposit_km",
    ]].to_csv(out_pred, index=False)
    log.info("Saved predictions → %s", out_pred)

    model_bundle = {
        "pipeline":             calibrated,
        "feature_cols":         FEATURES,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "best_model_name":      best_name,
        "metrics":              best,
        "all_results":          results,
    }
    model_path = MODELS_DIR / "prospectivity_model.pkl"
    joblib.dump(model_bundle, model_path)
    log.info("Saved model bundle → %s  (%.1f KB)",
             model_path.name, model_path.stat().st_size / 1024)

    (MODELS_DIR / "model_metadata.json").write_text(json.dumps({
        "best_model":  best_name,
        "roc_auc":     best["roc_auc_mean"],
        "pr_auc":      best["pr_auc_mean"],
        "n_train":     len(train_df),
        "n_features":  len(FEATURES),
        "features":    FEATURES,
        "region":      "Central/Southern Africa Copperbelt",
        "commodity":   "Copper · Cobalt · Nickel",
        "grid_points": len(grid_df),
        "score_range": [float(proba.min()), float(proba.max())],
    }, indent=2))

    log.info("=" * 65)
    log.info("TRAINING COMPLETE")
    log.info("Best model : %s", best_name)
    log.info("ROC-AUC    : %.3f ± %.3f (spatial CV)", best["roc_auc_mean"], best["roc_auc_std"])
    log.info("PR-AUC     : %.3f ± %.3f", best["pr_auc_mean"], best["pr_auc_std"])
    log.info("Grid scored: %d points  |  score range: %.3f – %.3f",
             len(grid_df), proba.min(), proba.max())
    log.info("=" * 65)


if __name__ == "__main__":
    main()
