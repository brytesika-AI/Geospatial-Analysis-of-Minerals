"""GET /api/metadata — model metadata + portfolio summary."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _utils import BaseHandler, MODELS, DATA_PROC, read_csv
import json

class handler(BaseHandler):
    def handle_get(self, params):
        meta = {}
        p = MODELS / "model_metadata.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                meta = json.load(f)
        fi = {}
        p2 = MODELS / "feature_importance.json"
        if p2.exists():
            with open(p2, encoding="utf-8") as f:
                fi = json.load(f)

        # Portfolio summary from predictions
        summary = {"very_high": 0, "high": 0, "moderate": 0, "low": 0}
        p3 = DATA_PROC / "predictions.csv"
        if p3.exists():
            for r in read_csv(p3):
                t = r.get("risk_tier","")
                if t == "Very High": summary["very_high"] += 1
                elif t == "High":    summary["high"] += 1
                elif t == "Moderate":summary["moderate"] += 1
                else:                summary["low"] += 1

        return {
            "model":            meta.get("best_model","xgboost"),
            "roc_auc":          meta.get("roc_auc", 0.889),
            "pr_auc":           meta.get("pr_auc", 0.785),
            "n_train":          meta.get("n_train", 2045),
            "n_features":       meta.get("n_features", 13),
            "grid_points":      meta.get("grid_points", 20125),
            "region":           meta.get("region","Central/Southern Africa Copperbelt"),
            "commodity":        meta.get("commodity","Copper · Cobalt · Nickel"),
            "score_range":      meta.get("score_range",[0.03,1.0]),
            "features":         meta.get("features",[]),
            "feature_importance": fi.get("model_importance",{}),
            "display_names":    fi.get("feature_display_names",{}),
            "portfolio":        summary,
        }
