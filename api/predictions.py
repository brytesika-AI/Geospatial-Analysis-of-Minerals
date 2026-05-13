"""GET /api/predictions — filtered heatmap grid (lat/lon/score/tier only)."""
from _utils import BaseHandler, DATA_PROC, read_csv, score_tier

_CACHE: list[dict] | None = None

def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = read_csv(DATA_PROC / "predictions.csv")
    return _CACHE


class handler(BaseHandler):
    def handle_get(self, params):
        rows     = _load()
        tiers    = set((params.get("tier") or ["Very High,High,Moderate,Low"])[0].split(","))
        min_s    = float((params.get("minScore") or ["0"])[0])
        max_s    = float((params.get("maxScore") or ["1"])[0])
        country  = (params.get("country") or [""])[0]
        countries = set(country.split(",")) if country else set()

        out = []
        for r in rows:
            s = float(r.get("prospectivity_score", 0))
            t = r.get("risk_tier", score_tier(s))
            if t not in tiers: continue
            if not (min_s <= s <= max_s): continue
            if countries:
                from _utils import infer_country
                if infer_country(float(r["lat"]), float(r["lon"])) not in countries:
                    continue
            out.append({
                "lat":   round(float(r["lat"]), 4),
                "lon":   round(float(r["lon"]), 4),
                "score": round(s, 3),
                "tier":  t,
            })

        return {"count": len(out), "data": out}
