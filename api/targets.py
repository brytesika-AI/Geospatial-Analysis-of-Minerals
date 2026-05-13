"""GET /api/targets?n=50&minScore=0.5&tier=&country= — enriched priority targets."""
import math
from _utils import BaseHandler, DATA_PROC, read_csv, infer_country, score_tier

_CACHE = None

def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = read_csv(DATA_PROC / "predictions.csv")
    return _CACHE


def _enrich(rows: list[dict], n: int, min_score: float, tiers: set, countries: set) -> list[dict]:
    filtered = []
    for r in rows:
        s = float(r.get("prospectivity_score", 0))
        t = r.get("risk_tier", score_tier(s))
        if s < min_score: continue
        if tiers and t not in tiers: continue
        lat, lon = float(r["lat"]), float(r["lon"])
        if countries:
            if infer_country(lat, lon) not in countries: continue
        filtered.append((s, r))

    filtered.sort(key=lambda x: x[0], reverse=True)
    top = filtered[:n]

    results = []
    for i, (s, r) in enumerate(top):
        lat  = float(r["lat"])
        lon  = float(r["lon"])
        t    = r.get("risk_tier", score_tier(s))
        cu   = float(r.get("log_cu_ppm", 0) or 0)
        co   = float(r.get("log_co_ppm", 0) or 0)
        ni   = float(r.get("log_ni_ppm", 0) or 0)
        fd   = float(r.get("dist_fault_km", 30) or 30)
        dd   = float(r.get("dist_deposit_km", 50) or 50)
        co_cu_pct = round((math.expm1(co) / max(math.expm1(cu), 1)) * 100, 1) if cu else 0
        ni_cu_pct = round((math.expm1(ni) / max(math.expm1(cu), 1)) * 100, 1) if cu else 0
        unc  = round(0.12 + 0.35 * min(dd, 80) / 80 + 0.18 * min(fd, 50) / 50, 3)

        if s >= 0.85:   program = "Diamond Core (DD) 300–500 m + IP/CSAMT"
        elif s >= 0.70: program = "Reverse Circulation (RC) 150–250 m + IP survey"
        elif s >= 0.50: program = "RC scout 50–150 m + soil geochemistry"
        else:           program = "Regional EM/magnetics + recon mapping"

        if s >= 0.85:   cost = "$600k–$1.5M"
        elif s >= 0.70: cost = "$250k–$600k"
        elif s >= 0.50: cost = "$80k–$200k"
        else:           cost = "$30k–$80k"

        results.append({
            "id":          f"AFR-{i+1:03d}",
            "lat":         round(lat, 4),
            "lon":         round(lon, 4),
            "score":       round(s, 3),
            "tier":        t,
            "country":     infer_country(lat, lon),
            "co_cu_pct":   co_cu_pct,
            "ni_cu_pct":   ni_cu_pct,
            "uncertainty": unc,
            "program":     program,
            "est_cost":    cost,
            "decision":    "Advance" if s >= 0.70 else "Hold",
        })
    return results


class handler(BaseHandler):
    def handle_get(self, params):
        rows      = _load()
        n         = int((params.get("n") or ["50"])[0])
        min_score = float((params.get("minScore") or ["0.3"])[0])
        tier_str  = (params.get("tier") or [""])[0]
        ctry_str  = (params.get("country") or [""])[0]
        tiers     = set(tier_str.split(",")) if tier_str else set()
        countries = set(ctry_str.split(",")) if ctry_str else set()
        data      = _enrich(rows, n, min_score, tiers, countries)
        return {"count": len(data), "data": data}
