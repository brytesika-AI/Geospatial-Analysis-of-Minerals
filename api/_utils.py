"""Shared utilities for Vercel serverless API handlers."""
from __future__ import annotations
import csv, json, math
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS    = ROOT / "models"

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class BaseHandler(BaseHTTPRequestHandler):
    """Base class with CORS + JSON helpers."""

    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        params = self._params()
        try:
            data = self.handle_get(params)
            self._send(200, data)
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def handle_get(self, params: dict) -> dict | list:
        raise NotImplementedError

    def _params(self) -> dict:
        return parse_qs(urlparse(self.path).query, keep_blank_values=False)

    def _param(self, params: dict, key: str, default=None):
        vals = params.get(key)
        return vals[0] if vals else default

    def _send(self, status: int, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_geojson_points(path: Path) -> list[dict]:
    gj = read_json(path)
    rows = []
    for feat in gj.get("features", []):
        geom  = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        if geom.get("type") == "Point":
            c = geom.get("coordinates", [None, None])
            rows.append({"lon": c[0], "lat": c[1], **props})
    return rows


def infer_country(lat: float, lon: float) -> str:
    if -15 <= lat <= -8  and 22 <= lon <= 34: return "Zambia"
    if -13 <= lat <= -4  and 18 <= lon <= 31: return "DRC"
    if -27 <= lat <= -18 and 20 <= lon <= 30: return "Botswana"
    if -23 <= lat <= -15 and 26 <= lon <= 34: return "Zimbabwe"
    if -29 <= lat <= -17 and 11 <= lon <= 25: return "Namibia"
    if -35 <= lat <= -22 and 17 <= lon <= 33: return "South Africa"
    if -27 <= lat <= -10 and 32 <= lon <= 36: return "Mozambique"
    if -12 <= lat <= 0   and 29 <= lon <= 41: return "Tanzania"
    if -5  <= lat <= 5   and 33 <= lon <= 42: return "Kenya"
    if -18 <= lat <= -5  and 12 <= lon <= 25: return "Angola"
    if -3  <= lat <= 5   and 29 <= lon <= 36: return "Uganda"
    if -18 <= lat <= -9  and 33 <= lon <= 36: return "Malawi"
    if -3  <= lat <= 12  and -4 <= lon <= 2:  return "Ghana"
    if -4  <= lat <= 4   and 8  <= lon <= 16: return "Gabon"
    return "Africa"


def score_tier(s: float) -> str:
    if s >= 0.70: return "Very High"
    if s >= 0.50: return "High"
    if s >= 0.30: return "Moderate"
    return "Low"


def idw_score(lat: float, lon: float, rows: list[dict], k: int = 4) -> dict:
    dists = [
        (math.hypot(float(r["lon"]) - lon, float(r["lat"]) - lat), r)
        for r in rows
    ]
    dists.sort(key=lambda x: x[0])
    nearest = dists[:k]
    weights = [1.0 / max(d[0], 1e-6) ** 2 for d in nearest]
    total_w = sum(weights)
    score   = sum(w * float(r["prospectivity_score"]) for w, (_, r) in zip(weights, nearest)) / total_w
    top     = nearest[0][1]
    feats   = {k: float(v) for k, v in top.items()
               if k not in ("lat","lon","prospectivity_score","risk_tier") and v}
    return {
        "score":     round(score, 4),
        "tier":      score_tier(score),
        "country":   infer_country(lat, lon),
        "features":  feats,
        "nearest_km": round(nearest[0][0] * 111.0, 2),
    }
