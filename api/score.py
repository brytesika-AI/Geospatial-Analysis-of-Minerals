"""GET /api/score?lat=&lon= — IDW prospectivity score for any point."""
from _utils import BaseHandler, DATA_PROC, read_csv, idw_score

_CACHE = None

def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = read_csv(DATA_PROC / "predictions.csv")
    return _CACHE


class handler(BaseHandler):
    def handle_get(self, params):
        lat = params.get("lat")
        lon = params.get("lon")
        if not lat or not lon:
            return {"error": "lat and lon are required"}
        lat, lon = float(lat[0]), float(lon[0])
        if not (-35 <= lat <= 18 and -18 <= lon <= 53):
            return {"error": "Coordinates outside Sub-Saharan Africa study area"}
        rows = _load()
        return {"lat": lat, "lon": lon, **idw_score(lat, lon, rows)}
