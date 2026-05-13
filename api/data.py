"""GET /api/data?type=deposits|cities|oilgas|ports — context layer data."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import json
from _utils import BaseHandler, DATA_RAW, read_csv, read_geojson_points

OIL_GAS_FALLBACK = [
    {"name":"Cabinda Block 0","country":"Angola","lat":-5.50,"lon":12.20,"type":"Oil","status":"Producing"},
    {"name":"Rovuma LNG","country":"Mozambique","lat":-10.60,"lon":40.60,"type":"Gas","status":"Development"},
    {"name":"Jubilee","country":"Ghana","lat":4.65,"lon":-1.96,"type":"Oil","status":"Producing"},
    {"name":"Lokichar","country":"Kenya","lat":2.38,"lon":35.95,"type":"Oil","status":"Appraisal"},
    {"name":"Kingfisher/Jobi-Rii","country":"Uganda","lat":1.80,"lon":31.00,"type":"Oil","status":"Development"},
    {"name":"Niger Delta","country":"Nigeria","lat":5.50,"lon":6.00,"type":"Oil","status":"Producing"},
    {"name":"Orange Basin","country":"Namibia","lat":-28.00,"lon":15.50,"type":"Oil","status":"Exploration"},
    {"name":"Orange Basin SA","country":"South Africa","lat":-30.00,"lon":17.50,"type":"Oil","status":"Exploration"},
    {"name":"Ruvu Basin","country":"Tanzania","lat":-7.20,"lon":39.50,"type":"Gas","status":"Exploration"},
    {"name":"Kizomba Block 15","country":"Angola","lat":-9.50,"lon":12.80,"type":"Oil","status":"Producing"},
    {"name":"Pointe-Noire","country":"Congo","lat":-4.77,"lon":11.86,"type":"Oil","status":"Producing"},
    {"name":"Port-Gentil","country":"Gabon","lat":-1.00,"lon":8.80,"type":"Oil","status":"Producing"},
    {"name":"Walvis Basin","country":"Namibia","lat":-24.00,"lon":12.00,"type":"Gas","status":"Exploration"},
    {"name":"Kalahari Basin","country":"Botswana","lat":-21.00,"lon":21.50,"type":"Gas","status":"Exploration"},
    {"name":"Luangwa Basin","country":"Zambia","lat":-13.00,"lon":32.50,"type":"Gas","status":"Exploration"},
    {"name":"Kivu Methane","country":"Rwanda","lat":-2.00,"lon":29.30,"type":"Gas","status":"Producing"},
    {"name":"Mnazi Bay","country":"Tanzania","lat":-11.00,"lon":40.42,"type":"Gas","status":"Producing"},
    {"name":"Ogaden Basin","country":"Ethiopia","lat":7.00,"lon":45.00,"type":"Gas","status":"Exploration"},
    {"name":"Tema/Accra Offshore","country":"Ghana","lat":5.30,"lon":-0.50,"type":"Oil","status":"Exploration"},
    {"name":"Palmeira","country":"Mozambique","lat":-14.50,"lon":40.50,"type":"Gas","status":"Development"},
    {"name":"Puntland Offshore","country":"Somalia","lat":9.00,"lon":48.50,"type":"Oil","status":"Exploration"},
    {"name":"Cabora Bassa","country":"Zimbabwe","lat":-16.50,"lon":32.00,"type":"Gas","status":"Exploration"},
]


class handler(BaseHandler):
    def handle_get(self, params):
        dtype = (params.get("type") or ["deposits"])[0]

        if dtype == "deposits":
            path = DATA_RAW / "mrds_africa_copper.geojson"
            if path.exists():
                rows = read_geojson_points(path)
                return {"count": len(rows), "data": rows[:800]}
            return {"count": 0, "data": []}

        if dtype == "cities":
            path = DATA_RAW / "cities_africa.csv"
            if path.exists():
                rows = read_csv(path)
                out  = [{"lat": float(r["lat"]), "lon": float(r["lon"]),
                          "name": r.get("name",""), "country": r.get("country",""),
                          "population": int(r.get("population", 0) or 0)}
                        for r in rows if r.get("lat") and r.get("lon")]
                return {"count": len(out), "data": out}
            return {"count": 0, "data": []}

        if dtype == "oilgas":
            path = DATA_RAW / "oil_gas_africa.csv"
            if path.exists():
                rows = read_csv(path)
                out  = [{"lat": float(r["lat"]), "lon": float(r["lon"]),
                          "name": r.get("name",""), "country": r.get("country",""),
                          "type": r.get("type",""), "status": r.get("status","")}
                        for r in rows if r.get("lat") and r.get("lon")]
                return {"count": len(out), "data": out}
            return {"count": len(OIL_GAS_FALLBACK), "data": OIL_GAS_FALLBACK}

        if dtype == "ports":
            path = DATA_RAW / "ports_africa.csv"
            if path.exists():
                rows = read_csv(path)
                out  = [{"lat": float(r["lat"]), "lon": float(r["lon"]),
                          "name": r.get("name",""), "country": r.get("country","")}
                        for r in rows if r.get("lat") and r.get("lon")]
                return {"count": len(out), "data": out}
            return {"count": 0, "data": []}

        if dtype == "railways":
            path = DATA_RAW / "railways_africa.geojson"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    gj = json.load(f)
                lines = []
                for feat in gj.get("features", [])[:300]:
                    geom = feat.get("geometry") or {}
                    name = (feat.get("properties") or {}).get("name", "Railway") or "Railway"
                    if geom.get("type") == "LineString":
                        coords = [[c[1], c[0]] for c in geom.get("coordinates", [])]
                        if len(coords) >= 2:
                            lines.append({"coords": coords[::3], "name": name})
                return {"count": len(lines), "data": lines}
            return {"count": 0, "data": []}

        return {"error": f"Unknown type: {dtype}"}
