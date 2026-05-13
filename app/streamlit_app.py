"""
GeoExplorer AI — Sub-Saharan Africa Edition
Professional mineral prospectivity + resource intelligence platform.

Coverage   : Zambia · DRC · Botswana · Zimbabwe · Mozambique · Namibia ·
             South Africa · Tanzania · Kenya · Angola · Ghana · Gabon ·
             Uganda · Malawi · Ethiopia · and more
Commodities: Cu · Co · Ni · Li · Mn · REE · Graphite · V  + Oil & Gas
Deposit styles: SHSC (Lufilian Arc), Magmatic Cu-Ni (Bushveld),
                Carbonatite REE, SEDEX (Northern Cape), Pegmatite Li
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.geo_utils import (
    REGION_CENTER, humanise_features, infer_country,
    within_africa_study_area, within_model_bbox,
)
from app.llm_interpreter import GeoInterpreter
from app.mineral_systems import (
    FEATURE_SYSTEM_MAP, ProspectTarget, feature_component_summary,
)

st.set_page_config(
    page_title="GeoExplorer AI | Sub-Saharan Africa",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional dark-navy theme ──────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --bg:       #0c1420;
    --panel:    #162032;
    --panel2:   #1c2b40;
    --border:   #1e3a5f;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --gold:     #f59e0b;
    --gold-lt:  #fcd34d;
    --blue:     #60a5fa;
    --green:    #34d399;
    --teal:     #2dd4bf;
    --purple:   #a78bfa;
    --orange:   #fb923c;
    --red:      #f87171;
    --vh:       #f87171;
    --hi:       #f59e0b;
    --mod:      #60a5fa;
    --low:      #64748b;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg); color: var(--text);
    font-family: Inter, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stSidebar"] {
    background: #0a1624;
    border-right: 1px solid var(--border);
}
[data-testid="metric-container"] {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 18px;
}
[data-testid="stMetricValue"] {
    color: var(--gold); font-size: 1.75rem; font-weight: 800;
}
[data-testid="stMetricLabel"] {
    color: var(--muted); font-size: 0.72rem; letter-spacing: 0.07em;
    text-transform: uppercase; font-weight: 600;
}
[data-testid="stMetricDelta"] { color: var(--green); }
h1,h2,h3 { color: var(--text) !important; font-weight: 800 !important; }
h1 { font-size: 2.6rem !important; line-height: 1.05 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.15rem !important; }
a { color: var(--gold) !important; }
[data-testid="stTabs"] button {
    color: var(--muted) !important; font-weight: 600 !important;
    font-size: 0.82rem !important; letter-spacing: 0.04em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}
.stButton > button {
    background: var(--gold); color: #0c1420;
    border: none; border-radius: 6px;
    font-weight: 700; font-size: 0.88rem; letter-spacing: 0.03em;
    padding: 9px 20px;
}
.stButton > button:hover { background: var(--gold-lt); }
[data-testid="stDownloadButton"] > button {
    background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    font-weight: 600; font-size: 0.84rem;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--gold); color: var(--gold); }
.hero { border-bottom: 1px solid var(--border); padding: 4px 0 22px 0; margin-bottom: 20px; }
.eyebrow { color: var(--gold); font-size: 0.72rem; font-weight: 700;
           letter-spacing: 0.16em; text-transform: uppercase; }
.lede { max-width: 980px; color: var(--muted); font-size: 1.0rem; line-height: 1.6; margin-top:6px; }
.info-box { background: var(--panel); border-left: 3px solid var(--blue);
            border-radius: 0 6px 6px 0; padding: 12px 16px;
            color: var(--muted); margin: 8px 0 16px 0; font-size:0.9rem; }
.warn-box  { background: var(--panel); border-left: 3px solid var(--gold);
             border-radius: 0 6px 6px 0; padding: 12px 16px;
             color: var(--muted); margin: 8px 0 16px 0; font-size:0.9rem; }
.ok-box    { background: var(--panel); border-left: 3px solid var(--green);
             border-radius: 0 6px 6px 0; padding: 12px 16px;
             color: var(--muted); margin: 8px 0 16px 0; font-size:0.9rem; }
.score-badge { display:inline-block; padding:8px 16px; border-radius:6px;
               font-weight:800; font-size:1.1rem; color:#0c1420; background:var(--gold); }
.tier-VH { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--vh); color:#fff; }
.tier-H  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--hi); color:#0c1420; }
.tier-M  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--mod); color:#0c1420; }
.tier-L  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--panel2); color:var(--muted);
           border:1px solid var(--border); }
.action-card { background:var(--panel); border:1px solid var(--border);
               border-radius:8px; padding:16px 18px; margin-bottom:8px; }
.action-card h4 { color:var(--gold); font-size:0.8rem; font-weight:700;
                  letter-spacing:0.08em; text-transform:uppercase; margin:0 0 6px 0; }
.action-card p  { color:var(--text); font-size:0.88rem; margin:0; line-height:1.5; }
.layer-chip { display:inline-block; padding:3px 9px; border-radius:4px; font-size:0.75rem;
              font-weight:600; margin: 2px 3px; }
.caption { color:var(--muted); font-size:0.78rem; line-height:1.45; }
hr { border-color: var(--border) !important; }
.sidebar-brand { padding: 6px 0 18px 0; }
.sidebar-brand .title { font-size:1.25rem; font-weight:800; line-height:1.15; }
.sidebar-brand .sub   { font-size:0.78rem; color:var(--muted); margin-top:5px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#162032",
    plot_bgcolor="#162032",
    font=dict(color="#e2e8f0", family="Inter"),
    margin=dict(l=0, r=0, t=44, b=0),
)

TIER_COLORS = {"Very High": "#f87171", "High": "#f59e0b", "Moderate": "#60a5fa", "Low": "#64748b"}

COMMODITY_COLORS = {
    "Cu":        "#34d399",
    "Cu-Co":     "#60a5fa",
    "Cu-Au":     "#fcd34d",
    "Cu-Ni":     "#f87171",
    "Co":        "#818cf8",
    "Ni":        "#f59e0b",
    "Li":        "#2dd4bf",
    "Mn":        "#c084fc",
    "REE":       "#fb923c",
    "Graphite":  "#94a3b8",
    "V":         "#e879f9",
    "PGM":       "#e2e8f0",
    "Oil & Gas": "#fbbf24",
    "default":   "#475569",
}

ALL_COUNTRIES = [
    "Angola", "Botswana", "DRC", "Ethiopia", "Gabon", "Ghana",
    "Kenya", "Malawi", "Mozambique", "Namibia", "Rwanda", "South Africa",
    "Tanzania", "Uganda", "Zambia", "Zimbabwe",
]

RESOURCE_CATEGORIES = [
    "Cu · Co · Ni (Copperbelt)",
    "Green Metals (Li · Mn · REE · Graphite · V)",
    "Oil & Gas",
    "Infrastructure (Cities · Railways · Ports)",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_deposits() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError:
        return pd.DataFrame(columns=["lon", "lat", "name", "commod1"])
    path = ROOT / "data" / "raw" / "mrds_africa_copper.geojson"
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "name", "commod1"])
    try:
        gdf = gpd.read_file(path)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        cols = ["lon", "lat"] + [c for c in gdf.columns if c not in ("lon","lat","geometry")]
        return pd.DataFrame(gdf[cols])
    except Exception:
        return pd.DataFrame(columns=["lon", "lat", "name", "commod1"])


@st.cache_data(show_spinner=False)
def load_cities() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "cities_africa.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lon","lat","name","population","country"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["lon","lat","name","population","country"])


@st.cache_data(show_spinner=False)
def load_oil_gas() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "oil_gas_africa.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lon","lat","name","type","country","status"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["lon","lat","name","type","country","status"])


@st.cache_data(show_spinner=False)
def load_railways() -> pd.DataFrame:
    """Returns list of [lat,lon] polylines for railway rendering."""
    path = ROOT / "data" / "raw" / "railways_africa.geojson"
    if not path.exists():
        return pd.DataFrame(columns=["coords","name"])
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        rows = []
        for _, row in gdf.iterrows():
            if row.geometry is None:
                continue
            if row.geometry.geom_type == "LineString":
                coords = [[y, x] for x, y in row.geometry.coords]
                rows.append({"coords": coords, "name": row.get("name","Railway")})
            elif row.geometry.geom_type == "MultiLineString":
                for line in row.geometry.geoms:
                    coords = [[y, x] for x, y in line.coords]
                    rows.append({"coords": coords, "name": row.get("name","Railway")})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["coords","name"])
    except Exception:
        return pd.DataFrame(columns=["coords","name"])


@st.cache_data(show_spinner=False)
def load_ports() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "ports_africa.csv"
    if not path.exists():
        return pd.DataFrame(columns=["lon","lat","name","country"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["lon","lat","name","country"])


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = ROOT / "models" / "prospectivity_model.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    path = ROOT / "models" / "model_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    path = ROOT / "models" / "feature_importance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def tier_chip(tier: str) -> str:
    cls = {"Very High": "tier-VH", "High": "tier-H", "Moderate": "tier-M", "Low": "tier-L"}
    return f'<span class="{cls.get(tier,"tier-L")}">{tier}</span>'


def score_to_tier(s: float) -> str:
    if s >= 0.70: return "Very High"
    if s >= 0.50: return "High"
    if s >= 0.30: return "Moderate"
    return "Low"


def to_dms(dd: float, is_lat: bool) -> str:
    hemi = ("S" if dd < 0 else "N") if is_lat else ("W" if dd < 0 else "E")
    dd = abs(dd)
    d, m = int(dd), int((dd % 1) * 60)
    s = round(((dd % 1) * 60 % 1) * 60, 1)
    return f"{d}°{m:02d}'{s:04.1f}\" {hemi}"


def _commodity_color(c: str) -> str:
    c = str(c)
    for key, col in COMMODITY_COLORS.items():
        if key in c:
            return col
    return COMMODITY_COLORS["default"]


def _classify_commodity(c: str) -> str:
    c = str(c).upper()
    if any(x in c for x in ("LI","LITHIUM")):  return "Green Metals"
    if any(x in c for x in ("MN","MANGANESE")): return "Green Metals"
    if any(x in c for x in ("REE","RARE EARTH","CE","LA","ND")): return "Green Metals"
    if "GRAPHIT" in c: return "Green Metals"
    if any(x in c for x in ("VANADIUM"," V,")): return "Green Metals"
    if any(x in c for x in ("NI","NICKEL")):  return "Cu-Co-Ni"
    if any(x in c for x in ("CO","COBALT")):  return "Cu-Co-Ni"
    if any(x in c for x in ("CU","COPPER")):  return "Cu-Co-Ni"
    return "Other"


def _build_targets(predictions: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    fc = [c for c in ["log_cu_ppm","log_co_ppm","log_ni_ppm",
                       "dist_fault_km","dist_deposit_km","elevation_m"]
          if c in predictions.columns]
    top = predictions.nlargest(n, "prospectivity_score")[
        ["lat","lon","prospectivity_score","risk_tier"] + fc
    ].copy().reset_index(drop=True)

    top["target_id"] = [f"AFR-{i+1:03d}" for i in range(len(top))]
    top["country"]   = top.apply(lambda r: infer_country(r.lat, r.lon), axis=1)

    top["uncertainty"] = (
        0.12
        + 0.35 * top.get("dist_deposit_km", pd.Series([50]*len(top))).clip(0,80) / 80
        + 0.18 * top.get("dist_fault_km",   pd.Series([30]*len(top))).clip(0,50) / 50
    ).round(3)

    if "log_co_ppm" in top.columns and "log_cu_ppm" in top.columns:
        top["co_cu_pct"] = (
            np.expm1(top["log_co_ppm"]) /
            np.expm1(top["log_cu_ppm"]).clip(lower=1) * 100
        ).round(1)

    if "log_ni_ppm" in top.columns and "log_cu_ppm" in top.columns:
        top["ni_cu_pct"] = (
            np.expm1(top["log_ni_ppm"]) /
            np.expm1(top["log_cu_ppm"]).clip(lower=1) * 100
        ).round(1)

    def _program(row):
        s = row.prospectivity_score
        if s >= 0.85: return "Diamond core drilling (DD) 300–500 m + geophysics"
        if s >= 0.70: return "RC drilling 150–250 m + IP/CSAMT survey"
        if s >= 0.50: return "Infill soil geochemistry + IP ground survey"
        return "Regional airborne EM/magnetics, recon mapping"

    def _cost(row):
        s = row.prospectivity_score
        if s >= 0.85: return "$600k–$1.5M"
        if s >= 0.70: return "$250k–$600k"
        if s >= 0.50: return "$80k–$200k"
        return "$30k–$80k"

    top["program"]  = top.apply(_program, axis=1)
    top["est_cost"] = top.apply(_cost, axis=1)
    top["decision"] = np.where(top["prospectivity_score"] >= 0.70, "Advance", "Hold")
    return top


def _apply_filters(df: pd.DataFrame, countries: list, tiers: list,
                   score_min: float, score_max: float) -> pd.DataFrame:
    if "country" not in df.columns:
        df = df.copy()
        df["country"] = df.apply(lambda r: infer_country(r.lat, r.lon), axis=1)
    mask = (
        df["country"].isin(countries) &
        df["risk_tier"].isin(tiers) &
        (df["prospectivity_score"] >= score_min) &
        (df["prospectivity_score"] <= score_max)
    )
    return df[mask].copy()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(meta: dict, model_ok: bool):
    uploaded_df = None
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="eyebrow">GeoExplorer AI · Africa</div>'
            '<div class="title">Sub-Saharan Resource Intelligence</div>'
            '<div class="sub">Cu · Co · Ni · Li · Mn · REE · Graphite<br>'
            'Oil & Gas · Infrastructure</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Global filters ──
        st.markdown("**Global Filters**")

        sel_countries = st.multiselect(
            "Country / Territory",
            ALL_COUNTRIES,
            default=ALL_COUNTRIES,
            key="f_countries",
        )

        sel_tiers = st.multiselect(
            "Prospectivity tier",
            ["Very High", "High", "Moderate", "Low"],
            default=["Very High", "High", "Moderate"],
            key="f_tiers",
        )

        sel_score = st.slider(
            "Score range", 0.0, 1.0, (0.30, 1.0), 0.05,
            key="f_score",
        )

        sel_resources = st.multiselect(
            "Resource categories (map layers)",
            RESOURCE_CATEGORIES,
            default=RESOURCE_CATEGORIES,
            key="f_resources",
        )

        st.divider()

        # ── Point query ──
        st.markdown("**Query Location**")
        lat = st.number_input("Latitude (°S negative, °N positive)", value=-12.5,
                              min_value=-35.0, max_value=18.0, format="%.4f")
        lon = st.number_input("Longitude (°E positive, °W negative)", value=28.2,
                              min_value=-18.0, max_value=53.0, format="%.4f")
        score_btn = st.button("Score location", use_container_width=True)

        st.divider()
        st.markdown("**Batch Samples**")
        uploaded = st.file_uploader(
            "Upload feature CSV", type=["csv"],
            help="Columns: elevation_m, slope_deg, log_cu_ppm, log_co_ppm, "
                 "log_ni_ppm, log_au_ppb, fe_pct, log_pb_ppm, log_zn_ppm, "
                 "log_mo_ppm, log_as_ppm, dist_fault_km, dist_deposit_km",
        )
        if uploaded:
            uploaded_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(uploaded_df):,} rows")
            if not model_ok:
                st.caption("Model bundle not found — preview only.")

        st.divider()
        if meta:
            st.markdown("**Model snapshot**")
            st.markdown(
                f'<div class="info-box">'
                f'<strong>{meta.get("best_model","—").replace("_"," ").title()}</strong><br>'
                f'ROC-AUC <strong>{meta.get("roc_auc",0):.3f}</strong> · spatial CV<br>'
                f'{meta.get("n_train","?"):,} training samples<br>'
                f'{meta.get("grid_points","?"):,} scored grid cells<br>'
                f'<span style="color:#f59e0b">{meta.get("commodity","Cu · Co · Ni")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="caption">USGS MRDS · GEM Active Faults · '
            'Natural Earth · OSM mines · curated oil & gas.<br>'
            'Not a resource estimate or regulatory disclosure.</div>',
            unsafe_allow_html=True,
        )

    return (
        float(lat), float(lon), score_btn, uploaded_df,
        sel_countries, sel_tiers, sel_score, sel_resources,
    )


# ── Exploration Map ───────────────────────────────────────────────────────────

def render_map(
    predictions, deposits, cities, oil_gas_df, railways, ports,
    sel_resources, sel_tiers, sel_score,
    q_lat=None, q_lon=None, q_score=None,
):
    m = folium.Map(
        location=[REGION_CENTER["lat"], REGION_CENTER["lon"]],
        zoom_start=4,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    show_cuconi   = "Cu · Co · Ni (Copperbelt)"       in sel_resources
    show_green    = "Green Metals (Li · Mn · REE · Graphite · V)" in sel_resources
    show_oilgas   = "Oil & Gas"                        in sel_resources
    show_infra    = "Infrastructure (Cities · Railways · Ports)"  in sel_resources

    # ── Prospectivity heatmap ──
    if show_cuconi and not predictions.empty:
        from folium.plugins import HeatMap
        filtered = predictions[
            predictions["risk_tier"].isin(sel_tiers) &
            (predictions["prospectivity_score"] >= sel_score[0]) &
            (predictions["prospectivity_score"] <= sel_score[1])
        ]
        heat_data = [
            [r.lat, r.lon, r.prospectivity_score]
            for _, r in filtered.iterrows()
            if r.prospectivity_score > 0.2
        ]
        if heat_data:
            HeatMap(
                heat_data, name="Cu·Co·Ni Prospectivity",
                min_opacity=0.2, max_val=0.9,
                radius=22, blur=28,
                gradient={
                    0.2: "#1e3a5f", 0.4: "#1d4ed8", 0.55: "#34d399",
                    0.68: "#f59e0b", 0.82: "#f87171", 0.95: "#fef3c7",
                },
            ).add_to(m)

    # ── Known Cu/Co/Ni mineral deposits ──
    if show_cuconi and not deposits.empty:
        dep_layer = folium.FeatureGroup(name="Cu·Co·Ni Deposits", show=True)
        cu_deposits = deposits[
            deposits.get("commod1", pd.Series(dtype=str)).apply(
                lambda c: _classify_commodity(str(c)) == "Cu-Co-Ni"
            )
        ] if "commod1" in deposits.columns else deposits
        for _, d in cu_deposits.head(600).iterrows():
            c = str(d.get("commod1", "Cu"))
            colour = _commodity_color(c)
            folium.CircleMarker(
                location=[d.lat, d.lon], radius=4,
                color=colour, fill=True, fill_color=colour,
                fill_opacity=0.85, weight=1,
                tooltip=f"<b>{d.get('district', d.get('name','deposit'))}</b><br>{c}",
            ).add_to(dep_layer)
        dep_layer.add_to(m)

    # ── Green metals deposits ──
    if show_green and not deposits.empty and "commod1" in deposits.columns:
        green_layer = folium.FeatureGroup(name="Green Metals (Li·Mn·REE·Graphite)", show=True)
        green_deps = deposits[
            deposits["commod1"].apply(lambda c: _classify_commodity(str(c)) == "Green Metals")
        ]
        for _, d in green_deps.head(400).iterrows():
            c = str(d.get("commod1", "Li"))
            colour = _commodity_color(c)
            icon = folium.DivIcon(
                html=f'<div style="background:{colour};width:10px;height:10px;'
                     f'border-radius:50%;border:1.5px solid #fff;opacity:0.9"></div>',
                icon_size=(10, 10), icon_anchor=(5, 5),
            )
            folium.Marker(
                location=[d.lat, d.lon], icon=icon,
                tooltip=f"<b>{d.get('district', d.get('name','Green metal'))}</b><br>{c}",
            ).add_to(green_layer)
        green_layer.add_to(m)

    # ── Oil & Gas fields ──
    if show_oilgas and not oil_gas_df.empty:
        og_layer = folium.FeatureGroup(name="Oil & Gas Fields", show=True)
        for _, f in oil_gas_df.iterrows():
            status = str(f.get("status","")).lower()
            colour = "#fbbf24" if "produc" in status else "#f97316"
            folium.RegularPolygonMarker(
                location=[f.lat, f.lon],
                number_of_sides=3, radius=9,
                color=colour, fill=True, fill_color=colour, fill_opacity=0.85,
                tooltip=(
                    f"<b>{f.get('name','O&G field')}</b><br>"
                    f"{f.get('type','')} · {f.get('country','')}<br>"
                    f"Status: {f.get('status','unknown')}"
                ),
            ).add_to(og_layer)
        og_layer.add_to(m)

    # ── Urban centres ──
    if show_infra and not cities.empty:
        city_layer = folium.FeatureGroup(name="Urban Centres", show=True)
        for _, c in cities.iterrows():
            pop = c.get("population", 0)
            r   = 4 if pop < 500_000 else 7 if pop < 2_000_000 else 10
            folium.CircleMarker(
                location=[c.lat, c.lon], radius=r,
                color="#e2e8f0", fill=True, fill_color="#e2e8f0",
                fill_opacity=0.6, weight=1,
                tooltip=f"<b>{c.get('name','City')}</b><br>{c.get('country','')}",
            ).add_to(city_layer)
        city_layer.add_to(m)

    # ── Railways ──
    if show_infra and not railways.empty:
        rail_layer = folium.FeatureGroup(name="Railways", show=True)
        for _, r in railways.iterrows():
            if r.coords:
                folium.PolyLine(
                    locations=r.coords, color="#475569", weight=1.5,
                    opacity=0.7, tooltip=r.get("name","Railway"),
                ).add_to(rail_layer)
        rail_layer.add_to(m)

    # ── Seaports ──
    if show_infra and not ports.empty:
        port_layer = folium.FeatureGroup(name="Seaports", show=True)
        for _, p in ports.iterrows():
            folium.Marker(
                location=[p.lat, p.lon],
                icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
                tooltip=f"<b>{p.get('name','Port')}</b><br>{p.get('country','')}",
            ).add_to(port_layer)
        port_layer.add_to(m)

    # ── Query point ──
    if q_lat is not None and q_lon is not None:
        folium.Marker(
            location=[q_lat, q_lon],
            popup=f"<b>Score: {q_score:.2f}</b>" if q_score else "Query site",
            tooltip="Query site",
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = (
        '<div style="background:#162032;border:1px solid #1e3a5f;border-radius:6px;'
        'padding:10px 14px;font-size:0.78rem;color:#94a3b8;line-height:1.8;">'
        '<b style="color:#e2e8f0">Layer Legend</b><br>'
        '<span style="color:#f87171">■</span> Very High prospectivity &nbsp;'
        '<span style="color:#f59e0b">■</span> High &nbsp;'
        '<span style="color:#34d399">■</span> Cu deposit &nbsp;'
        '<span style="color:#60a5fa">■</span> Cu-Co &nbsp;'
        '<span style="color:#2dd4bf">■</span> Li/REE &nbsp;'
        '<span style="color:#fbbf24">▲</span> Oil field &nbsp;'
        '<span style="color:#e2e8f0">●</span> City &nbsp;'
        '<span style="color:#60a5fa">⚓</span> Port'
        '</div>'
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:1000">{legend_html}</div>'
    ))

    st_folium(m, width="100%", height=620, returned_objects=[])


# ── Executive Summary ─────────────────────────────────────────────────────────

def render_executive_summary(
    predictions: pd.DataFrame, meta: dict,
    sel_countries, sel_tiers, sel_score,
) -> None:
    if predictions.empty:
        st.warning("Run the data pipeline to generate the prediction grid.")
        return

    df = predictions.copy()
    df["country"] = df.apply(lambda r: infer_country(r.lat, r.lon), axis=1)
    filtered = _apply_filters(df, sel_countries, sel_tiers, sel_score[0], sel_score[1])

    if filtered.empty:
        st.warning("No predictions match the current filters.")
        return

    targets = _build_targets(filtered, min(50, len(filtered)))

    n_vh  = int((filtered["prospectivity_score"] >= 0.70).sum())
    n_hi  = int(((filtered["prospectivity_score"] >= 0.50) & (filtered["prospectivity_score"] < 0.70)).sum())
    n_adv = int((targets["decision"] == "Advance").sum())
    auc   = meta.get("roc_auc", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filtered cells",      f"{len(filtered):,}", help="After applying sidebar filters")
    c2.metric("Very High targets",   f"{n_vh:,}", help="Score ≥ 0.70")
    c3.metric("High targets",        f"{n_hi:,}", help="Score 0.50–0.70")
    c4.metric("Advance recommended", f"{n_adv}",  help="Top-50 with score ≥ 0.70")
    c5.metric("Model ROC-AUC",       f"{auc:.3f}", help="Spatial cross-validation")

    st.divider()
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("#### Portfolio by Tier")
        tier_counts = filtered["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Tier"] = pd.Categorical(
            tier_counts["Tier"],
            categories=["Very High","High","Moderate","Low"], ordered=True
        )
        tier_counts = tier_counts.sort_values("Tier")
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            hole=0.55,
            marker_colors=[TIER_COLORS.get(t,"#64748b") for t in tier_counts["Tier"]],
            textinfo="label+percent", textfont_size=11,
        ))
        fig_donut.update_layout(
            **PLOTLY_THEME, height=280, showlegend=False,
            annotations=[dict(text="Targets", x=0.5, y=0.5, font_size=13,
                              showarrow=False, font_color="#e2e8f0")],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("#### Coverage by Country")
        ctry = targets["country"].value_counts().reset_index()
        ctry.columns = ["Country", "Targets"]
        fig_bar = px.bar(
            ctry, x="Targets", y="Country", orientation="h",
            color="Targets", color_continuous_scale=["#1e3a5f","#f59e0b"],
            template="plotly_dark",
        )
        fig_bar.update_layout(**PLOTLY_THEME, height=250,
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("#### Target Prioritisation Matrix")
        st.caption("Prospectivity vs. geological uncertainty — size = score")
        fig_mat = px.scatter(
            targets, x="uncertainty", y="prospectivity_score",
            size="prospectivity_score", color="decision",
            hover_name="target_id",
            hover_data={"country": True, "est_cost": True, "program": True,
                        "lat": ":.3f", "lon": ":.3f"},
            color_discrete_map={"Advance": "#f59e0b", "Hold": "#60a5fa"},
            template="plotly_dark",
        )
        fig_mat.add_hrect(y0=0.70, y1=1.0, fillcolor="#f87171", opacity=0.07,
                          line_width=0, annotation_text="Very High",
                          annotation_position="right")
        fig_mat.add_hrect(y0=0.50, y1=0.70, fillcolor="#f59e0b", opacity=0.07,
                          line_width=0, annotation_text="High",
                          annotation_position="right")
        fig_mat.add_vline(x=0.40, line_dash="dash", line_color="#475569",
                          annotation_text="High uncertainty →",
                          annotation_position="top right")
        fig_mat.update_layout(
            **PLOTLY_THEME, height=460,
            xaxis_title="Geological uncertainty proxy",
            yaxis_title="Prospectivity score",
            legend=dict(orientation="h", y=-0.12),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

    st.divider()
    st.markdown("#### Recommended Action Plan")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        adv    = targets[targets["decision"] == "Advance"]
        ctries = ", ".join(adv["country"].unique()[:3])
        st.markdown(
            f'<div class="action-card"><h4>⚡ Immediate (0–3 months)</h4>'
            f'<p>Advance <strong>{len(adv)} priority targets</strong> in {ctries}.<br>'
            f'IP geophysics + RC drilling scoping. Budget: <strong>$1.5M–$4M</strong>.</p></div>',
            unsafe_allow_html=True)
    with a2:
        st.markdown(
            '<div class="action-card"><h4>📋 Short-term (3–6 months)</h4>'
            '<p>Infill soil geochemistry on High-tier targets. Geological mapping '
            'for SHSC stratigraphic confirmation. Budget: <strong>$400k–$800k</strong>.</p></div>',
            unsafe_allow_html=True)
    with a3:
        st.markdown(
            '<div class="action-card"><h4>🔬 Medium-term (6–18 months)</h4>'
            '<p>Diamond core drilling on best RC intercepts. Resource estimation '
            'scoping. Engage permitting in priority jurisdictions. '
            'Budget: <strong>$2M–$8M</strong>.</p></div>',
            unsafe_allow_html=True)
    with a4:
        st.markdown(
            '<div class="action-card"><h4>🌍 Green Metals Track</h4>'
            '<p>Evaluate Li (DRC Manono, Zambia Bikita), Mn (Kalahari), '
            'REE (Tanzania Ngualla, Kenya Mrima). Rapidly growing EV demand '
            'justifies parallel track.</p></div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Top 10 Priority Targets")
    top10 = targets.head(10)[
        [c for c in ["target_id","country","lat","lon","prospectivity_score",
                     "risk_tier","co_cu_pct","est_cost","decision","program"]
         if c in targets.columns]
    ].rename(columns={
        "target_id":"ID","country":"Country",
        "prospectivity_score":"Score","risk_tier":"Tier",
        "co_cu_pct":"Co/Cu %","est_cost":"Est. Cost",
        "decision":"Decision","program":"Recommended Program",
    })
    st.dataframe(top10, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇  Download top-10 summary (CSV)",
        top10.to_csv(index=False).encode(),
        "geoexplorer_priority_targets.csv", "text/csv",
    )


# ── Priority Targets ──────────────────────────────────────────────────────────

def render_priority_targets(
    predictions: pd.DataFrame,
    sel_countries, sel_tiers, sel_score,
) -> None:
    if predictions.empty:
        st.warning("Prediction grid unavailable.")
        return

    df = predictions.copy()
    df["country"] = df.apply(lambda r: infer_country(r.lat, r.lon), axis=1)
    filtered = _apply_filters(df, sel_countries, sel_tiers, sel_score[0], sel_score[1])

    n_show  = st.slider("Number of targets to display", 10, 200, 50, step=10)
    targets = _build_targets(filtered, min(n_show, len(filtered)))

    # Additional country filter inside tab
    avail_countries = sorted(targets["country"].unique().tolist())
    tab_countries   = st.multiselect(
        "Narrow by country (tab-level)", avail_countries, default=avail_countries,
        key="pt_countries",
    )
    targets = targets[targets["country"].isin(tab_countries)]

    # Deposit style diagnostic banner
    if "co_cu_pct" in targets.columns:
        katanga_n = int((targets["co_cu_pct"] >= 10).sum())
        bushveld_n = (
            int((targets.get("ni_cu_pct", pd.Series([])) >= 20).sum())
            if "ni_cu_pct" in targets.columns else 0
        )
        cols_info = st.columns(3)
        cols_info[0].metric("DRC Katanga-type (Co/Cu≥10%)", katanga_n)
        cols_info[1].metric("Bushveld-type (Ni/Cu≥20%)", bushveld_n)
        cols_info[2].metric("Advance decisions", int((targets["decision"]=="Advance").sum()))

    st.markdown(
        '<div class="warn-box">'
        '<strong>Co/Cu ≥ 10%</strong> → DRC Katanga SHSC (Cailteux et al. 2005). '
        '<strong>Ni/Cu ≥ 20%</strong> → Bushveld magmatic sulphide. '
        'Uncertainty proxy guides recommended field programme.</div>',
        unsafe_allow_html=True,
    )

    display_cols = [c for c in [
        "target_id","country","lat","lon","prospectivity_score","risk_tier",
        "co_cu_pct","ni_cu_pct","uncertainty","est_cost","decision","program"
    ] if c in targets.columns]

    st.dataframe(
        targets[display_cols].rename(columns={
            "target_id":"ID","country":"Country",
            "prospectivity_score":"Score","risk_tier":"Tier",
            "co_cu_pct":"Co/Cu %","ni_cu_pct":"Ni/Cu %",
            "uncertainty":"Uncertainty","est_cost":"Est. Cost",
            "decision":"Decision","program":"Recommended Program",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇  Export targets (CSV)",
        targets[display_cols].to_csv(index=False).encode(),
        "geoexplorer_targets.csv", "text/csv",
    )


# ── Resource Intelligence (Green Metals + Oil & Gas) ─────────────────────────

def render_resource_intelligence(deposits: pd.DataFrame, oil_gas_df: pd.DataFrame) -> None:
    st.markdown("### Sub-Saharan Africa Resource Intelligence")
    st.markdown(
        '<div class="info-box">'
        'Extended resource coverage beyond Cu/Co/Ni: critical minerals for the energy '
        'transition (Li, Mn, REE, Graphite, Vanadium) and conventional hydrocarbon basins.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_a, tab_b = st.tabs(["🔋 Green & Critical Metals", "🛢 Oil & Gas Basins"])

    with tab_a:
        st.markdown("#### Critical & Green Metal Deposits")
        green_entries = [
            ("DRC", "Manono",     "Li",       -7.30,  27.43, "Pegmatite", "Pre-development", "Largest hard-rock Li deposit in Africa; 400 Mt at 1.65% Li₂O"),
            ("Zimbabwe", "Bikita", "Li",       -20.12, 31.68, "Pegmatite", "Operating",        "Historic Li producer; spodumene + lepidolite"),
            ("Namibia", "Karibib","Li",        -21.96, 15.85, "Pegmatite", "Exploration",       "Spodumene-bearing pegmatites"),
            ("DRC", "Kisenge",    "Mn",        -9.12,  26.75, "Sedimentary","Exploration",       "High-grade Mn in Katanga sediments"),
            ("South Africa","Kalahari Mn","Mn",-27.1,  22.6,  "Sedimentary","Operating",        "World's largest Mn ore body; ~80% of global reserves"),
            ("Gabon","Moanda",    "Mn",        -1.56,  13.24, "Sedimentary","Operating",        "High-grade Mn; Compagnie Minière de l'Ogooué"),
            ("Tanzania","Ngualla","REE",       -8.71,  32.42, "Carbonatite","Pre-feasibility",  "Nd-Pr dominant; Peak Rare Earths"),
            ("Kenya","Mrima Hill","REE",       -4.25,  39.25, "Carbonatite","Exploration",       "Elevated Nb, REE in coastal carbonatite"),
            ("Malawi","Songwe Hill","REE",     -9.85,  33.70, "Carbonatite","Pre-development",  "Mkango Resources; Nd-Pr focus"),
            ("Ghana","Ewoyaa",    "Li",        5.20,   -0.95, "Pegmatite", "Permitting",        "Atlantic Lithium; spodumene to battery grade"),
            ("Tanzania","Mahenge","Graphite",  -8.73,  36.71, "Metamorphic","Development",       "Large flake graphite; NEXT Graphite"),
            ("Tanzania","Kabanga","Ni",        -2.72,  30.52, "Magmatic",  "Pre-feasibility",   "High-grade Ni sulphide; Kabanga Nickel"),
            ("Zambia","Munali",   "Ni",        -15.88, 28.98, "Magmatic",  "Care & maintenance","Magmatic Ni-Cu in Lusaka area"),
        ]
        df_green = pd.DataFrame(green_entries, columns=[
            "Country","District","Commodity","Lat","Lon","Style","Status","Notes"
        ])
        st.dataframe(df_green, use_container_width=True, hide_index=True)

        if not deposits.empty and "commod1" in deposits.columns:
            green_mrds = deposits[
                deposits["commod1"].apply(lambda c: _classify_commodity(str(c)) == "Green Metals")
            ]
            if not green_mrds.empty:
                st.markdown(f"**MRDS database:** {len(green_mrds)} additional green metal records found")
                st.dataframe(
                    green_mrds[["lon","lat"] + [c for c in green_mrds.columns
                                                if c not in ("lon","lat","geometry")]].head(50),
                    use_container_width=True, hide_index=True,
                )

        st.download_button(
            "⬇  Export green metals table (CSV)",
            df_green.to_csv(index=False).encode(),
            "ssa_green_metals.csv", "text/csv",
        )

    with tab_b:
        st.markdown("#### Oil & Gas Basins — Sub-Saharan Africa")

        if not oil_gas_df.empty:
            n_prod = int((oil_gas_df.get("status","").str.lower().str.contains("produc", na=False)).sum())
            n_exp  = len(oil_gas_df) - n_prod
            c1, c2, c3 = st.columns(3)
            c1.metric("Total O&G records", len(oil_gas_df))
            c2.metric("Producing fields", n_prod)
            c3.metric("Exploration / development", n_exp)
            st.dataframe(oil_gas_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Export O&G data (CSV)",
                oil_gas_df.to_csv(index=False).encode(),
                "ssa_oil_gas.csv", "text/csv",
            )
        else:
            curated = [
                ("Angola","Cabinda Block 0",     -5.50,  12.20,"Oil","Producing","TotalEnergies/Chevron; >700 kbbl/d"),
                ("Angola","Block 15 (Kizomba)",  -9.50,  12.80,"Oil","Producing","ExxonMobil operator"),
                ("Mozambique","Rovuma LNG",      -10.60, 40.60,"Gas","Development","ENI/TotalEnergies; 18 tcf reserves"),
                ("Tanzania","Ruvu basin",         -7.20,  39.50,"Gas","Exploration","Offshore gas; smaller volumes"),
                ("Kenya","Lokichar basin",         2.38,  35.95,"Oil","Appraisal","Tullow/Africa Oil"),
                ("Uganda","Kingfisher/Jobi-Rii",   1.80,  31.00,"Oil","Development","TotalEnergies; ~1.4 Bb"),
                ("Ghana","Jubilee field",           4.65,  -1.96,"Oil","Producing","Tullow/Ghana; first deepwater"),
                ("Nigeria","Niger Delta",           5.50,   6.00,"Oil","Producing","Major producing province"),
                ("Gabon","Port-Gentil basin",       -1.00,   8.80,"Oil","Producing","TotalEnergies; mature"),
                ("Congo","Pointe-Noire",            -4.77,  11.86,"Oil","Producing","Eni, TotalEnergies"),
                ("Namibia","Orange Basin",         -28.00,  15.50,"Oil","Exploration","Shell/TotalEnergies; 2+ Bb potential"),
                ("South Africa","Orange Basin SA", -30.00,  17.50,"Oil","Exploration","TotalEnergies deepwater"),
                ("Tanzania","Wentworth basin",      -5.00,  39.50,"Gas","Production","Mnazi Bay gas field"),
                ("Ethiopia","Ogaden basin",          7.00,  45.00,"Gas","Exploration","CNPC; stalled development"),
                ("Malawi","Karoo basin",             -13.50, 34.00,"Gas","Exploration","RAK Gas; limited data"),
                ("Rwanda","Kivu lake",               -2.00,  29.30,"Gas","Production","Methane from lake"),
                ("Zambia","Luangwa basin",           -13.00, 32.50,"Gas","Exploration","Early stage"),
                ("Botswana","Kalahari basin",        -21.00, 21.50,"Gas","Exploration","African Energy; CBM"),
                ("Zimbabwe","Cabora Bassa",          -16.50, 32.00,"Gas","Exploration","Early stage"),
                ("Mozambique","Palmeira",            -14.50, 40.50,"Gas","Development","Adjacent to Rovuma"),
                ("Somalia","Puntland",                9.00,  48.50,"Oil","Exploration","Somalia offshore"),
            ]
            df_og = pd.DataFrame(curated, columns=["Country","Field","Lat","Lon","Type","Status","Notes"])
            c1, c2 = st.columns(2)
            c1.metric("Total O&G fields", len(df_og))
            c2.metric("Producing", int((df_og["Status"] == "Producing").sum()))
            st.dataframe(df_og, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇  Export O&G data (CSV)",
                df_og.to_csv(index=False).encode(),
                "ssa_oil_gas.csv", "text/csv",
            )


# ── Infrastructure ────────────────────────────────────────────────────────────

def render_infrastructure(cities: pd.DataFrame, railways: pd.DataFrame,
                           ports: pd.DataFrame) -> None:
    st.markdown("### Infrastructure Intelligence")
    st.markdown(
        '<div class="info-box">'
        'Proximity to cities, railways and seaports is critical for project logistics, '
        'concentrate transport, labour sourcing and capex estimation.'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_c, tab_r, tab_p = st.tabs(["🏙 Urban Centres", "🚂 Railways", "⚓ Seaports"])

    with tab_c:
        if not cities.empty:
            st.markdown(f"**{len(cities):,} cities / towns** (Natural Earth, pop > 100k)")
            country_f = st.multiselect(
                "Filter by country", sorted(cities["country"].dropna().unique().tolist()),
                default=[], key="city_f",
            )
            d = cities if not country_f else cities[cities["country"].isin(country_f)]
            st.dataframe(d.head(100), use_container_width=True, hide_index=True)
        else:
            _show_fallback_cities()

    with tab_r:
        if not railways.empty:
            st.success(f"**{len(railways):,} railway segments** loaded from Natural Earth")
            st.markdown(
                '<div class="warn-box">'
                'Major corridors: TAZARA (Dar es Salaam → Lusaka), Lobito Corridor '
                '(Lobito → Kolwezi), South Africa Transnet, Zimbabwe NRZ, '
                'Zambia ZRL, Kenya–Uganda Standard Gauge Railway.'
                '</div>', unsafe_allow_html=True,
            )
        else:
            _show_fallback_railways()

    with tab_p:
        if not ports.empty:
            st.markdown(f"**{len(ports):,} seaports** (Natural Earth)")
            st.dataframe(ports.head(50), use_container_width=True, hide_index=True)
        else:
            _show_fallback_ports()


def _show_fallback_cities():
    st.info("Cities data file not yet downloaded — run Phase 1 pipeline.")
    cities_list = [
        ("Lusaka", "Zambia", -15.42, 28.28, 2_900_000),
        ("Lubumbashi", "DRC", -11.68, 27.47, 2_500_000),
        ("Johannesburg", "South Africa", -26.20, 28.04, 5_600_000),
        ("Cape Town", "South Africa", -33.93, 18.42, 4_600_000),
        ("Nairobi", "Kenya", -1.29, 36.82, 4_400_000),
        ("Dar es Salaam", "Tanzania", -6.79, 39.21, 3_400_000),
        ("Kinshasa", "DRC", -4.32, 15.32, 16_000_000),
        ("Harare", "Zimbabwe", -17.83, 31.05, 1_600_000),
        ("Maputo", "Mozambique", -25.97, 32.59, 1_100_000),
        ("Gaborone", "Botswana", -24.65, 25.90, 280_000),
        ("Windhoek", "Namibia", -22.56, 17.08, 430_000),
        ("Accra", "Ghana", 5.56, -0.20, 2_600_000),
        ("Kampala", "Uganda", 0.33, 32.58, 1_700_000),
        ("Luanda", "Angola", -8.84, 13.23, 8_300_000),
    ]
    st.dataframe(
        pd.DataFrame(cities_list, columns=["City","Country","Lat","Lon","Population"]),
        use_container_width=True, hide_index=True,
    )


def _show_fallback_railways():
    corridors = [
        ("TAZARA Railway", "Dar es Salaam → Kapiri Mposhi", "Tanzania/Zambia", 1860),
        ("Lobito Corridor", "Lobito → Kolwezi (planned restore)", "Angola/DRC/Zambia", 1300),
        ("Kenya-Uganda SGR", "Mombasa → Kampala", "Kenya/Uganda", 960),
        ("Transnet Freight Rail", "Johannesburg → Cape Town / Durban", "South Africa", 22000),
        ("Zimbabwe NRZ", "Bulawayo → Beira / Victoria Falls", "Zimbabwe", 3100),
        ("Zambia ZRL", "Livingstone → Ndola", "Zambia", 900),
        ("Mozambique CFM", "Beira → Harare · Maputo → Jo'burg", "Mozambique", 3100),
        ("DRC SNCC", "Lubumbashi → Sakania", "DRC", 2600),
        ("Tanzania TRL", "Dar es Salaam → Kigoma / Mwanza", "Tanzania", 2700),
        ("Ethiopia-Djibouti SGR", "Addis Ababa → Djibouti", "Ethiopia/Djibouti", 752),
    ]
    st.dataframe(
        pd.DataFrame(corridors, columns=["Corridor","Route","Countries","Length km"]),
        use_container_width=True, hide_index=True,
    )


def _show_fallback_ports():
    p = [
        ("Durban","South Africa",-29.86,31.03,"Container · bulk · general"),
        ("Cape Town","South Africa",-33.91,18.42,"Container · bulk"),
        ("Richards Bay","South Africa",-28.81,32.07,"Coal · bulk export"),
        ("Dar es Salaam","Tanzania",-6.82,39.30,"Container · general"),
        ("Mombasa","Kenya",-4.05,39.67,"Container · bulk"),
        ("Beira","Mozambique",-19.84,34.84,"Bulk · container (Copperbelt route)"),
        ("Maputo","Mozambique",-25.97,32.59,"Bulk · container"),
        ("Nacala","Mozambique",-14.54,40.67,"Deep-water · bulk"),
        ("Lobito","Angola",-12.35,13.54,"General · mineral export corridor"),
        ("Walvis Bay","Namibia",-22.96,14.51,"Container · bulk"),
        ("Luanda","Angola",-8.79,13.23,"Container · general"),
        ("Tema","Ghana",5.62,-0.02,"Container · bulk"),
        ("Libreville","Gabon",0.39,9.45,"General"),
        ("Djibouti","Djibouti",11.60,43.13,"Hub · transshipment"),
    ]
    st.dataframe(
        pd.DataFrame(p, columns=["Port","Country","Lat","Lon","Type"]),
        use_container_width=True, hide_index=True,
    )


# ── Site Analysis ─────────────────────────────────────────────────────────────

def render_site_analysis(lat: float, lon: float, predictions: pd.DataFrame) -> None:
    from app.geo_utils import score_point
    with st.spinner("Scoring …"):
        result   = score_point(lat, lon, predictions)
    score    = result["score"]
    tier     = result["risk_tier"]
    features = result["features"]
    country  = infer_country(lat, lon)

    st.markdown(
        f'<div class="score-badge">Score: {score:.3f}</div>&nbsp;'
        f'{tier_chip(tier)}&nbsp;'
        f'<span style="color:var(--muted);font-size:0.88rem;margin-left:8px;">'
        f'{country} · {to_dms(lat, True)}, {to_dms(lon, False)}</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prospectivity", f"{score:.2%}")
    c2.metric("Tier", tier)
    c3.metric("Country", country)
    c4.metric("Nearest grid", f"{result['nearest_grid_distance_km']} km")

    log_co = features.get("log_co_ppm", 0)
    log_cu = features.get("log_cu_ppm", 1)
    co_cu  = round((np.expm1(log_co) / max(np.expm1(log_cu), 1)) * 100, 1)
    log_ni = features.get("log_ni_ppm", 0)
    ni_cu  = round((np.expm1(log_ni) / max(np.expm1(log_cu), 1)) * 100, 1)
    c5.metric("Co/Cu", f"{co_cu:.1f} %",
              help="≥ 10% → DRC Katanga-type · <10% → Zambia / SA system")

    if ni_cu > 20 and co_cu < 10:
        style_msg = "🇿🇦 **Ni-dominant** — consistent with Bushveld-type magmatic Cu-Ni sulphide."
    elif co_cu >= 10:
        style_msg = "🇨🇩 **Co-dominant** — consistent with DRC Katanga-style SHSC Cu-Co."
    elif score >= 0.5:
        style_msg = "🇿🇲 **Cu-dominant** — consistent with Zambia Copperbelt SHSC."
    else:
        style_msg = "No diagnostic commodity signature above threshold."
    st.info(style_msg)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        target = ProspectTarget(
            target_id="QUERY", lat=lat, lon=lon,
            prospectivity_score=score, risk_tier=tier,
            uncertainty_proxy=0.0, features=features,
        )
        ev = target.mineral_system_evidence
        if any(ev.values()):
            st.markdown("##### Mineral System Evidence")
            for comp, items in ev.items():
                if items:
                    st.markdown(f"**{comp.upper()}:** " + " · ".join(items))
        st.markdown("##### Geological Interpretation")
        with st.spinner("Generating LLM interpretation …"):
            interp = GeoInterpreter().interpret_score(lat, lon, score, features)
        st.markdown(interp)

    with col_b:
        st.markdown("##### Feature Profile")
        human_feats = humanise_features(features)
        if human_feats:
            df_feat = pd.DataFrame(human_feats)
            colours = ["#f59e0b" if "Cobalt" in r["label"] or "Nickel" in r["label"]
                       else "#34d399" for _, r in df_feat.iterrows()]
            fig = go.Figure(go.Bar(
                x=df_feat["value"], y=df_feat["label"],
                orientation="h", marker_color=colours,
                text=df_feat["value"].round(2),
            ))
            fig.update_layout(**PLOTLY_THEME, height=380, xaxis_title="Feature value")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ── Drill Program ─────────────────────────────────────────────────────────────

def render_drill_program(
    predictions: pd.DataFrame,
    sel_countries, sel_tiers, sel_score,
) -> None:
    st.markdown("### Field-Ready Drill Program Generator")
    st.markdown(
        '<div class="warn-box">'
        'GPS-precise WGS84 coordinates, drilling method, depth range, indicative '
        'cost and permit authority per jurisdiction. Export CSV for field use.'
        '</div>',
        unsafe_allow_html=True,
    )

    if predictions.empty:
        st.warning("Run the pipeline to generate predictions.")
        return

    df = predictions.copy()
    df["country"] = df.apply(lambda r: infer_country(r.lat, r.lon), axis=1)
    filtered = _apply_filters(df, sel_countries, sel_tiers, sel_score[0], sel_score[1])

    c1, c2 = st.columns(2)
    n_targets  = c1.slider("Number of drill targets", 3, 30, 8)
    min_score  = c2.slider("Minimum prospectivity score", 0.30, 0.95, 0.60, 0.05,
                           key="dp_minscore")

    country_opts = sorted(filtered["country"].unique().tolist())
    sel_dp_ctry  = st.multiselect("Countries", country_opts, default=country_opts,
                                  key="dp_countries")

    targets = _build_targets(filtered, min(200, len(filtered)))
    targets = targets[
        (targets["prospectivity_score"] >= min_score) &
        (targets["country"].isin(sel_dp_ctry))
    ].head(n_targets).copy()

    if targets.empty:
        st.warning("No targets match the current filters.")
        return

    targets["lat_dms"] = targets["lat"].apply(lambda x: to_dms(x, True))
    targets["lon_dms"] = targets["lon"].apply(lambda x: to_dms(x, False))

    def drill_type(s):
        if s >= 0.85: return "Diamond Core (DD)"
        if s >= 0.70: return "Reverse Circulation (RC)"
        if s >= 0.50: return "RC Scout + IP Survey"
        return "Soil Geochem + Ground Magnetics"

    def drill_depth(s):
        if s >= 0.85: return "300–500 m"
        if s >= 0.70: return "150–250 m"
        if s >= 0.50: return "50–150 m"
        return "Surface only"

    def holes(s):
        if s >= 0.85: return "3–5 holes"
        if s >= 0.70: return "5–10 holes"
        if s >= 0.50: return "2–5 holes"
        return "N/A"

    permit_map = {
        "South Africa": "MPRDA licence (DMRE)",
        "Zambia":       "Exploration licence (MMMD)",
        "DRC":          "Permis de Recherches Minières (CAMI)",
        "Botswana":     "Prospecting Licence (MMWE)",
        "Zimbabwe":     "Special Grant (MMCZ)",
        "Namibia":      "Exclusive Prospecting Licence (MME)",
        "Mozambique":   "Licença de Prospeção (MIREME)",
        "Tanzania":     "Prospecting Licence (MEM)",
        "Kenya":        "Prospecting Licence (Mining Act 2016)",
        "Ghana":        "Reconnaissance Licence (Minerals Commission)",
        "Angola":       "Licença de Prospecção (MIREMPET)",
        "Uganda":       "Exploration Licence (DGSM)",
    }

    targets["drill_type"]  = targets["prospectivity_score"].apply(drill_type)
    targets["depth"]       = targets["prospectivity_score"].apply(drill_depth)
    targets["holes"]       = targets["prospectivity_score"].apply(holes)
    targets["permit_note"] = targets["country"].map(permit_map).fillna("Check national mining authority")

    field_cols = [c for c in [
        "target_id","country","lat","lon","lat_dms","lon_dms",
        "prospectivity_score","risk_tier","co_cu_pct","ni_cu_pct",
        "drill_type","depth","holes","est_cost","permit_note","program",
    ] if c in targets.columns]

    st.markdown("#### Drill Target Sheet")
    st.dataframe(
        targets[field_cols].rename(columns={
            "target_id":"Target ID","country":"Country",
            "lat":"Lat (DD)","lon":"Lon (DD)",
            "lat_dms":"Lat (DMS)","lon_dms":"Lon (DMS)",
            "prospectivity_score":"Score","risk_tier":"Tier",
            "co_cu_pct":"Co/Cu %","ni_cu_pct":"Ni/Cu %",
            "drill_type":"Method","depth":"Depth","holes":"Holes",
            "est_cost":"Est. Cost","permit_note":"Permit Authority","program":"Program",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇  Export drill program (CSV)",
        targets[field_cols].to_csv(index=False).encode(),
        "geoexplorer_drill_program.csv", "text/csv",
    )

    st.markdown("#### Target Locations — Map")
    m = folium.Map(
        location=[targets["lat"].mean(), targets["lon"].mean()],
        zoom_start=5, tiles="CartoDB dark_matter",
    )
    for _, r in targets.iterrows():
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=10,
            color="#f59e0b", fill=True, fill_color="#f59e0b", fill_opacity=0.9,
            tooltip=(
                f"<b>{r.target_id}</b> — {r.country}<br>"
                f"Score: {r.prospectivity_score:.3f} ({r.risk_tier})<br>"
                f"Method: {r.drill_type}<br>Cost: {r.est_cost}"
            ),
            popup=folium.Popup(
                f"<b>{r.target_id}</b><br>{r.lat_dms}, {r.lon_dms}<br>"
                f"<b>{r.program}</b>", max_width=300,
            ),
        ).add_to(m)
    st_folium(m, width="100%", height=400, returned_objects=[])


# ── Mineral Systems ───────────────────────────────────────────────────────────

def render_mineral_system_panel() -> None:
    st.markdown("### Mineral Systems Semantic Model — Africa")
    st.markdown(
        '<div class="info-box">'
        'Each ML feature maps to a component of the SHSC mineral systems framework '
        '(<strong>Source → Pathway → Trap → Modifier</strong>), grounding model '
        'outputs in geological theory for the Lufilian Arc and Bushveld Complex.'
        '</div>',
        unsafe_allow_html=True,
    )
    rows = [
        {
            "Feature":     feat,
            "Component":   info["component"].value.title(),
            "Styles":      ", ".join(s.value for s in info.get("styles",[])),
            "Description": info.get("description",""),
            "Threshold":   str(info.get("anomaly_threshold_log","—")),
        }
        for feat, info in FEATURE_SYSTEM_MAP.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### South Africa — Deposit Style Context")
    sa_rows = [
        ("Bushveld Complex (BIC)", "Ni-Cu-PGE magmatic sulphide",
         "Ni dominant > Cu; Co low; Eastern/Western limb structures", "Limpopo/Mpumalanga"),
        ("Phalaborwa/Palabora", "Alkaline carbonatite Cu",
         "Cu anomaly; low Co/Cu; associated with P, F, REE", "Limpopo"),
        ("Namaqualand/O'Kiep", "Proterozoic SHSC-analog Cu",
         "Cu signal; similar age to Lufilian Arc; fault-controlled", "Northern Cape"),
        ("Aggeneys/Black Mountain", "SEDEX Pb-Zn-Cu-Ag",
         "Elevated Zn and Pb relative to Cu; different from SHSC", "Northern Cape"),
        ("Kalahari Copper Belt", "SHSC extension",
         "Co/Cu > 5%; Cu anomaly; Proterozoic sediments", "Northern Cape/Botswana border"),
    ]
    st.dataframe(
        pd.DataFrame(sa_rows, columns=["District","Style","Key Indicators","Province"]),
        use_container_width=True, hide_index=True,
    )


# ── Model Analytics ───────────────────────────────────────────────────────────

def render_model_analytics(fi: dict, predictions: pd.DataFrame) -> None:
    comp_path = ROOT / "models" / "model_comparison.json"
    if comp_path.exists():
        results = json.loads(comp_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(results)
        df["model"] = df["name"].str.replace("_"," ").str.title()
        fig = go.Figure()
        fig.add_bar(name="ROC-AUC", x=df.model, y=df.roc_auc_mean,
                    error_y=dict(type="data", array=df.roc_auc_std.tolist()),
                    marker_color="#f59e0b")
        fig.add_bar(name="PR-AUC",  x=df.model, y=df.pr_auc_mean,
                    marker_color="#60a5fa")
        fig.update_layout(
            **PLOTLY_THEME, barmode="group", height=320,
            title="Model comparison — 5-fold spatial CV",
            yaxis=dict(range=[0,1], title="Score"),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

    if fi:
        imp     = fi.get("model_importance", {})
        display = fi.get("feature_display_names", {})
        if imp:
            df_fi = pd.DataFrame(
                [{"feature": display.get(k,k), "importance": v} for k,v in imp.items()]
            ).sort_values("importance")
            colours = ["#f59e0b" if any(x in r["feature"]
                       for x in ("Cobalt","Nickel","Iron")) else "#34d399"
                       for _, r in df_fi.iterrows()]
            fig_fi = go.Figure(go.Bar(
                x=df_fi.importance, y=df_fi.feature, orientation="h",
                marker_color=colours, text=df_fi.importance.round(3),
            ))
            fig_fi.update_layout(
                **PLOTLY_THEME, height=400,
                title="Feature importance (gold = geochemical pathfinders)",
            )
            fig_fi.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_fi, use_container_width=True)

    if not predictions.empty:
        fig_hist = px.histogram(
            predictions, x="prospectivity_score", nbins=50,
            color_discrete_sequence=["#34d399"],
            template="plotly_dark",
            title="Score distribution — Africa coverage grid",
        )
        fig_hist.update_layout(
            **PLOTLY_THEME, height=280,
            xaxis_title="Prospectivity score", yaxis_title="Grid cells", bargap=0.05,
        )
        fig_hist.add_vline(x=0.50, line_color="#f59e0b", line_dash="dash",
                           annotation_text="High")
        fig_hist.add_vline(x=0.70, line_color="#f87171", line_dash="dash",
                           annotation_text="Very High")
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    predictions  = load_predictions()
    deposits     = load_deposits()
    cities       = load_cities()
    oil_gas_df   = load_oil_gas()
    railways     = load_railways()
    ports        = load_ports()
    meta         = load_model_metadata()
    fi           = load_feature_importance()
    model_bundle = load_model_bundle()

    (
        query_lat, query_lon, score_btn, uploaded_df,
        sel_countries, sel_tiers, sel_score, sel_resources,
    ) = render_sidebar(meta, model_bundle is not None)

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">AI-Assisted Mineral & Resource Exploration · Sub-Saharan Africa</div>
          <h1>GeoExplorer AI — Resource Intelligence Platform</h1>
          <p class="lede">
            Machine-learning prospectivity screening for copper, cobalt, nickel and
            <strong>green critical metals (Li · Mn · REE · Graphite · V)</strong> across
            Sub-Saharan Africa — integrated with oil &amp; gas basin intelligence,
            urban centres, railways and seaport infrastructure context.
            Zambia · DRC · South Africa · Tanzania · Kenya · Angola · Botswana ·
            Zimbabwe · Mozambique · Namibia · Ghana · Uganda · Gabon · Malawi.
            XGBoost · 20,125 scored grid cells · ROC-AUC 0.889 spatial CV.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🗺  Exploration Map",
        "📋  Executive Summary",
        "🎯  Priority Targets",
        "🔬  Site Analysis",
        "⛏  Drill Program",
        "🔋  Resource Intelligence",
        "🏗  Infrastructure",
        "🧬  Mineral Systems",
        "📊  Model Analytics",
    ])

    with tabs[0]:
        render_map(
            predictions, deposits, cities, oil_gas_df, railways, ports,
            sel_resources, sel_tiers, sel_score,
            query_lat, query_lon, st.session_state.get("last_score"),
        )

    with tabs[1]:
        render_executive_summary(predictions, meta, sel_countries, sel_tiers, sel_score)

    with tabs[2]:
        render_priority_targets(predictions, sel_countries, sel_tiers, sel_score)

    with tabs[3]:
        if score_btn or st.session_state.get("scored"):
            if not within_model_bbox(query_lat, query_lon):
                st.warning(
                    "Coordinates fall outside the ML scoring grid "
                    "(15°E–38°E, 35°S–0° — Central/Southern Africa Copperbelt). "
                    "The prospectivity model was trained on this region. "
                    "Adjust coordinates in the sidebar or use the nearest grid cell result."
                )
            elif predictions.empty:
                st.error("No predictions loaded — run the pipeline first.")
            else:
                st.session_state["scored"] = True
                render_site_analysis(query_lat, query_lon, predictions)
        else:
            st.markdown(
                '<div class="info-box">Enter coordinates in the sidebar and click '
                '<strong>Score location</strong> for a site-specific AI geological '
                'assessment with mineral system evidence and LLM interpretation.</div>',
                unsafe_allow_html=True,
            )
            if not predictions.empty:
                best = predictions.nlargest(1, "prospectivity_score").iloc[0]
                st.info(
                    f"Highest-scoring cell: {abs(best.lat):.3f}°S, {best.lon:.3f}°E "
                    f"— score {best.prospectivity_score:.3f} "
                    f"({infer_country(best.lat, best.lon)})"
                )

    with tabs[4]:
        render_drill_program(predictions, sel_countries, sel_tiers, sel_score)

    with tabs[5]:
        render_resource_intelligence(deposits, oil_gas_df)

    with tabs[6]:
        render_infrastructure(cities, railways, ports)

    with tabs[7]:
        render_mineral_system_panel()

    with tabs[8]:
        render_model_analytics(fi, predictions)


if __name__ == "__main__":
    main()
