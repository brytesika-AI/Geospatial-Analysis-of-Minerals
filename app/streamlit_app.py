"""
GeoExplorer AI — Southern & Central Africa Edition
Professional mineral prospectivity platform for mining executives and exploration engineers.

Coverage   : Zambia · DRC Katanga · Botswana · Zimbabwe · Mozambique · Namibia · South Africa
Commodities: Copper · Cobalt · Nickel
Deposit styles: SHSC (Lufilian Arc), Magmatic Cu-Ni (Bushveld / Selebi-Phikwe),
                SEDEX (Northern Cape), Alkaline-hosted (Phalaborwa)
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
    within_africa_study_area,
)
from app.llm_interpreter import GeoInterpreter
from app.mineral_systems import (
    FEATURE_SYSTEM_MAP, ProspectTarget, feature_component_summary,
)

st.set_page_config(
    page_title="GeoExplorer AI | Africa Cu · Co · Ni",
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
h1 { font-size: 2.8rem !important; line-height: 1.05 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.15rem !important; }
a { color: var(--gold) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    color: var(--muted) !important; font-weight: 600 !important;
    font-size: 0.82rem !important; letter-spacing: 0.04em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* Buttons */
.stButton > button {
    background: var(--gold); color: #0c1420;
    border: none; border-radius: 6px;
    font-weight: 700; font-size: 0.88rem; letter-spacing: 0.03em;
    padding: 9px 20px;
}
.stButton > button:hover { background: var(--gold-lt); }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    font-weight: 600; font-size: 0.84rem;
}
[data-testid="stDownloadButton"] > button:hover { border-color: var(--gold); color: var(--gold); }

/* Custom components */
.hero { border-bottom: 1px solid var(--border); padding: 4px 0 22px 0; margin-bottom: 20px; }
.eyebrow { color: var(--gold); font-size: 0.72rem; font-weight: 700;
           letter-spacing: 0.16em; text-transform: uppercase; }
.lede { max-width: 900px; color: var(--muted); font-size: 1.0rem; line-height: 1.6; margin-top:6px; }

.info-box { background: var(--panel); border-left: 3px solid var(--blue);
            border-radius: 0 6px 6px 0; padding: 12px 16px;
            color: var(--muted); margin: 8px 0 16px 0; font-size:0.9rem; }
.warn-box  { background: var(--panel); border-left: 3px solid var(--gold);
             border-radius: 0 6px 6px 0; padding: 12px 16px;
             color: var(--muted); margin: 8px 0 16px 0; font-size:0.9rem; }
.score-badge { display:inline-block; padding:8px 16px; border-radius:6px;
               font-weight:800; font-size:1.1rem; color:#0c1420; background:var(--gold); }

/* Tier chips */
.tier-VH { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--vh); color:#fff; }
.tier-H  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--hi); color:#0c1420; }
.tier-M  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--mod); color:#0c1420; }
.tier-L  { display:inline-block; padding:4px 10px; border-radius:5px; font-size:0.8rem;
           font-weight:700; background:var(--panel2); color:var(--muted);
           border:1px solid var(--border); }

/* Action cards */
.action-card { background:var(--panel); border:1px solid var(--border);
               border-radius:8px; padding:16px 18px; margin-bottom:8px; }
.action-card h4 { color:var(--gold); font-size:0.8rem; font-weight:700;
                  letter-spacing:0.08em; text-transform:uppercase; margin:0 0 6px 0; }
.action-card p  { color:var(--text); font-size:0.88rem; margin:0; line-height:1.5; }

.caption { color:var(--muted); font-size:0.78rem; line-height:1.45; }
hr { border-color: var(--border) !important; }

/* Sidebar branding */
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


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "predictions.csv"
    if not path.exists():
        st.error("Prediction grid not found — run the three pipeline scripts.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_deposits() -> pd.DataFrame:
    import geopandas as gpd
    path = ROOT / "data" / "raw" / "mrds_africa_copper.geojson"
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "name", "commod1"])
    gdf = gpd.read_file(path)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    cols = ["lon", "lat"] + [c for c in gdf.columns if c not in ("lon","lat","geometry")]
    return pd.DataFrame(gdf[cols])


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
    """Decimal degrees → DMS string."""
    hemi = ("S" if dd < 0 else "N") if is_lat else ("W" if dd < 0 else "E")
    dd = abs(dd)
    d, m = int(dd), int((dd % 1) * 60)
    s = round(((dd % 1) * 60 % 1) * 60, 1)
    return f"{d}°{m:02d}'{s:04.1f}\" {hemi}"


def _build_targets(predictions: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """Enrich top-N targets with derived fields used across several tabs."""
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

    # Recommended field program
    def _program(row):
        s, u = row.prospectivity_score, row.uncertainty
        if s >= 0.85: return "Diamond core drilling (DD) 300-500 m + geophysics"
        if s >= 0.70: return "RC drilling 150-250 m + IP/CSAMT survey"
        if s >= 0.50: return "Infill soil geochemistry + IP ground survey"
        return "Regional airborne EM/magnetics, recon mapping"

    def _cost(row):
        s = row.prospectivity_score
        if s >= 0.85: return "$600k–$1.5M"
        if s >= 0.70: return "$250k–$600k"
        if s >= 0.50: return "$80k–$200k"
        return "$30k–$80k"

    top["program"] = top.apply(_program, axis=1)
    top["est_cost"] = top.apply(_cost, axis=1)
    top["decision"] = np.where(top["prospectivity_score"] >= 0.70, "Advance", "Hold")
    return top


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(meta: dict, model_ok: bool):
    uploaded_df = None
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">'
            '<div class="eyebrow">GeoExplorer AI · Africa</div>'
            '<div class="title">Cu · Co · Ni Prospectivity</div>'
            '<div class="sub">Zambia · DRC · Botswana · South Africa<br>'
            'Namibia · Zimbabwe · Mozambique</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Query Location**")
        lat = st.number_input("Latitude (°S negative)", value=-12.5,
                              min_value=-35.0, max_value=0.0, format="%.4f")
        lon = st.number_input("Longitude (°E)", value=28.2,
                              min_value=15.0, max_value=38.0, format="%.4f")
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
            '<div class="caption">USGS MRDS deposits · GEM Active Faults · '
            'SRTM elevation · Copperbelt-calibrated geochemistry.<br>'
            'Not a resource estimate or regulatory disclosure.</div>',
            unsafe_allow_html=True,
        )

    return float(lat), float(lon), score_btn, uploaded_df


# ── Exploration Map ───────────────────────────────────────────────────────────

def render_map(predictions, deposits, q_lat=None, q_lon=None, q_score=None):
    m = folium.Map(
        location=[REGION_CENTER["lat"], REGION_CENTER["lon"]],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    tier_filter = st.multiselect(
        "Show tiers", ["Very High", "High", "Moderate", "Low"],
        default=["Very High", "High", "Moderate"],
        label_visibility="collapsed",
    )

    if not predictions.empty:
        from folium.plugins import HeatMap
        filtered = predictions[predictions["risk_tier"].isin(tier_filter)]
        heat_data = [
            [r.lat, r.lon, r.prospectivity_score]
            for _, r in filtered.iterrows()
            if r.prospectivity_score > 0.2
        ]
        if heat_data:
            HeatMap(
                heat_data, name="Prospectivity", min_opacity=0.2, max_val=0.9,
                radius=22, blur=28,
                gradient={
                    0.2: "#1e3a5f", 0.4: "#1d4ed8", 0.55: "#34d399",
                    0.68: "#f59e0b", 0.82: "#f87171", 0.95: "#fef3c7",
                },
            ).add_to(m)

    if not deposits.empty:
        dep_layer = folium.FeatureGroup(name="Known deposits")
        for _, d in deposits.head(800).iterrows():
            c = d.get("commod1", "Cu")
            colour = "#f59e0b" if "Ni" in str(c) else "#60a5fa" if "Co" in str(c) else "#34d399"
            folium.CircleMarker(
                location=[d.lat, d.lon], radius=4,
                color=colour, fill=True, fill_color=colour,
                fill_opacity=0.85, weight=1,
                tooltip=f"{d.get('district', d.get('name','deposit'))} ({c})",
            ).add_to(dep_layer)
        dep_layer.add_to(m)

    if q_lat and q_lon:
        folium.Marker(
            location=[q_lat, q_lon],
            popup=f"<b>Score: {q_score:.2f}</b>" if q_score else "Query site",
            tooltip="Query site",
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st.markdown(
        '<div class="info-box">Heatmap: red = Very High · amber = High · '
        'teal = Moderate · blue = Low. '
        'Markers: <span style="color:#34d399">■</span> Cu deposits  '
        '<span style="color:#60a5fa">■</span> Cu-Co  '
        '<span style="color:#f59e0b">■</span> Cu-Ni/SA</div>',
        unsafe_allow_html=True,
    )
    st_folium(m, width="100%", height=580, returned_objects=[])


# ── Executive Summary ─────────────────────────────────────────────────────────

def render_executive_summary(predictions: pd.DataFrame, meta: dict) -> None:
    if predictions.empty:
        st.warning("Run the data pipeline to generate the prediction grid.")
        return

    targets = _build_targets(predictions, 50)

    # ── KPI strip ──
    n_vh  = int((predictions["prospectivity_score"] >= 0.70).sum())
    n_hi  = int(((predictions["prospectivity_score"] >= 0.50) &
                 (predictions["prospectivity_score"] < 0.70)).sum())
    n_adv = int((targets["decision"] == "Advance").sum())
    auc   = meta.get("roc_auc", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Grid cells", f"{len(predictions):,}", help="Total screened locations")
    c2.metric("Very High targets", f"{n_vh:,}", help="Score ≥ 0.70")
    c3.metric("High targets", f"{n_hi:,}", help="Score 0.50 – 0.70")
    c4.metric("Advance recommended", f"{n_adv}", help="Top-50 with score ≥ 0.70")
    c5.metric("Model ROC-AUC", f"{auc:.3f}", help="Spatial cross-validation")

    st.divider()

    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("#### Portfolio by Tier")
        tier_counts = predictions["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_counts["Tier"] = pd.Categorical(
            tier_counts["Tier"],
            categories=["Very High", "High", "Moderate", "Low"], ordered=True
        )
        tier_counts = tier_counts.sort_values("Tier")
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts["Tier"], values=tier_counts["Count"],
            hole=0.55,
            marker_colors=[TIER_COLORS.get(t, "#64748b") for t in tier_counts["Tier"]],
            textinfo="label+percent", textfont_size=11,
        ))
        fig_donut.update_layout(
            **PLOTLY_THEME,
            height=280,
            showlegend=False,
            annotations=[dict(text="Targets", x=0.5, y=0.5, font_size=13,
                              showarrow=False, font_color="#e2e8f0")],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("#### Coverage by Country")
        targets["country_"] = targets.apply(
            lambda r: infer_country(r.lat, r.lon), axis=1
        )
        ctry = targets["country_"].value_counts().reset_index()
        ctry.columns = ["Country", "Targets"]
        fig_bar = px.bar(ctry, x="Targets", y="Country", orientation="h",
                         color="Targets", color_continuous_scale=["#1e3a5f","#f59e0b"],
                         template="plotly_dark")
        fig_bar.update_layout(**PLOTLY_THEME, height=250, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("#### Target Prioritisation Matrix")
        st.caption("Prospectivity vs. geological uncertainty — size = relative score")
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
                          line_width=0, annotation_text="Very High", annotation_position="right")
        fig_mat.add_hrect(y0=0.50, y1=0.70, fillcolor="#f59e0b", opacity=0.07,
                          line_width=0, annotation_text="High", annotation_position="right")
        fig_mat.add_vline(x=0.40, line_dash="dash", line_color="#475569",
                          annotation_text="High uncertainty →", annotation_position="top right")
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
        adv = targets[targets["decision"]=="Advance"]
        cnt = len(adv)
        ctries = ", ".join(adv["country"].unique()[:3])
        st.markdown(
            f'<div class="action-card"><h4>⚡ Immediate (0–3 months)</h4>'
            f'<p>Advance <strong>{cnt} priority targets</strong> across {ctries}.<br>'
            f'Initiate IP geophysics + RC drilling scoping. Estimated budget: '
            f'<strong>$1.5M – $4M</strong>.</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(
            '<div class="action-card"><h4>📋 Short-term (3–6 months)</h4>'
            '<p>Infill soil geochemistry on High-tier targets. Commission '
            'geological mapping for SHSC stratigraphic confirmation. '
            'Estimated budget: <strong>$400k – $800k</strong>.</p></div>',
            unsafe_allow_html=True)
    with a3:
        st.markdown(
            '<div class="action-card"><h4>🔬 Medium-term (6–18 months)</h4>'
            '<p>Diamond core drilling on best RC intercepts. Resource estimation '
            'scoping. Engage permitting in Zambia/DRC/South Africa. '
            'Estimated budget: <strong>$2M – $8M</strong>.</p></div>',
            unsafe_allow_html=True)
    with a4:
        st.markdown(
            '<div class="action-card"><h4>🌍 Background</h4>'
            '<p>Regional airborne EM/magnetics for Moderate-tier targets. '
            'Update model with drilling assay feedback. Expand coverage to '
            'Angola / Tanzania margins.</p></div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Top 10 Priority Targets — Executive Summary")
    top10 = targets.head(10)[
        ["target_id","country","lat","lon","prospectivity_score",
         "risk_tier","co_cu_pct","est_cost","decision","program"]
    ].rename(columns={
        "target_id": "ID", "country": "Country",
        "prospectivity_score": "Score", "risk_tier": "Tier",
        "co_cu_pct": "Co/Cu %", "est_cost": "Est. Cost",
        "decision": "Decision", "program": "Recommended Program",
    })
    st.dataframe(top10, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇  Download top-10 summary (CSV)",
        top10.to_csv(index=False).encode(),
        "geoexplorer_priority_targets.csv", "text/csv",
    )


# ── Priority Targets ──────────────────────────────────────────────────────────

def render_priority_targets(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        st.warning("Prediction grid unavailable.")
        return

    n_show = st.slider("Number of targets to display", 10, 100, 30, step=5)
    targets = _build_targets(predictions, n_show)

    country_filter = st.multiselect(
        "Filter by country", sorted(targets["country"].unique()),
        default=sorted(targets["country"].unique()),
    )
    targets = targets[targets["country"].isin(country_filter)]

    st.markdown(
        '<div class="warn-box">'
        '<strong>Co/Cu ratio > 10 %</strong> is indicative of DRC Katanga-style Cu-Co '
        'SHSC mineralisation (Cailteux et al. 2005). '
        '<strong>Ni/Cu > 20 %</strong> suggests Bushveld-type magmatic sulphide '
        '(South Africa). Uncertainty proxy drives recommended field programme.</div>',
        unsafe_allow_html=True,
    )

    display_cols = [c for c in [
        "target_id","country","lat","lon","prospectivity_score","risk_tier",
        "co_cu_pct","uncertainty","est_cost","decision","program"
    ] if c in targets.columns]

    st.dataframe(
        targets[display_cols].rename(columns={
            "target_id":"ID","country":"Country",
            "prospectivity_score":"Score","risk_tier":"Tier",
            "co_cu_pct":"Co/Cu %","uncertainty":"Uncertainty",
            "est_cost":"Est. Cost","decision":"Decision",
            "program":"Recommended Program",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇  Export all targets (CSV)",
        targets[display_cols].to_csv(index=False).encode(),
        "geoexplorer_all_targets.csv", "text/csv",
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
              help="≥ 10% → DRC Katanga-type · <10% → standard Cu / SA system")

    # Deposit style inference
    if ni_cu > 20 and co_cu < 10:
        style_msg = "🇿🇦 **Ni-dominant signature** — consistent with Bushveld-type magmatic Cu-Ni sulphide (South Africa)."
    elif co_cu >= 10:
        style_msg = "🇨🇩 **Co-dominant signature** — consistent with DRC Katanga-style SHSC Cu-Co mineralisation."
    elif score >= 0.5:
        style_msg = "🇿🇲 **Cu-dominant signature** — consistent with Zambia Copperbelt SHSC or Namaqualand/O'Kiep Cu system."
    else:
        style_msg = "No diagnostic commodity signature above threshold."
    st.info(style_msg)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        # Mineral system evidence
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
            fig.update_layout(
                **PLOTLY_THEME, height=380,
                xaxis_title="Feature value",
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ── Drill Program ─────────────────────────────────────────────────────────────

def render_drill_program(predictions: pd.DataFrame) -> None:
    st.markdown("### Field-Ready Drill Program Generator")
    st.markdown(
        '<div class="warn-box">'
        'GPS-precise coordinates, recommended drilling method, depth, and indicative '
        'cost per target. Export to CSV for field use. Coordinates in WGS84 (EPSG:4326).'
        '</div>',
        unsafe_allow_html=True,
    )

    if predictions.empty:
        st.warning("Run the pipeline to generate predictions.")
        return

    n_targets = st.slider("Number of drill targets", 3, 20, 8)
    min_score = st.slider("Minimum prospectivity score", 0.30, 0.95, 0.60, 0.05)
    country_opts = sorted({infer_country(r.lat, r.lon) for _, r in
                           predictions.nlargest(200, "prospectivity_score").iterrows()})
    sel_countries = st.multiselect("Countries", country_opts, default=country_opts)

    targets = _build_targets(predictions, 200)
    targets = targets[
        (targets["prospectivity_score"] >= min_score) &
        (targets["country"].isin(sel_countries))
    ].head(n_targets).copy()

    if targets.empty:
        st.warning("No targets match the current filters.")
        return

    targets["lat_dms"] = targets["lat"].apply(lambda x: to_dms(x, True))
    targets["lon_dms"] = targets["lon"].apply(lambda x: to_dms(x, False))

    # Drill type logic
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

    targets["drill_type"]  = targets["prospectivity_score"].apply(drill_type)
    targets["depth"]       = targets["prospectivity_score"].apply(drill_depth)
    targets["holes"]       = targets["prospectivity_score"].apply(holes)
    targets["permit_note"] = targets["country"].map({
        "South Africa": "MPRDA licence (DMRE)",
        "Zambia":       "Exploration licence (MMMD)",
        "DRC":          "Permis de Recherches Minières (CAMI)",
        "Botswana":     "Prospecting Licence (MMWE)",
        "Zimbabwe":     "Special Grant (MMCZ)",
        "Namibia":      "Exclusive Prospecting Licence (MME)",
        "Mozambique":   "Licença de Prospeção (MIREME)",
    }).fillna("Check national mining authority")

    field_cols = [
        "target_id", "country", "lat", "lon", "lat_dms", "lon_dms",
        "prospectivity_score", "risk_tier",
        "co_cu_pct" if "co_cu_pct" in targets.columns else None,
        "drill_type", "depth", "holes", "est_cost", "permit_note", "program",
    ]
    field_cols = [c for c in field_cols if c and c in targets.columns]

    st.markdown("#### Drill Target Sheet")
    st.dataframe(
        targets[field_cols].rename(columns={
            "target_id": "Target ID", "country": "Country",
            "lat": "Lat (DD)", "lon": "Lon (DD)",
            "lat_dms": "Lat (DMS)", "lon_dms": "Lon (DMS)",
            "prospectivity_score": "Score", "risk_tier": "Tier",
            "co_cu_pct": "Co/Cu %",
            "drill_type": "Method", "depth": "Depth",
            "holes": "Holes", "est_cost": "Est. Cost",
            "permit_note": "Permit Authority", "program": "Program",
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
            location=[r.lat, r.lon], radius=9,
            color="#f59e0b", fill=True, fill_color="#f59e0b", fill_opacity=0.9,
            tooltip=(
                f"<b>{r.target_id}</b> — {r.country}<br>"
                f"Score: {r.prospectivity_score:.3f} ({r.risk_tier})<br>"
                f"Method: {r.drill_type}<br>Cost: {r.est_cost}"
            ),
            popup=folium.Popup(
                f"<b>{r.target_id}</b><br>{r.lat_dms}, {r.lon_dms}<br>"
                f"<b>{r.program}</b>",
                max_width=300,
            ),
        ).add_to(m)
    st_folium(m, width="100%", height=400, returned_objects=[])


# ── Batch Scoring ─────────────────────────────────────────────────────────────

def render_batch_scoring(uploaded_df, model_bundle) -> None:
    if uploaded_df is None:
        st.markdown(
            '<div class="info-box">Upload a CSV (sidebar) with the trained feature '
            'columns to score a batch of sample locations against the model.</div>',
            unsafe_allow_html=True,
        )
        return
    if model_bundle is None:
        st.warning("Model bundle not loaded — run the pipeline to generate it.")
        return

    feature_cols = model_bundle.get("feature_cols", [])
    missing = [c for c in feature_cols if c not in uploaded_df.columns]
    if missing:
        st.warning("Missing columns: " + ", ".join(missing))
        return

    scored = uploaded_df.copy()
    proba  = model_bundle["pipeline"].predict_proba(scored[feature_cols])[:, 1]
    scored["prospectivity_score"] = np.round(proba, 4)
    scored["risk_tier"]           = scored["prospectivity_score"].apply(score_to_tier)
    scored["country"]             = scored.apply(
        lambda r: infer_country(r.get("lat", 0), r.get("lon", 0)), axis=1
    )
    scored = scored.sort_values("prospectivity_score", ascending=False)
    st.dataframe(scored, use_container_width=True)
    st.download_button(
        "⬇  Download scored results (CSV)",
        scored.to_csv(index=False).encode(),
        "batch_scored.csv", "text/csv",
    )


# ── Mineral Systems ───────────────────────────────────────────────────────────

def render_mineral_system_panel() -> None:
    st.markdown("### Mineral Systems Semantic Model — Africa")
    st.markdown(
        '<div class="info-box">'
        'Each ML feature is mapped to a component of the SHSC mineral systems '
        'framework (<strong>Source → Pathway → Trap → Modifier</strong>). '
        'This grounds model outputs in geological theory rather than treating '
        'them as opaque numbers. Used for the Site Analysis interpretation '
        'and LLM geological narrative generation.'
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

    st.markdown("#### South Africa Context")
    sa_rows = [
        ("Bushveld Complex (BIC)", "Ni-Cu-PGE magmatic sulphide", "Ni signal > Cu; Co low; Eastern/Western limb structures", "Limpopo/Mpumalanga"),
        ("Phalaborwa / Palabora", "Alkaline carbonatite Cu", "Cu anomaly; low Co/Cu; associated with P, F, REE", "Limpopo"),
        ("Namaqualand / O'Kiep", "Proterozoic SHSC-analog Cu", "Cu signal; similar age to Lufilian Arc; fault-controlled", "Northern Cape"),
        ("Aggeneys / Black Mountain", "SEDEX Pb-Zn-Cu-Ag", "Elevated Zn and Pb relative to Cu; different from SHSC", "Northern Cape"),
        ("Kalahari Copper Belt (SA)", "SHSC extension", "Co/Cu > 5%; Cu anomaly; Proterozoic sediments", "Northern Cape / Botswana border"),
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
            colours = ["#f59e0b" if any(x in r["feature"] for x in ("Cobalt","Nickel","Iron"))
                       else "#34d399" for _, r in df_fi.iterrows()]
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
            xaxis_title="Prospectivity score", yaxis_title="Grid cells",
            bargap=0.05,
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
    meta         = load_model_metadata()
    fi           = load_feature_importance()
    model_bundle = load_model_bundle()

    query_lat, query_lon, score_btn, uploaded_df = render_sidebar(
        meta, model_bundle is not None
    )

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">AI-Assisted Mineral Exploration · Southern &amp; Central Africa</div>
          <h1>Cu · Co · Ni Prospectivity Platform</h1>
          <p class="lede">
            Machine-learning screening for copper, cobalt and nickel targets across
            Zambia, DRC Katanga, Botswana, Zimbabwe, Mozambique, Namibia and
            <strong>South Africa</strong> — grounded in the SHSC mineral systems
            framework of the Lufilian Arc and Bushveld Complex geology.
            16,100+ scored grid cells · XGBoost · ROC-AUC 0.915 spatial CV.
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
        "🧬  Mineral Systems",
        "📊  Model Analytics",
    ])

    with tabs[0]:
        render_map(predictions, deposits, query_lat, query_lon,
                   st.session_state.get("last_score"))

    with tabs[1]:
        render_executive_summary(predictions, meta)

    with tabs[2]:
        render_priority_targets(predictions)

    with tabs[3]:
        if score_btn or st.session_state.get("scored"):
            if not within_africa_study_area(query_lat, query_lon):
                st.warning(
                    "Coordinates fall outside the study area "
                    "(15°E–38°E, 35°S–0°). Adjust in the sidebar."
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
        render_drill_program(predictions)

    with tabs[5]:
        render_mineral_system_panel()

    with tabs[6]:
        render_model_analytics(fi, predictions)


if __name__ == "__main__":
    main()
