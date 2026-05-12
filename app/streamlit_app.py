"""
GeoExplorer AI — Africa Copperbelt Edition
Streamlit frontend for Cu / Co / Ni mineral prospectivity screening
across the Central / Southern Africa Copperbelt.

Operational context: KoBold Metals Africa — Zambia, DRC Katanga, Botswana.
Deposit style     : Sediment-Hosted Stratiform Copper (SHSC) / Lufilian Arc.
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
    REGION_CENTER, humanise_features, score_to_hex,
    within_africa_study_area,
)
from app.llm_interpreter import GeoInterpreter
from app.mineral_systems import (
    FEATURE_SYSTEM_MAP, ProspectTarget, feature_component_summary,
)

st.set_page_config(
    page_title="GeoExplorer AI | Africa Copperbelt",
    page_icon="⛏",
    layout="wide",
    initial_sidebar_state="expanded",
)

KOBOLD_CSS = """
<style>
:root {
    --bg: #07130f;
    --panel: #0d1f18;
    --panel-2: #132b21;
    --line: #285342;
    --text: #edf7ef;
    --muted: #9fb6a8;
    --green: #74d680;
    --lime: #c6ff6b;
    --copper: #d8924c;
    --cobalt: #5b9cf6;
    --warning: #ff6b57;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg); color: var(--text);
    font-family: Inter, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stSidebar"] { background: #081713; border-right: 1px solid var(--line); }
[data-testid="metric-container"] {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 6px; padding: 14px 16px;
}
[data-testid="stMetricValue"] { color: var(--lime); font-size: 1.65rem; font-weight: 760; }
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.76rem; letter-spacing: 0.05em; text-transform: uppercase; }
h1, h2, h3 { color: var(--text) !important; }
h1 { font-size: 2.6rem !important; line-height: 1.02 !important; }
a { color: var(--green) !important; }
.stButton > button {
    background: var(--lime); color: #07130f;
    border: 1px solid var(--lime); border-radius: 6px; font-weight: 760;
}
.stButton > button:hover { background: var(--green); border-color: var(--green); }
.hero { border-bottom: 1px solid var(--line); padding: 10px 0 18px 0; margin-bottom: 16px; }
.eyebrow { color: var(--green); font-size: 0.78rem; font-weight: 760; letter-spacing: 0.14em; text-transform: uppercase; }
.lede { max-width: 920px; color: var(--muted); font-size: 1.04rem; line-height: 1.55; }
.score-badge { display: inline-block; padding: 7px 13px; border-radius: 6px; font-weight: 760; color: #07130f; background: var(--lime); margin-bottom: 8px; }
.tier-VH { display: inline-block; border-radius: 6px; padding: 5px 10px; font-size: 0.84rem; font-weight: 760; background: var(--warning); color: #fff; }
.tier-H  { display: inline-block; border-radius: 6px; padding: 5px 10px; font-size: 0.84rem; font-weight: 760; background: var(--copper); color: #07130f; }
.tier-M  { display: inline-block; border-radius: 6px; padding: 5px 10px; font-size: 0.84rem; font-weight: 760; background: var(--green); color: #07130f; }
.tier-L  { display: inline-block; border-radius: 6px; padding: 5px 10px; font-size: 0.84rem; font-weight: 760; background: var(--panel); color: var(--muted); border: 1px solid var(--line); }
.info-box { background: var(--panel); border-left: 4px solid var(--green); border-radius: 0 6px 6px 0; padding: 12px 15px; color: var(--muted); margin: 8px 0 14px 0; }
.cobalt-box { background: var(--panel); border-left: 4px solid var(--cobalt); border-radius: 0 6px 6px 0; padding: 12px 15px; color: var(--muted); margin: 8px 0 14px 0; }
.caption { color: var(--muted); font-size: 0.78rem; line-height: 1.45; }
hr { border-color: var(--line); }
</style>
"""
st.markdown(KOBOLD_CSS, unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "predictions.csv"
    if not path.exists():
        st.error("Prediction grid not found. Run `make phase1` or the three pipeline scripts.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_deposits() -> pd.DataFrame:
    import geopandas as gpd
    path = ROOT / "data" / "raw" / "mrds_africa_copper.geojson"
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "name", "commod1", "source"])
    gdf = gpd.read_file(path)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    cols = ["lon", "lat"] + [c for c in gdf.columns if c not in ("lon", "lat", "geometry")]
    return pd.DataFrame(gdf[cols])


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = ROOT / "models" / "prospectivity_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict:
    path = ROOT / "models" / "model_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    path = ROOT / "models" / "feature_importance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ── UI helpers ────────────────────────────────────────────────────────────────

def tier_chip(tier: str) -> str:
    cls = {"Very High": "tier-VH", "High": "tier-H", "Moderate": "tier-M", "Low": "tier-L"}
    return f'<span class="{cls.get(tier, "tier-L")}">{tier}</span>'


def score_to_tier(score: float) -> str:
    if score >= 0.70: return "Very High"
    if score >= 0.50: return "High"
    if score >= 0.30: return "Moderate"
    return "Low"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(
    meta: dict, model_available: bool
) -> tuple[float, float, bool, pd.DataFrame | None]:
    uploaded_df = None
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 14px 0;">
              <div class="eyebrow">GeoExplorer AI · Africa</div>
              <div style="font-size:1.38rem; font-weight:800; line-height:1.1; margin-top:5px;">
                Copperbelt target screen
              </div>
              <div class="caption" style="margin-top:8px;">
                Cu · Co · Ni prospectivity · Zambia, DRC Katanga, Botswana
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Query Location**")
        lat = st.number_input("Latitude (°S negative)",  value=-12.5, min_value=-28.0, max_value=0.0,   format="%.4f")
        lon = st.number_input("Longitude (°E)",          value=28.2,  min_value=15.0,  max_value=38.0,  format="%.4f")
        score_btn = st.button("Score location", use_container_width=True)

        st.divider()
        st.markdown("**Batch Samples**")
        uploaded = st.file_uploader(
            "Upload feature CSV",
            type=["csv"],
            help="Columns: elevation_m, slope_deg, log_cu_ppm, log_co_ppm, log_ni_ppm, "
                 "log_au_ppb, fe_pct, log_pb_ppm, log_zn_ppm, log_mo_ppm, log_as_ppm, "
                 "dist_fault_km, dist_deposit_km",
        )
        if uploaded:
            uploaded_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(uploaded_df):,} rows")
            st.dataframe(uploaded_df.head(4), use_container_width=True)
            if not model_available:
                st.caption("Model bundle not found — preview only.")

        st.divider()
        if meta:
            st.markdown("**Model Snapshot**")
            model_name = meta.get("best_model", "model").replace("_", " ").title()
            st.markdown(
                f'<div class="info-box">'
                f'<b>{model_name}</b><br>'
                f'ROC-AUC <b>{meta.get("roc_auc", 0):.3f}</b> spatial CV<br>'
                f'{meta.get("n_train", "?")} training samples<br>'
                f'{meta.get("grid_points", "?")} scored grid cells<br>'
                f'Commodity: {meta.get("commodity", "Cu")}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="caption">Africa Copperbelt edition. '
            'USGS MRDS deposits · GEM Active Faults · SRTM elevation · '
            'Copperbelt-calibrated geochemistry.</div>',
            unsafe_allow_html=True,
        )

    return float(lat), float(lon), score_btn, uploaded_df


# ── Prospectivity map ─────────────────────────────────────────────────────────

def render_map(
    predictions: pd.DataFrame,
    deposits: pd.DataFrame,
    query_lat: float | None = None,
    query_lon: float | None = None,
    query_score: float | None = None,
) -> None:
    m = folium.Map(
        location=[REGION_CENTER["lat"], REGION_CENTER["lon"]],
        zoom_start=5,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    if not predictions.empty:
        from folium.plugins import HeatMap
        heat_data = [
            [row.lat, row.lon, row.prospectivity_score]
            for _, row in predictions.iterrows()
            if row.prospectivity_score > 0.25
        ]
        if heat_data:
            HeatMap(
                heat_data,
                name="Prospectivity heatmap",
                min_opacity=0.22, max_val=0.95,
                radius=22, blur=28,
                gradient={
                    0.20: "#10251d",
                    0.40: "#2d6a4f",
                    0.55: "#74d680",
                    0.68: "#5b9cf6",   # cobalt blue tier
                    0.80: "#d8924c",
                    0.92: "#fff2b2",
                },
            ).add_to(m)

    if not deposits.empty:
        deposit_layer = folium.FeatureGroup(name="Cu/Co/Ni deposit training points")
        for _, dep in deposits.head(700).iterrows():
            commod = dep.get("commod1", "Cu")
            colour = "#5b9cf6" if "Co" in str(commod) else "#c6ff6b" if "Ni" not in str(commod) else "#f5a623"
            folium.CircleMarker(
                location=[dep.lat, dep.lon],
                radius=4, color=colour, fill=True,
                fill_color=colour, fill_opacity=0.82, weight=1,
                tooltip=f"{dep.get('name', dep.get('district', 'deposit'))} ({commod})",
            ).add_to(deposit_layer)
        deposit_layer.add_to(m)

    if query_lat and query_lon:
        score_str = f"{query_score:.2f}" if query_score is not None else "pending"
        folium.Marker(
            location=[query_lat, query_lon],
            popup=f"<b>Score: {score_str}</b>",
            tooltip=f"Query — score {score_str}",
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=580, returned_objects=[])


# ── Site scorer panel ─────────────────────────────────────────────────────────

def render_score_panel(lat: float, lon: float, predictions: pd.DataFrame) -> None:
    from app.geo_utils import score_point
    with st.spinner("Scoring location …"):
        result = score_point(lat, lon, predictions)

    score    = result["score"]
    tier     = result["risk_tier"]
    features = result["features"]

    st.markdown(
        f'<div class="score-badge">Score: {score:.3f}</div>&nbsp;{tier_chip(tier)}',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prospectivity", f"{score:.2%}")
    c2.metric("Tier", tier)
    c3.metric("Nearest Grid", f"{result['nearest_grid_distance_km']} km")

    # Co/Cu ratio display — diagnostic for DRC Katanga type
    log_co = features.get("log_co_ppm", 0)
    log_cu = features.get("log_cu_ppm", 1)
    co_cu  = round((np.expm1(log_co) / max(np.expm1(log_cu), 1)) * 100, 1)
    c4.metric("Co/Cu ratio", f"{co_cu:.1f} %", help="≥ 10 % is consistent with DRC Katanga-style Cu-Co SHSC mineralisation")

    # Mineral system evidence
    target = ProspectTarget(
        target_id="QUERY", lat=lat, lon=lon,
        prospectivity_score=score, risk_tier=tier,
        uncertainty_proxy=0.0, features=features,
    )
    ev = target.mineral_system_evidence
    if any(ev.values()):
        st.markdown("#### Mineral System Evidence")
        for comp, items in ev.items():
            if items:
                st.markdown(f"**{comp.title()}:** " + " · ".join(items))

    st.markdown("#### Geological Interpretation")
    with st.spinner("Generating interpretation …"):
        interpretation = GeoInterpreter().interpret_score(lat, lon, score, features)
    st.markdown(interpretation)

    st.markdown("#### Feature Breakdown")
    human_feats = humanise_features(features)
    if human_feats:
        fig = px.bar(
            pd.DataFrame(human_feats), x="value", y="label",
            orientation="h", text="value", color="value",
            color_continuous_scale=["#10251d", "#74d680", "#c6ff6b"],
            title="Site feature values", template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18",
            font_color="#edf7ef", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=42, b=0),
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ── Batch scoring ─────────────────────────────────────────────────────────────

def render_uploaded_scores(uploaded_df: pd.DataFrame | None, model_bundle) -> None:
    if uploaded_df is None:
        st.markdown(
            '<div class="info-box">Upload a CSV with the trained model feature columns '
            'to score a batch of sample locations.</div>',
            unsafe_allow_html=True,
        )
        return
    if model_bundle is None:
        st.warning("Model bundle not available. Run the pipeline to generate it.")
        return

    feature_cols = model_bundle.get("feature_cols", [])
    missing = [c for c in feature_cols if c not in uploaded_df.columns]
    if missing:
        st.warning("Missing feature columns: " + ", ".join(missing))
        return

    scored = uploaded_df.copy()
    proba  = model_bundle["pipeline"].predict_proba(scored[feature_cols])[:, 1]
    scored["prospectivity_score"] = np.round(proba, 4)
    scored["risk_tier"]           = scored["prospectivity_score"].apply(score_to_tier)
    st.dataframe(scored.sort_values("prospectivity_score", ascending=False), use_container_width=True)


# ── Target portfolio ──────────────────────────────────────────────────────────

def render_target_portfolio(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        st.warning("Prediction grid unavailable. Run the pipeline.")
        return

    target_cols  = ["lat", "lon", "prospectivity_score", "risk_tier"]
    feature_cols = [
        c for c in ["elevation_m", "log_cu_ppm", "log_co_ppm", "log_ni_ppm",
                     "dist_fault_km", "dist_deposit_km"]
        if c in predictions.columns
    ]
    targets = predictions.nlargest(20, "prospectivity_score")[target_cols + feature_cols].copy()
    targets["target_id"] = [f"AFR-{i:03d}" for i in range(1, len(targets) + 1)]

    targets["uncertainty_proxy"] = (
        0.15
        + 0.35 * (targets.get("dist_deposit_km", 50).clip(0, 80) / 80)
        + 0.15 * (targets.get("dist_fault_km",   50).clip(0, 50) / 50)
    ).round(3)

    targets["field_program"] = np.where(
        targets["uncertainty_proxy"] >= 0.45,
        "Recon mapping + infill soil geochem",
        "Priority mapping + IP geophysics + rock sampling",
    )
    targets["decision"] = np.where(
        targets["prospectivity_score"] >= 0.70, "Advance", "Hold for data"
    )

    # Co/Cu ratio (diagnostic for SHSC type)
    if "log_co_ppm" in targets.columns and "log_cu_ppm" in targets.columns:
        targets["co_cu_pct"] = (
            np.expm1(targets["log_co_ppm"]) /
            np.expm1(targets["log_cu_ppm"]).clip(lower=1)
        * 100).round(1)

    st.markdown(
        '<div class="cobalt-box">'
        'Top-20 grid cells ranked by prospectivity score. '
        'Co/Cu % > 10 is consistent with DRC Katanga-style Cu-Co SHSC mineralisation. '
        'Uncertainty proxy drives the recommended field programme.</div>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "target_id", "lat", "lon", "prospectivity_score", "risk_tier",
        "co_cu_pct" if "co_cu_pct" in targets.columns else None,
        "uncertainty_proxy", "field_program", "decision",
    ]
    display_cols = [c for c in display_cols if c]
    st.dataframe(targets[display_cols], use_container_width=True)

    fig = px.scatter(
        targets, x="uncertainty_proxy", y="prospectivity_score",
        size="prospectivity_score", color="decision",
        hover_name="target_id",
        hover_data={"lat": True, "lon": True, "field_program": True},
        title="Target value vs uncertainty (Africa Copperbelt)",
        template="plotly_dark",
        color_discrete_map={"Advance": "#c6ff6b", "Hold for data": "#d8924c"},
    )
    fig.update_layout(
        paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18",
        font_color="#edf7ef",
        xaxis_title="Uncertainty proxy", yaxis_title="Prospectivity score",
        margin=dict(l=0, r=0, t=42, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Semantic model panel ──────────────────────────────────────────────────────

def render_mineral_system_panel() -> None:
    st.markdown("### Mineral Systems Semantic Model — Africa Copperbelt")
    st.markdown(
        '<div class="cobalt-box">'
        'Each ML feature is mapped to a component of the SHSC mineral systems '
        'framework (Source → Pathway → Trap → Modifier). This grounds model '
        'outputs in geological theory rather than treating them as opaque numbers.'
        '</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for feat, info in FEATURE_SYSTEM_MAP.items():
        rows.append({
            "Feature":    feat,
            "Component":  info["component"].value.title(),
            "Styles":     ", ".join(s.value for s in info.get("styles", [])),
            "Description": info.get("description", ""),
            "Threshold":  str(info.get("anomaly_threshold_log", "—")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── Role capabilities ─────────────────────────────────────────────────────────

def render_role_capabilities(meta: dict) -> None:
    st.markdown("### Role-aligned capabilities — Africa Copperbelt")
    rows = [
        ("Geoscience data system",
         "Curates USGS MRDS deposits (global), GEM Active Faults, SRTM elevation, "
         "and Copperbelt-calibrated geochemistry. Cu + Co + Ni feature suite."),
        ("SHSC mineral systems framework",
         "Semantic model maps each ML feature to Source / Pathway / Trap / Modifier "
         "components of the Lufilian Arc SHSC framework (mineral_systems.py)."),
        ("Co/Cu ratio diagnostic",
         "Co/Cu % displayed per target — key distinguisher of DRC Katanga-type "
         "Cu-Co mineralisation vs background Cu."),
        ("Spatial prospectivity",
         "Scores a 0.2 ° grid (≈ 22 km) across Zambia, DRC Katanga, Botswana, "
         "Zimbabwe, Mozambique, Namibia using spatially cross-validated ML."),
        ("Uncertainty-driven field prioritisation",
         "Target Portfolio ranks locations by score + uncertainty proxy and recommends "
         "IP geophysics, soil sampling, or geological mapping."),
        ("LLM geological interpretation",
         "HuggingFace-powered narrative grounded in SHSC geology — not porphyry. "
         "References Co/Cu ratio, Lufilian Arc structure, and redox trap context."),
        ("Cloud scoring API",
         "Cloudflare Worker endpoint for coordinate scoring and LLM interpretation. "
         "Extendable to real-time KV-backed grid lookup."),
        ("Software engineering practice",
         "Git, pytest, GitHub Actions CI, modular pipeline scripts, documented "
         "deployment, and reproducible synthetic data with published references."),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["KoBold requirement", "Project evidence"]),
        use_container_width=True,
    )
    if meta:
        st.caption(
            f"Model: {meta.get('best_model','—').replace('_',' ').title()} · "
            f"ROC-AUC {meta.get('roc_auc', 0):.3f} (spatial CV) · "
            f"{meta.get('n_train', 0):,} training rows · "
            f"Region: {meta.get('region', 'Africa Copperbelt')} · "
            f"Commodity: {meta.get('commodity', 'Cu · Co · Ni')}"
        )


# ── Model analytics ───────────────────────────────────────────────────────────

def render_model_comparison() -> None:
    comp_path = ROOT / "models" / "model_comparison.json"
    if not comp_path.exists():
        return
    results = json.loads(comp_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(results)
    df["model"] = df["name"].str.replace("_", " ").str.title()

    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=df.model, y=df.roc_auc_mean,
                error_y=dict(type="data", array=df.roc_auc_std.tolist()),
                marker_color="#c6ff6b")
    fig.add_bar(name="PR-AUC",  x=df.model, y=df.pr_auc_mean, marker_color="#5b9cf6")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18",
        font_color="#edf7ef", barmode="group",
        title="Model comparison — spatial CV (Africa)",
        yaxis=dict(range=[0, 1], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(fi: dict) -> None:
    imp     = fi.get("model_importance", {}) if fi else {}
    display = fi.get("feature_display_names", {}) if fi else {}
    if not imp:
        return
    df = pd.DataFrame(
        [{"feature": display.get(k, k), "importance": v} for k, v in imp.items()]
    ).sort_values("importance", ascending=True)

    # Colour cobalt and nickel bars distinctly
    colours = ["#5b9cf6" if "Cobalt" in r["feature"] or "Nickel" in r["feature"]
               else "#74d680" for _, r in df.iterrows()]

    fig = go.Figure(go.Bar(
        x=df.importance, y=df.feature, orientation="h",
        marker_color=colours, text=df.importance.round(3),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18",
        font_color="#edf7ef", title="Feature importance (Co/Ni highlighted in blue)",
        margin=dict(l=0, r=0, t=42, b=0),
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution(predictions: pd.DataFrame) -> None:
    if predictions.empty or "prospectivity_score" not in predictions.columns:
        return
    fig = px.histogram(
        predictions, x="prospectivity_score", nbins=50,
        template="plotly_dark", title="Prospectivity score distribution (Africa grid)",
        color_discrete_sequence=["#74d680"],
    )
    fig.update_layout(
        paper_bgcolor="#0d1f18", plot_bgcolor="#0d1f18", font_color="#edf7ef",
        bargap=0.05, margin=dict(l=0, r=0, t=42, b=0),
        xaxis_title="Prospectivity score", yaxis_title="Grid cells",
    )
    fig.add_vline(x=0.5,  line_color="#d8924c", line_dash="dash", annotation_text="High")
    fig.add_vline(x=0.70, line_color="#ff6b57", line_dash="dash", annotation_text="Very High")
    st.plotly_chart(fig, use_container_width=True)


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
          <div class="eyebrow">AI-assisted mineral exploration · Africa</div>
          <h1>Copper · Cobalt · Nickel targets across the Copperbelt.</h1>
          <p class="lede">
            GeoExplorer AI screens the Central and Southern African Copperbelt
            for Cu-Co-Ni prospectivity using geochemical, structural, terrain,
            and proximity features — grounded in the Sediment-Hosted Stratiform
            Copper (SHSC) mineral systems framework of the Lufilian Arc.
            Covering Zambia, DRC Katanga, Botswana, Zimbabwe, Mozambique, and
            Namibia. Not a reserve estimate or regulatory disclosure.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not predictions.empty and meta:
        k1, k2, k3, k4, k5 = st.columns(5)
        n_high   = int((predictions["prospectivity_score"] >= 0.5).sum())
        n_v_high = int((predictions["prospectivity_score"] >= 0.7).sum())
        k1.metric("Grid Cells Scored",  f"{len(predictions):,}")
        k2.metric("High Potential",     f"{n_high:,}")
        k3.metric("Very High Potential",f"{n_v_high:,}")
        k4.metric("Training Points",    f"{meta.get('n_train', 0):,}")
        k5.metric("Best ROC-AUC",       f"{meta.get('roc_auc', 0):.3f}")

    tabs = st.tabs([
        "Prospectivity Map",
        "Target Portfolio",
        "Site Scorer",
        "Batch Scoring",
        "Mineral System",
        "Model Analytics",
        "Role Fit",
        "API",
    ])

    with tabs[0]:
        st.markdown(
            '<div class="info-box">Heatmap of precomputed Cu/Co/Ni prospectivity scores. '
            'Green markers = Cu deposits · Blue markers = Cu-Co deposits · '
            'Orange markers = Cu-Ni deposits.</div>',
            unsafe_allow_html=True,
        )
        render_map(predictions, deposits, query_lat, query_lon)

    with tabs[1]:
        render_target_portfolio(predictions)

    with tabs[2]:
        if score_btn or st.session_state.get("scored"):
            if not within_africa_study_area(query_lat, query_lon):
                st.warning("Coordinates are outside the Africa Copperbelt study area "
                           "(15 °E – 38 °E, 28 °S – 0 °).")
            elif predictions.empty:
                st.error("No predictions loaded. Run the pipeline first.")
            else:
                st.session_state["scored"] = True
                st.markdown(f"**Site:** {abs(query_lat):.4f} °S, {query_lon:.4f} °E")
                render_score_panel(query_lat, query_lon, predictions)
        else:
            st.markdown(
                '<div class="info-box">Enter coordinates in the sidebar and click '
                '<b>Score location</b> for an AI-assisted geological assessment.</div>',
                unsafe_allow_html=True,
            )
            if not predictions.empty:
                best = predictions.nlargest(1, "prospectivity_score").iloc[0]
                st.info(
                    f"Highest-scoring grid cell: {abs(best.lat):.3f} °S, "
                    f"{best.lon:.3f} °E — score {best.prospectivity_score:.3f}."
                )

    with tabs[3]:
        render_uploaded_scores(uploaded_df, model_bundle)

    with tabs[4]:
        render_mineral_system_panel()

    with tabs[5]:
        col1, col2 = st.columns(2)
        with col1:
            render_model_comparison()
        with col2:
            render_feature_importance(fi)
        render_score_distribution(predictions)

    with tabs[6]:
        render_role_capabilities(meta)

    with tabs[7]:
        st.markdown("### Cloudflare Workers API — Africa Copperbelt")
        st.markdown(
            """
            `POST /score`

            ```json
            {
              "lat": -12.37,
              "lon":  27.85,
              "features": { "cu_ppm": 850, "co_ppm": 120 },
              "interpret": true
            }
            ```

            Returns prospectivity score, risk tier, and (optionally) an
            LLM-generated geological interpretation grounded in SHSC geology.
            """
        )


if __name__ == "__main__":
    main()
