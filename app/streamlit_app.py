"""
GeoExplorer AI Streamlit frontend.

Interactive copper prospectivity mapping for Arizona and Nevada. The visual
direction is inspired by KoBold Metals' science-forward public site, without
using official KoBold assets or implying endorsement.
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

from app.geo_utils import REGION_CENTER, humanise_features, score_to_hex, within_az_nv
from app.llm_interpreter import GeoInterpreter


st.set_page_config(
    page_title="GeoExplorer AI | Copper Prospectivity",
    page_icon="GE",
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
    --warning: #ff6b57;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--text);
    font-family: Inter, "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stSidebar"] {
    background: #081713;
    border-right: 1px solid var(--line);
}
[data-testid="metric-container"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 14px 16px;
}
[data-testid="stMetricValue"] {
    color: var(--lime);
    font-size: 1.65rem;
    font-weight: 760;
}
[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.76rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
h1, h2, h3 { color: var(--text) !important; letter-spacing: 0; }
h1 { font-size: 2.6rem !important; line-height: 1.02 !important; }
a { color: var(--green) !important; }
.stButton > button {
    background: var(--lime);
    color: #07130f;
    border: 1px solid var(--lime);
    border-radius: 6px;
    font-weight: 760;
}
.stButton > button:hover {
    background: var(--green);
    border-color: var(--green);
}
.hero {
    border-bottom: 1px solid var(--line);
    padding: 10px 0 18px 0;
    margin-bottom: 16px;
}
.eyebrow {
    color: var(--green);
    font-size: 0.78rem;
    font-weight: 760;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.lede {
    max-width: 920px;
    color: var(--muted);
    font-size: 1.04rem;
    line-height: 1.55;
}
.score-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 6px;
    font-weight: 760;
    color: #07130f;
    background: var(--lime);
    margin-bottom: 8px;
}
.tier-VH, .tier-H, .tier-M, .tier-L {
    display: inline-block;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 0.84rem;
    font-weight: 760;
}
.tier-VH { background: var(--warning); color: #fff; }
.tier-H  { background: var(--copper); color: #07130f; }
.tier-M  { background: var(--green); color: #07130f; }
.tier-L  { background: var(--panel); color: var(--muted); border: 1px solid var(--line); }
.info-box {
    background: var(--panel);
    border-left: 4px solid var(--green);
    border-radius: 0 6px 6px 0;
    padding: 12px 15px;
    color: var(--muted);
    margin: 8px 0 14px 0;
}
.caption {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.45;
}
hr { border-color: var(--line); }
</style>
"""
st.markdown(KOBOLD_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "predictions.csv"
    if not path.exists():
        st.error("Prediction grid not found. Run `make phase1` first.")
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_deposits() -> pd.DataFrame:
    import geopandas as gpd

    path = ROOT / "data" / "raw" / "mrds_copper_az_nv.geojson"
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "name", "source"])
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_importance() -> dict:
    path = ROOT / "models" / "feature_importance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def tier_chip(tier: str) -> str:
    cls = {"Very High": "tier-VH", "High": "tier-H", "Moderate": "tier-M", "Low": "tier-L"}
    return f'<span class="{cls.get(tier, "tier-L")}">{tier}</span>'


def score_to_tier(score: float) -> str:
    if score >= 0.70:
        return "Very High"
    if score >= 0.50:
        return "High"
    if score >= 0.30:
        return "Moderate"
    return "Low"


def render_sidebar(meta: dict, model_available: bool) -> tuple[float, float, bool, pd.DataFrame | None]:
    uploaded_df = None
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 4px 0 14px 0;">
              <div class="eyebrow">GeoExplorer AI</div>
              <div style="font-size:1.38rem; font-weight:800; line-height:1.1; margin-top:5px;">
                Critical minerals target screen
              </div>
              <div class="caption" style="margin-top:8px;">
                Copper prospectivity across Arizona and Nevada.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Query Location**")
        lat = st.number_input("Latitude", value=33.45, min_value=31.0, max_value=42.0, format="%.4f")
        lon = st.number_input("Longitude", value=-110.80, min_value=-120.0, max_value=-109.0, format="%.4f")
        score_btn = st.button("Score location", use_container_width=True)

        st.divider()
        st.markdown("**Batch Samples**")
        uploaded = st.file_uploader(
            "Upload feature CSV",
            type=["csv"],
            help="If the trained model bundle is present, rows with model feature columns will be scored.",
        )
        if uploaded:
            uploaded_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(uploaded_df):,} rows")
            st.dataframe(uploaded_df.head(4), use_container_width=True)
            if not model_available:
                st.caption("Model bundle not found, so uploaded rows are previewed only.")

        st.divider()
        if meta:
            model_name = meta.get("best_model", "model").replace("_", " ").title()
            st.markdown("**Model Snapshot**")
            st.markdown(
                f'<div class="info-box">'
                f'<b>{model_name}</b><br>'
                f'ROC-AUC <b>{meta.get("roc_auc", 0):.3f}</b> using spatial folds<br>'
                f'{meta.get("n_train", "?")} training samples<br>'
                f'{meta.get("grid_points", "?")} scored grid cells'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="caption">Demo system. Current checked-in artifacts may be synthetic fallbacks '
            'when public source endpoints are unavailable.</div>',
            unsafe_allow_html=True,
        )

    return float(lat), float(lon), score_btn, uploaded_df


def render_map(
    predictions: pd.DataFrame,
    deposits: pd.DataFrame,
    query_lat: float | None = None,
    query_lon: float | None = None,
    query_score: float | None = None,
) -> None:
    m = folium.Map(
        location=[REGION_CENTER["lat"], REGION_CENTER["lon"]],
        zoom_start=6,
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
                min_opacity=0.22,
                max_val=0.95,
                radius=18,
                blur=24,
                gradient={
                    0.20: "#10251d",
                    0.40: "#2d6a4f",
                    0.58: "#74d680",
                    0.72: "#d8924c",
                    0.88: "#fff2b2",
                },
            ).add_to(m)

    if not deposits.empty:
        deposit_layer = folium.FeatureGroup(name="Deposit training points")
        for _, dep in deposits.head(600).iterrows():
            source = dep.get("source", "unknown")
            folium.CircleMarker(
                location=[dep.lat, dep.lon],
                radius=4,
                color="#c6ff6b",
                fill=True,
                fill_color="#c6ff6b",
                fill_opacity=0.82,
                weight=1,
                tooltip=f"{dep.get('name', 'Copper deposit')} ({source})",
            ).add_to(deposit_layer)
        deposit_layer.add_to(m)

    if query_lat and query_lon:
        score_str = f"{query_score:.2f}" if query_score is not None else "pending"
        folium.Marker(
            location=[query_lat, query_lon],
            popup=f"<b>Score: {score_str}</b>",
            tooltip=f"Query point - score {score_str}",
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width="100%", height=560, returned_objects=[])


def render_score_panel(lat: float, lon: float, predictions: pd.DataFrame) -> None:
    from app.geo_utils import score_point

    with st.spinner("Scoring location..."):
        result = score_point(lat, lon, predictions)

    score = result["score"]
    tier = result["risk_tier"]
    features = result["features"]

    st.markdown(
        f'<div class="score-badge">Score: {score:.3f}</div>&nbsp;{tier_chip(tier)}',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].metric("Prospectivity", f"{score:.2%}")
    cols[1].metric("Tier", tier)
    cols[2].metric("Nearest Grid", f"{result['nearest_grid_distance_km']} km")

    st.markdown("#### Geological Interpretation")
    with st.spinner("Generating interpretation..."):
        interpretation = GeoInterpreter().interpret_score(lat, lon, score, features)
    st.markdown(interpretation)

    st.markdown("#### Feature Breakdown")
    human_feats = humanise_features(features)
    if human_feats:
        fig = px.bar(
            pd.DataFrame(human_feats),
            x="value",
            y="label",
            orientation="h",
            text="value",
            color="value",
            color_continuous_scale=["#10251d", "#74d680", "#c6ff6b"],
            title="Site feature values",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0d1f18",
            plot_bgcolor="#0d1f18",
            font_color="#edf7ef",
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=42, b=0),
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def render_uploaded_scores(uploaded_df: pd.DataFrame | None, model_bundle: dict | None) -> None:
    if uploaded_df is None:
        st.markdown(
            '<div class="info-box">Upload a CSV in the sidebar to score rows with the trained model bundle. '
            'Required columns are shown in the model metadata.</div>',
            unsafe_allow_html=True,
        )
        return
    if model_bundle is None:
        st.warning("Model bundle is not available. The uploaded data can be previewed, but not scored.")
        return

    feature_cols = model_bundle.get("feature_cols", [])
    missing = [c for c in feature_cols if c not in uploaded_df.columns]
    if missing:
        st.warning("Uploaded file is missing required feature columns: " + ", ".join(missing))
        return

    scored = uploaded_df.copy()
    proba = model_bundle["pipeline"].predict_proba(scored[feature_cols])[:, 1]
    scored["prospectivity_score"] = np.round(proba, 4)
    scored["risk_tier"] = scored["prospectivity_score"].apply(score_to_tier)
    st.dataframe(scored.sort_values("prospectivity_score", ascending=False), use_container_width=True)


def render_model_comparison() -> None:
    comp_path = ROOT / "models" / "model_comparison.json"
    if not comp_path.exists():
        return
    results = json.loads(comp_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(results)
    df["model"] = df["name"].str.replace("_", " ").str.title()

    fig = go.Figure()
    fig.add_bar(
        name="ROC-AUC",
        x=df.model,
        y=df.roc_auc_mean,
        error_y=dict(type="data", array=df.roc_auc_std.tolist()),
        marker_color="#c6ff6b",
    )
    fig.add_bar(name="PR-AUC", x=df.model, y=df.pr_auc_mean, marker_color="#d8924c")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1f18",
        plot_bgcolor="#0d1f18",
        font_color="#edf7ef",
        barmode="group",
        title="Model comparison - spatial cross-validation",
        yaxis=dict(range=[0, 1], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(fi: dict) -> None:
    imp = fi.get("model_importance", {}) if fi else {}
    display = fi.get("feature_display_names", {}) if fi else {}
    if not imp:
        return

    df = pd.DataFrame(
        [{"feature": display.get(k, k), "importance": v} for k, v in imp.items()]
    ).sort_values("importance", ascending=True)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#10251d", "#74d680", "#c6ff6b"],
        template="plotly_dark",
        title="Feature importance",
        text="importance",
    )
    fig.update_layout(
        paper_bgcolor="#0d1f18",
        plot_bgcolor="#0d1f18",
        font_color="#edf7ef",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=42, b=0),
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution(predictions: pd.DataFrame) -> None:
    if predictions.empty or "prospectivity_score" not in predictions.columns:
        return
    fig = px.histogram(
        predictions,
        x="prospectivity_score",
        nbins=50,
        template="plotly_dark",
        title="Prospectivity score distribution",
        color_discrete_sequence=["#74d680"],
    )
    fig.update_layout(
        paper_bgcolor="#0d1f18",
        plot_bgcolor="#0d1f18",
        font_color="#edf7ef",
        bargap=0.05,
        margin=dict(l=0, r=0, t=42, b=0),
        xaxis_title="Prospectivity score",
        yaxis_title="Grid cells",
    )
    fig.add_vline(x=0.5, line_color="#d8924c", line_dash="dash", annotation_text="High threshold")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    predictions = load_predictions()
    deposits = load_deposits()
    meta = load_model_metadata()
    fi = load_feature_importance()
    model_bundle = load_model_bundle()

    query_lat, query_lon, score_btn, uploaded_df = render_sidebar(meta, model_bundle is not None)

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">AI-assisted mineral exploration</div>
          <h1>Finding copper signals with data, models, and geology.</h1>
          <p class="lede">
            GeoExplorer AI screens Arizona and Nevada for copper prospectivity using geochemical,
            structural, terrain, and proximity features. It is a portfolio-grade exploration
            decision support tool, not a reserve estimate or regulatory disclosure.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not predictions.empty and meta:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        n_high = int((predictions["prospectivity_score"] >= 0.5).sum())
        kpi1.metric("Grid Cells Scored", f"{len(predictions):,}")
        kpi2.metric("High Potential", f"{n_high:,}")
        kpi3.metric("Training Points", f"{meta.get('n_train', 0):,}")
        kpi4.metric("Best ROC-AUC", f"{meta.get('roc_auc', 0):.3f}")

    tab_map, tab_score, tab_batch, tab_model, tab_api = st.tabs(
        ["Prospectivity Map", "Site Scorer", "Batch Scoring", "Model Analytics", "API"]
    )

    with tab_map:
        st.markdown(
            '<div class="info-box">The heatmap visualizes precomputed grid scores. '
            'Deposit markers show the training/source points available in this local artifact set.</div>',
            unsafe_allow_html=True,
        )
        render_map(predictions, deposits, query_lat, query_lon)

    with tab_score:
        if score_btn or st.session_state.get("scored"):
            if not within_az_nv(query_lat, query_lon):
                st.warning("Coordinates are outside the AZ/NV study area.")
            elif predictions.empty:
                st.error("No predictions loaded. Run `make phase1` first.")
            else:
                st.session_state["scored"] = True
                st.markdown(f"**Site:** {query_lat:.4f} N, {query_lon:.4f} W")
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
                    f"Highest-scoring grid cell: {best.lat:.3f} N, {best.lon:.3f} W "
                    f"with score {best.prospectivity_score:.3f}."
                )

    with tab_batch:
        render_uploaded_scores(uploaded_df, model_bundle)

    with tab_model:
        col1, col2 = st.columns(2)
        with col1:
            render_model_comparison()
        with col2:
            render_feature_importance(fi)
        render_score_distribution(predictions)

    with tab_api:
        st.markdown("### Cloudflare Workers API")
        st.markdown(
            """
            The Worker exposes demo endpoints for coordinate scoring and LLM interpretation.
            KV-backed grid lookup requires a real Cloudflare KV namespace; without KV, local
            worker development returns a deterministic mock score.

            `POST /score`

            ```json
            {
              "lat": 33.45,
              "lon": -110.80,
              "features": { "cu_ppm": 450 },
              "interpret": true
            }
            ```
            """
        )


if __name__ == "__main__":
    main()
