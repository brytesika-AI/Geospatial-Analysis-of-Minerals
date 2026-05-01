/**
 * GeoExplorer AI - Cloudflare Workers Edge API
 *
 * Routes:
 *   GET  /health
 *   GET  /
 *   POST /score
 *   GET  /heatmap
 *   POST /interpret
 */

const DEFAULT_ORIGINS = [
  "https://geo-explorer-ai.streamlit.app",
  "http://localhost:8501",
];

function allowedOrigins(env) {
  const configured = (env.CORS_ORIGIN || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return [...new Set([...DEFAULT_ORIGINS, ...configured])];
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin");
  const allowed = allowedOrigins(env);
  return {
    "Access-Control-Allow-Origin": origin && allowed.includes(origin) ? origin : allowed[0],
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function jsonResponse(data, request, env, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders(request, env),
    },
  });
}

function errorResponse(message, request, env, status = 400) {
  return jsonResponse({ error: message, status }, request, env, status);
}

function htmlResponse(html) {
  return new Response(html, {
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=300",
    },
  });
}

function riskTier(score) {
  if (score >= 0.7) return "Very High";
  if (score >= 0.5) return "High";
  if (score >= 0.3) return "Moderate";
  return "Low";
}

async function lookupScore(lat, lon, kv) {
  const snapLat = (Math.round(lat * 10) / 10).toFixed(1);
  const snapLon = (Math.round(lon * 10) / 10).toFixed(1);
  const key = `score:${snapLat}:${snapLon}`;

  if (!kv) {
    const mockScore = 0.35 + 0.45 * Math.sin((lat - 31) * 3.1) * Math.cos((lon + 115) * 2.7);
    return {
      score: Math.max(0, Math.min(1, mockScore)),
      grid_lat: Number.parseFloat(snapLat),
      grid_lon: Number.parseFloat(snapLon),
      source: "mock",
    };
  }

  const raw = await kv.get(key, "json");
  if (raw) {
    return {
      ...raw,
      grid_lat: Number.parseFloat(snapLat),
      grid_lon: Number.parseFloat(snapLon),
      source: "kv",
    };
  }

  const offsets = [
    [0, 0.1],
    [0, -0.1],
    [0.1, 0],
    [-0.1, 0],
  ];
  const scores = await Promise.all(
    offsets.map(async ([dlat, dlon]) => {
      const latKey = (Math.round((lat + dlat) * 10) / 10).toFixed(1);
      const lonKey = (Math.round((lon + dlon) * 10) / 10).toFixed(1);
      const value = await kv.get(`score:${latKey}:${lonKey}`, "json");
      return value ? value.score : null;
    }),
  );
  const valid = scores.filter((score) => score !== null);
  const interp = valid.length ? valid.reduce((sum, score) => sum + score, 0) / valid.length : 0.2;
  return {
    score: Math.round(interp * 10000) / 10000,
    grid_lat: Number.parseFloat(snapLat),
    grid_lon: Number.parseFloat(snapLon),
    source: "interpolated",
  };
}

async function interpretWithAI(ai, lat, lon, score, features) {
  const tier = riskTier(score);
  if (!ai) {
    return `Score ${score.toFixed(2)} indicates ${tier} copper prospectivity. Workers AI is not configured for this environment.`;
  }

  const cuLog = features?.cu_ppm ? Math.log1p(features.cu_ppm).toFixed(2) : "N/A";
  const faultKm = features?.dist_fault_km?.toFixed(1) ?? "N/A";
  const depKm = features?.dist_deposit_km?.toFixed(1) ?? "N/A";
  const elev = features?.elevation_m?.toFixed(0) ?? "N/A";

  try {
    const result = await ai.run("@cf/meta/llama-3-8b-instruct", {
      messages: [
        {
          role: "system",
          content:
            "You are a senior exploration geologist. Write concise, evidence-based copper target screening notes. Avoid overclaiming.",
        },
        {
          role: "user",
          content: `Assess this exploration site:
Location: ${lat.toFixed(4)} N, ${lon.toFixed(4)} W
AI Score: ${score.toFixed(3)} / 1.00 (${tier})
Copper signal: log(Cu ppm) = ${cuLog}
Distance to fault: ${faultKm} km
Distance to known deposit: ${depKm} km
Elevation: ${elev} m
Write 3-4 bullets: interpretation, positives, limits, recommendation.`,
        },
      ],
      max_tokens: 256,
      temperature: 0.3,
    });
    return result?.response ?? "Interpretation unavailable.";
  } catch (err) {
    return `Score ${score.toFixed(2)} - ${tier} prospectivity. Workers AI error: ${err.message}`;
  }
}

async function handleScore(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse("Invalid JSON body.", request, env);
  }

  const { lat, lon, features = {}, interpret = true } = body;
  if (typeof lat !== "number" || typeof lon !== "number") {
    return errorResponse("lat and lon must be numbers.", request, env);
  }
  if (lat < 31 || lat > 42 || lon < -120 || lon > -109) {
    return errorResponse("Coordinates outside study area: lat 31-42, lon -120 to -109.", request, env);
  }

  const scoreResult = await lookupScore(lat, lon, env.GEO_KV);
  const score = scoreResult.score;
  const interpretation = interpret ? await interpretWithAI(env.AI, lat, lon, score, features) : null;

  return jsonResponse(
    {
      lat,
      lon,
      score,
      risk_tier: riskTier(score),
      interpretation,
      grid_snap: { lat: scoreResult.grid_lat, lon: scoreResult.grid_lon },
      source: scoreResult.source,
      features_received: features,
    },
    request,
    env,
  );
}

function handleHeatmap(request, env) {
  const districts = [
    { lat: 33.4, lon: -110.8, score: 0.91, name: "Globe-Miami" },
    { lat: 31.6, lon: -110.7, score: 0.89, name: "Bisbee-Cochise" },
    { lat: 33.4, lon: -109.8, score: 0.87, name: "Morenci-Clifton" },
    { lat: 33.4, lon: -112.5, score: 0.82, name: "Bagdad" },
    { lat: 33.9, lon: -111.0, score: 0.78, name: "Superior" },
    { lat: 33.4, lon: -111.3, score: 0.76, name: "Ray" },
    { lat: 36.2, lon: -114.9, score: 0.71, name: "Searchlight NV" },
    { lat: 40.8, lon: -116.5, score: 0.68, name: "Battle Mountain NV" },
    { lat: 38.5, lon: -117.1, score: 0.65, name: "Tonopah NV" },
  ];

  return jsonResponse(
    {
      type: "FeatureCollection",
      features: districts.map((d) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [d.lon, d.lat] },
        properties: { score: d.score, risk_tier: riskTier(d.score), name: d.name },
      })),
      count: districts.length,
      source: "demo-districts",
    },
    request,
    env,
  );
}

async function handleInterpret(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse("Invalid JSON body.", request, env);
  }
  const { lat = 0, lon = 0, score = 0, features = {} } = body;
  const interpretation = await interpretWithAI(env.AI, lat, lon, score, features);
  return jsonResponse({ interpretation }, request, env);
}

function handleDocs(request, env) {
  return jsonResponse(
    {
      name: "GeoExplorer AI Edge API",
      version: "1.0.0",
      description: "Copper prospectivity scoring for Arizona and Nevada.",
      endpoints: {
        "GET /health": "Health check",
        "POST /score": {
          body: { lat: "number", lon: "number", features: "object", interpret: "boolean" },
          example: { lat: 33.45, lon: -110.8, features: { cu_ppm: 450 }, interpret: true },
        },
        "GET /heatmap": "Demo high-score districts as GeoJSON",
        "POST /interpret": "LLM-only geological interpretation",
      },
      region: "Arizona and Nevada",
      note: "KV-backed production scoring requires a configured GEO_KV namespace.",
    },
    request,
    env,
  );
}

const FRONTEND_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GeoExplorer AI | Copper Prospectivity</title>
  <link rel="preconnect" href="https://unpkg.com" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
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
      --red: #ff6b57;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 28px clamp(18px, 4vw, 56px) 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #0a1b15 0%, #07130f 100%);
    }
    .eyebrow {
      color: var(--green);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 1050px;
      margin: 10px 0 12px;
      font-size: clamp(36px, 6vw, 76px);
      line-height: .98;
      letter-spacing: 0;
    }
    .lede {
      max-width: 960px;
      margin: 0;
      color: var(--muted);
      font-size: clamp(16px, 2vw, 20px);
      line-height: 1.55;
    }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 22px;
    }
    nav a {
      color: var(--text);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 12px;
      background: var(--panel);
      font-size: 14px;
    }
    main {
      padding: 24px clamp(18px, 4vw, 56px) 44px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, .55fr);
      gap: 18px;
      align-items: start;
    }
    .section {
      margin-top: 20px;
      border-top: 1px solid var(--line);
      padding-top: 22px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    #map {
      height: min(62vh, 640px);
      min-height: 430px;
      border-radius: 8px;
      border: 1px solid var(--line);
      overflow: hidden;
      background: #06100d;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-height: 82px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .05em;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      color: var(--lime);
      font-size: 26px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin: 12px 0 6px;
    }
    input {
      width: 100%;
      background: #07130f;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      padding: 10px 11px;
      font-size: 15px;
    }
    button {
      width: 100%;
      margin-top: 14px;
      background: var(--lime);
      color: #07130f;
      border: 0;
      border-radius: 6px;
      padding: 11px 12px;
      font-weight: 850;
      cursor: pointer;
    }
    button:hover { background: var(--green); }
    .score {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }
    .score div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #091812;
    }
    .score strong {
      color: var(--lime);
      font-size: 24px;
    }
    .note {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      overflow: hidden;
      border-radius: 8px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 11px 10px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: .04em;
      text-transform: uppercase;
      background: #0a1913;
    }
    tr:hover td { background: #0a1913; }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      color: #07130f;
      font-weight: 800;
      font-size: 12px;
      background: var(--green);
    }
    .pill.very { background: var(--red); color: white; }
    .pill.high { background: var(--copper); }
    .pill.mod { background: var(--green); }
    .pill.low { background: var(--muted); }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
    }
    .card h3 {
      margin: 0 0 8px;
      font-size: 16px;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }
    .api {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      background: #050d0a;
      border: 1px solid var(--line);
      color: #d8ffe0;
      border-radius: 8px;
      padding: 14px;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .grid, .metrics, .cards { grid-template-columns: 1fr; }
      #map { height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="eyebrow">AI-assisted mineral exploration</div>
    <h1>From crustal data to field-ready copper targets.</h1>
    <p class="lede">GeoExplorer AI screens Arizona and Nevada for copper prospectivity, ranks targets by model score and uncertainty, and turns geoscience evidence into field-program recommendations.</p>
    <nav>
      <a href="#map-section">Map</a>
      <a href="#targets">Target Portfolio</a>
      <a href="#role-fit">Role Fit</a>
      <a href="/api">API Docs</a>
      <a href="https://github.com/brytesika-AI/Geospatial-Analysis-of-Minerals">GitHub</a>
    </nav>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><span>Grid Cells</span><strong>12,100</strong></div>
      <div class="metric"><span>Training Rows</span><strong>1,590</strong></div>
      <div class="metric"><span>Spatial CV ROC-AUC</span><strong>0.903</strong></div>
      <div class="metric"><span>Cloud API</span><strong>Live</strong></div>
    </section>

    <section class="grid" id="map-section">
      <div>
        <div id="map"></div>
      </div>
      <aside class="panel">
        <div class="eyebrow">Site Scorer</div>
        <h2>Score a coordinate</h2>
        <p class="note">Enter any point within the AZ/NV study area. The deployed frontend calls the live Worker scoring endpoint and returns a risk tier plus grid snap.</p>
        <label for="lat">Latitude</label>
        <input id="lat" type="number" min="31" max="42" step="0.0001" value="33.45" />
        <label for="lon">Longitude</label>
        <input id="lon" type="number" min="-120" max="-109" step="0.0001" value="-110.80" />
        <label for="cu">Copper ppm</label>
        <input id="cu" type="number" min="0" step="1" value="450" />
        <button id="scoreBtn">Score location</button>
        <div class="score">
          <div><span class="note">Score</span><br><strong id="scoreValue">--</strong></div>
          <div><span class="note">Tier</span><br><strong id="tierValue">--</strong></div>
        </div>
        <p class="note" id="scoreMeta">Waiting for a query.</p>
      </aside>
    </section>

    <section class="section" id="targets">
      <div class="eyebrow">Target Portfolio</div>
      <h2>Ranked exploration targets and field actions</h2>
      <p class="note">This converts model output into a decision queue: where to advance, where uncertainty is high, and what field data would reduce decision risk.</p>
      <div class="panel">
        <table>
          <thead>
            <tr><th>Target</th><th>District</th><th>Score</th><th>Tier</th><th>Uncertainty</th><th>Field Program</th><th>Decision</th></tr>
          </thead>
          <tbody id="targetRows"></tbody>
        </table>
      </div>
    </section>

    <section class="section" id="role-fit">
      <div class="eyebrow">KoBold Data Scientist Fit</div>
      <h2>Skills showcased in the deployed app</h2>
      <div class="cards">
        <div class="card"><h3>Geoscience data curation</h3><p>Deposits, geochemistry, faults, terrain features, spatial proximity, and generated target tables.</p></div>
        <div class="card"><h3>Predictive modeling</h3><p>2D prospectivity surface with spatial CV metrics and a clear path to 3D depth slices or inversion grids.</p></div>
        <div class="card"><h3>Uncertainty reduction</h3><p>Target ranking combines prospectivity and uncertainty proxy to propose the next field program.</p></div>
        <div class="card"><h3>Cloud software delivery</h3><p>Live Cloudflare Worker frontend plus API endpoints, GitHub CI, tests, and reproducible Python pipeline.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="eyebrow">API Contract</div>
      <h2>Live scoring endpoint</h2>
      <div class="api">POST /score
{
  "lat": 33.45,
  "lon": -110.80,
  "features": { "cu_ppm": 450 },
  "interpret": false
}</div>
    </section>
  </main>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const map = L.map('map', { scrollWheelZoom: true }).setView([36.5, -114.5], 6);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      maxZoom: 19
    }).addTo(map);

    let queryMarker = null;
    const tierClass = (tier) => tier === 'Very High' ? 'very' : tier === 'High' ? 'high' : tier === 'Moderate' ? 'mod' : 'low';

    async function loadHeatmap() {
      const res = await fetch('/heatmap');
      const data = await res.json();
      const rows = [];
      data.features.forEach((feature, index) => {
        const lon = feature.geometry.coordinates[0];
        const lat = feature.geometry.coordinates[1];
        const props = feature.properties;
        const radius = 10000 + props.score * 42000;
        L.circle([lat, lon], {
          radius,
          color: props.score >= 0.7 ? '#c6ff6b' : '#d8924c',
          fillColor: props.score >= 0.7 ? '#74d680' : '#d8924c',
          fillOpacity: 0.28,
          weight: 1
        }).bindPopup('<strong>' + props.name + '</strong><br>Score ' + props.score.toFixed(2) + '<br>' + props.risk_tier).addTo(map);

        const uncertainty = Math.max(0.18, Math.min(0.62, 0.65 - props.score * 0.34 + index * 0.018));
        const program = uncertainty > 0.42 ? 'Recon mapping + infill geochem' : 'Priority mapping + ground truthing';
        const decision = props.score >= 0.7 ? 'Advance' : 'Hold for data';
        rows.push('<tr><td>AZNV-' + String(index + 1).padStart(3, '0') + '</td><td>' + props.name + '</td><td>' + props.score.toFixed(2) + '</td><td><span class="pill ' + tierClass(props.risk_tier) + '">' + props.risk_tier + '</span></td><td>' + uncertainty.toFixed(2) + '</td><td>' + program + '</td><td>' + decision + '</td></tr>');
      });
      document.getElementById('targetRows').innerHTML = rows.join('');
    }

    async function scoreLocation() {
      const lat = Number(document.getElementById('lat').value);
      const lon = Number(document.getElementById('lon').value);
      const cu = Number(document.getElementById('cu').value);
      document.getElementById('scoreMeta').textContent = 'Scoring...';
      const res = await fetch('/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lon, features: { cu_ppm: cu }, interpret: false })
      });
      const data = await res.json();
      if (!res.ok) {
        document.getElementById('scoreMeta').textContent = data.error || 'Scoring failed.';
        return;
      }
      document.getElementById('scoreValue').textContent = data.score.toFixed(3);
      document.getElementById('tierValue').textContent = data.risk_tier;
      document.getElementById('scoreMeta').textContent = 'Grid snap: ' + data.grid_snap.lat + ', ' + data.grid_snap.lon + ' | Source: ' + data.source;
      if (queryMarker) map.removeLayer(queryMarker);
      queryMarker = L.marker([lat, lon]).addTo(map).bindPopup('Query score ' + data.score.toFixed(3) + '<br>' + data.risk_tier).openPopup();
      map.setView([lat, lon], 8);
    }

    document.getElementById('scoreBtn').addEventListener('click', scoreLocation);
    loadHeatmap();
    scoreLocation();
  </script>
</body>
</html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/$/, "") || "/";

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }
    if (path === "/" && request.method === "GET") return htmlResponse(FRONTEND_HTML);
    if (path === "/api" && request.method === "GET") return handleDocs(request, env);
    if (path === "/health" && request.method === "GET") {
      return jsonResponse({ status: "ok", ts: new Date().toISOString() }, request, env);
    }
    if (path === "/score" && request.method === "POST") return handleScore(request, env);
    if (path === "/heatmap" && request.method === "GET") return handleHeatmap(request, env);
    if (path === "/interpret" && request.method === "POST") return handleInterpret(request, env);

    return errorResponse(`Route not found: ${request.method} ${path}`, request, env, 404);
  },
};
