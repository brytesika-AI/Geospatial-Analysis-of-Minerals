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

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/$/, "") || "/";

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    }
    if (path === "/" && request.method === "GET") return handleDocs(request, env);
    if (path === "/health" && request.method === "GET") {
      return jsonResponse({ status: "ok", ts: new Date().toISOString() }, request, env);
    }
    if (path === "/score" && request.method === "POST") return handleScore(request, env);
    if (path === "/heatmap" && request.method === "GET") return handleHeatmap(request, env);
    if (path === "/interpret" && request.method === "POST") return handleInterpret(request, env);

    return errorResponse(`Route not found: ${request.method} ${path}`, request, env, 404);
  },
};
