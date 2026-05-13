"""GET /api/interpret — LLM geological + ESG + logistics narrative for a target.
Priority: Cache → OpenAI (gpt-4o-mini) → Claude (Haiku) → Groq → HuggingFace → rule-based.

Cost guide (per unique target click):
  - OpenAI gpt-4o-mini : ~$0.0005  (set OPENAI_API_KEY)
  - Claude Haiku       : ~$0.003   (set ANTHROPIC_API_KEY)
  - Groq               : free       (set GROQ_API_KEY)
  - HuggingFace        : free       (set HF_API_KEY)
  - rule-based         : $0         (always available fallback)
In-memory cache means repeated clicks on the same target cost nothing.
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import json, math, time, urllib.request, urllib.error
from _utils import BaseHandler, DATA_PROC, DATA_RAW, read_csv, infer_country, score_tier

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
CLAUDE_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_KEY    = os.environ.get("GROQ_API_KEY", "")
HF_KEY      = os.environ.get("HF_API_KEY", "")

OPENAI_URL  = "https://api.openai.com/v1/chat/completions"
CLAUDE_URL  = "https://api.anthropic.com/v1/messages"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
HF_BASE     = "https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "gemma2-9b-it"),
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

HF_MODELS = [
    os.environ.get("HF_MODEL", ""),
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
HF_MODELS = [m for m in HF_MODELS if m]

# In-memory cache — same target (lat/lon rounded to 2dp) costs $0 on repeat clicks
_CACHE: dict = {}

_GRID  = None
_DUMPS = None


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 2)}_{round(lon, 2)}"


def _load_grid():
    global _GRID
    if _GRID is None:
        _GRID = read_csv(DATA_PROC / "predictions.csv")
    return _GRID


def _load_dumps():
    global _DUMPS
    if _DUMPS is None:
        p = DATA_RAW / "mining_dumps_africa.csv"
        _DUMPS = read_csv(p) if p.exists() else []
    return _DUMPS


def _nearest_dump(lat: float, lon: float):
    dumps = _load_dumps()
    if not dumps:
        return None, 999
    best = min(dumps, key=lambda d: math.hypot(float(d["lat"]) - lat, float(d["lon"]) - lon))
    dist = math.hypot(float(best["lat"]) - lat, float(best["lon"]) - lon) * 111
    return best, round(dist, 1)


def _build_prompt(lat, lon, score, tier, co_cu, ni_cu, country, features_text):
    dump, dump_dist = _nearest_dump(lat, lon)

    if dump and dump_dist < 120:
        dump_note = (
            f"Historical mining dump/tailings '{dump['name']}' ({dump.get('commodity','minerals')}) "
            f"is {dump_dist} km away (grade: {dump.get('est_grade','unknown')}; "
            f"volume: {dump.get('volume_mt','?')} Mt; status: {dump.get('status','unknown')}). "
            f"Tailings reprocessing may be viable alongside primary exploration."
        )
    else:
        dump_note = "No significant historical mining dumps within 120 km."

    co_flag = " — DRC Katanga-style SHSC Cu-Co (Cailteux et al. 2005)"  if co_cu >= 10 else ""
    ni_flag = " — Bushveld/Kabanga magmatic sulphide signature"           if ni_cu >= 20 else ""

    return f"""You are a senior economic geologist and ESG strategist advising a junior mining company board on a drill target in Sub-Saharan Africa.

TARGET PROFILE:
- Location: {country} ({lat:.4f}°, {lon:.4f}°)
- Prospectivity Score: {score:.3f}/1.0 ({tier} tier)
- Co/Cu Ratio: {co_cu:.1f}%{co_flag}
- Ni/Cu Ratio: {ni_cu:.1f}%{ni_flag}
- Key Geological Signals: {features_text or "Cu, Co, Ni geochemical anomalism; structural proximity to mapped faults"}
- Historical Dumps / Tailings: {dump_note}

Provide a structured assessment with EXACTLY these four headings (no extra text before the first heading):

GEOLOGICAL RATIONALE
Write 2-3 sentences. Apply the SHSC mineral systems framework (Source-Pathway-Trap-Modifier). Name the most likely deposit type. Cite one real African analogue deposit by name.

ESG CONSIDERATIONS
Write exactly 5 bullet points:
• Water: stewardship strategy for {country}
• ASM: artisanal & small-scale mining conflict risk and mitigation
• Biodiversity: ecosystem sensitivity and ESIA trigger assessment
• Community: social licence timeline and FPIC requirements
• Carbon: Scope 3 pathway, net-zero alignment, responsible sourcing (LME/IRMA)

LOGISTICS & INFRASTRUCTURE
Write exactly 4 bullet points:
• Port & Offtake: nearest deep-water port, concentrate routing, smelter options
• Road/Rail: access road condition, rail connectivity for bulk ore
• Power: grid availability vs solar-diesel hybrid for drill camp and future processing
• Water Source: nearest permanent water source for drilling and future processing plant

RECOMMENDATION
State ADVANCE / HOLD / DROP clearly. Give next 3 actions with a 90-day timeline. State estimated cost to first drill result in USD.

Be specific to {country}. Use geological terminology. Write for a board-level audience — precise, no filler."""


def _call_openai(prompt: str) -> tuple[str, str]:
    payload = json.dumps({
        "model":       OPENAI_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  900,
        "temperature": 0.35,
    }).encode()
    req = urllib.request.Request(
        OPENAI_URL, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {OPENAI_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        return OPENAI_MODEL, text


def _call_claude(prompt: str) -> tuple[str, str]:
    payload = json.dumps({
        "model":      CLAUDE_MODEL,
        "max_tokens": 900,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        CLAUDE_URL, data=payload,
        headers={"Content-Type":      "application/json",
                 "x-api-key":         CLAUDE_KEY,
                 "anthropic-version":  "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=35) as r:
        data = json.loads(r.read())
        text = data["content"][0]["text"].strip()
        return CLAUDE_MODEL, text


def _call_groq(prompt: str) -> tuple[str, str]:
    last_err = ""
    for model in GROQ_MODELS:
        payload = json.dumps({
            "model":       model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  900,
            "temperature": 0.35,
        }).encode()
        req = urllib.request.Request(
            GROQ_URL, data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {GROQ_KEY}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=28) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"].strip()
                return model, text
        except urllib.error.HTTPError as e:
            last_err = f"{model}: HTTP {e.code}"
            continue
        except Exception as e:
            last_err = f"{model}: {e}"
            continue
    raise RuntimeError(f"All Groq models failed. Last: {last_err}")


def _call_hf_model(model: str, prompt: str) -> str:
    url     = HF_BASE.format(model=model)
    payload = json.dumps({
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  900,
        "temperature": 0.35,
        "stream":      False,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {HF_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=55) as r:
            data = json.loads(r.read())
            if "choices" in data:
                return model, data["choices"][0]["message"]["content"].strip()
            raise ValueError(str(data)[:200])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise ValueError(f"HTTP {e.code}: {body[:150]}")


def _call_hf(prompt: str) -> tuple[str, str]:
    last_err = "No HF models configured"
    for model in HF_MODELS:
        try:
            return _call_hf_model(model, prompt)
        except ValueError as e:
            last_err = f"{model}: {e}"
            if "503" in str(e):
                time.sleep(15)
            continue
    raise RuntimeError(f"All HF models failed. Last: {last_err}")


def _rule_based(lat, lon, score, tier, co_cu, ni_cu, country):
    if co_cu >= 30:
        system  = "high-grade SHSC Cu-Co system — analogue: Tenke Fungurume (DRC)"
        deposit = "SHSC stratabound Cu-Co"
    elif co_cu >= 10:
        system  = "Sediment-Hosted Stratiform Copper-Cobalt (SHSC) — analogue: Nkana or Konkola (Zambia)"
        deposit = "SHSC Cu-Co"
    elif ni_cu >= 20:
        system  = "magmatic sulphide system — analogue: Munali (Zambia) or Kabanga (Tanzania)"
        deposit = "Ni-Cu-PGM magmatic sulphide"
    else:
        system  = "porphyry or IOCG Cu system — analogue: regional Copperbelt district targets"
        deposit = "porphyry/IOCG Cu"

    rec  = "ADVANCE" if score >= 0.70 else ("HOLD" if score >= 0.50 else "DROP")
    cost = "$400k–$900k" if score >= 0.85 else ("$150k–$400k" if score >= 0.70 else "$60k–$150k")
    timeline = "90 days to RC scout result" if score >= 0.70 else "120 days to EM/soil geochemistry"

    dump, dump_dist = _nearest_dump(lat, lon)
    dump_line = ""
    if dump and dump_dist < 120:
        dump_line = f"\n• Tailings Reprocessing: Historical {dump.get('commodity','mineral')} dump '{dump['name']}' at {dump_dist} km — evaluate low-capex reprocessing as revenue bridge alongside primary exploration."

    return f"""GEOLOGICAL RATIONALE
This {tier} target is consistent with a {system}. The Source-Pathway-Trap geometry is supported by geochemical anomalism and structural proximity to mapped fault corridors in {country}. The dominant expected deposit style is {deposit}.

ESG CONSIDERATIONS
• Water: Implement closed-loop water management; obtain national water authority permit in {country} before mobilisation.
• ASM: Conduct 10 km community mapping; establish formal artisanal miner coexistence or exclusion protocol.
• Biodiversity: Desktop ESIA required before field programme; check IUCN Red List habitat and protected area buffers.
• Community: Minimum 6-month FPIC engagement and community development agreement before drill permit.
• Carbon: Solar-hybrid power for drill camp; align with LME Responsible Sourcing and IRMA framework for future operations.{dump_line}

LOGISTICS & INFRASTRUCTURE
• Port & Offtake: Route Cu/Co concentrate to nearest deep-water port; negotiate term sheet with regional smelter before resource definition.
• Road/Rail: Assess seasonal road access; engage mining contractor for wet-season logistics and camp establishment.
• Power: Diesel genset for initial drilling; evaluate 200–500 kW solar-diesel hybrid for extended campaign and future PEA.
• Water Source: Identify nearest permanent water source within 15 km; design borehole + lined dam for 200 m³/day processing requirement.

RECOMMENDATION
{rec} — {timeline}. Next actions: (1) Commission airborne EM/magnetic survey over 50 km²; (2) Deploy soil geochemistry grid at 200×50 m; (3) Submit drill permit application to relevant {country} mining authority. Estimated cost to first drill result: {cost}."""


def _parse_sections(text: str) -> dict:
    sections = {}
    current, buf = None, []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in ("GEOLOGICAL RATIONALE", "ESG CONSIDERATIONS",
                        "LOGISTICS & INFRASTRUCTURE", "RECOMMENDATION"):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = stripped, []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


class handler(BaseHandler):
    def handle_get(self, params):
        def _p(k, default=""):
            v = params.get(k)
            return (v[0] if isinstance(v, list) else v) if v else default

        lat      = float(_p("lat",  "-12.5"))
        lon      = float(_p("lon",   "28.2"))
        score    = float(_p("score", "0.5"))
        tier     = _p("tier", score_tier(score))
        co_cu    = float(_p("co_cu", "0"))
        ni_cu    = float(_p("ni_cu", "0"))
        country  = _p("country", infer_country(lat, lon))
        features = _p("features", "")

        prompt = _build_prompt(lat, lon, score, tier, co_cu, ni_cu, country, features)
        errors = []

        # Serve from cache if available — free on repeat clicks
        ck = _cache_key(lat, lon)
        if ck in _CACHE:
            cached = dict(_CACHE[ck])
            cached["cached"] = True
            return cached

        if OPENAI_KEY:
            try:
                model_used, text = _call_openai(prompt)
                result = {"source": f"openai-{model_used}", "sections": _parse_sections(text), "raw": text}
                _CACHE[ck] = result
                return result
            except Exception as e:
                errors.append(f"openai: {e}")

        if CLAUDE_KEY:
            try:
                model_used, text = _call_claude(prompt)
                label = model_used.split("-")[1] if "-" in model_used else model_used
                result = {"source": f"claude-{label}", "sections": _parse_sections(text), "raw": text}
                _CACHE[ck] = result
                return result
            except Exception as e:
                errors.append(f"claude: {e}")

        if GROQ_KEY:
            try:
                model_used, text = _call_groq(prompt)
                label = model_used.replace("-versatile","").replace("-instant","")
                result = {"source": f"groq-{label}", "sections": _parse_sections(text), "raw": text}
                _CACHE[ck] = result
                return result
            except Exception as e:
                errors.append(f"groq: {e}")

        if HF_KEY:
            try:
                model_used, text = _call_hf(prompt)
                label = model_used.split("/")[-1]
                result = {"source": f"huggingface-{label}", "sections": _parse_sections(text), "raw": text}
                _CACHE[ck] = result
                return result
            except Exception as e:
                errors.append(f"hf: {e}")

        text = _rule_based(lat, lon, score, tier, co_cu, ni_cu, country)
        result = {"source": "rule-based", "sections": _parse_sections(text), "raw": text, "errors": errors}
        _CACHE[ck] = result
        return result
