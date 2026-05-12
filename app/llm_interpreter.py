"""
LLM geological interpretation layer — Africa Copperbelt edition.

Uses HuggingFace Inference API when configured, and falls back to a
deterministic geological summary when the API is unavailable.

Geological context: Sediment-Hosted Stratiform Copper (SHSC) systems of
the Lufilian Arc, Zambia Copperbelt, and DRC Katanga — the primary target
environment for KoBold Metals in Central Africa.
"""

from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HF_MODEL   = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_BASE    = "https://api-inference.huggingface.co/models"
TIMEOUT    = 30

GEOLOGY_CONTEXT = """
African Copperbelt geology context (KoBold Metals operational area):

Deposit style
  The dominant deposit type is Sediment-Hosted Stratiform Copper (SHSC),
  hosted in the Neoproterozoic Katanga Supergroup of the Lufilian Arc
  (~880–550 Ma; Selley et al. 2005).  Unlike porphyry copper, SHSC deposits
  form by diagenetic or low-grade metamorphic oxidising fluids (basinal brines)
  precipitating Cu (and Co) at a redox boundary — typically at the interface
  between oxidised continental redbeds and reduced marine shales or evaporites.

Key districts
  Zambia Copperbelt: Nchanga, Nkana-Kitwe, Mufulira, Luanshya, Kansanshi.
  KoBold Mingomba project: ~Chililabombwe area, Zambia.
  DRC Katanga: Kolwezi, Tenke-Fungurume, Likasi, Lubumbashi.
  Botswana: Selebi-Phikwe (magmatic Cu-Ni sulphide — different system).

Geochemical indicator suite
  Primary:   Cu, Co (key — Co/Cu ratio > 0.1 is strongly Katanga-type),
             Zn, Pb, As (pathfinders)
  Secondary: Fe (redox proxy), Mn (sedimentary sequence indicator)
  Note:      Mo is LOW in SHSC vs porphyry; high Mo may indicate a different
             mineral system.

Structural control
  Syn-sedimentary faults and fold hinges in the Lufilian Arc controlled
  basin formation and fluid focussing. Proximity to these structures is a
  key exploration criterion.

Interpretation caveats
  Scores are a screening tool, not a resource estimate.  Soil geochemical
  anomalies require follow-up: rock geochemistry, ground geophysics (IP/CSAMT),
  and geological mapping to confirm stratigraphic position and redox context.
"""


class GeoInterpreter:
    """Concise geological interpretations for African Copperbelt prospectivity scores."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key  = api_key or HF_API_KEY
        self.model    = model or HF_MODEL
        self.endpoint = f"{HF_BASE}/{self.model}"
        self._client  = httpx.Client(timeout=TIMEOUT)

    def interpret_score(
        self,
        lat: float,
        lon: float,
        score: float,
        features: dict[str, Any],
        nearest_deposit: str | None = None,
    ) -> str:
        prompt   = self._build_prompt(lat, lon, score, features, nearest_deposit)
        response = self._call_hf(prompt)
        if response:
            return self._clean_response(response)
        return self._fallback_interpretation(score, features)

    def _build_prompt(
        self,
        lat: float,
        lon: float,
        score: float,
        features: dict,
        nearest_deposit: str | None,
    ) -> str:
        tier      = self._tier(score)
        cu        = features.get("log_cu_ppm",      0)
        co        = features.get("log_co_ppm",      0)
        ni        = features.get("log_ni_ppm",      0)
        zn        = features.get("log_zn_ppm",      0)
        fe        = features.get("fe_pct",           0)
        fault_km  = features.get("dist_fault_km",  99)
        dep_km    = features.get("dist_deposit_km", 99)
        elev      = features.get("elevation_m",   1100)

        nearest_str = (
            f"Nearest known deposit: {nearest_deposit} (~{dep_km:.1f} km)"
            if nearest_deposit
            else f"Distance to nearest deposit: {dep_km:.1f} km"
        )

        # Infer likely hemisphere of the study area from coords
        hemisphere = "Zambia/DRC Katanga Copperbelt" if lon > 20 and lat < -5 else "Africa study area"

        return textwrap.dedent(
            f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a senior exploration geologist specialising in African sediment-hosted
            copper-cobalt (SHSC) systems of the Lufilian Arc.  Write concise,
            evidence-based target screening notes.  Reference African geology, not porphyry.
            {GEOLOGY_CONTEXT}
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            Write a concise geological assessment in 4-6 bullets, max 200 words.

            Location     : {lat:.4f} °S, {lon:.4f} °E  ({hemisphere})
            AI score     : {score:.2f} / 1.00  ({tier} prospectivity)

            Key indicators:
            - Copper signal    : log(Cu ppm) = {cu:.2f}
            - Cobalt signal    : log(Co ppm) = {co:.2f}   ← Co/Cu context critical
            - Nickel signal    : log(Ni ppm) = {ni:.2f}
            - Zinc signal      : log(Zn ppm) = {zn:.2f}
            - Iron content     : Fe = {fe:.2f} %  (redox proxy)
            - Distance to fault: {fault_km:.1f} km
            - {nearest_str}
            - Elevation        : {elev:.0f} m

            Structure: score interpretation → SHSC mineral system evidence
            (source / pathway / trap) → limiting factors → recommended next step.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        ).strip()

    def _call_hf(self, prompt: str) -> str | None:
        if not self.api_key:
            log.warning("No HuggingFace API key configured; using fallback.")
            return None
        try:
            resp = self._client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 320,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "return_full_text": False,
                        "stop": ["<|eot_id|>", "<|end_of_text|>"],
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            return None
        except httpx.HTTPStatusError as exc:
            log.warning("HF API error %s; using fallback.", exc.response.status_code)
            return None
        except Exception as exc:
            log.warning("HF call failed: %s", exc)
            return None

    @staticmethod
    def _clean_response(text: str) -> str:
        for marker in ["<|eot_id|>", "<|end_of_text|>", "assistant"]:
            text = text.split(marker)[0]
        return text.strip()

    @staticmethod
    def _tier(score: float) -> str:
        if score >= 0.70: return "Very High"
        if score >= 0.50: return "High"
        if score >= 0.30: return "Moderate"
        return "Low"

    @classmethod
    def _fallback_interpretation(cls, score: float, features: dict) -> str:
        tier     = cls._tier(score)
        cu       = features.get("log_cu_ppm",      0)
        co       = features.get("log_co_ppm",      0)
        ni       = features.get("log_ni_ppm",      0)
        fault    = features.get("dist_fault_km",  99)
        dep      = features.get("dist_deposit_km", 99)
        fe       = features.get("fe_pct",           0)

        bullets = [f"**AI score: {score:.2f} — {tier} prospectivity (African Copperbelt screen).**"]

        # Copper
        if cu > 6.2:
            bullets.append("- **Strong Cu anomaly** (≥ 500 ppm equiv.) — primary follow-up indicator.")
        elif cu > 5.3:
            bullets.append("- Moderate Cu anomaly — warrants scout soil sampling.")
        else:
            bullets.append("- Cu near background; geochemical contrast is low.")

        # Cobalt — the DRC Katanga fingerprint
        if co > 4.6:
            bullets.append("- **Co anomaly present** — elevated Co/Cu ratio consistent with DRC Katanga-style SHSC mineralisation.")
        elif co > 3.9:
            bullets.append("- Weak Co signal; could reflect distal Katanga-style alteration halo.")

        # Nickel — Botswana magmatic system indicator
        if ni > 4.1:
            bullets.append("- Ni signal elevated — consider magmatic Cu-Ni sulphide (Selebi-Phikwe type) as an alternative target.")

        # Structural
        if fault < 5:
            bullets.append("- Very close to mapped structure — favourable for fluid focussing and redox trap development.")
        elif fault < 15:
            bullets.append("- Moderate structural proximity — potential fault-controlled fluid pathway.")
        else:
            bullets.append("- Distal to mapped faults in current dataset; structural context needs ground-truth.")

        # Iron / redox proxy
        if fe > 5:
            bullets.append("- Elevated Fe may reflect hematite-rich redbeds above the redox boundary — a classic SHSC trap indicator.")

        # District proximity
        if dep < 15:
            bullets.append("- Within 15 km of known mineralisation — strong district-scale endorsement.")

        # Recommendation
        if score >= 0.70:
            bullets.append("- **Recommendation: priority target** — initiate rock sampling, IP geophysics, and geological mapping to confirm stratigraphic position.")
        elif score >= 0.50:
            bullets.append("- **Recommendation: secondary target** — low-cost infill geochemistry and remote sensing review.")
        elif score >= 0.30:
            bullets.append("- Recommendation: monitor; compile additional regional data before field commitment.")
        else:
            bullets.append("- Recommendation: low priority under current screening criteria.")

        return "\n".join(bullets)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
