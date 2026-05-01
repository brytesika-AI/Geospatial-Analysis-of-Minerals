"""
LLM geological interpretation layer.

Uses HuggingFace Inference API when configured, and falls back to a
deterministic geological summary when the API is unavailable.
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
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_BASE = "https://api-inference.huggingface.co/models"
TIMEOUT = 30

GEOLOGY_CONTEXT = """
Arizona-Nevada copper geology context:
- Porphyry copper deposits dominate: Morenci, Ray, Miami-Globe, Bagdad, Bisbee.
- Characteristic signatures include elevated Cu, Mo, Au, Fe, As, Pb, and Zn.
- Structural control often follows Basin-and-Range faults, caldera margins, and batholith contacts.
- Favourable elevations are commonly basin-and-range uplands and range-front settings.
- Scores are screening indicators, not resource estimates or drilling decisions.
"""


class GeoInterpreter:
    """Generate concise geological interpretations for prospectivity scores."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or HF_API_KEY
        self.model = model or HF_MODEL
        self.endpoint = f"{HF_BASE}/{self.model}"
        self._client = httpx.Client(timeout=TIMEOUT)

    def interpret_score(
        self,
        lat: float,
        lon: float,
        score: float,
        features: dict[str, Any],
        nearest_deposit: str | None = None,
    ) -> str:
        prompt = self._build_prompt(lat, lon, score, features, nearest_deposit)
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
        tier = self._tier(score)
        cu = features.get("log_cu_ppm", 0)
        au = features.get("log_au_ppb", 0)
        mo = features.get("log_mo_ppm", 0)
        fe = features.get("fe_pct", 0)
        fault_km = features.get("dist_fault_km", 99)
        dep_km = features.get("dist_deposit_km", 99)
        elev = features.get("elevation_m", 1000)
        nearest_str = (
            f"Nearest known deposit: {nearest_deposit} (~{dep_km:.1f} km away)"
            if nearest_deposit
            else f"Distance to nearest deposit: {dep_km:.1f} km"
        )

        return textwrap.dedent(
            f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a senior exploration geologist specialising in porphyry copper systems.
            Write concise, evidence-based target screening notes. Use technical but accessible
            language and avoid overclaiming.
            {GEOLOGY_CONTEXT}
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            Write a concise geological assessment in 4-6 bullets, max 180 words.

            Location: {lat:.4f} N, {lon:.4f} W
            AI prospectivity score: {score:.2f} / 1.00 ({tier} potential)

            Key indicators:
            - Copper signal: log(Cu ppm) = {cu:.2f}
            - Gold signal: log(Au ppb) = {au:.2f}
            - Molybdenum signal: log(Mo ppm) = {mo:.2f}
            - Iron content: Fe = {fe:.2f}%
            - Distance to fault: {fault_km:.1f} km
            - {nearest_str}
            - Elevation: {elev:.0f} m

            Structure: score interpretation, positive indicators, limiting factors, recommendation.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        ).strip()

    def _call_hf(self, prompt: str) -> str | None:
        if not self.api_key:
            log.warning("No HuggingFace API key configured; using fallback interpretation.")
            return None
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "return_full_text": False,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"],
                },
            }
            resp = self._client.post(self.endpoint, headers=headers, json=payload)
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
        if score >= 0.70:
            return "Very High"
        if score >= 0.50:
            return "High"
        if score >= 0.30:
            return "Moderate"
        return "Low"

    @classmethod
    def _fallback_interpretation(cls, score: float, features: dict) -> str:
        tier = cls._tier(score)
        cu = features.get("log_cu_ppm", 0)
        fault = features.get("dist_fault_km", 99)
        dep = features.get("dist_deposit_km", 99)

        bullets = [f"**AI score: {score:.2f} - {tier} prospectivity.**"]

        if cu > 3.5:
            bullets.append("- Elevated copper signal supports follow-up target screening.")
        elif cu > 2.5:
            bullets.append("- Copper is moderately anomalous and may justify scout sampling.")
        else:
            bullets.append("- Copper is near background in the current feature set.")

        if fault < 5:
            bullets.append("- The site is close to mapped structure, a plausible fluid pathway.")
        elif fault < 20:
            bullets.append("- Fault proximity is moderate, so structural support is present but not dominant.")
        else:
            bullets.append("- The site is distal to mapped faults in the current dataset.")

        if dep < 20:
            bullets.append("- Nearby known mineralisation improves the regional context.")

        if score >= 0.70:
            bullets.append("- Recommendation: priority target for data review, mapping, and geochemical follow-up.")
        elif score >= 0.50:
            bullets.append("- Recommendation: secondary target for low-cost validation.")
        elif score >= 0.30:
            bullets.append("- Recommendation: monitor until additional supporting data is available.")
        else:
            bullets.append("- Recommendation: low priority based on current screening data.")

        return "\n".join(bullets)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
