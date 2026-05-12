"""
app/mineral_systems.py
======================
Semantic model for the African Copperbelt.

Maps ML features to mineral system components following the
Sediment-Hosted Stratiform Copper (SHSC) / Lufilian Arc framework.

Deposit style: Cu-Co SHSC dominates Zambia Copperbelt and DRC Katanga.
Botswana focus: Cu-Ni magmatic sulphide systems (Selebi-Phikwe type).

References
----------
  Hitzman et al. (2010)  Sediment-hosted stratabound copper deposits.
  Cailteux et al. (2005) Katanga Supergroup Cu-Co systems.
  Selley et al. (2005)   Central African Copperbelt geology.
  Maier et al. (2013)    Ni-Cu(-PGE) deposits of Botswana.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Mineral system ontology ───────────────────────────────────────────────────

class MineralSystemComponent(str, Enum):
    SOURCE   = "source"    # metal and fluid source
    PATHWAY  = "pathway"   # structural/stratigraphic conduits
    TRAP     = "trap"      # depositional site / redox boundary
    MODIFIER = "modifier"  # preservation, erosion, detectability


class DepositStyle(str, Enum):
    SHSC  = "Sediment-Hosted Stratiform Cu"    # Zambia / DRC Copperbelt
    CU_NI = "Magmatic Cu-Ni Sulphide"          # Botswana (Selebi-Phikwe)
    VMS   = "Volcanogenic Massive Sulphide"
    IOCG  = "Iron-Oxide Copper-Gold"
    PORPH = "Porphyry Copper"


# ── Feature → mineral system semantic mapping ─────────────────────────────────
# Each entry tags an ML feature with its geological interpretation for
# sediment-hosted Cu-Co systems of the Lufilian Arc.

FEATURE_SYSTEM_MAP: dict[str, dict[str, Any]] = {
    "log_cu_ppm": {
        "component":   MineralSystemComponent.SOURCE,
        "styles":      [DepositStyle.SHSC, DepositStyle.CU_NI],
        "description": "Copper anomaly in soil — direct indicator of metal source or trap",
        "anomaly_threshold_log": 6.2,   # ~500 ppm; ore-grade ≥ 1 000 ppm in Copperbelt
        "background_note": "Regional background 50–150 ppm (Tembo et al. 2009)",
    },
    "log_co_ppm": {
        "component":   MineralSystemComponent.SOURCE,
        "styles":      [DepositStyle.SHSC],
        "description": "Cobalt anomaly — hallmark of DRC Katanga-style Cu-Co mineralisation",
        "anomaly_threshold_log": 4.6,   # ~100 ppm; Katanga Co anomaly threshold
        "background_note": "Regional background 30–80 ppm (Cailteux et al. 2005)",
    },
    "log_ni_ppm": {
        "component":   MineralSystemComponent.SOURCE,
        "styles":      [DepositStyle.CU_NI],
        "description": "Nickel signal — supports mafic source input or Ni-Cu systems (Botswana)",
        "anomaly_threshold_log": 4.1,   # ~60 ppm
        "background_note": "Regional background 25–75 ppm (Maier et al. 2013)",
    },
    "dist_fault_km": {
        "component":   MineralSystemComponent.PATHWAY,
        "styles":      [DepositStyle.SHSC, DepositStyle.CU_NI],
        "description": (
            "Proximity to active faults — syn-sedimentary and reactivated structures "
            "control basinal brine migration and metal precipitation in SHSC systems"
        ),
        "favourable_below_km": 10.0,
    },
    "dist_deposit_km": {
        "component":   MineralSystemComponent.TRAP,
        "styles":      [DepositStyle.SHSC, DepositStyle.CU_NI],
        "description": "Proximity to known mineralisation — reflects district-scale footprint",
        "favourable_below_km": 25.0,
    },
    "log_zn_ppm": {
        "component":   MineralSystemComponent.SOURCE,
        "styles":      [DepositStyle.SHSC],
        "description": "Zn pathfinder — Cu-Pb-Zn association typical in carbonate-hosted SHSC",
        "anomaly_threshold_log": 5.3,   # ~200 ppm
    },
    "fe_pct": {
        "component":   MineralSystemComponent.TRAP,
        "styles":      [DepositStyle.SHSC],
        "description": (
            "Fe-oxide index — redox boundary indicator; hematite (oxidised) vs "
            "pyrite (reduced) contrast drives Cu precipitation in SHSC"
        ),
    },
    "log_as_ppm": {
        "component":   MineralSystemComponent.PATHWAY,
        "styles":      [DepositStyle.SHSC],
        "description": "As pathfinder — fluid pathway indicator in reducing ore environments",
    },
    "elevation_m": {
        "component":   MineralSystemComponent.MODIFIER,
        "styles":      [DepositStyle.SHSC],
        "description": "Topographic proxy for erosion level and preservation of ore horizons",
    },
    "slope_deg": {
        "component":   MineralSystemComponent.MODIFIER,
        "styles":      [DepositStyle.SHSC],
        "description": "Slope — influences geochemical dispersion and sampling representativeness",
    },
}


# ── Typed entity schemas ───────────────────────────────────────────────────────

@dataclass
class GeochemSample:
    sample_id: str
    lat: float
    lon: float
    cu_ppm:  float | None = None
    co_ppm:  float | None = None
    ni_ppm:  float | None = None
    au_ppb:  float | None = None
    fe_pct:  float | None = None
    zn_ppm:  float | None = None
    pb_ppm:  float | None = None
    mo_ppm:  float | None = None
    as_ppm:  float | None = None
    source:  str = "unknown"

    @property
    def co_cu_ratio(self) -> float | None:
        """Co/Cu ratio — diagnostic for DRC Katanga-type Cu-Co mineralisation."""
        if self.co_ppm and self.cu_ppm and self.cu_ppm > 0:
            return self.co_ppm / self.cu_ppm
        return None

    @property
    def is_cu_anomalous(self) -> bool:
        return bool(self.cu_ppm and self.cu_ppm >= 500)

    @property
    def is_co_anomalous(self) -> bool:
        return bool(self.co_ppm and self.co_ppm >= 100)


@dataclass
class MineralDeposit:
    dep_id:            str
    name:              str
    lat:               float
    lon:               float
    primary_commodity: str
    deposit_style:     DepositStyle = DepositStyle.SHSC
    district:          str = ""
    source:            str = "unknown"


@dataclass
class ProspectTarget:
    target_id:        str
    lat:              float
    lon:              float
    prospectivity_score: float
    risk_tier:        str
    uncertainty_proxy: float
    features:         dict[str, float] = field(default_factory=dict)

    @property
    def mineral_system_evidence(self) -> dict[str, list[str]]:
        """Group anomalous features by mineral system component."""
        evidence: dict[str, list[str]] = {c.value: [] for c in MineralSystemComponent}
        for feat, val in self.features.items():
            if feat not in FEATURE_SYSTEM_MAP:
                continue
            info     = FEATURE_SYSTEM_MAP[feat]
            comp     = info["component"].value
            thresh   = info.get("anomaly_threshold_log")
            fav_km   = info.get("favourable_below_km")
            if thresh and val >= thresh:
                evidence[comp].append(f"{feat}: {val:.2f} (anomalous)")
            elif fav_km and "dist" in feat and val < fav_km:
                evidence[comp].append(f"{feat}: {val:.1f} km (favourable)")
        return evidence

    @property
    def decision_rationale(self) -> str:
        ev = self.mineral_system_evidence
        parts = []
        if ev[MineralSystemComponent.SOURCE.value]:
            parts.append("Source: " + "; ".join(ev[MineralSystemComponent.SOURCE.value]))
        if ev[MineralSystemComponent.PATHWAY.value]:
            parts.append("Pathway: " + "; ".join(ev[MineralSystemComponent.PATHWAY.value]))
        if ev[MineralSystemComponent.TRAP.value]:
            parts.append("Trap: " + "; ".join(ev[MineralSystemComponent.TRAP.value]))
        return ". ".join(parts) or "Insufficient indicators for component classification."

    @property
    def dominant_deposit_style(self) -> DepositStyle:
        """Infer likely deposit style from feature signature."""
        co = self.features.get("log_co_ppm", 0)
        ni = self.features.get("log_ni_ppm", 0)
        cu = self.features.get("log_cu_ppm", 0)
        if ni > 4.0 and cu > 4.0:
            return DepositStyle.CU_NI
        if co > 4.0 and cu > 5.0:
            return DepositStyle.SHSC
        return DepositStyle.SHSC


def feature_component_summary(features: dict[str, float]) -> dict[str, str]:
    """
    Return a human-readable mapping of features to their mineral system roles.
    Used in the Streamlit Site Scorer geological breakdown panel.
    """
    summary = {}
    for feat, val in features.items():
        if feat in FEATURE_SYSTEM_MAP:
            info = FEATURE_SYSTEM_MAP[feat]
            summary[feat] = (
                f"[{info['component'].value.upper()}] {info['description']}"
            )
    return summary
