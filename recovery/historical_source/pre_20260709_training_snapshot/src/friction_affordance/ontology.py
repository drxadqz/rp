from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


FRICTION_STATES = [
    "dry",
    "damp",
    "wet",
    "very_wet",
    "water",
    "fresh_snow",
    "melted_snow",
    "packed_snow",
    "partial_snow",
    "ice",
]

MATERIALS = [
    "asphalt",
    "concrete",
    "cobblestone",
    "dirt_mud",
    "gravel",
]

UNEVENNESS = ["smooth", "slight", "severe"]
WETNESS = ["dry", "damp", "wet", "very_wet", "water"]
SNOW = ["none", "partial_snow", "fresh_snow", "packed_snow", "melted_snow", "ice"]
RISK = ["very_low", "low", "medium", "high", "very_high"]

TASKS = {
    "friction": FRICTION_STATES,
    "material": MATERIALS,
    "unevenness": UNEVENNESS,
    "wetness": WETNESS,
    "snow": SNOW,
    "risk": RISK,
}

IGNORE_INDEX = -1


FRICTION_INTERVAL_REFERENCE_SOURCES = {
    "zhao_2025_mssp": {
        "citation": (
            "Zhao et al., Tire-Road friction coefficients adaptive estimation "
            "through image and vehicle dynamics integration, Mechanical Systems "
            "and Signal Processing, 2025."
        ),
        "doi": "https://doi.org/10.1016/j.ymssp.2024.112039",
        "role": (
            "Real-vehicle image+dynamics fusion paper. Table 2 reports measured "
            "reference adhesion coefficients for dry/semi-wet asphalt, "
            "snow-covered roads, rough ice, and smooth ice; Eqs. (4)-(5) give "
            "wet/waterlogged speed-dependent references."
        ),
    },
    "liu_2025_tits_review": {
        "citation": (
            "Liu et al., A Survey and Comprehensive Taxonomy of Tire-Road "
            "Adhesion Coefficient Estimation for Intelligent Vehicles, IEEE "
            "Transactions on Intelligent Transportation Systems, 2025."
        ),
        "doi": "https://doi.org/10.1109/TITS.2025.3565542",
        "role": (
            "Survey taxonomy. Its fusion-framework figure uses road-type "
            "adhesion ranges as weak constraints for image/dynamics fusion."
        ),
    },
    "nhtsa_utqg_ecfr_49cfr575_104": {
        "citation": (
            "U.S. eCFR, 49 CFR 575.104, Uniform Tire Quality Grading Standards, "
            "traction grading coefficients for wet asphalt and wet concrete."
        ),
        "doi": "https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-575/subpart-B/section-575.104",
        "role": (
            "Official regulatory wet-surface tire-traction sanity check. The "
            "UTQG thresholds are not visual dataset labels, but they provide "
            "independent wet-asphalt coefficient anchors for the weak friction "
            "affordance ontology."
        ),
    },
    "fhwa_pavement_friction_safety_primer": {
        "citation": (
            "Federal Highway Administration, Pavement Friction for Road Safety: "
            "Primer on Friction Measurement and Management Methods, FHWA-SA-23-007."
        ),
        "doi": "https://highways.dot.gov/safety/rwd/keep-vehicles-road/pavement-friction/pavement-friction-road-safety-primer-friction",
        "role": (
            "Official pavement-friction management reference. It supports the "
            "paper framing that friction is a tire-pavement interaction affected "
            "by measurement method, texture, roadway context, and safety demand, "
            "so image labels should supervise conservative affordance intervals."
        ),
    },
    "trb_sr115_snow_ice_skid_resistance": {
        "citation": (
            "Transportation Research Board Special Report 115, Skid Resistance of "
            "Snow- or Ice-Covered Roads."
        ),
        "doi": "https://onlinepubs.trb.org/Onlinepubs/sr/sr115/115-010.pdf",
        "role": (
            "Public snow/ice skid-resistance table. It anchors winter-condition "
            "interval sanity checks for ice, new snow, old/compacted snow, and "
            "treated snow without treating those values as image-level labels."
        ),
    },
}


FRICTION_INTERVAL_BENCHMARKS = [
    {
        "anchor": "dry_asphalt",
        "mapped_state": "dry",
        "mapped_material": "asphalt",
        "reference_low": 0.86,
        "reference_high": 0.92,
        "source": "zhao_2025_mssp_table2",
    },
    {
        "anchor": "semi_wet_asphalt",
        "mapped_state": "damp",
        "mapped_material": "asphalt",
        "reference_low": 0.72,
        "reference_high": 0.77,
        "source": "zhao_2025_mssp_table2",
    },
    {
        "anchor": "wet_asphalt_reference",
        "mapped_state": "wet",
        "mapped_material": "asphalt",
        "reference_low": 0.50,
        "reference_high": 0.70,
        "source": "liu_2025_tits_fig6_and_zhao_2025_eq4",
    },
    {
        "anchor": "waterlogged_reference",
        "mapped_state": "water",
        "mapped_material": "asphalt",
        "reference_low": 0.45,
        "reference_high": 0.56,
        "source": "zhao_2025_mssp_eq5",
    },
    {
        "anchor": "snow_covered",
        "mapped_state": "fresh_snow",
        "mapped_material": None,
        "reference_low": 0.37,
        "reference_high": 0.38,
        "source": "zhao_2025_mssp_table2",
    },
    {
        "anchor": "rough_ice",
        "mapped_state": "ice",
        "mapped_material": None,
        "reference_low": 0.21,
        "reference_high": 0.23,
        "source": "zhao_2025_mssp_table2",
    },
    {
        "anchor": "smooth_ice",
        "mapped_state": "ice",
        "mapped_material": None,
        "reference_low": 0.09,
        "reference_high": 0.10,
        "source": "zhao_2025_mssp_table2_and_liu_2025_tits_fig6",
    },
    {
        "anchor": "loose_snow_review",
        "mapped_state": "partial_snow",
        "mapped_material": None,
        "reference_low": 0.20,
        "reference_high": 0.25,
        "source": "liu_2025_tits_fig6",
    },
    {
        "anchor": "trb_ice_snow_road",
        "mapped_state": "ice",
        "mapped_material": None,
        "reference_low": 0.10,
        "reference_high": 0.20,
        "source": "trb_sr115_table1",
    },
    {
        "anchor": "trb_new_snow_road",
        "mapped_state": "fresh_snow",
        "mapped_material": None,
        "reference_low": 0.20,
        "reference_high": 0.25,
        "source": "trb_sr115_table1",
    },
    {
        "anchor": "trb_old_or_compacted_snow",
        "mapped_state": "packed_snow",
        "mapped_material": None,
        "reference_low": 0.25,
        "reference_high": 0.30,
        "source": "trb_sr115_table1",
    },
    {
        "anchor": "trb_refrozen_snow",
        "mapped_state": "melted_snow",
        "mapped_material": None,
        "reference_low": 0.30,
        "reference_high": 0.40,
        "source": "trb_sr115_table1",
    },
    {
        "anchor": "trb_sand_or_chloride_treated_snow",
        "mapped_state": "partial_snow",
        "mapped_material": None,
        "reference_low": 0.30,
        "reference_high": 0.40,
        "source": "trb_sr115_table1",
    },
    {
        "anchor": "utqg_wet_asphalt_aa_boundary",
        "mapped_state": "wet",
        "mapped_material": "asphalt",
        "reference_low": 0.54,
        "reference_high": 0.54,
        "source": "nhtsa_utqg_ecfr_49cfr575_104_table2",
    },
    {
        "anchor": "utqg_wet_asphalt_a_boundary",
        "mapped_state": "wet",
        "mapped_material": "asphalt",
        "reference_low": 0.47,
        "reference_high": 0.47,
        "source": "nhtsa_utqg_ecfr_49cfr575_104_table2",
    },
    {
        "anchor": "utqg_wet_asphalt_b_c_boundary",
        "mapped_state": "very_wet",
        "mapped_material": "asphalt",
        "reference_low": 0.38,
        "reference_high": 0.38,
        "source": "nhtsa_utqg_ecfr_49cfr575_104_table2",
    },
]


@dataclass(frozen=True)
class LabelRecord:
    friction: str | None = None
    material: str | None = None
    unevenness: str | None = None
    wetness: str | None = None
    snow: str | None = None
    risk: str | None = None
    mu_low: float | None = None
    mu_high: float | None = None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "null", "unknown", "-1"}:
        return None
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    aliases = {
        "dirt": "dirt_mud",
        "mud": "dirt_mud",
        "dirtmud": "dirt_mud",
        "dirt_mud": "dirt_mud",
        "cobble": "cobblestone",
        "cobbles": "cobblestone",
        "cobblestone": "cobblestone",
        "semi_wet": "damp",
        "verywet": "very_wet",
        "waterlogged": "water",
        "fully_packed": "packed_snow",
        "fully_packed_snow": "packed_snow",
        "packed": "packed_snow",
        "partially_covered": "partial_snow",
        "partially_covered_snow": "partial_snow",
        "partial": "partial_snow",
        "fresh_fallen_snow": "fresh_snow",
        "snow_covered": "fresh_snow",
        "snow": "fresh_snow",
        "rough_ice": "ice",
        "smooth_ice": "ice",
    }
    return aliases.get(text, text)


def label_to_index(task: str, value: Any) -> int:
    label = normalize_label(value)
    if label is None:
        return IGNORE_INDEX
    labels = TASKS[task]
    return labels.index(label) if label in labels else IGNORE_INDEX


def index_to_label(task: str, idx: int) -> str:
    return TASKS[task][idx]


def weak_mu_interval_from_state(
    friction: str | None = None,
    wetness: str | None = None,
    snow: str | None = None,
    material: str | None = None,
) -> tuple[float | None, float | None]:
    """Return a conservative visual-evidence-compatible friction interval.

    These are weak prior intervals, not physical measurements. They are wide on
    purpose because unobserved tire, speed, temperature, and load variables are
    not available in public visual datasets.
    """
    friction = normalize_label(friction)
    wetness = normalize_label(wetness)
    snow = normalize_label(snow)
    material = normalize_label(material)

    state = friction or wetness
    if snow and snow != "none":
        state = snow

    intervals = {
        "dry": (0.70, 1.10),
        "damp": (0.55, 0.90),
        "wet": (0.45, 0.80),
        "very_wet": (0.30, 0.65),
        "water": (0.20, 0.60),
        "partial_snow": (0.20, 0.55),
        "fresh_snow": (0.18, 0.45),
        "melted_snow": (0.15, 0.45),
        "packed_snow": (0.12, 0.40),
        "ice": (0.03, 0.25),
    }
    if state not in intervals:
        return None, None
    low, high = intervals[state]
    if material in {"gravel", "dirt_mud"} and state == "dry":
        low, high = 0.45, 0.85
    if material == "cobblestone" and state in {"wet", "very_wet", "water"}:
        low = max(0.10, low - 0.05)
        high = max(low + 0.05, high - 0.05)
    return low, high


def risk_from_mu_interval(mu_low: float | None, mu_high: float | None) -> str | None:
    if mu_low is None or mu_high is None:
        return None
    midpoint = (float(mu_low) + float(mu_high)) / 2.0
    if midpoint >= 0.80:
        return "very_low"
    if midpoint >= 0.60:
        return "low"
    if midpoint >= 0.40:
        return "medium"
    if midpoint >= 0.22:
        return "high"
    return "very_high"


def parse_rscd_label(label: str) -> LabelRecord:
    label = normalize_label(label)
    if label is None:
        return LabelRecord()
    parts = label.split("_")
    friction = None
    material = None
    unevenness = None

    if label in {"fresh_snow", "melted_snow", "ice"}:
        friction = label
        snow = "ice" if label == "ice" else label
        low, high = weak_mu_interval_from_state(friction=friction, snow=snow)
        return LabelRecord(
            friction=friction,
            snow=snow,
            risk=risk_from_mu_interval(low, high),
            mu_low=low,
            mu_high=high,
        )

    if parts:
        friction = normalize_label(parts[0])
    if len(parts) >= 2:
        material = normalize_label(parts[1])
    if len(parts) >= 3:
        unevenness = normalize_label(parts[2])
    wetness = friction if friction in WETNESS else None
    low, high = weak_mu_interval_from_state(friction=friction, wetness=wetness, material=material)
    return LabelRecord(
        friction=friction,
        material=material,
        unevenness=unevenness,
        wetness=wetness,
        snow="none" if friction not in {"fresh_snow", "melted_snow", "ice"} else friction,
        risk=risk_from_mu_interval(low, high),
        mu_low=low,
        mu_high=high,
    )


def parse_roadsaw_label(label: str) -> LabelRecord:
    label = normalize_label(label)
    if label is None:
        return LabelRecord()
    parts = label.split("_")
    material = normalize_label(parts[0]) if parts else None
    wetness = normalize_label("_".join(parts[1:])) if len(parts) > 1 else None
    friction = wetness if wetness in FRICTION_STATES else None
    low, high = weak_mu_interval_from_state(friction=friction, wetness=wetness, material=material)
    return LabelRecord(
        friction=friction,
        material=material,
        wetness=wetness,
        snow="none",
        risk=risk_from_mu_interval(low, high),
        mu_low=low,
        mu_high=high,
    )


def parse_roadsc_label(label: str) -> LabelRecord:
    label = normalize_label(label)
    if label is None:
        return LabelRecord()
    if label in {"fresh_snow", "packed_snow", "partial_snow", "melted_snow", "ice"}:
        snow = label
        friction = label if label in FRICTION_STATES else None
        low, high = weak_mu_interval_from_state(friction=friction, snow=snow)
        return LabelRecord(
            friction=friction,
            snow=snow,
            risk=risk_from_mu_interval(low, high),
            mu_low=low,
            mu_high=high,
        )
    return parse_roadsaw_label(label)


def record_to_manifest_fields(record: LabelRecord) -> dict[str, Any]:
    return {
        "friction_label": record.friction,
        "material_label": record.material,
        "unevenness_label": record.unevenness,
        "wetness_label": record.wetness,
        "snow_label": record.snow,
        "risk_label": record.risk,
        "mu_low": record.mu_low,
        "mu_high": record.mu_high,
    }


def infer_record(dataset_name: str, label: str) -> LabelRecord:
    name = normalize_label(dataset_name) or ""
    if "rscd" in name:
        return parse_rscd_label(label)
    if "roadsaw" in name:
        return parse_roadsaw_label(label)
    if "roadsc" in name:
        return parse_roadsc_label(label)
    return parse_rscd_label(label)
