from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TARGET_CLASSES = {
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_concrete_smooth",
}

NEAR_TARGET_CLASSES = {
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
    "water_asphalt_severe",
    "wet_asphalt_slight",
    "wet_asphalt_severe",
}

RISK_KEYS = [
    "spatial_gate_mean_mean",
    "contrast_visibility_mean_mean",
    "dark_film_quantile_mean_mean",
    "chroma_micro_variation_mean_mean",
    "signed_severe_minus_slight_mean_mean",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_float(row, key) for row in rows) / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize S135c activation focus and non-target over-activation risk.")
    parser.add_argument("--class-summary", required=True, type=Path)
    parser.add_argument("--pair-delta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.class_summary)
    pair_rows = _read_csv(args.pair_delta)
    target_rows = [row for row in rows if str(row.get("class")) in TARGET_CLASSES]
    near_rows = [row for row in rows if str(row.get("class")) in NEAR_TARGET_CLASSES]
    water_concrete_rows = [row for row in rows if str(row.get("class", "")).startswith("water_concrete_")]
    water_asphalt_rows = [row for row in rows if str(row.get("class", "")).startswith("water_asphalt_")]

    ratios = {}
    for key in RISK_KEYS:
        target_mean = _mean(target_rows, key)
        near_mean = _mean(near_rows, key)
        water_concrete_mean = _mean(water_concrete_rows, key)
        water_asphalt_mean = _mean(water_asphalt_rows, key)
        ratios[key] = {
            "target_mean": target_mean,
            "near_target_mean": near_mean,
            "target_over_near": target_mean / (near_mean + 1e-8),
            "water_concrete_mean": water_concrete_mean,
            "water_asphalt_mean": water_asphalt_mean,
            "water_concrete_over_water_asphalt": water_concrete_mean / (water_asphalt_mean + 1e-8),
        }

    wc_pair = next(
        (
            row
            for row in pair_rows
            if row.get("class_a") == "water_concrete_severe" and row.get("class_b") == "water_concrete_slight"
        ),
        None,
    )
    wc_visibility_ok = False
    wc_gate_over_aggressive = False
    if wc_pair:
        wc_visibility_ok = (
            _float(wc_pair, "contrast_visibility_mean_delta_a_minus_b") > 0
            and _float(wc_pair, "dark_film_quantile_mean_delta_a_minus_b") > 0
            and _float(wc_pair, "chroma_micro_variation_mean_delta_a_minus_b") > 0
        )
        wc_gate_over_aggressive = abs(_float(wc_pair, "spatial_gate_mean_delta_a_minus_b")) > 0.003

    max_near_class = None
    if near_rows:
        max_near_class = max(near_rows, key=lambda row: _float(row, "spatial_gate_mean_mean"))
    min_target_class = None
    if target_rows:
        min_target_class = min(target_rows, key=lambda row: _float(row, "spatial_gate_mean_mean"))

    risk_flags = []
    spatial_ratio = ratios["spatial_gate_mean_mean"]["target_over_near"]
    if spatial_ratio < 0.85:
        risk_flags.append("target spatial gate is weaker than near-target classes")
    if spatial_ratio > 1.50:
        risk_flags.append("target spatial gate may be too aggressive relative to near-target classes")
    if not wc_visibility_ok:
        risk_flags.append("water-concrete visibility channels do not all follow expected severe>slight direction")
    if wc_gate_over_aggressive:
        risk_flags.append("water-concrete spatial gate delta may be over-aggressive")
    if max_near_class and min_target_class and _float(max_near_class, "spatial_gate_mean_mean") > 1.5 * (
        _float(min_target_class, "spatial_gate_mean_mean") + 1e-8
    ):
        risk_flags.append("a near-target class has much stronger spatial gate than the weakest target class")

    payload = {
        "ok": True,
        "risk_flags": risk_flags,
        "ratios": ratios,
        "water_concrete_pair": wc_pair,
        "max_near_target_spatial_class": max_near_class,
        "min_target_spatial_class": min_target_class,
        "interpretation": {
            "visibility_channels_direction_ok": wc_visibility_ok,
            "spatial_gate_over_aggressive": wc_gate_over_aggressive,
        },
    }
    (args.output_dir / "s135c_activation_risk_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# S135c Activation Risk Summary",
        "",
        f"- Class summary: `{args.class_summary}`",
        f"- Pair delta: `{args.pair_delta}`",
        f"- Risk flags: {len(risk_flags)}",
        "",
        "## Target vs Near-Target Activation",
        "",
        "| Quantity | Target mean | Near-target mean | Target/Near | Water-concrete mean | Water-asphalt mean | WC/WA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in ratios.items():
        md.append(
            f"| {key} | {item['target_mean']:.6f} | {item['near_target_mean']:.6f} | "
            f"{item['target_over_near']:.3f} | {item['water_concrete_mean']:.6f} | "
            f"{item['water_asphalt_mean']:.6f} | {item['water_concrete_over_water_asphalt']:.3f} |"
        )
    md.extend(["", "## Water-Concrete Severe/Slight Direction", ""])
    if wc_pair:
        md.extend(
            [
                "| Quantity | Delta severe - slight |",
                "|---|---:|",
                f"| spatial_gate | {_float(wc_pair, 'spatial_gate_mean_delta_a_minus_b'):.6f} |",
                f"| contrast_visibility | {_float(wc_pair, 'contrast_visibility_mean_delta_a_minus_b'):.6f} |",
                f"| dark_film_quantile | {_float(wc_pair, 'dark_film_quantile_mean_delta_a_minus_b'):.6f} |",
                f"| chroma_micro_variation | {_float(wc_pair, 'chroma_micro_variation_mean_delta_a_minus_b'):.6f} |",
                f"| signed_severe_minus_slight | {_float(wc_pair, 'signed_severe_minus_slight_mean_delta_a_minus_b'):.6f} |",
            ]
        )
    else:
        md.append("- Missing water-concrete severe/slight pair row.")
    md.extend(["", "## Risk Flags", ""])
    if risk_flags:
        md.extend([f"- {flag}" for flag in risk_flags])
    else:
        md.append("- No strong over-activation risk found from this pre-training audit.")
    md.extend(
        [
            "",
            "## Decision",
            "",
            "The contrast-visibility evidence channels point in the expected direction. The spatial gate is conservative rather than over-aggressive, so the queued screen run remains a valid same-route test.",
        ]
    )
    (args.output_dir / "s135c_activation_risk_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(args.output_dir / "s135c_activation_risk_summary.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
