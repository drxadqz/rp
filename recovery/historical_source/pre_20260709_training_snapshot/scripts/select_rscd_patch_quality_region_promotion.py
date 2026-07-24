from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_patch_quality_region_decision")

REFERENCE = "fast_physics_texture_quality"
CANDIDATES = [
    {
        "name": "fast_physics_texture_quality_patch_stats",
        "formal_name": "formal_physics_texture_quality_patch_stats",
        "formal_args": [
            "--use-physics-branch",
            "--physics-quality-cues",
            "--no-physics-quality-region-cues",
            "--physics-dim",
            "96",
        ],
        "note": "192px ConvNeXt patch-invariant quality statistics.",
    },
    {
        "name": "fast_physics_texture_quality_patch_stats_224",
        "formal_name": "formal_physics_texture_quality_patch_stats_224",
        "formal_args": [
            "--use-physics-branch",
            "--physics-quality-cues",
            "--no-physics-quality-region-cues",
            "--physics-dim",
            "96",
        ],
        "note": "224px ConvNeXt patch-invariant quality statistics for the RSCD fine-texture resolution hypothesis.",
    },
]


def main() -> None:
    reference_payload = load_payload(REFERENCE)
    reference = summary(reference_payload)
    reference_slices = slice_metrics(reference_payload)

    rows = []
    for cfg in CANDIDATES:
        candidate_payload = load_payload(cfg["name"])
        candidate = summary(candidate_payload)
        candidate_slices = slice_metrics(candidate_payload)
        rows.append(
            {
                "reference": REFERENCE,
                "candidate": cfg["name"],
                "formal_name": cfg["formal_name"],
                "formal_args": cfg["formal_args"],
                "note": cfg["note"],
                "reference_available": reference is not None,
                "candidate_available": candidate is not None,
                "reference_summary": reference,
                "candidate_summary": candidate,
                "reference_slices": reference_slices,
                "candidate_slices": candidate_slices,
                "delta_top1": delta(candidate, reference, "top1"),
                "delta_macro_f1": delta(candidate, reference, "macro_f1"),
                "delta_wet_water_f1": delta(candidate_slices, reference_slices, "wet_water_f1"),
                "delta_water_concrete_f1": delta(candidate_slices, reference_slices, "water_concrete_f1"),
            }
        )

    promoted = choose_promotion(rows)
    decision = choose_text_decision(rows, promoted)
    result = {
        "claim_boundary": (
            "This selector answers whether RSCD should keep bottom-vs-top "
            "quality-region cues inside PhysicsTexture. It is based on the fast "
            "RSCD subset until a promoted formal run exists. The 224px row also "
            "tests whether RSCD patch fine texture is being lost by 192px resizing."
        ),
        "promotion_rule": (
            "Promote patch-invariant PhysicsTexture to a formal RSCD run if it "
            "improves fast Macro-F1 by at least 0.3 percentage point without Top-1 "
            "loss, or improves wet/water or water-concrete F1 by at least 1.0 "
            "percentage point without losing more than 0.2 percentage point "
            "Macro-F1. If it is neutral, keep the code option but remove the "
            "contact-region explanation for RSCD. If it is worse, keep old weights "
            "for existing formal comparisons and describe region cues only as a "
            "front-view/general-dataset option."
        ),
        "rows": rows,
        "decision": decision,
        "promoted": promoted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def choose_promotion(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible: list[tuple[dict[str, Any], list[str]]] = []
    for row in rows:
        if not row["reference_available"] or not row["candidate_available"]:
            continue
        d_f1 = float(row.get("delta_macro_f1") or 0.0)
        d_top1 = float(row.get("delta_top1") or 0.0)
        d_wet = float(row.get("delta_wet_water_f1") or 0.0)
        d_wc = float(row.get("delta_water_concrete_f1") or 0.0)
        reasons = []
        if d_f1 >= 0.003 and d_top1 >= 0.0:
            reasons.append("Patch-invariant quality statistics improve fast Macro-F1 without Top-1 loss.")
        if (d_wet >= 0.010 or d_wc >= 0.010) and d_f1 >= -0.002:
            reasons.append("Patch-invariant quality statistics improve a hard wet/water slice without a material main-score loss.")
        if reasons:
            eligible.append((row, reasons))

    if not eligible:
        return None

    row, reasons = max(
        eligible,
        key=lambda item: (
            float(item[0].get("delta_macro_f1") or 0.0),
            float(item[0].get("delta_top1") or 0.0),
            float(item[0].get("delta_wet_water_f1") or 0.0),
        ),
    )
    return {
        "name": row["candidate"],
        "formal_output_dir": str(ROOT / row["formal_name"] / ""),
        "formal_args": row["formal_args"],
        "promotion_reasons": reasons,
        "delta_top1": row.get("delta_top1"),
        "delta_macro_f1": row.get("delta_macro_f1"),
        "delta_wet_water_f1": row.get("delta_wet_water_f1"),
        "delta_water_concrete_f1": row.get("delta_water_concrete_f1"),
    }


def choose_text_decision(rows: list[dict[str, Any]], promoted: dict[str, Any] | None) -> dict[str, str]:
    if not any(row["candidate_available"] for row in rows):
        return {
            "status": "pending_fast_screen",
            "action": "Wait for patch-invariant quality fast screens at 192px and 224px.",
        }
    if promoted:
        return {
            "status": "promote_patch_invariant_quality",
            "action": "Run formal RSCD with --no-physics-quality-region-cues and update the RSCD method if formal evidence confirms the fast gain.",
        }
    available = [row for row in rows if row["candidate_available"]]
    best = max(available, key=lambda row: float(row.get("delta_macro_f1") or -999.0))
    d_f1 = best.get("delta_macro_f1")
    if d_f1 is not None and float(d_f1) < -0.003:
        return {
            "status": "do_not_replace_by_performance",
            "action": "Keep old formal result for performance, but remove the tire-contact-zone explanation for RSCD because image geometry does not support it.",
        }
    return {
        "status": "neutral_keep_as_ablation",
        "action": "Use patch-invariant wording in the paper; keep the switch for RSCD-specific ablations and avoid claiming bottom-contact semantics.",
    }


def load_payload(name: str) -> dict[str, Any] | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    item = payload.get("summary")
    return item if isinstance(item, dict) else None


def delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    if a.get(key) is None or b.get(key) is None:
        return None
    return float(a[key]) - float(b[key])


def slice_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    if not payload:
        return {}
    report = payload.get("classification_report", {})
    if not isinstance(report, dict):
        return {}
    rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict) or "support" not in metrics:
            continue
        rows.append(
            {
                "label": str(label),
                "f1": float(metrics.get("f1-score") or 0.0),
                "support": int(float(metrics.get("support") or 0.0)),
            }
        )
    return {
        "wet_water_f1": weighted_f1(rows, lambda row: friction_state(row["label"]) in {"wet", "very_wet", "water"}),
        "water_concrete_f1": weighted_f1(rows, lambda row: row["label"].startswith("water_concrete")),
    }


def friction_state(label: str) -> str:
    label = label.strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    if label.startswith("very_wet"):
        return "very_wet"
    return label.split("_")[0] if label else "unknown"


def weighted_f1(rows: list[dict[str, Any]], pred: Any) -> float:
    selected = [row for row in rows if pred(row)]
    support = sum(int(row["support"]) for row in selected)
    if support <= 0:
        return 0.0
    return sum(float(row["f1"]) * int(row["support"]) for row in selected) / support


def pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    lines = [
        "# RSCD Patch Quality-Region Decision",
        "",
        result["claim_boundary"],
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Action: {decision['action']}",
        "",
        "## Rule",
        "",
        result["promotion_rule"],
        "",
        "## Fast Comparison",
        "",
        "| item | Top-1 | Macro-F1 | wet/water F1 | water-concrete F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    if result["rows"]:
        ref = result["rows"][0]
        lines.append(
            "| reference region-cues | {rtop} | {rf1} | {rwet} | {rwc} |".format(
                rtop=pct((ref.get("reference_summary") or {}).get("top1")),
                rf1=pct((ref.get("reference_summary") or {}).get("macro_f1")),
                rwet=pct((ref.get("reference_slices") or {}).get("wet_water_f1")),
                rwc=pct((ref.get("reference_slices") or {}).get("water_concrete_f1")),
            )
        )
    for row in result["rows"]:
        lines.append(
            "| candidate `{name}` | {ctop} | {cf1} | {cwet} | {cwc} |".format(
                name=row["candidate"],
                ctop=pct((row.get("candidate_summary") or {}).get("top1")),
                cf1=pct((row.get("candidate_summary") or {}).get("macro_f1")),
                cwet=pct((row.get("candidate_slices") or {}).get("wet_water_f1")),
                cwc=pct((row.get("candidate_slices") or {}).get("water_concrete_f1")),
            )
        )
        lines.append(
            "| delta vs reference | {dtop} | {df1} | {dwet} | {dwc} |".format(
                dtop=pct(row.get("delta_top1"), signed=True),
                df1=pct(row.get("delta_macro_f1"), signed=True),
                dwet=pct(row.get("delta_wet_water_f1"), signed=True),
                dwc=pct(row.get("delta_water_concrete_f1"), signed=True),
            )
        )
    lines.append("")
    if result["promoted"]:
        lines.append(f"Promoted formal candidate: `{result['promoted']['name']}`")
    else:
        lines.append("Promoted formal candidate: none yet.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
