from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_foundation_promotion_decision")

BASELINE = "fast_convnext_tiny"
PHYSICS = "fast_physics_texture_quality"

CANDIDATES = {
    "fast_dinov2_global_rscd": [
        "--backbone",
        "timm:vit_small_patch14_dinov2",
        "--embedding-dim",
        "384",
        "--pretrained",
    ],
    "fast_dinov2_physics_texture_rscd": [
        "--backbone",
        "timm:vit_small_patch14_dinov2",
        "--embedding-dim",
        "384",
        "--pretrained",
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim",
        "96",
    ],
}

SKIP_MARKERS = {
    "fast_dinov2_physics_texture_rscd": "skipped_after_global_failure.json",
}


def main() -> None:
    baseline_payload = load_payload(BASELINE)
    physics_payload = load_payload(PHYSICS)
    baseline = summary(baseline_payload)
    physics = summary(physics_payload)
    baseline_slices = slice_metrics(baseline_payload)
    physics_slices = slice_metrics(physics_payload)

    rows: list[dict[str, Any]] = []
    for name, args in CANDIDATES.items():
        payload = load_payload(name)
        row_summary = summary(payload)
        if row_summary is None:
            marker = skip_marker(name)
            if marker is not None:
                rows.append(
                    {
                        "name": name,
                        "status": "skipped_pruned",
                        "reason": marker.get("reason"),
                        "formal_args": args,
                    }
                )
                continue
            rows.append({"name": name, "status": "missing", "formal_args": args})
            continue
        slices = slice_metrics(payload)
        status = "available"
        reason = None
        if name == "fast_dinov2_global_rscd" and row_summary["macro_f1"] < 0.50:
            status = "pruned_fast_screen"
            reason = (
                "End-to-end DINOv2 fast screen is far below the ConvNeXt/PhysicsTexture references; "
                "do not promote this protocol. Revisit only as frozen feature extraction or linear probing."
            )
        rows.append(
            {
                "name": name,
                "status": status,
                "reason": reason,
                "top1": row_summary["top1"],
                "macro_f1": row_summary["macro_f1"],
                "balanced_accuracy": row_summary.get("balanced_accuracy"),
                "wet_water_f1": slices.get("wet_water_f1"),
                "water_concrete_f1": slices.get("water_concrete_f1"),
                "delta_top1_vs_baseline": delta(row_summary, baseline, "top1"),
                "delta_macro_f1_vs_baseline": delta(row_summary, baseline, "macro_f1"),
                "delta_top1_vs_physics": delta(row_summary, physics, "top1"),
                "delta_macro_f1_vs_physics": delta(row_summary, physics, "macro_f1"),
                "delta_wet_water_f1_vs_physics": delta(slices, physics_slices, "wet_water_f1"),
                "delta_water_concrete_f1_vs_physics": delta(slices, physics_slices, "water_concrete_f1"),
                "formal_args": args,
            }
        )

    promoted = choose_promotion(rows)
    result = {
        "references": {
            "baseline": BASELINE,
            "physics": PHYSICS,
        },
        "promotion_rule": (
            "Promote a DINOv2 foundation candidate only if the fast screen gives a "
            "material signal: DINOv2+PhysicsTexture must improve Macro-F1 over "
            "PhysicsTexture by at least 0.5 percentage point without reducing Top-1; "
            "or DINOv2 global must improve Macro-F1 over ConvNeXt by at least 1.0 "
            "percentage point; or a wet/water or water-concrete hard slice must gain "
            "at least 1.0 percentage point without a main Macro-F1 loss greater than "
            "0.2 percentage point versus PhysicsTexture."
        ),
        "rows": rows,
        "promoted": promoted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def choose_promotion(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    available = [row for row in rows if row.get("status") == "available"]
    if not available:
        return None

    promoted_rows = []
    for row in available:
        name = row["name"]
        d_f1_physics = float(row.get("delta_macro_f1_vs_physics") or 0.0)
        d_top1_physics = float(row.get("delta_top1_vs_physics") or 0.0)
        d_f1_base = float(row.get("delta_macro_f1_vs_baseline") or 0.0)
        d_wet = float(row.get("delta_wet_water_f1_vs_physics") or 0.0)
        d_water_concrete = float(row.get("delta_water_concrete_f1_vs_physics") or 0.0)
        reasons = []
        if name == "fast_dinov2_physics_texture_rscd" and d_f1_physics >= 0.005 and d_top1_physics >= 0.0:
            reasons.append("DINOv2+PhysicsTexture beats the validated PhysicsTexture fast reference on Macro-F1 without Top-1 loss.")
        if name == "fast_dinov2_global_rscd" and d_f1_base >= 0.010:
            reasons.append("DINOv2 global beats the clean ConvNeXt fast baseline by at least 1.0pp Macro-F1.")
        if (d_wet >= 0.010 or d_water_concrete >= 0.010) and d_f1_physics >= -0.002:
            reasons.append("Hard wet/water slice improves enough without a material main Macro-F1 loss.")
        if reasons:
            item = dict(row)
            item["promotion_reasons"] = reasons
            promoted_rows.append(item)

    if not promoted_rows:
        return None

    best = max(
        promoted_rows,
        key=lambda row: (
            float(row.get("macro_f1") or 0.0),
            float(row.get("top1") or 0.0),
            float(row.get("wet_water_f1") or 0.0),
        ),
    )
    formal_name = best["name"].replace("fast_", "formal_")
    return {
        "name": best["name"],
        "formal_output_dir": str(ROOT / formal_name / ""),
        "formal_args": best["formal_args"],
        "promotion_reasons": best["promotion_reasons"],
        "macro_gain_vs_physics": best.get("delta_macro_f1_vs_physics"),
        "top1_gain_vs_physics": best.get("delta_top1_vs_physics"),
        "macro_gain_vs_baseline": best.get("delta_macro_f1_vs_baseline"),
        "wet_water_gain_vs_physics": best.get("delta_wet_water_f1_vs_physics"),
        "water_concrete_gain_vs_physics": best.get("delta_water_concrete_f1_vs_physics"),
    }


def load_payload(name: str) -> dict[str, Any] | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def skip_marker(name: str) -> dict[str, Any] | None:
    marker_name = SKIP_MARKERS.get(name)
    if marker_name is None:
        return None
    path = ROOT / name / marker_name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    lines = [
        "# RSCD Foundation Promotion Decision",
        "",
        f"Baseline reference: `{result['references']['baseline']}`",
        f"Physics reference: `{result['references']['physics']}`",
        "",
        result["promotion_rule"],
        "",
        "| candidate | status | Top-1 | Macro-F1 | wet/water F1 | water-concrete F1 | dF1 vs base | dF1 vs physics | dWet vs physics |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| `{name}` | {status} | {top1} | {mf1} | {wet} | {wc} | {dfb} | {dfp} | {dwet} |".format(
                name=row["name"],
                status=row["status"],
                top1=pct(row.get("top1")),
                mf1=pct(row.get("macro_f1")),
                wet=pct(row.get("wet_water_f1")),
                wc=pct(row.get("water_concrete_f1")),
                dfb=pct(row.get("delta_macro_f1_vs_baseline"), signed=True),
                dfp=pct(row.get("delta_macro_f1_vs_physics"), signed=True),
                dwet=pct(row.get("delta_wet_water_f1_vs_physics"), signed=True),
            )
        )
    lines.append("")
    if result["promoted"]:
        promoted = result["promoted"]
        lines.append(f"Promoted: `{promoted['name']}`")
        lines.append("")
        for reason in promoted["promotion_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append(f"Formal output dir: `{promoted['formal_output_dir']}`")
    else:
        lines.append("Promoted: none yet.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
