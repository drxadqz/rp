from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT = Path("reports/paper_protocol_summary/rscd_class_slice_comparison")
HARD_CORE_6 = {
    "water_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_asphalt_severe",
    "water_asphalt_severe",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RSCD class-wise and physical-slice metrics across runs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--baseline", default="fast_convnext_tiny")
    parser.add_argument("--reference", default="fast_physics_texture_quality")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Optional run-name prefix to include. Can be passed multiple times.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    runs = load_runs(args.root)
    rows = []
    for name, payload in sorted(runs.items()):
        if name.startswith("smoke_"):
            continue
        if args.include_prefix and not any(name.startswith(prefix) for prefix in args.include_prefix):
            continue
        report = payload.get("classification_report", {})
        per_class = class_rows(report)
        slices = slice_rows(per_class)
        rows.append(
            {
                "name": name,
                "summary": payload.get("summary", {}),
                "slices": slices,
                "worst_classes": sorted(per_class, key=lambda x: x["f1"])[:10],
                "best_classes": sorted(per_class, key=lambda x: x["f1"], reverse=True)[:10],
            }
        )

    baseline = next((r for r in rows if r["name"] == args.baseline), None)
    reference = next((r for r in rows if r["name"] == args.reference), None)
    result = {
        "claim_boundary": (
            "Class-slice comparisons diagnose RSCD-27 road-surface classification. "
            "They do not prove measured tire-road friction regression."
        ),
        "baseline": args.baseline,
        "reference": args.reference,
        "include_prefix": list(args.include_prefix),
        "runs": rows,
        "deltas_vs_baseline": deltas(rows, baseline),
        "deltas_vs_reference": deltas(rows, reference),
    }
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(args.out.with_suffix(".md"))


def load_runs(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(root.glob("*/evaluate_test.json")):
        try:
            out[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def class_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        rows.append(
            {
                "class_label": label,
                "friction": friction_state(label),
                "material": material_state(label),
                "roughness": roughness_state(label),
                "precision": float(item.get("precision") or 0.0),
                "recall": float(item.get("recall") or 0.0),
                "f1": float(item.get("f1-score") or 0.0),
                "support": int(item.get("support") or 0),
            }
        )
    return rows


def slice_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"friction:{row['friction']}"].append(row)
        if row["material"]:
            groups[f"material:{row['material']}"].append(row)
        if row["roughness"]:
            groups[f"roughness:{row['roughness']}"].append(row)
        if row["friction"] in {"wet", "water", "fresh_snow", "melted_snow", "ice"}:
            groups["safety:low_friction_visual"].append(row)
        if row["friction"] in {"wet", "water"}:
            groups["safety:wet_water"].append(row)
        if row["friction"] in {"fresh_snow", "melted_snow", "ice"}:
            groups["safety:winter"].append(row)
        if row["class_label"] in HARD_CORE_6:
            groups["safety:hard_core_6"].append(row)
        if row["friction"] in {"wet", "water"} and row["material"] == "concrete":
            groups["coupling:wet_water_concrete"].append(row)
        if row["friction"] in {"wet", "water"} and row["material"] == "asphalt":
            groups["coupling:wet_water_asphalt"].append(row)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
    support = sum(int(x["support"]) for x in items)
    if support <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    return {
        "precision": sum(x["precision"] * x["support"] for x in items) / support,
        "recall": sum(x["recall"] * x["support"] for x in items) / support,
        "f1": sum(x["f1"] * x["support"] for x in items) / support,
        "support": support,
    }


def deltas(rows: list[dict[str, Any]], base: dict[str, Any] | None) -> list[dict[str, Any]]:
    if base is None:
        return []
    out = []
    base_slices = base["slices"]
    for row in rows:
        slice_deltas = {}
        for name, metrics in row["slices"].items():
            if name not in base_slices:
                continue
            slice_deltas[name] = {
                "delta_f1": float(metrics["f1"]) - float(base_slices[name]["f1"]),
                "delta_recall": float(metrics["recall"]) - float(base_slices[name]["recall"]),
                "support": metrics["support"],
            }
        out.append({"name": row["name"], "slice_deltas": slice_deltas})
    return out


def friction_state(label: str) -> str:
    label = canonical(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    parts = label.split("_")
    return parts[0] if parts else "unknown"


def material_state(label: str) -> str | None:
    label = canonical(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else None


def roughness_state(label: str) -> str | None:
    label = canonical(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[2] if len(parts) >= 3 else None


def canonical(label: str) -> str:
    return str(label).strip().lower().replace("-", "_")


def pct(value: float, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RSCD Class Slice Comparison",
        "",
        result["claim_boundary"],
        "",
        "## Run Slices",
        "",
        "| run | Top-1 | Macro-F1 | hard-core-6 F1 | wet/water concrete F1 | wet/water asphalt F1 | wet/water F1 | winter F1 | low-friction visual F1 | dry F1 | water F1 | ice F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["runs"], key=lambda r: float(r["summary"].get("macro_f1") or 0.0), reverse=True):
        slices = row["slices"]
        summary = row["summary"]
        lines.append(
            "| `{name}` | {top1} | {mf1} | {hard} | {wwconcrete} | {wwasphalt} | {wetwater} | {winter} | {low} | {dry} | {water} | {ice} |".format(
                name=row["name"],
                top1=pct(float(summary.get("top1") or 0.0)),
                mf1=pct(float(summary.get("macro_f1") or 0.0)),
                hard=pct(float(slices.get("safety:hard_core_6", {}).get("f1") or 0.0)),
                wwconcrete=pct(float(slices.get("coupling:wet_water_concrete", {}).get("f1") or 0.0)),
                wwasphalt=pct(float(slices.get("coupling:wet_water_asphalt", {}).get("f1") or 0.0)),
                wetwater=pct(float(slices.get("safety:wet_water", {}).get("f1") or 0.0)),
                winter=pct(float(slices.get("safety:winter", {}).get("f1") or 0.0)),
                low=pct(float(slices.get("safety:low_friction_visual", {}).get("f1") or 0.0)),
                dry=pct(float(slices.get("friction:dry", {}).get("f1") or 0.0)),
                water=pct(float(slices.get("friction:water", {}).get("f1") or 0.0)),
                ice=pct(float(slices.get("friction:ice", {}).get("f1") or 0.0)),
            )
        )
    lines.extend(["", f"## Deltas Vs `{result['reference']}`", ""])
    reference_deltas = result.get("deltas_vs_reference", [])
    lines.extend(delta_table(reference_deltas))
    lines.extend(["", "## Worst Classes", ""])
    for row in sorted(result["runs"], key=lambda r: float(r["summary"].get("macro_f1") or 0.0), reverse=True):
        worst = ", ".join(f"{x['class_label']}={pct(x['f1'])}" for x in row["worst_classes"][:5])
        lines.append(f"- `{row['name']}`: {worst}")
    lines.append("")
    return "\n".join(lines)


def delta_table(delta_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| run | d wet/water F1 | d winter F1 | d low-friction F1 | d water F1 | d ice F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        d = row["slice_deltas"]
        lines.append(
            "| `{name}` | {wetwater} | {winter} | {low} | {water} | {ice} |".format(
                name=row["name"],
                wetwater=pct(float(d.get("safety:wet_water", {}).get("delta_f1") or 0.0), signed=True),
                winter=pct(float(d.get("safety:winter", {}).get("delta_f1") or 0.0), signed=True),
                low=pct(float(d.get("safety:low_friction_visual", {}).get("delta_f1") or 0.0), signed=True),
                water=pct(float(d.get("friction:water", {}).get("delta_f1") or 0.0), signed=True),
                ice=pct(float(d.get("friction:ice", {}).get("delta_f1") or 0.0), signed=True),
            )
        )
    return lines


if __name__ == "__main__":
    main()
