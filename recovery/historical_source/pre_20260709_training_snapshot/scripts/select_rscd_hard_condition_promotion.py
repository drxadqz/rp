from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_hard_condition_promotion_decision")

CANDIDATES = {
    "fast_physics_texture_hard_condition_boost035": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--physics-dim",
        "96",
        "--hard-condition-boost",
        "0.35",
    ],
    "fast_physics_texture_hard_condition_hier_boost035": [
        "--use-physics-branch",
        "--physics-quality-cues",
        "--physics-dim",
        "96",
        "--hard-condition-boost",
        "0.35",
        "--hierarchical-smoothing",
        "0.08",
    ],
}


def main() -> None:
    reference = load_summary("fast_physics_texture_quality")
    rows = []
    for name, args in CANDIDATES.items():
        summary = load_summary(name)
        if summary is None:
            rows.append({"name": name, "status": "missing"})
            continue
        rows.append(
            {
                "name": name,
                "status": "available",
                "top1": summary["top1"],
                "macro_f1": summary["macro_f1"],
                "balanced_accuracy": summary.get("balanced_accuracy"),
                "delta_top1_vs_physics": summary["top1"] - reference["top1"] if reference else None,
                "delta_macro_f1_vs_physics": summary["macro_f1"] - reference["macro_f1"] if reference else None,
                "formal_args": args,
            }
        )

    available = [r for r in rows if r.get("status") == "available"]
    promoted = None
    if reference and available:
        best = max(available, key=lambda r: (float(r["macro_f1"]), float(r["top1"])))
        macro_gain = float(best["macro_f1"]) - float(reference["macro_f1"])
        top1_gain = float(best["top1"]) - float(reference["top1"])
        if macro_gain > 0.001 or (top1_gain > 0.003 and macro_gain >= -0.001):
            promoted = {
                "name": best["name"],
                "formal_output_dir": str(ROOT / best["name"].replace("fast_", "formal_") / ""),
                "formal_args": best["formal_args"],
                "macro_gain": macro_gain,
                "top1_gain": top1_gain,
            }

    result = {
        "reference": "fast_physics_texture_quality",
        "promotion_rule": (
            "Promote hard-condition candidate only if it improves fast Macro-F1 by more "
            "than 0.1 percentage point, or improves Top-1 by more than 0.3 percentage "
            "point without losing more than 0.1 percentage point Macro-F1."
        ),
        "rows": rows,
        "promoted": promoted,
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def load_summary(name: str) -> dict | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary")


def pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def to_markdown(result: dict) -> str:
    lines = [
        "# RSCD Hard-Condition Promotion Decision",
        "",
        f"Reference: `{result['reference']}`",
        "",
        result["promotion_rule"],
        "",
        "| candidate | status | Top-1 | Macro-F1 | dTop1 | dMacro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            "| `{name}` | {status} | {top1} | {mf1} | {dt} | {df} |".format(
                name=row["name"],
                status=row["status"],
                top1=pct(row.get("top1")),
                mf1=pct(row.get("macro_f1")),
                dt=pct(row.get("delta_top1_vs_physics"), signed=True),
                df=pct(row.get("delta_macro_f1_vs_physics"), signed=True),
            )
        )
    lines.append("")
    if result["promoted"]:
        promoted = result["promoted"]
        lines.append(f"Promoted: `{promoted['name']}`")
        lines.append("")
        lines.append(f"Formal output dir: `{promoted['formal_output_dir']}`")
    else:
        lines.append("Promoted: none yet.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
