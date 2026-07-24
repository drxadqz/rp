from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
SUMMARY = Path("reports/paper_protocol_summary")
OUT = SUMMARY / "rscd_formal_hard_class_diagnosis"

BASELINE = "formal_convnext_tiny_b12e20_resume"
PHYSICS = "formal_physics_texture_quality_b12e20_resume"


def main() -> None:
    baseline = _load_report(BASELINE)
    physics = _load_report(PHYSICS)
    if not baseline or not physics:
        raise SystemExit("Missing formal ConvNeXt or PhysicsTexture evaluate_test.json")

    rows = []
    base_rows = _class_rows(baseline.get("classification_report", {}))
    phys_rows = _class_rows(physics.get("classification_report", {}))
    labels = sorted(set(base_rows) | set(phys_rows))
    for label in labels:
        b = base_rows.get(label, {})
        p = phys_rows.get(label, {})
        rows.append(
            {
                "class_label": label,
                "friction": _friction(label),
                "material": _material(label),
                "roughness": _roughness(label),
                "support": int(p.get("support") or b.get("support") or 0),
                "baseline_precision": _num(b.get("precision")),
                "baseline_recall": _num(b.get("recall")),
                "baseline_f1": _num(b.get("f1")),
                "physics_precision": _num(p.get("precision")),
                "physics_recall": _num(p.get("recall")),
                "physics_f1": _num(p.get("f1")),
            }
        )
    for row in rows:
        row["delta_precision"] = row["physics_precision"] - row["baseline_precision"]
        row["delta_recall"] = row["physics_recall"] - row["baseline_recall"]
        row["delta_f1"] = row["physics_f1"] - row["baseline_f1"]

    hard_labels = {
        "water_concrete_slight",
        "water_asphalt_slight",
        "wet_concrete_slight",
        "water_concrete_severe",
        "dry_asphalt_severe",
    }
    result = {
        "claim_boundary": (
            "Formal hard-class diagnosis for RSCD-27 class-label classification. "
            "It compares the matched formal ConvNeXt baseline with formal PhysicsTexture; "
            "it does not claim measured tire-road friction regression."
        ),
        "baseline": BASELINE,
        "physics": PHYSICS,
        "baseline_summary": _summary_with_mean_pr(baseline),
        "physics_summary": _summary_with_mean_pr(physics),
        "rows": sorted(rows, key=lambda item: item["physics_f1"]),
        "hard_route_rows": [row for row in rows if row["class_label"] in hard_labels],
        "largest_f1_gains": sorted(rows, key=lambda item: item["delta_f1"], reverse=True)[:10],
        "largest_f1_losses": sorted(rows, key=lambda item: item["delta_f1"])[:10],
    }

    SUMMARY.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(_to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def _load_report(name: str) -> dict[str, Any] | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _class_rows(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    out = {}
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        out[str(label)] = {
            "precision": _num(item.get("precision")),
            "recall": _num(item.get("recall")),
            "f1": _num(item.get("f1-score")),
            "support": int(item.get("support") or 0),
        }
    return out


def _summary_with_mean_pr(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary", {}))
    report = payload.get("classification_report", {})
    rows = _class_rows(report)
    if rows:
        summary.setdefault("mean_precision", sum(row["precision"] for row in rows.values()) / len(rows))
        summary.setdefault("mean_recall", sum(row["recall"] for row in rows.values()) / len(rows))
    return summary


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _canon(label: str) -> str:
    return str(label).strip().lower().replace("-", "_")


def _friction(label: str) -> str:
    label = _canon(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    return label.split("_")[0] if label else "unknown"


def _material(label: str) -> str | None:
    label = _canon(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else None


def _roughness(label: str) -> str | None:
    label = _canon(label)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[2] if len(parts) >= 3 else None


def _pct(value: float, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    b = result["baseline_summary"]
    p = result["physics_summary"]
    lines = [
        "# RSCD Formal Hard-Class Diagnosis",
        "",
        result["claim_boundary"],
        "",
        "## Formal Summary",
        "",
        "| model | Top-1 | Mean-F1 | Mean-P | Mean-R |",
        "|---|---:|---:|---:|---:|",
        (
            f"| `{result['baseline']}` | {_pct(_num(b.get('top1')))} | "
            f"{_pct(_num(b.get('macro_f1')))} | {_pct(_num(b.get('mean_precision')))} | "
            f"{_pct(_num(b.get('mean_recall')))} |"
        ),
        (
            f"| `{result['physics']}` | {_pct(_num(p.get('top1')))} | "
            f"{_pct(_num(p.get('macro_f1')))} | {_pct(_num(p.get('mean_precision')))} | "
            f"{_pct(_num(p.get('mean_recall')))} |"
        ),
        "",
        "## Target Hard Classes",
        "",
        "| class | support | ConvNeXt F1 | Physics F1 | dF1 | ConvNeXt recall | Physics recall | dRecall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["hard_route_rows"], key=lambda item: item["physics_f1"]):
        lines.append(_row_line(row))

    lines.extend(
        [
            "",
            "## Worst PhysicsTexture Classes",
            "",
            "| class | support | ConvNeXt F1 | Physics F1 | dF1 | ConvNeXt recall | Physics recall | dRecall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"][:10]:
        lines.append(_row_line(row))

    lines.extend(
        [
            "",
            "## Largest Formal F1 Gains From PhysicsTexture",
            "",
            "| class | support | ConvNeXt F1 | Physics F1 | dF1 | dRecall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["largest_f1_gains"]:
        lines.append(
            f"| `{row['class_label']}` | {row['support']} | {_pct(row['baseline_f1'])} | "
            f"{_pct(row['physics_f1'])} | {_pct(row['delta_f1'], signed=True)} | "
            f"{_pct(row['delta_recall'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## Design Implication",
            "",
            (
                "PhysicsTexture is retained because it improves the formal ConvNeXt baseline and "
                "most friction-relevant slices. The next algorithm should not add generic attention "
                "complexity; it should target material-conditioned wetness and fine patch texture, "
                "especially wet/water concrete and slight/severe roughness confusions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _row_line(row: dict[str, Any]) -> str:
    return (
        f"| `{row['class_label']}` | {row['support']} | {_pct(row['baseline_f1'])} | "
        f"{_pct(row['physics_f1'])} | {_pct(row['delta_f1'], signed=True)} | "
        f"{_pct(row['baseline_recall'])} | {_pct(row['physics_recall'])} | "
        f"{_pct(row['delta_recall'], signed=True)} |"
    )


if __name__ == "__main__":
    main()
