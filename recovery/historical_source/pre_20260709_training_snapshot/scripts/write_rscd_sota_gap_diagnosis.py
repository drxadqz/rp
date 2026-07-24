from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
SUMMARY = Path("reports/paper_protocol_summary")
OUT = SUMMARY / "rscd_sota_gap_diagnosis"

BASELINE = "formal_convnext_tiny_b12e20_resume"
PHYSICS = "formal_physics_texture_quality_b12e20_resume"
STRICT_TARGET = {
    "name": "RoadFormer-L original RSCD-27 context",
    "top1": 0.9286,
    "mean_precision": 0.8617,
    "mean_recall": 0.8395,
    "macro_f1": 0.8499,
}


def main() -> None:
    baseline = _load(BASELINE)
    physics = _load(PHYSICS)
    if not baseline or not physics:
        raise SystemExit("Missing formal baseline or PhysicsTexture result.")

    rows = [_row(BASELINE, baseline), _row(PHYSICS, physics)]
    for row in rows:
        row["gap_top1_to_roadformer_l"] = row["top1"] - STRICT_TARGET["top1"]
        row["gap_macro_f1_to_roadformer_l"] = row["macro_f1"] - STRICT_TARGET["macro_f1"]
        row["gap_mean_precision_to_roadformer_l"] = row["mean_precision"] - STRICT_TARGET["mean_precision"]
        row["gap_mean_recall_to_roadformer_l"] = row["mean_recall"] - STRICT_TARGET["mean_recall"]

    result = {
        "claim_boundary": (
            "SOTA-gap diagnosis for RSCD-27 class-label classification. "
            "RoadFormer/RoadMamba numbers are external context until split, preprocessing, "
            "training schedule, input size, and label mapping are exactly matched."
        ),
        "strict_external_context": STRICT_TARGET,
        "rows": rows,
        "interpretation": [
            (
                "PhysicsTexture is a validated local improvement: it raises formal ConvNeXt "
                "Top-1 and Mean-F1 under the same local protocol."
            ),
            (
                "Mean-F1 is close to the RoadFormer-L context target, but Top-1 remains far lower. "
                "This is a protocol and exact-class-discrimination warning, not an SOTA claim."
            ),
            (
                "The next route must improve exact fine-grained RSCD class decisions, especially "
                "wet/water concrete and slight/severe roughness, while preserving the hard-slice gains."
            ),
        ],
    }
    SUMMARY.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(_to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def _load(name: str) -> dict[str, Any] | None:
    path = ROOT / name / "evaluate_test.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary", {}))
    mean_precision, mean_recall = _mean_pr(payload)
    summary.setdefault("mean_precision", mean_precision)
    summary.setdefault("mean_recall", mean_recall)
    return {
        "name": name,
        "top1": _num(summary.get("top1")),
        "mean_precision": _num(summary.get("mean_precision")),
        "mean_recall": _num(summary.get("mean_recall")),
        "macro_f1": _num(summary.get("macro_f1")),
        "weighted_f1": _num(summary.get("weighted_f1")),
        "balanced_accuracy": _num(summary.get("balanced_accuracy")),
        "num_samples": int(summary.get("num_samples") or 0),
        "num_classes": int(summary.get("num_classes") or 0),
    }


def _mean_pr(payload: dict[str, Any]) -> tuple[float, float]:
    report = payload.get("classification_report", {})
    precisions = []
    recalls = []
    for label, item in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(item, dict) and "precision" in item and "recall" in item:
            precisions.append(_num(item.get("precision")))
            recalls.append(_num(item.get("recall")))
    if not precisions:
        return 0.0, 0.0
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct(value: float, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    target = result["strict_external_context"]
    lines = [
        "# RSCD SOTA Gap Diagnosis",
        "",
        result["claim_boundary"],
        "",
        "## Strict External Context",
        "",
        "| context | Top-1 | Mean-P | Mean-R | Mean-F1 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {target['name']} | {_pct(target['top1'])} | {_pct(target['mean_precision'])} | "
            f"{_pct(target['mean_recall'])} | {_pct(target['macro_f1'])} |"
        ),
        "",
        "## Local Formal Rows",
        "",
        "| run | Top-1 | Mean-P | Mean-R | Mean-F1 | dTop-1 to RoadFormer-L | dMean-F1 to RoadFormer-L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['name']}` | {_pct(row['top1'])} | {_pct(row['mean_precision'])} | "
            f"{_pct(row['mean_recall'])} | {_pct(row['macro_f1'])} | "
            f"{_pct(row['gap_top1_to_roadformer_l'], signed=True)} | "
            f"{_pct(row['gap_macro_f1_to_roadformer_l'], signed=True)} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in result["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
