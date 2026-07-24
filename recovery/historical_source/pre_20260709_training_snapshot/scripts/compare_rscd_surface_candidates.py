from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
DEFAULT_OUT = Path("reports/paper_protocol_summary/rscd_surface_candidate_comparison")


def main() -> None:
    rows = []
    for path in sorted(DEFAULT_ROOT.glob("*/evaluate_test.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = payload.get("summary", {})
        if not summary:
            continue
        name = path.parent.name
        if name.startswith("smoke_"):
            continue
        mean_precision, mean_recall = mean_pr_from_payload(payload)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "top1": _num(summary.get("top1")),
                "mean_precision": _num(summary.get("mean_precision", mean_precision)),
                "mean_recall": _num(summary.get("mean_recall", mean_recall)),
                "macro_f1": _num(summary.get("macro_f1")),
                "weighted_f1": _num(summary.get("weighted_f1")),
                "balanced_accuracy": _num(summary.get("balanced_accuracy")),
                "num_samples": int(summary.get("num_samples") or 0),
                "num_classes": int(summary.get("num_classes") or 0),
            }
        )
    rows.sort(key=lambda r: (r["macro_f1"], r["top1"]), reverse=True)
    baselines = {
        "fast_convnext_tiny": _find(rows, "fast_convnext_tiny"),
        "fast_physics_texture_quality": _find(rows, "fast_physics_texture_quality"),
    }
    for row in rows:
        row["delta_top1_vs_fast_convnext"] = _delta(row, baselines["fast_convnext_tiny"], "top1")
        row["delta_macro_f1_vs_fast_convnext"] = _delta(row, baselines["fast_convnext_tiny"], "macro_f1")
        row["delta_top1_vs_fast_physics"] = _delta(row, baselines["fast_physics_texture_quality"], "top1")
        row["delta_macro_f1_vs_fast_physics"] = _delta(row, baselines["fast_physics_texture_quality"], "macro_f1")

    result = {
        "claim_boundary": (
            "These rows are local RSCD-27 class-label results. They are comparable "
            "only when protocol settings match; weak friction-interval results are a separate target."
        ),
        "root": str(DEFAULT_ROOT),
        "rows": rows,
    }
    DEFAULT_OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    DEFAULT_OUT.with_suffix(".md").write_text(_to_markdown(result), encoding="utf-8")
    print(DEFAULT_OUT.with_suffix(".md"))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean_pr_from_payload(payload: dict[str, Any]) -> tuple[float, float]:
    report = payload.get("classification_report", {})
    precisions = []
    recalls = []
    for name, item in report.items():
        if not isinstance(item, dict) or "precision" not in item or "recall" not in item:
            continue
        if name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precisions.append(_num(item.get("precision")))
        recalls.append(_num(item.get("recall")))
    if not precisions:
        return 0.0, 0.0
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def _find(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row["name"] == name:
            return row
    return None


def _delta(row: dict[str, Any], base: dict[str, Any] | None, key: str) -> float | None:
    if base is None:
        return None
    return float(row[key]) - float(base[key])


def _pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    rows = result["rows"]
    lines = [
        "# RSCD Surface Candidate Comparison",
        "",
        result["claim_boundary"],
        "",
        "| rank | run | Top-1 | Mean-P | Mean-R | Mean-F1 | Balanced Acc | samples | dTop1 vs ConvNeXt | dF1 vs ConvNeXt | dTop1 vs Physics | dF1 vs Physics |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | `{name}` | {top1} | {mp} | {mr} | {f1} | {bal} | {samples} | {dtb} | {dfb} | {dtp} | {dfp} |".format(
                rank=idx,
                name=row["name"],
                top1=_pct(row["top1"]),
                mp=_pct(row["mean_precision"]),
                mr=_pct(row["mean_recall"]),
                f1=_pct(row["macro_f1"]),
                bal=_pct(row["balanced_accuracy"]),
                samples=row["num_samples"],
                dtb=_pct(row["delta_top1_vs_fast_convnext"], signed=True),
                dfb=_pct(row["delta_macro_f1_vs_fast_convnext"], signed=True),
                dtp=_pct(row["delta_top1_vs_fast_physics"], signed=True),
                dfp=_pct(row["delta_macro_f1_vs_fast_physics"], signed=True),
            )
        )
    if not rows:
        lines.append("")
        lines.append("No `evaluate_test.json` files found.")
    lines.extend(
        [
            "",
            "Promotion rule:",
            "",
            "- Promote a candidate only if it improves Macro-F1 or Top-1 under the same RSCD-27 fast protocol.",
            "- If it improves only wet/water/ice classes, keep it as a targeted hard-condition module and require class-wise evidence.",
            "- If it loses to `fast_physics_texture_quality`, prune it or merge only the useful cues.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
