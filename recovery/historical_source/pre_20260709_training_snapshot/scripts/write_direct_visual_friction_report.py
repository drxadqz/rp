from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\direct_visual_friction")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir)
    out_json = args.out_json or args.summary_dir / "direct_visual_friction_report.json"
    out_md = args.out_md or args.summary_dir / "direct_visual_friction_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_md}")


def build_report(root: Path, summary_dir: Path) -> dict[str, Any]:
    class_dir = root / "extreme_road_class_baseline_fast"
    base_dir = root / "extreme_road_global_convnext_fast"
    faf_dir = root / "extreme_road_quality_physics_fast"
    pair_dir = root / "fair_pairwise"

    dataset_audit = _load_json(summary_dir / "extreme_road_dataset_audit.json")
    forward_check = _load_json(summary_dir / "extreme_road_direct_forward_loss_check.json")
    class_test = _load_json(class_dir / "evaluate_test.json")
    base_boot = _first_existing_json(
        base_dir / "bootstrap_metrics_best_safety.json",
        base_dir / "bootstrap_metrics.json",
    )
    faf_boot = _first_existing_json(
        faf_dir / "bootstrap_metrics_best_safety.json",
        faf_dir / "bootstrap_metrics.json",
    )
    pairwise = _load_json(pair_dir / "extreme_road_faf_vs_global_convnext_paired_bootstrap.json")

    return {
        "root": str(root),
        "claim_boundary": (
            "ExtremeRoad is a separate direct-visual-friction validation protocol. "
            "It should not be merged into the RSCD/RoadSaW/RoadSC main paper-protocol tables."
        ),
        "dataset": _dataset_summary(dataset_audit),
        "forward_check": _forward_summary(forward_check),
        "six_class_surface_baseline": _class_summary(class_test),
        "same_task_global_convnext": _faf_summary(base_boot),
        "same_task_faf": _faf_summary(faf_boot),
        "paired_faf_minus_global": _paired_summary(pairwise),
        "status": _status(class_test, base_boot, faf_boot, pairwise),
        "evidence_files": {
            "dataset_audit": str(summary_dir / "extreme_road_dataset_audit.json"),
            "forward_check": str(summary_dir / "extreme_road_direct_forward_loss_check.json"),
            "class_test": str(class_dir / "evaluate_test.json"),
            "global_bootstrap": str(base_dir / "bootstrap_metrics_best_safety.json"),
            "faf_bootstrap": str(faf_dir / "bootstrap_metrics_best_safety.json"),
            "paired_bootstrap": str(pair_dir / "extreme_road_faf_vs_global_convnext_paired_bootstrap.json"),
        },
    }


def _dataset_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"status": "missing"}
    classes = data.get("classes") or data.get("class_rows") or []
    if isinstance(classes, dict):
        class_items = list(classes.values())
        rows = sum(int(item.get("num_images") or 0) for item in class_items if isinstance(item, dict))
        split_counts: dict[str, int] = {}
        for item in class_items:
            if not isinstance(item, dict):
                continue
            for split, count in (item.get("split_counts") or {}).items():
                split_counts[str(split)] = split_counts.get(str(split), 0) + int(count)
        class_count = len(class_items)
    else:
        rows = data.get("rows")
        split_counts = data.get("split_counts") or {}
        class_count = len(classes) if isinstance(classes, list) else None
    return {
        "status": "complete",
        "rows": rows,
        "root": data.get("root"),
        "class_count": class_count,
        "split_counts": split_counts,
    }


def _forward_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"status": "missing"}
    failures = data.get("failures") or []
    rows = data.get("rows") or []
    return {
        "status": "pass" if not failures else "fail",
        "checks": data.get("checks"),
        "failures": len(failures),
        "checked_configs": [row.get("config") for row in rows if isinstance(row, dict)],
    }


def _class_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"status": "pending"}
    summary = data.get("summary") or {}
    return {
        "status": "complete",
        "top1": _num(summary.get("top1")),
        "macro_f1": _num(summary.get("macro_f1")),
        "weighted_f1": _num(summary.get("weighted_f1")),
        "balanced_accuracy": _num(summary.get("balanced_accuracy")),
        "num_samples": summary.get("num_samples"),
        "num_classes": summary.get("num_classes"),
    }


def _faf_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"status": "pending"}
    return {
        "status": "complete",
        "friction_macro_f1": _point(data, "classification", "friction", "macro_f1"),
        "risk_macro_f1": _point(data, "classification", "risk", "macro_f1"),
        "low_friction_recall": _point(data, "low_friction_detection", "recall"),
        "raw_coverage": _point(data, "mu_interval", "raw_coverage"),
        "raw_width": _point(data, "mu_interval", "raw_width"),
        "calibrated_coverage": _point(data, "mu_interval", "calibrated_coverage"),
        "calibrated_width": _point(data, "mu_interval", "calibrated_width"),
        "checkpoint": data.get("checkpoint"),
    }


def _paired_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"status": "pending"}
    metrics = data.get("metrics") or {}
    return {
        "status": "complete",
        "delta_definition": data.get("delta_definition"),
        "friction_macro_f1_delta": _delta(metrics.get("friction_macro_f1_delta")),
        "risk_macro_f1_delta": _delta(metrics.get("risk_macro_f1_delta")),
        "low_friction_recall_delta": _delta(metrics.get("low_friction_recall_delta")),
        "raw_interval_coverage_delta": _delta(metrics.get("raw_interval_coverage_delta")),
        "calibrated_interval_coverage_delta": _delta(metrics.get("calibrated_interval_coverage_delta")),
        "calibrated_interval_width_delta": _delta(metrics.get("calibrated_interval_width_delta")),
    }


def _status(
    class_test: dict[str, Any] | None,
    base_boot: dict[str, Any] | None,
    faf_boot: dict[str, Any] | None,
    pairwise: dict[str, Any] | None,
) -> str:
    if class_test and base_boot and faf_boot and pairwise:
        return "complete"
    if base_boot or faf_boot or class_test:
        return "partial"
    return "pending"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Direct Visual Friction Report",
        "",
        report["claim_boundary"],
        "",
        f"Status: `{report['status']}`",
        "",
        "## Dataset",
        "",
        "| item | value |",
        "|---|---:|",
        f"| rows | {_fmt(report['dataset'].get('rows'))} |",
        f"| classes | {_fmt(report['dataset'].get('class_count'))} |",
        "",
        "## Forward/Loss Check",
        "",
        "| status | checks | failures | configs |",
        "|---|---:|---:|---|",
        (
            f"| {report['forward_check'].get('status')} | "
            f"{_fmt(report['forward_check'].get('checks'))} | "
            f"{_fmt(report['forward_check'].get('failures'))} | "
            f"{', '.join(str(x) for x in report['forward_check'].get('checked_configs', []) if x)} |"
        ),
        "",
        "## Six-Class Surface Baseline",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| top-1 | {_pct(report['six_class_surface_baseline'].get('top1'))} |",
        f"| macro F1 | {_pct(report['six_class_surface_baseline'].get('macro_f1'))} |",
        f"| weighted F1 | {_pct(report['six_class_surface_baseline'].get('weighted_f1'))} |",
        f"| balanced accuracy | {_pct(report['six_class_surface_baseline'].get('balanced_accuracy'))} |",
        "",
        "## Same-Task Friction/Interval Models",
        "",
        "| model | friction F1 | risk F1 | low recall | raw coverage | calibrated coverage | calibrated width |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _model_row("Global ConvNeXt", report["same_task_global_convnext"]),
        _model_row("FAF quality physics", report["same_task_faf"]),
        "",
        "## Paired Bootstrap Delta",
        "",
        "Delta is `FAF quality physics - Global ConvNeXt` on the same ExtremeRoad test split.",
        "",
        "| metric | point | 95% CI |",
        "|---|---:|---:|",
    ]
    paired = report["paired_faf_minus_global"]
    for key, label, percent in [
        ("friction_macro_f1_delta", "friction macro-F1", True),
        ("risk_macro_f1_delta", "risk macro-F1", True),
        ("low_friction_recall_delta", "low-friction recall", True),
        ("raw_interval_coverage_delta", "raw interval coverage", True),
        ("calibrated_interval_coverage_delta", "calibrated interval coverage", True),
        ("calibrated_interval_width_delta", "calibrated interval width", False),
    ]:
        lines.append(_delta_row(label, paired.get(key), percent=percent))
    lines.extend(
        [
            "",
            "## Evidence Files",
            "",
        ]
    )
    for name, path in report["evidence_files"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _model_row(name: str, data: dict[str, Any]) -> str:
    return (
        f"| {name} | {_pct(data.get('friction_macro_f1'))} | {_pct(data.get('risk_macro_f1'))} | "
        f"{_pct(data.get('low_friction_recall'))} | {_pct(data.get('raw_coverage'))} | "
        f"{_pct(data.get('calibrated_coverage'))} | {_fmt_float(data.get('calibrated_width'))} |"
    )


def _delta_row(label: str, data: dict[str, Any] | None, *, percent: bool) -> str:
    if not data:
        return f"| {label} | - | - |"
    fmt = _signed_pct if percent else _signed_float
    return f"| {label} | {fmt(data.get('point'))} | [{fmt(data.get('ci_low'))}, {fmt(data.get('ci_high'))}] |"


def _point(data: dict[str, Any], *path: str) -> float | None:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, dict):
        return _num(cur.get("point"))
    return _num(cur)


def _delta(data: Any) -> dict[str, float | None] | None:
    if not isinstance(data, dict):
        return None
    return {
        "point": _num(data.get("point")),
        "ci_low": _num(data.get("ci_low")),
        "ci_high": _num(data.get("ci_high")),
    }


def _first_existing_json(*paths: Path) -> dict[str, Any] | None:
    for path in paths:
        data = _load_json(path)
        if data is not None:
            return data
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _pct(value: Any) -> str:
    number = _num(value)
    return "-" if number is None else f"{number * 100:.2f}"


def _signed_pct(value: Any) -> str:
    number = _num(value)
    return "-" if number is None else f"{number * 100:+.2f}"


def _fmt_float(value: Any) -> str:
    number = _num(value)
    return "-" if number is None else f"{number:.4f}"


def _signed_float(value: Any) -> str:
    number = _num(value)
    return "-" if number is None else f"{number:+.4f}"


if __name__ == "__main__":
    main()
