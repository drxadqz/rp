from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")

CORE_ORDER = [
    "Global-only",
    "+ PhysicsTexture",
    "+ FrictionSet",
    "+ DG losses",
    "+ EvidenceField aux",
    "Full model",
]

ADJACENT = [
    ("PhysicsTexture", "Global-only", "+ PhysicsTexture"),
    ("FrictionSet", "+ PhysicsTexture", "+ FrictionSet"),
    ("DG losses", "+ FrictionSet", "+ DG losses"),
    ("EvidenceField aux", "+ DG losses", "+ EvidenceField aux"),
    ("Full fusion", "+ EvidenceField aux", "Full model"),
]

METRICS = [
    ("friction_macro_f1", "friction F1", True),
    ("risk_macro_f1", "risk F1", True),
    ("low_friction_recall", "low-friction recall", True),
    ("calibrated_coverage", "calibrated coverage", True),
    ("calibrated_width", "calibrated width", False),
    ("worst_dataset_f1", "worst dataset F1", True),
    ("dataset_id_balanced_accuracy", "dataset-ID bal acc", True),
]

PRIMARY_METRICS = [
    "friction_macro_f1",
    "risk_macro_f1",
    "low_friction_recall",
    "calibrated_coverage",
    "worst_dataset_f1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "p0_claim_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "p0_claim_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "p0_claim_deltas.csv")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out_csv, report["adjacent_deltas"])
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = load_json(summary_dir / "paper_protocol_summary.json") or {}
    core_rows = {row.get("method"): row for row in summary.get("core_ablation", [])}
    ablation_rows = {row.get("method"): row for row in summary.get("ablation", [])}
    rows: list[dict[str, Any]] = []
    for method in CORE_ORDER:
        row = dict(ablation_rows.get(method, core_rows.get(method, {"method": method, "status": "missing"})))
        rows.append(row)

    deltas = []
    for module, prev_name, cur_name in ADJACENT:
        prev = ablation_rows.get(prev_name)
        cur = ablation_rows.get(cur_name)
        if not prev or not cur or prev.get("status") != "complete" or cur.get("status") != "complete":
            deltas.append({"module": module, "previous": prev_name, "current": cur_name, "status": "pending"})
            continue
        item = {
            "module": module,
            "previous": prev_name,
            "current": cur_name,
            "status": "complete",
        }
        for key, _, _ in METRICS:
            item.update(delta_fields(key, prev, cur))
        item["claim_recommendation"] = recommend_module(item)
        deltas.append(item)

    return {
        "summary_dir": str(summary_dir),
        "core_status": "complete" if all(row.get("status") == "complete" for row in rows) else "incomplete",
        "rows": rows,
        "adjacent_deltas": deltas,
        "methodology_note": (
            "Confidence intervals for individual rows come from per-run bootstrap metrics. "
            "Delta intervals here are conservative independent-CI bounds unless an explicit paired comparison is later available."
        ),
        "claim_rule": (
            "Keep a module only when it improves at least two primary metrics or one safety-critical metric "
            "without a clear worst-dataset, low-friction, interval-width, or dataset-shortcut regression."
        ),
    }


def delta_fields(key: str, prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    prev_point = num(prev.get(key))
    cur_point = num(cur.get(key))
    if prev_point is None or cur_point is None:
        return {
            f"{key}_delta": None,
            f"{key}_delta_ci_low": None,
            f"{key}_delta_ci_high": None,
            f"{key}_delta_ci_policy": "missing",
        }
    prev_low = num(prev.get(f"{key}_ci_low"))
    prev_high = num(prev.get(f"{key}_ci_high"))
    cur_low = num(cur.get(f"{key}_ci_low"))
    cur_high = num(cur.get(f"{key}_ci_high"))
    out = {f"{key}_delta": cur_point - prev_point}
    if None not in (prev_low, prev_high, cur_low, cur_high):
        out[f"{key}_delta_ci_low"] = cur_low - prev_high
        out[f"{key}_delta_ci_high"] = cur_high - prev_low
        out[f"{key}_delta_ci_policy"] = "conservative_independent_bootstrap_bound"
    else:
        out[f"{key}_delta_ci_low"] = None
        out[f"{key}_delta_ci_high"] = None
        out[f"{key}_delta_ci_policy"] = "point_only"
    return out


def recommend_module(row: dict[str, Any]) -> str:
    if row.get("status") != "complete":
        return "pending"
    primary_improvements = 0
    primary_regressions = 0
    for key in PRIMARY_METRICS:
        delta = num(row.get(f"{key}_delta"))
        if delta is None:
            continue
        if delta >= 0.005:
            primary_improvements += 1
        if delta <= -0.02:
            primary_regressions += 1
    width_delta = num(row.get("calibrated_width_delta"))
    shortcut_delta = num(row.get("dataset_id_balanced_accuracy_delta"))
    if primary_regressions or (width_delta is not None and width_delta >= 0.05):
        return "rework_or_remove"
    if primary_improvements >= 2:
        return "keep"
    if shortcut_delta is not None and shortcut_delta <= -0.02:
        return "keep_for_shortcut_reduction"
    return "merge_or_simplify"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P0 Claim Report",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        f"P0 status: `{report['core_status']}`",
        "",
        report["methodology_note"],
        "",
        "## Core P0 Table With Confidence Intervals",
        "",
        "| Method | Status | friction F1 | risk F1 | low recall | calibrated coverage | calibrated width | worst dataset F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {width} | {worst} |".format(
                method=row.get("method"),
                status=row.get("status"),
                friction=fmt_metric(row, "friction_macro_f1", True),
                risk=fmt_metric(row, "risk_macro_f1", True),
                low=fmt_metric(row, "low_friction_recall", True),
                cov=fmt_metric(row, "calibrated_coverage", True),
                width=fmt_metric(row, "calibrated_width", False),
                worst=fmt_metric(row, "worst_dataset_f1", True),
            )
        )

    lines.extend(["", "## Adjacent Module Delta Claims", ""])
    lines.append(
        "| Module | Status | Recommendation | d friction F1 | d risk F1 | d low recall | d calibrated coverage | d width | d worst dataset F1 | d dataset-ID |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["adjacent_deltas"]:
        lines.append(
            "| {module} | {status} | {rec} | {friction} | {risk} | {low} | {cov} | {width} | {worst} | {dataset_id} |".format(
                module=row.get("module"),
                status=row.get("status"),
                rec=row.get("claim_recommendation", "-"),
                friction=fmt_delta(row, "friction_macro_f1", True),
                risk=fmt_delta(row, "risk_macro_f1", True),
                low=fmt_delta(row, "low_friction_recall", True),
                cov=fmt_delta(row, "calibrated_coverage", True),
                width=fmt_delta(row, "calibrated_width", False),
                worst=fmt_delta(row, "worst_dataset_f1", True),
                dataset_id=fmt_delta(row, "dataset_id_balanced_accuracy", True),
            )
        )

    lines.extend(["", "## Claim Rule", "", report["claim_rule"], ""])
    return "\n".join(lines)


def fmt_metric(row: dict[str, Any], key: str, percent: bool) -> str:
    value = num(row.get(key))
    if value is None:
        return "-"
    low = num(row.get(f"{key}_ci_low"))
    high = num(row.get(f"{key}_ci_high"))
    point = fmt_value(value, percent)
    if low is None or high is None:
        return point
    return f"{point} [{fmt_value(low, percent)}, {fmt_value(high, percent)}]"


def fmt_delta(row: dict[str, Any], key: str, percent: bool) -> str:
    value = num(row.get(f"{key}_delta"))
    if value is None:
        return "-"
    low = num(row.get(f"{key}_delta_ci_low"))
    high = num(row.get(f"{key}_delta_ci_high"))
    point = fmt_value(value, percent, signed=True)
    if low is None or high is None:
        return point
    return f"{point} [{fmt_value(low, percent, signed=True)}, {fmt_value(high, percent, signed=True)}]"


def fmt_value(value: float, percent: bool, signed: bool = False) -> str:
    if percent:
        scaled = 100.0 * float(value)
        return f"{scaled:+.2f}" if signed else f"{scaled:.2f}"
    return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"


def num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
