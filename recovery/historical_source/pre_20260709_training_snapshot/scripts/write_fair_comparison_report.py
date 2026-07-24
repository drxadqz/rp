from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DATASETS = [
    ("RSCD", "rscd"),
    ("RoadSaW", "roadsaw"),
    ("RoadSC", "roadsc"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "fair_comparison_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "fair_comparison_report.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    config_audit = _load_json(summary_dir / "protocol_config_audit.json") or {}
    single_rows = {row.get("method"): row for row in summary.get("single_dataset", [])}
    baseline_rows = {row.get("method"): row for row in summary.get("fair_baselines", [])}
    delta_rows = {row.get("dataset"): row for row in summary.get("fair_single_dataset_deltas", [])}

    rows = []
    for display, slug in DATASETS:
        faf_label = f"{display} only"
        base_label = f"{display} global ConvNeXt"
        faf = single_rows.get(faf_label, {})
        baseline = baseline_rows.get(base_label, {})
        delta = delta_rows.get(display, {"status": "pending"})
        paired = _load_json(
            summary_dir
            / "fair_pairwise"
            / f"{slug}_faf_vs_global_convnext_paired_bootstrap.json"
        )
        rows.append(_dataset_row(display, faf, baseline, delta, paired))

    verdict = "pending"
    if all(row["status"] == "complete" for row in rows):
        any_risk_gain = any(_num(row.get("paired_risk_f1_delta")) and _num(row.get("paired_risk_f1_delta")) > 0 for row in rows)
        all_split_fair = config_audit.get("verdict") == "pass"
        if all_split_fair and any_risk_gain:
            verdict = "direct_fair_comparisons_available"
        elif all_split_fair:
            verdict = "fair_but_no_clear_risk_f1_gain"
        else:
            verdict = "comparison_protocol_problem"
    return {
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "protocol_config_verdict": config_audit.get("verdict"),
        "rows": rows,
        "rules": [
            "Use these rows for direct fair comparison because FAF and ConvNeXt baselines share the same dataset, split, label mapping, and metrics.",
            "Do not use multi-dataset FAF rows as direct comparisons against single-dataset papers.",
            "Treat external paper numbers as contextual unless the dataset split and label mapping exactly match.",
            "For friction intervals, compare against same-manifest rule and conformal baselines because most public classifiers do not report interval coverage.",
        ],
    }


def _dataset_row(
    display: str,
    faf: dict[str, Any],
    baseline: dict[str, Any],
    delta: dict[str, Any],
    paired: dict[str, Any] | None,
) -> dict[str, Any]:
    status = "complete" if faf.get("status") == "complete" and baseline.get("status") == "complete" else "pending"
    row = {
        "dataset": display,
        "status": status,
        "faf_status": faf.get("status", "missing"),
        "baseline_status": baseline.get("status", "missing"),
        "faf_friction_f1": faf.get("friction_macro_f1"),
        "baseline_friction_f1": baseline.get("friction_macro_f1"),
        "faf_risk_f1": faf.get("risk_macro_f1"),
        "baseline_risk_f1": baseline.get("risk_macro_f1"),
        "faf_low_friction_recall": faf.get("low_friction_recall"),
        "baseline_low_friction_recall": baseline.get("low_friction_recall"),
        "faf_calibrated_coverage": faf.get("calibrated_coverage"),
        "baseline_calibrated_coverage": baseline.get("calibrated_coverage"),
        "delta_status": delta.get("status"),
    }
    if isinstance(delta, dict):
        for key, value in delta.items():
            if key.startswith("delta_"):
                row[key] = value
    if isinstance(paired, dict):
        row["paired_status"] = "complete"
        metrics = paired.get("metrics", {})
        for metric_name, out_key in [
            ("friction_macro_f1_delta", "paired_friction_f1_delta"),
            ("risk_macro_f1_delta", "paired_risk_f1_delta"),
            ("low_friction_recall_delta", "paired_low_friction_recall_delta"),
            ("calibrated_interval_coverage_delta", "paired_calibrated_coverage_delta"),
            ("calibrated_interval_width_delta", "paired_calibrated_width_delta"),
        ]:
            item = metrics.get(metric_name)
            if isinstance(item, dict):
                row[out_key] = item.get("point")
                row[f"{out_key}_ci_low"] = item.get("ci_low")
                row[f"{out_key}_ci_high"] = item.get("ci_high")
    else:
        row["paired_status"] = "pending"
    return row


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fair Single-Dataset Comparison Report",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        f"Verdict: `{report['verdict']}`",
        f"Protocol config verdict: `{report['protocol_config_verdict']}`",
        "",
        "## Direct Fair Comparison Rows",
        "",
        "| Dataset | Status | FAF risk F1 | Baseline risk F1 | Paired d risk F1 | FAF friction F1 | Baseline friction F1 | Paired d friction F1 | d low recall | d calib cov | d calib width |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {dataset} | {status} | {faf_risk} | {base_risk} | {drisk} | {faf_fric} | {base_fric} | {dfric} | {dlow} | {dcov} | {dwidth} |".format(
                dataset=row["dataset"],
                status=row["status"],
                faf_risk=_fmt_percent(row.get("faf_risk_f1")),
                base_risk=_fmt_percent(row.get("baseline_risk_f1")),
                drisk=_fmt_ci_percent(row, "paired_risk_f1_delta"),
                faf_fric=_fmt_percent(row.get("faf_friction_f1")),
                base_fric=_fmt_percent(row.get("baseline_friction_f1")),
                dfric=_fmt_ci_percent(row, "paired_friction_f1_delta"),
                dlow=_fmt_ci_percent(row, "paired_low_friction_recall_delta"),
                dcov=_fmt_ci_percent(row, "paired_calibrated_coverage_delta"),
                dwidth=_fmt_ci_abs(row, "paired_calibrated_width_delta"),
            )
        )
    lines.extend(["", "## Rules", ""])
    for idx, rule in enumerate(report["rules"], start=1):
        lines.append(f"{idx}. {rule}")
    lines.append("")
    return "\n".join(lines)


def _fmt_percent(value: Any) -> str:
    num = _num(value)
    return "-" if num is None else f"{100.0 * num:.2f}"


def _fmt_ci_percent(row: dict[str, Any], key: str) -> str:
    value = _num(row.get(key))
    if value is None:
        return "-"
    low = _num(row.get(f"{key}_ci_low"))
    high = _num(row.get(f"{key}_ci_high"))
    if low is None or high is None:
        return f"{100.0 * value:+.2f}"
    return f"{100.0 * value:+.2f} [{100.0 * low:+.2f}, {100.0 * high:+.2f}]"


def _fmt_ci_abs(row: dict[str, Any], key: str) -> str:
    value = _num(row.get(key))
    if value is None:
        return "-"
    low = _num(row.get(f"{key}_ci_low"))
    high = _num(row.get(f"{key}_ci_high"))
    if low is None or high is None:
        return f"{value:+.4f}"
    return f"{value:+.4f} [{low:+.4f}, {high:+.4f}]"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
