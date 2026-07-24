from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
SUMMARY = ROOT / "reports" / "paper_protocol_summary"


PAIRS = [
    ("RoadSaW", "single_roadsaw_full_faf", "baseline_single_roadsaw_global_convnext"),
    ("RSCD", "single_rscd_full_faf", "baseline_single_rscd_global_convnext"),
    ("RoadSC", "single_roadsc_full_faf", "baseline_single_roadsc_global_convnext"),
]


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    rows = [row(dataset, faf, base) for dataset, faf, base in PAIRS]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": "Same-dataset FAF and ConvNeXt baselines share local manifests, weak-label mapping, and detailed-test metrics. These are public-label friction-affordance comparisons, not measured tire-road friction coefficients.",
        "rows": rows,
        "verdict": verdict(rows),
    }
    (SUMMARY / "fair_comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (SUMMARY / "fair_comparison_report.md").write_text(markdown(report), encoding="utf-8")
    print(SUMMARY / "fair_comparison_report.md")
    print(SUMMARY / "fair_comparison_report.json")


def row(dataset: str, faf_name: str, base_name: str) -> dict[str, Any]:
    faf = read_json(EXP_ROOT / faf_name / "detailed_test.json")
    base = read_json(EXP_ROOT / base_name / "detailed_test.json")
    return {
        "dataset": dataset,
        "faf_run": faf_name,
        "baseline_run": base_name,
        "status": "complete" if faf and base else "pending",
        "faf_friction_macro_f1": metric(faf, "friction", "macro_f1"),
        "baseline_friction_macro_f1": metric(base, "friction", "macro_f1"),
        "delta_friction_macro_f1": diff(metric(faf, "friction", "macro_f1"), metric(base, "friction", "macro_f1")),
        "faf_risk_macro_f1": metric(faf, "risk", "macro_f1"),
        "baseline_risk_macro_f1": metric(base, "risk", "macro_f1"),
        "delta_risk_macro_f1": diff(metric(faf, "risk", "macro_f1"), metric(base, "risk", "macro_f1")),
        "faf_low_friction_recall": low_recall(faf),
        "baseline_low_friction_recall": low_recall(base),
        "delta_low_friction_recall": diff(low_recall(faf), low_recall(base)),
        "faf_raw_coverage": mu(faf, "coverage"),
        "baseline_raw_coverage": mu(base, "coverage"),
        "delta_raw_coverage": diff(mu(faf, "coverage"), mu(base, "coverage")),
        "faf_width": mu(faf, "width_mean"),
        "baseline_width": mu(base, "width_mean"),
        "delta_width": diff(mu(faf, "width_mean"), mu(base, "width_mean")),
    }


def verdict(rows: list[dict[str, Any]]) -> str:
    if not all(r["status"] == "complete" for r in rows):
        return "pending"
    wins = sum(1 for r in rows if (r.get("delta_friction_macro_f1") or 0) > 0)
    losses = sum(1 for r in rows if (r.get("delta_friction_macro_f1") or 0) < 0)
    if wins and losses:
        return "mixed_dataset_specific"
    return "faf_wins_all" if wins else "baseline_wins_all"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fair Single-Dataset Comparison Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        report["claim_boundary"],
        "",
        "| Dataset | Status | FAF friction F1 | ConvNeXt friction F1 | Delta friction | FAF risk F1 | ConvNeXt risk F1 | Delta risk | Delta low recall | Delta raw cov | FAF width | ConvNeXt width |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        lines.append(
            "| {dataset} | {status} | {faf_friction} | {base_friction} | {d_friction} | {faf_risk} | {base_risk} | {d_risk} | {d_low} | {d_cov} | {faf_width} | {base_width} |".format(
                dataset=r["dataset"],
                status=r["status"],
                faf_friction=pct(r["faf_friction_macro_f1"]),
                base_friction=pct(r["baseline_friction_macro_f1"]),
                d_friction=signed_pct(r["delta_friction_macro_f1"]),
                faf_risk=pct(r["faf_risk_macro_f1"]),
                base_risk=pct(r["baseline_risk_macro_f1"]),
                d_risk=signed_pct(r["delta_risk_macro_f1"]),
                d_low=signed_pct(r["delta_low_friction_recall"]),
                d_cov=signed_pct(r["delta_raw_coverage"]),
                faf_width=num(r["faf_width"]),
                base_width=num(r["baseline_width"]),
            )
        )
    lines += [
        "",
        "Interpretation:",
        "- RoadSaW and RoadSC show task-F1 gains for FAF over the matched ConvNeXt baseline.",
        "- RSCD shows the opposite: the strong ConvNeXt baseline beats the current FAF on friction/risk and raw interval coverage.",
        "- Therefore the current evidence supports a dataset-specific and mechanism-focused story, not a blanket superiority claim.",
    ]
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def metric(report: dict[str, Any], task: str, name: str) -> Any:
    return ((report.get("tasks") or {}).get(task) or {}).get(name) if report else None


def low_recall(report: dict[str, Any]) -> Any:
    if not report:
        return None
    low = report.get("low_friction_detection") or {}
    if low.get("applicable") is False:
        return None
    return low.get("recall")


def mu(report: dict[str, Any], name: str) -> Any:
    return (report.get("mu_interval") or {}).get(name) if report else None


def diff(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def pct(x: Any) -> str:
    if x is None:
        return "-"
    return f"{100 * float(x):.2f}%"


def signed_pct(x: Any) -> str:
    if x is None:
        return "-"
    return f"{100 * float(x):+.2f}%"


def num(x: Any) -> str:
    if x is None:
        return "-"
    return f"{float(x):.4f}"


if __name__ == "__main__":
    main()
