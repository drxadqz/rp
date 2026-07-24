from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "evidence_failure_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "evidence_failure_report.json")
    args = parser.parse_args()

    report = build_report(args.root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        audit_path = run_dir / "evidence_field_audit.json"
        if not audit_path.exists():
            continue
        audit = load_json(audit_path)
        if not isinstance(audit, dict):
            continue
        row = summarize_audit(run_dir, audit)
        rows.append(row)
        examples.extend(examples_from_audit(run_dir, audit))
        examples.extend(examples_from_metadata(run_dir))

    rows.sort(key=lambda item: item["run"])
    examples.sort(key=lambda item: (item.get("priority", 99), item.get("run", ""), item.get("tag", "")))
    return {
        "root": str(root),
        "num_evidence_runs": len(rows),
        "runs": rows,
        "examples": examples[:80],
        "interpretation": [
            "EvidenceField is useful for the paper only if failures reveal actionable localization or domain-shift patterns.",
            "RoadSaW failures are especially important because they test wet-state transfer rather than in-domain memorization.",
            "Low-friction failures are safety-critical and should guide ROI, pseudo-road, and interval-coverage constraints.",
        ],
    }


def summarize_audit(run_dir: Path, audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary", {}) if isinstance(audit.get("summary"), dict) else {}
    all_row = summary.get("all", {})
    success = summary.get("risk_success", {})
    failure = summary.get("risk_failure", {})
    low_success = summary.get("low_friction_success", {})
    low_failure = summary.get("low_friction_failure", {})
    roadsaw = summary.get("dataset::roadsaw", {})
    roadsaw_failure = summary.get("dataset::roadsaw::risk_failure", {})
    return {
        "run": run_dir.name,
        "output_dir": str(run_dir),
        "num_records": audit.get("num_records"),
        "risk_accuracy_sampled": all_row.get("risk_accuracy"),
        "friction_accuracy_sampled": all_row.get("friction_accuracy"),
        "raw_interval_coverage_sampled": all_row.get("raw_interval_coverage"),
        "risk_failure_count": failure.get("num_samples", 0),
        "low_friction_failure_count": low_failure.get("num_samples", 0),
        "roadsaw_count": roadsaw.get("num_samples", 0),
        "roadsaw_risk_accuracy_sampled": roadsaw.get("risk_accuracy"),
        "roadsaw_failure_count": roadsaw_failure.get("num_samples", 0),
        "success_bottom_mass": metric_mean(success, "attention_bottom_half_mass"),
        "failure_bottom_mass": metric_mean(failure, "attention_bottom_half_mass"),
        "failure_minus_success_bottom_mass": diff_metric(failure, success, "attention_bottom_half_mass"),
        "success_top_mass": metric_mean(success, "attention_top_half_mass"),
        "failure_top_mass": metric_mean(failure, "attention_top_half_mass"),
        "failure_minus_success_top_mass": diff_metric(failure, success, "attention_top_half_mass"),
        "success_road_likelihood": metric_mean(success, "attention_weighted_road_likelihood_mean"),
        "failure_road_likelihood": metric_mean(failure, "attention_weighted_road_likelihood_mean"),
        "failure_minus_success_road_likelihood": diff_metric(
            failure,
            success,
            "attention_weighted_road_likelihood_mean",
        ),
        "low_success_bottom_mass": metric_mean(low_success, "attention_bottom_half_mass"),
        "low_failure_bottom_mass": metric_mean(low_failure, "attention_bottom_half_mass"),
        "low_failure_minus_success_bottom_mass": diff_metric(
            low_failure,
            low_success,
            "attention_bottom_half_mass",
        ),
    }


def examples_from_audit(run_dir: Path, audit: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    examples = audit.get("examples", {}) if isinstance(audit.get("examples"), dict) else {}
    priority = {
        "roadsaw_failures": 1,
        "lowest_bottom_mass_failures": 2,
        "highest_top_mass_failures": 3,
    }
    for tag, items in examples.items():
        if not isinstance(items, list):
            continue
        for item in items[:12]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "run": run_dir.name,
                    "source": "audit",
                    "tag": tag,
                    "priority": priority.get(tag, 8),
                    "dataset": item.get("dataset"),
                    "group_key": item.get("group_key"),
                    "true_risk": item.get("true_risk"),
                    "pred_risk": item.get("pred_risk"),
                    "true_friction": item.get("true_friction"),
                    "pred_friction": item.get("pred_friction"),
                    "attention_bottom_half_mass": item.get("attention_bottom_half_mass"),
                    "attention_top_half_mass": item.get("attention_top_half_mass"),
                    "image_path": item.get("image_path"),
                    "overlay_path": None,
                }
            )
    return out


def examples_from_metadata(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evidence_maps" / "metadata.json"
    payload = load_json(path)
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    priority = {
        "low_failure": 1,
        "risk_failure": 2,
        "low_success": 4,
        "risk_success": 5,
    }
    for item in payload:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("selection_tag", "unknown"))
        out.append(
            {
                "run": run_dir.name,
                "source": "overlay",
                "tag": tag,
                "priority": priority.get(tag, 7),
                "dataset": item.get("dataset"),
                "group_key": item.get("group_key"),
                "true_risk": item.get("true_risk"),
                "pred_risk": item.get("pred_risk"),
                "true_friction": item.get("true_friction"),
                "pred_friction": item.get("pred_friction"),
                "attention_bottom_half_mass": item.get("attention_bottom_half_mass"),
                "attention_top_half_mass": item.get("attention_top_half_mass"),
                "image_path": item.get("image_path"),
                "overlay_path": str(run_dir / "evidence_maps" / str(item.get("file"))),
            }
        )
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence Failure Report",
        "",
        f"Root: `{report['root']}`",
        f"EvidenceField runs audited: {report['num_evidence_runs']}",
        "",
        "## Run-Level Attention And Failure Summary",
        "",
        "| Run | sampled risk acc | raw cov | failures | low-friction failures | RoadSaW acc | RoadSaW failures | fail-success bottom | fail-success top | fail-success roadness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            "| {run} | {risk} | {cov} | {fail} | {lowfail} | {roadsaw_acc} | {roadsaw_fail} | {db} | {dt} | {dr} |".format(
                run=row["run"],
                risk=fmt_pct(row.get("risk_accuracy_sampled")),
                cov=fmt_pct(row.get("raw_interval_coverage_sampled")),
                fail=row.get("risk_failure_count", 0),
                lowfail=row.get("low_friction_failure_count", 0),
                roadsaw_acc=fmt_pct(row.get("roadsaw_risk_accuracy_sampled")),
                roadsaw_fail=row.get("roadsaw_failure_count", 0),
                db=fmt_signed(row.get("failure_minus_success_bottom_mass")),
                dt=fmt_signed(row.get("failure_minus_success_top_mass")),
                dr=fmt_signed(row.get("failure_minus_success_road_likelihood")),
            )
        )

    lines.extend(["", "## Interpretation", ""])
    for item in report.get("interpretation", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Candidate Figure/Failure Examples", ""])
    examples = report.get("examples", [])[:36]
    if not examples:
        lines.append("- No examples found yet.")
    else:
        lines.append("| Run | Source | Tag | Dataset | true -> pred risk | true -> pred friction | bottom | top | overlay/image |")
        lines.append("|---|---|---|---|---|---|---:|---:|---|")
        for item in examples:
            link = item.get("overlay_path") or item.get("image_path") or "-"
            lines.append(
                "| {run} | {source} | {tag} | {dataset} | {tr} -> {pr} | {tf} -> {pf} | {bottom} | {top} | `{link}` |".format(
                    run=item.get("run"),
                    source=item.get("source"),
                    tag=item.get("tag"),
                    dataset=item.get("dataset"),
                    tr=item.get("true_risk"),
                    pr=item.get("pred_risk"),
                    tf=item.get("true_friction"),
                    pf=item.get("pred_friction"),
                    bottom=fmt_abs(item.get("attention_bottom_half_mass")),
                    top=fmt_abs(item.get("attention_top_half_mass")),
                    link=link,
                )
            )
    lines.append("")
    return "\n".join(lines)


def metric_mean(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, dict) and value.get("mean") is not None:
        return float(value["mean"])
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def diff_metric(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = metric_mean(left, key)
    right_value = metric_mean(right, key)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def fmt_signed(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
