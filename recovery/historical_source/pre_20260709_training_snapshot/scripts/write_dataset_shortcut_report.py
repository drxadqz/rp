from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_protocol_progress import ROWS


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY / "dataset_shortcut_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY / "dataset_shortcut_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY / "dataset_shortcut_report.csv")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    report = build_report(args.root, threshold=float(args.threshold))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    write_csv(report, args.out_csv)
    print(render_markdown(report))


def build_report(root: Path, threshold: float) -> dict[str, Any]:
    rows = []
    for run in ROWS:
        rows.append(inspect_run(root / run, threshold))
    complete = [row for row in rows if row.get("status") == "complete"]
    high = [
        row
        for row in complete
        if _num(row.get("overall_balanced_accuracy"), 0.0) > threshold
        or _num(row.get("risk_conditioned_balanced_accuracy"), 0.0) > threshold
        or _num(row.get("core_state_conditioned_balanced_accuracy"), 0.0) > threshold
    ]
    best = min(
        complete,
        key=lambda row: _num(row.get("core_state_conditioned_balanced_accuracy"), 1.0),
        default=None,
    )
    return {
        "root": str(root),
        "threshold": threshold,
        "rows": rows,
        "num_complete": len(complete),
        "num_high_shortcut": len(high),
        "best_completed_by_core_state_probe": best,
        "verdict": "pass" if complete and not high else "warn",
    }


def inspect_run(run_dir: Path, threshold: float) -> dict[str, Any]:
    if _is_single_dataset_scope(run_dir.name):
        return {
            "run": run_dir.name,
            "status": "not_applicable_single_dataset",
            "skip_reason": "Dataset-ID shortcut probe is undefined for single-dataset training/evaluation rows.",
        }
    path = run_dir / "dataset_id_diagnostic.json"
    if not path.exists():
        return {"run": run_dir.name, "status": "missing_dataset_id_diagnostic"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "run": run_dir.name,
        "status": "complete",
        "num_samples": payload.get("num_samples"),
        "overall_accuracy": payload.get("overall_dataset_id_accuracy"),
        "overall_balanced_accuracy": payload.get("overall_dataset_id_balanced_accuracy"),
        "risk_conditioned_accuracy": payload.get("risk_conditioned_common_dataset_id_accuracy"),
        "risk_conditioned_balanced_accuracy": payload.get("risk_conditioned_common_dataset_id_balanced_accuracy"),
        "risk_conditioned_num_samples": payload.get("risk_conditioned_common_num_samples"),
        "risk_conditioned_num_conditions": payload.get("risk_conditioned_common_num_conditions"),
        "core_state_conditioned_accuracy": payload.get("core_state_conditioned_common_dataset_id_accuracy"),
        "core_state_conditioned_balanced_accuracy": payload.get("core_state_conditioned_common_dataset_id_balanced_accuracy"),
        "core_state_conditioned_num_samples": payload.get("core_state_conditioned_common_num_samples"),
        "core_state_conditioned_num_conditions": payload.get("core_state_conditioned_common_num_conditions"),
    }
    row["shortcut_flag"] = bool(
        _num(row.get("overall_balanced_accuracy"), 0.0) > threshold
        or _num(row.get("risk_conditioned_balanced_accuracy"), 0.0) > threshold
        or _num(row.get("core_state_conditioned_balanced_accuracy"), 0.0) > threshold
    )
    return row


def render_markdown(report: dict[str, Any]) -> str:
    best = report.get("best_completed_by_core_state_probe") or {}
    lines = [
        "# Dataset Shortcut Report",
        "",
        f"Root: `{report['root']}`",
        f"Threshold: `{report['threshold']:.2f}` balanced accuracy.",
        f"Verdict: `{report['verdict']}`; completed rows: `{report['num_complete']}`; high-shortcut rows: `{report['num_high_shortcut']}`.",
        "",
        "The probe trains a lightweight classifier on frozen model features to predict dataset identity. High balanced accuracy means the representation still exposes dataset style/domain cues.",
        "",
        "## Best Completed Row By Core-State Probe",
        "",
        "- `{run}` core-state-conditioned balanced accuracy `{core}`, overall balanced accuracy `{overall}`.".format(
            run=best.get("run", "-"),
            core=_fmt_pct(best.get("core_state_conditioned_balanced_accuracy")),
            overall=_fmt_pct(best.get("overall_balanced_accuracy")),
        ),
        "",
        "## Run Summary",
        "",
        "| Run | Status | overall bal acc | risk-cond bal acc | core-state-cond bal acc | common core states | flag |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            "| {run} | {status} | {overall} | {risk} | {core} | {ncore} | {flag} |".format(
                run=row.get("run"),
                status=row.get("status"),
                overall=_fmt_pct(row.get("overall_balanced_accuracy")),
                risk=_fmt_pct(row.get("risk_conditioned_balanced_accuracy")),
                core=_fmt_pct(row.get("core_state_conditioned_balanced_accuracy")),
                ncore=row.get("core_state_conditioned_num_conditions", "-"),
                flag="high" if row.get("shortcut_flag") else "-",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- Prefer P1/final candidates that reduce risk-conditioned and core-state-conditioned dataset-ID balanced accuracy.",
            "- Do not claim domain-general shortcut resistance from overall test accuracy alone.",
            "- Use this report together with LODO, RoadSaW wetness, interval quality, and fair single-dataset comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "status",
        "num_samples",
        "overall_accuracy",
        "overall_balanced_accuracy",
        "risk_conditioned_accuracy",
        "risk_conditioned_balanced_accuracy",
        "risk_conditioned_num_samples",
        "risk_conditioned_num_conditions",
        "core_state_conditioned_accuracy",
        "core_state_conditioned_balanced_accuracy",
        "core_state_conditioned_num_samples",
        "core_state_conditioned_num_conditions",
        "shortcut_flag",
        "skip_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report.get("rows", []):
            writer.writerow({field: row.get(field) for field in fields})


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_single_dataset_scope(name: str) -> bool:
    return (
        name.startswith("single_")
        or name.startswith("baseline_single_")
        or name.startswith("final_single_")
    )


if __name__ == "__main__":
    main()
