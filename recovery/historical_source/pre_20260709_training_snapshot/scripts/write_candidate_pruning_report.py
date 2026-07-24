from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "candidate_pruning_report.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "candidate_pruning_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predeclare and apply candidate pruning rules for P1/P2/P3/final "
            "runs so modules are retained only with safety, generalization, "
            "interval, or interpretability evidence."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path) -> dict[str, Any]:
    hypothesis = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    rows = hypothesis.get("rows", []) if isinstance(hypothesis.get("rows"), list) else []
    metrics_by_run = _metrics_by_run(summary, rows)
    shortcut_by_run = {
        str(row.get("run")): row
        for row in shortcut.get("rows", [])
        if isinstance(row, dict) and row.get("run")
    }
    reference = _reference(metrics_by_run)
    prune_rows = []
    for row in rows:
        run = str(row.get("run", ""))
        if not _is_candidate_or_final(run, row):
            continue
        metrics = dict(row.get("available_metrics") or metrics_by_run.get(run) or {})
        if run in shortcut_by_run and "dataset_id_balanced_accuracy" not in metrics:
            metrics["dataset_id_balanced_accuracy"] = shortcut_by_run[run].get("overall_balanced_accuracy")
        prune_rows.append(_decision_row(row, metrics, reference))
    counts = _counts(prune_rows)
    verdict = "ready_pending_candidate_metrics"
    if counts["complete_pruned"] or counts["complete_keep"] or counts["complete_rescue"]:
        verdict = "candidate_metric_decisions_available"
    if not prune_rows:
        verdict = "missing_candidate_rows"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "This report is a pruning audit, not a final method claim. A module "
            "is retained only if completed metrics support safety, generalization, "
            "interval quality, or interpretability without major regression."
        ),
        "reference": reference,
        "counts": counts,
        "rows": prune_rows,
        "policy": pruning_policy(),
        "module_retention_verdict": module_retention.get("verdict"),
    }


def _metrics_by_run(summary: dict[str, Any], hypothesis_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section in [
        "ablation",
        "core_ablation",
        "lodo",
        "single_dataset",
        "fair_baselines",
        "final_lodo",
        "final_single_dataset",
    ]:
        for row in summary.get(section, []) if isinstance(summary.get(section), list) else []:
            run = _run_name(row)
            if run:
                out[run] = {k: v for k, v in row.items() if _is_metric_key(k)}
    for row in hypothesis_rows:
        run = str(row.get("run", ""))
        metrics = row.get("available_metrics")
        if run and isinstance(metrics, dict):
            out.setdefault(run, {}).update(metrics)
    return out


def _reference(metrics_by_run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    preferred = metrics_by_run.get("v1_physics_texture") or {}
    if preferred:
        return {"run": "v1_physics_texture", "label": "+ PhysicsTexture", "metrics": preferred}
    fallback = metrics_by_run.get("v0_global_only") or {}
    return {"run": "v0_global_only", "label": "Global-only", "metrics": fallback}


def _decision_row(
    row: dict[str, Any],
    metrics: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    run = str(row.get("run", ""))
    progress = str(row.get("progress_status") or row.get("contract_status") or "missing")
    phase = str(row.get("phase", ""))
    ref_metrics = reference.get("metrics", {}) if isinstance(reference.get("metrics"), dict) else {}
    deltas = {
        "risk_f1": _delta(metrics, ref_metrics, "risk_macro_f1"),
        "friction_f1": _delta(metrics, ref_metrics, "friction_macro_f1"),
        "low_recall": _delta(metrics, ref_metrics, "low_friction_recall"),
        "worst_dataset_f1": _delta(metrics, ref_metrics, "worst_dataset_f1"),
        "calibrated_coverage": _delta(metrics, ref_metrics, "calibrated_coverage"),
        "calibrated_width": _delta(metrics, ref_metrics, "calibrated_width"),
        "dataset_id_balanced_accuracy": _delta(metrics, ref_metrics, "dataset_id_balanced_accuracy"),
    }
    if progress != "complete":
        decision = "pending_result"
        reason = "Run is not complete; pruning is predeclared but not applied."
    else:
        decision, reason = _completed_decision(run, metrics, deltas)
    return {
        "run": run,
        "phase": phase,
        "progress_status": progress,
        "decision": decision,
        "reason": reason,
        "key_modules": row.get("key_modules", []),
        "primary_metrics": row.get("primary_metrics", []),
        "success_criteria": row.get("success_criteria"),
        "failure_action": row.get("failure_action"),
        "retention_rule": row.get("retention_rule"),
        "metrics": {k: metrics.get(k) for k in _primary_metric_keys()},
        "deltas_vs_reference": deltas,
        "reference_run": reference.get("run"),
    }


def _completed_decision(
    run: str,
    metrics: dict[str, Any],
    deltas: dict[str, float | None],
) -> tuple[str, str]:
    if run.startswith("baseline_single_"):
        return "required_control", "Fair ConvNeXt control rows are never pruned."
    if run.startswith("final_"):
        if _metric(metrics, "risk_macro_f1") is None:
            return "pending_result", "Final row lacks result metrics."
        return "final_evidence_available", "Final row is complete; use final-method selection report for architecture freeze."
    major_regressions = [
        name
        for name in ["risk_f1", "low_recall", "worst_dataset_f1"]
        if deltas.get(name) is not None and float(deltas[name]) < -0.02
    ]
    interval_bad = (
        _metric(metrics, "calibrated_coverage") is not None
        and _metric(metrics, "calibrated_coverage") < 0.88
    )
    width_inflation = (
        deltas.get("calibrated_width") is not None
        and float(deltas["calibrated_width"]) > 0.08
    )
    shortcut_gain = (
        deltas.get("dataset_id_balanced_accuracy") is not None
        and float(deltas["dataset_id_balanced_accuracy"]) < -0.05
    )
    primary_gains = [
        name
        for name in ["risk_f1", "friction_f1", "low_recall", "worst_dataset_f1", "calibrated_coverage"]
        if deltas.get(name) is not None and float(deltas[name]) > 0.005
    ]
    if major_regressions:
        return "prune_or_rework", f"Major safety/generalization regressions: {', '.join(major_regressions)}."
    if interval_bad:
        return "prune_or_rework", "Calibrated interval coverage is below the minimum pruning threshold."
    if width_inflation and not primary_gains:
        return "prune_or_rework", "Interval width inflates without a primary metric gain."
    if shortcut_gain and not major_regressions:
        return "rescue_candidate", "Dataset shortcut improves; keep only if paired safety metrics remain acceptable."
    if primary_gains:
        return "keep_candidate", f"Has primary gains: {', '.join(primary_gains)}."
    return "neutral_or_merge", "No clear primary gain; merge only if it adds interpretability or simplifies the final method."


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "rows": len(rows),
        "pending": 0,
        "complete_keep": 0,
        "complete_pruned": 0,
        "complete_rescue": 0,
        "complete_neutral": 0,
    }
    for row in rows:
        decision = row.get("decision")
        if decision == "pending_result":
            counts["pending"] += 1
        elif decision == "keep_candidate":
            counts["complete_keep"] += 1
        elif decision == "prune_or_rework":
            counts["complete_pruned"] += 1
        elif decision == "rescue_candidate":
            counts["complete_rescue"] += 1
        elif decision == "neutral_or_merge":
            counts["complete_neutral"] += 1
    return counts


def pruning_policy() -> list[dict[str, str]]:
    return [
        {
            "criterion": "Safety metrics",
            "keep": "Risk F1, low-friction recall, and worst-dataset F1 do not drop by more than 2 points.",
            "prune": "Any of those drops by more than 2 points unless a separate final-method row rescues it.",
        },
        {
            "criterion": "Shortcut reduction",
            "keep": "Dataset-ID balanced accuracy drops materially while safety metrics hold.",
            "prune": "Shortcut remains high and safety metrics do not improve.",
        },
        {
            "criterion": "Interval quality",
            "keep": "Coverage improves or stays usable with bounded calibrated width.",
            "prune": "Coverage improves only by excessive width, or coverage falls below 88%.",
        },
        {
            "criterion": "Interpretability",
            "keep": "Evidence maps/attention diagnostics support a metric gain or clear failure explanation.",
            "prune": "Visualizations are nicer but metrics or attention audits do not improve.",
        },
        {
            "criterion": "External/fair comparison",
            "keep": "Same-split ConvNeXt and final single-dataset rows support the claim.",
            "prune": "No fair baseline or final-method evidence supports the module.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Pruning Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        f"Reference: `{report['reference'].get('label')}` (`{report['reference'].get('run')}`)",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Counts",
        "",
        "| Rows | Pending | Keep | Rescue | Prune/Rework | Neutral/Merge |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {rows} | {pending} | {complete_keep} | {complete_rescue} | {complete_pruned} | {complete_neutral} |".format(
            **report["counts"]
        ),
        "",
        "## Candidate Decisions",
        "",
        "| Run | Phase | Status | Decision | Risk delta | Low-recall delta | Worst-F1 delta | Coverage delta | Width delta | Reason |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        d = row.get("deltas_vs_reference", {})
        lines.append(
            "| `{run}` | {phase} | `{status}` | `{decision}` | {risk} | {low} | {worst} | {cov} | {width} | {reason} |".format(
                run=row.get("run", "-"),
                phase=row.get("phase", "-"),
                status=row.get("progress_status", "-"),
                decision=row.get("decision", "-"),
                risk=_fmt_delta(d.get("risk_f1")),
                low=_fmt_delta(d.get("low_recall")),
                worst=_fmt_delta(d.get("worst_dataset_f1")),
                cov=_fmt_delta(d.get("calibrated_coverage")),
                width=_fmt_delta(d.get("calibrated_width")),
                reason=row.get("reason", "-"),
            )
        )
    lines.extend(["", "## Policy", ""])
    lines.extend(["| Criterion | Keep | Prune |", "|---|---|---|"])
    for item in report["policy"]:
        lines.append(f"| {item['criterion']} | {item['keep']} | {item['prune']} |")
    return "\n".join(lines) + "\n"


def _is_candidate_or_final(run: str, row: dict[str, Any]) -> bool:
    phase = str(row.get("phase", ""))
    return (
        run.startswith("v6")
        or run.startswith("v7")
        or run.startswith("v8")
        or run.startswith("v9")
        or run.startswith("v10")
        or run.startswith("v11")
        or run.startswith("v12")
        or run.startswith("v13")
        or run.startswith("v14")
        or run.startswith("v15")
        or run.startswith("v16")
        or run.startswith("v17")
        or run.startswith("v18")
        or run.startswith("v19")
        or run.startswith("v20")
        or run.startswith("v21")
        or run.startswith("v22")
        or run.startswith("v23")
        or run.startswith("v24")
        or run.startswith("final_")
        or "P1" in phase
        or "P2" in phase
        or "P3" in phase
    )


def _run_name(row: dict[str, Any]) -> str:
    output_dir = row.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).name
    method = str(row.get("method", ""))
    return method


def _is_metric_key(key: str) -> bool:
    return key.endswith("_f1") or key in set(_primary_metric_keys()) or "coverage" in key or "width" in key


def _primary_metric_keys() -> list[str]:
    return [
        "friction_macro_f1",
        "risk_macro_f1",
        "low_friction_recall",
        "worst_dataset_f1",
        "calibrated_coverage",
        "calibrated_width",
        "dataset_id_balanced_accuracy",
    ]


def _delta(metrics: dict[str, Any], ref: dict[str, Any], key: str) -> float | None:
    a = _metric(metrics, key)
    b = _metric(ref, key)
    if a is None or b is None:
        return None
    return a - b


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    try:
        value = metrics.get(key)
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:+.2f}"
    except (TypeError, ValueError):
        return "-"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
