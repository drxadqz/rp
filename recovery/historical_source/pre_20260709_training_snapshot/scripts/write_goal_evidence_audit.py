from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")
DEFAULT_OUT = DEFAULT_SUMMARY / "goal_evidence_audit.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY / "goal_evidence_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    summary = _load_json(args.summary_dir / "paper_protocol_summary.json") or {}
    completeness = _load_json(args.summary_dir / "protocol_completeness.json") or {}
    artifact_contract = _load_json(args.summary_dir / "artifact_contract_report.json") or {}
    _apply_artifact_contract_overrides(summary, completeness, artifact_contract)
    progress = _load_json(args.summary_dir / "paper_protocol_progress.json") or []
    dashboard = _load_latest_dashboard(args.summary_dir)
    queue = _load_json(args.summary_dir / "queue_recovery_report.json") or {}
    active_training_watch = _load_json(args.summary_dir / "active_training_watch_report.json") or {}
    live_trend = _load_active_live_trend(args.summary_dir, dashboard, active_training_watch, queue)
    roadsaw_lodo_protocol = _load_json(args.summary_dir / "roadsaw_lodo_protocol_audit.json") or {}
    lodo_generalization = _load_json(args.summary_dir / "lodo_generalization_report.json") or {}
    fair_comparison_protocol = _load_json(args.summary_dir / "fair_comparison_protocol_audit.json") or {}
    friction_interval_sources = _load_json(args.summary_dir / "friction_interval_source_audit.json") or {}
    quality_mondrian = _load_json(args.summary_dir / "quality_mondrian_summary.json") or {}
    asymmetric_mondrian = _load_json(args.summary_dir / "asymmetric_mondrian_summary.json") or {}
    region_mixture = _load_json(args.summary_dir / "region_mixture_summary.json") or {}
    structured = build_report(
        summary,
        completeness,
        progress,
        dashboard,
        active_training_watch,
        live_trend,
        queue,
        roadsaw_lodo_protocol,
        lodo_generalization,
        fair_comparison_protocol,
        friction_interval_sources,
        quality_mondrian,
        asymmetric_mondrian,
        region_mixture,
    )
    report = render_markdown(
        summary,
        completeness,
        progress,
        dashboard,
        active_training_watch,
        live_trend,
        queue,
        roadsaw_lodo_protocol,
        lodo_generalization,
        fair_comparison_protocol,
        friction_interval_sources,
        quality_mondrian,
        asymmetric_mondrian,
        region_mixture,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)


def _apply_artifact_contract_overrides(
    summary: dict[str, Any],
    completeness: dict[str, Any],
    artifact_contract: dict[str, Any],
) -> None:
    rows = artifact_contract.get("rows", []) if isinstance(artifact_contract, dict) else []
    if not isinstance(rows, list):
        return
    by_run = {str(row.get("name")): row for row in rows if isinstance(row, dict) and row.get("name")}
    root = Path(str(artifact_contract.get("root") or "")) if artifact_contract.get("root") else None

    single_map = {
        "RoadSaW only": "single_roadsaw_full_faf",
        "RSCD only": "single_rscd_full_faf",
        "RoadSC only": "single_roadsc_full_faf",
    }
    baseline_map = {
        "RoadSaW global ConvNeXt": "baseline_single_roadsaw_global_convnext",
        "RSCD global ConvNeXt": "baseline_single_rscd_global_convnext",
        "RoadSC global ConvNeXt": "baseline_single_roadsc_global_convnext",
    }
    _override_summary_rows(summary.get("single_dataset", []), single_map, by_run, root)
    _override_summary_rows(summary.get("fair_baselines", []), baseline_map, by_run, root)

    expected_fair = list(single_map.values()) + list(baseline_map.values())
    missing = [name for name in expected_fair if _contract_status(by_run.get(name)) != "complete"]
    for req in completeness.get("requirements", []) or []:
        if isinstance(req, dict) and req.get("name") == "fair_single_dataset_complete":
            req["missing"] = missing
            req["status"] = "complete" if not missing else "incomplete"


def _override_summary_rows(
    rows: Any,
    method_to_run: dict[str, str],
    contract_rows: dict[str, dict[str, Any]],
    root: Path | None,
) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_name = method_to_run.get(str(row.get("method")))
        if not run_name:
            continue
        contract = contract_rows.get(run_name)
        row["status"] = _contract_status(contract)
        if row["status"] == "complete":
            row.update(_bootstrap_metrics_for_run(root, run_name))


def _contract_status(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    if row.get("contract_status") == "complete":
        return "complete"
    if row.get("progress_status") == "running_or_partial":
        return "running"
    return str(row.get("contract_status") or row.get("progress_status") or "missing")


def _bootstrap_metrics_for_run(root: Path | None, run_name: str) -> dict[str, Any]:
    if root is None:
        return {}
    metrics = _load_json(root / run_name / "bootstrap_metrics.json")
    if not isinstance(metrics, dict):
        return {}
    classification = metrics.get("classification", {}) if isinstance(metrics.get("classification"), dict) else {}
    friction = classification.get("friction", {}) if isinstance(classification.get("friction"), dict) else {}
    risk = classification.get("risk", {}) if isinstance(classification.get("risk"), dict) else {}
    low = metrics.get("low_friction_detection", {}) if isinstance(metrics.get("low_friction_detection"), dict) else {}
    interval = metrics.get("mu_interval", {}) if isinstance(metrics.get("mu_interval"), dict) else {}
    low_applicable = _low_friction_applicable(root, run_name)
    low_recall = None if low.get("applicable") is False or low_applicable is False else _point(low.get("recall"))
    return {
        "friction_macro_f1": _point(friction.get("macro_f1")),
        "risk_macro_f1": _point(risk.get("macro_f1")),
        "worst_dataset_f1": _point(friction.get("worst_dataset_macro_f1")),
        "low_friction_recall": low_recall,
        "calibrated_coverage": _point(interval.get("calibrated_coverage")),
        "calibrated_width": _point(interval.get("calibrated_width")),
    }


def _point(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("point")
    return value


def _low_friction_applicable(root: Path | None, run_name: str) -> bool | None:
    if root is None:
        return None
    detailed = _load_json(root / run_name / "detailed_test.json")
    if not isinstance(detailed, dict):
        return None
    risk = (detailed.get("tasks") or {}).get("risk") if isinstance(detailed.get("tasks"), dict) else None
    if not isinstance(risk, dict):
        return None
    labels = risk.get("confusion_matrix_labels")
    matrix = risk.get("confusion_matrix")
    if not isinstance(labels, list) or not isinstance(matrix, list):
        return None
    positives = 0
    for label in ("high", "very_high"):
        if label not in labels:
            continue
        idx = labels.index(label)
        if idx < len(matrix) and isinstance(matrix[idx], list):
            positives += sum(int(value) for value in matrix[idx] if isinstance(value, (int, float)))
    return positives > 0


def build_report(
    summary: dict[str, Any],
    completeness: dict[str, Any],
    progress: list[dict[str, Any]],
    dashboard: dict[str, Any],
    active_training_watch: dict[str, Any],
    live_trend: dict[str, Any],
    queue: dict[str, Any],
    roadsaw_lodo_protocol: dict[str, Any],
    lodo_generalization: dict[str, Any],
    fair_comparison_protocol: dict[str, Any],
    friction_interval_sources: dict[str, Any],
    quality_mondrian: dict[str, Any],
    asymmetric_mondrian: dict[str, Any],
    region_mixture: dict[str, Any],
) -> dict[str, Any]:
    current = _fresh_current_run(progress, dashboard, active_training_watch, live_trend, queue)
    requirements = completeness.get("requirements", []) if isinstance(completeness.get("requirements"), list) else []
    incomplete_requirements = [row for row in requirements if row.get("status") != "complete"]
    lodo_rows = _authoritative_lodo_rows(summary, lodo_generalization)
    sections = {
        "p0_ablation": _section_counts(summary.get("ablation", [])[:6]),
        "p1_candidates": _section_counts(summary.get("ablation", [])[6:]),
        "lodo": _section_counts(lodo_rows),
        "single_dataset": _section_counts(summary.get("single_dataset", [])),
        "fair_baselines": _section_counts(summary.get("fair_baselines", [])),
        "final_lodo": _section_counts(summary.get("final_lodo", [])),
        "final_single_dataset": _section_counts(summary.get("final_single_dataset", [])),
    }
    audit_blockers = _collect_audit_blockers(summary)
    gates = {
        "module_value": _gate_status(
            summary.get("ablation", []),
            ["Global-only", "+ PhysicsTexture", "+ FrictionSet", "+ DG losses", "+ EvidenceField aux", "Full model"],
        ),
        "cross_dataset_generalization": _complete_count(lodo_rows),
        "domain_shortcut_control": _shortcut_status(summary.get("ablation", [])),
        "interval_quality": _interval_status(summary.get("ablation", [])),
        "conditional_calibration": _conditional_calibration_status(summary.get("ablation", [])),
        "fair_comparison": _complete_count(summary.get("fair_single_dataset_deltas", [])),
        "public_weak_label_validity": friction_interval_sources.get("verdict", "missing"),
    }
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "weak_supervised_visual_friction_affordance_public_road_condition_labels",
        "claim_boundary": "Public RSCD/RoadSaW/RoadSC labels define weak visual friction/risk intervals, not synchronized measured tire-road friction coefficients.",
        "current_execution": _compact_current(current),
        "requirements": requirements,
        "num_requirements": len(requirements),
        "num_incomplete_requirements": len(incomplete_requirements),
        "incomplete_requirements": [row.get("name") for row in incomplete_requirements],
        "sections": sections,
        "protocol_evidence": {
            "roadsaw_lodo_protocol": {
                "verdict": roadsaw_lodo_protocol.get("verdict", "missing"),
                "details": _roadsaw_protocol_details(roadsaw_lodo_protocol),
            },
            "fair_comparison_protocol": {
                "verdict": fair_comparison_protocol.get("verdict", "missing"),
                "pairs": fair_comparison_protocol.get("num_pairs"),
                "blocks": fair_comparison_protocol.get("num_blocks"),
                "warnings": fair_comparison_protocol.get("num_warnings"),
            },
            "friction_interval_sources": {
                "verdict": friction_interval_sources.get("verdict", "missing"),
                "anchors": len(friction_interval_sources.get("rows", []))
                if isinstance(friction_interval_sources.get("rows"), list)
                else None,
            },
            "quality_mondrian": {
                "verdict": quality_mondrian.get("verdict", "missing"),
                "rows": len(quality_mondrian.get("rows", []))
                if isinstance(quality_mondrian.get("rows"), list)
                else None,
                "supported": _quality_mondrian_supported(quality_mondrian),
            },
            "asymmetric_mondrian": {
                "verdict": asymmetric_mondrian.get("verdict", "missing"),
                "rows": len(asymmetric_mondrian.get("rows", []))
                if isinstance(asymmetric_mondrian.get("rows"), list)
                else None,
                "supported": _asymmetric_mondrian_supported(asymmetric_mondrian),
            },
            "region_mixture": {
                "verdict": region_mixture.get("verdict", "missing"),
                "rows": len(region_mixture.get("rows", []))
                if isinstance(region_mixture.get("rows"), list)
                else None,
                "supported": _region_mixture_supported(region_mixture),
            },
        },
        "audit_blockers": audit_blockers,
        "research_gates": gates,
        "next_best_actions": _next_actions(completeness, current),
    }


def render_markdown(
    summary: dict[str, Any],
    completeness: dict[str, Any],
    progress: list[dict[str, Any]],
    dashboard: dict[str, Any],
    active_training_watch: dict[str, Any],
    live_trend: dict[str, Any],
    queue: dict[str, Any],
    roadsaw_lodo_protocol: dict[str, Any],
    lodo_generalization: dict[str, Any],
    fair_comparison_protocol: dict[str, Any],
    friction_interval_sources: dict[str, Any],
    quality_mondrian: dict[str, Any],
    asymmetric_mondrian: dict[str, Any],
    region_mixture: dict[str, Any],
) -> str:
    lines = [
        "# Goal Evidence Audit",
        "",
        f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Scope: weak-supervised visual friction affordance and risk estimation from public road-condition labels. "
        "The labels define visual friction/risk intervals, not measured tire-road friction coefficients.",
        "",
    ]
    current = _fresh_current_run(progress, dashboard, active_training_watch, live_trend, queue)
    lodo_rows = _authoritative_lodo_rows(summary, lodo_generalization)
    lines.extend(["## Current Execution", ""])
    if current:
        checkpoint_epoch = current.get("epoch")
        checkpoint_epochs = current.get("epochs")
        active_epoch = current.get("active_epoch") or checkpoint_epoch
        active_epochs = current.get("active_epochs") or checkpoint_epochs
        lines.extend(
            [
                f"- Current run: `{current['name']}`",
                f"- Status: `{current.get('status')}`",
                f"- Active epoch: `{active_epoch}` / `{active_epochs}`",
                f"- Last checkpoint epoch: `{checkpoint_epoch}` / `{checkpoint_epochs}`",
                f"- Best validation loss: `{_fmt_raw(current.get('best_metric'))}`",
                f"- Stale epochs: `{current.get('stale_epochs')}`",
                _active_progress_line(current),
                "",
            ]
        )
    else:
        lines.extend(["- No active or pending run detected in progress report.", ""])

    lines.extend(["## Hard Requirements", ""])
    lines.append("| Requirement | Status | Missing Evidence |")
    lines.append("|---|---|---|")
    for req in completeness.get("requirements", []):
        missing = ", ".join(req.get("missing", [])) if req.get("missing") else "-"
        lines.append(f"| {req.get('name')} | {req.get('status')} | {missing} |")
    lines.append("")

    lines.extend(["## Protocol Evidence", ""])
    lines.append("| Evidence | Verdict | Details |")
    lines.append("|---|---|---|")
    lines.append(
        "| Held-out RoadSaW protocol | `{verdict}` | {details} |".format(
            verdict=roadsaw_lodo_protocol.get("verdict", "missing"),
            details=_roadsaw_protocol_details(roadsaw_lodo_protocol),
        )
    )
    lines.append(
        "| FAF vs ConvNeXt fair comparison protocol | `{verdict}` | pairs `{pairs}`, blocks `{blocks}`, warnings `{warnings}` |".format(
            verdict=fair_comparison_protocol.get("verdict", "missing"),
            pairs=fair_comparison_protocol.get("num_pairs", "-"),
            blocks=fair_comparison_protocol.get("num_blocks", "-"),
            warnings=fair_comparison_protocol.get("num_warnings", "-"),
        )
    )
    lines.append(
        "| Friction interval source audit | `{verdict}` | anchors `{anchors}`, source groups `{sources}` |".format(
            verdict=friction_interval_sources.get("verdict", "missing"),
            anchors=len(friction_interval_sources.get("rows", [])) if isinstance(friction_interval_sources.get("rows"), list) else "-",
            sources=len(friction_interval_sources.get("sources", {})) if isinstance(friction_interval_sources.get("sources"), dict) else "-",
        )
    )
    lines.append(
        "| Quality-Mondrian interval calibration | `{verdict}` | rows `{rows}`, supported `{supported}` |".format(
            verdict=quality_mondrian.get("verdict", "missing"),
            rows=len(quality_mondrian.get("rows", [])) if isinstance(quality_mondrian.get("rows"), list) else "-",
            supported=", ".join(_quality_mondrian_supported(quality_mondrian)) or "-",
        )
    )
    lines.append(
        "| Asymmetric-Mondrian interval calibration | `{verdict}` | rows `{rows}`, supported `{supported}` |".format(
            verdict=asymmetric_mondrian.get("verdict", "missing"),
            rows=len(asymmetric_mondrian.get("rows", [])) if isinstance(asymmetric_mondrian.get("rows"), list) else "-",
            supported=", ".join(_asymmetric_mondrian_supported(asymmetric_mondrian)) or "-",
        )
    )
    lines.append(
        "| Region-Mixture segmentation-style calibration | `{verdict}` | rows `{rows}`, supported `{supported}` |".format(
            verdict=region_mixture.get("verdict", "missing"),
            rows=len(region_mixture.get("rows", [])) if isinstance(region_mixture.get("rows"), list) else "-",
            supported=", ".join(_region_mixture_supported(region_mixture)) or "-",
        )
    )
    lines.append("")

    lines.extend(["## P0 Ablation Evidence", ""])
    ablation_rows = summary.get("ablation", [])
    lines.extend(_table_block(ablation_rows[:6], max_rows=6))
    lines.extend(["## P1 Candidate Evidence", ""])
    lines.extend(_table_block(ablation_rows[6:], max_rows=20))
    lines.extend(["## LODO Evidence", ""])
    lines.extend(_table_block(lodo_rows, max_rows=6))
    lines.extend(["## Fair Single-Dataset Evidence", ""])
    lines.extend(_table_block(summary.get("single_dataset", []), max_rows=6))
    lines.extend(["## Fair Baseline Evidence", ""])
    lines.extend(_table_block(summary.get("fair_baselines", []), max_rows=6))
    lines.extend(["## Final-Method LODO Evidence", ""])
    lines.extend(_table_block(summary.get("final_lodo", []), max_rows=6))
    lines.extend(["## Final-Method Single-Dataset Evidence", ""])
    lines.extend(_table_block(summary.get("final_single_dataset", []), max_rows=6))

    lines.extend(["## Audit Blockers To Clear", ""])
    blockers = _collect_audit_blockers(summary)
    if blockers:
        for item in blockers:
            lines.append(f"- `{item}`")
    else:
        lines.append("- No completed-run audit blockers found in the current summary.")
    lines.append("")

    lines.extend(["## Research Gates", ""])
    lines.append("| Gate | Evidence Needed | Current Status |")
    lines.append("|---|---|---|")
    lines.append(
        "| Module value | Adjacent P0 rows with CI and module decisions | "
        f"{_gate_status(summary.get('ablation', []), ['Global-only', '+ PhysicsTexture', '+ FrictionSet', '+ DG losses', '+ EvidenceField aux', 'Full model'])} |"
    )
    lines.append(
        "| Cross-dataset generalization | LODO rows, especially held-out RoadSaW | "
        f"{_complete_count(lodo_rows)} complete |"
    )
    lines.append(
        "| Domain shortcut control | dataset-ID diagnostics plus LODO/worst-dataset F1 | "
        f"{_shortcut_status(summary.get('ablation', []))} |"
    )
    lines.append(
        "| Interval quality | raw coverage, calibrated coverage, and width with CI | "
        f"{_interval_status(summary.get('ablation', []))} |"
    )
    lines.append(
        "| Conditional calibration | dataset, dataset::core-state, and risk conditional coverage-width | "
        f"{_conditional_calibration_status(summary.get('ablation', []))} |"
    )
    lines.append(
        "| EvidenceField interpretability | evidence maps with attention-road diagnostics | "
        "enabled for future evidence-field runs |"
    )
    lines.append(
        "| Fair comparison | paired bootstrap FAF vs ConvNeXt on identical single-dataset splits | "
        f"{_complete_count(summary.get('fair_single_dataset_deltas', []))} delta rows complete; protocol {fair_comparison_protocol.get('verdict', 'missing')} |"
    )
    lines.append(
        "| Public weak-label validity | interval anchors and claim boundary for public labels | "
        f"source audit {friction_interval_sources.get('verdict', 'missing')} |"
    )
    lines.append("")

    lines.extend(["## Next Best Actions", ""])
    next_actions = _next_actions(completeness, current)
    for action in next_actions:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def _table_block(rows: list[dict[str, Any]], max_rows: int) -> list[str]:
    lines = [
        "| Method | Status | friction F1 | risk F1 | low recall | calib cov | worst F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        lines.append("| - | missing | - | - | - | - | - |")
    for row in rows[:max_rows]:
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {worst} |".format(
                method=row.get("method", row.get("dataset", "-")),
                status=row.get("status", "-"),
                friction=_fmt_pct(row.get("friction_macro_f1")),
                risk=_fmt_pct(row.get("risk_macro_f1")),
                low=_fmt_pct(row.get("low_friction_recall")),
                cov=_fmt_pct(row.get("calibrated_coverage")),
                worst=_fmt_pct(row.get("worst_dataset_f1")),
            )
        )
    lines.append("")
    return lines


def _authoritative_lodo_rows(
    summary: dict[str, Any],
    lodo_generalization: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = lodo_generalization.get("rows") if isinstance(lodo_generalization, dict) else None
    if not isinstance(rows, list) or not rows:
        return list(summary.get("lodo", []) or [])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        held_out = str(row.get("held_out") or "-")
        out.append(
            {
                "method": f"held-out {held_out}",
                "status": row.get("status"),
                "friction_macro_f1": row.get("friction_f1"),
                "risk_macro_f1": row.get("risk_f1"),
                "low_friction_recall": row.get("low_friction_recall"),
                "calibrated_coverage": row.get("calibrated_coverage"),
                "worst_dataset_f1": row.get("worst_dataset_f1"),
            }
        )
    return out or list(summary.get("lodo", []) or [])


def _section_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return {
        "total": len(rows),
        "counts": counts,
        "complete": counts.get("complete", 0),
        "incomplete": len(rows) - counts.get("complete", 0),
        "missing": counts.get("missing", 0),
        "running_or_partial": counts.get("running", 0) + counts.get("running_or_partial", 0),
    }


def _compact_current(current: dict[str, Any] | None) -> dict[str, Any]:
    if not current:
        return {}
    return {
        "name": current.get("name"),
        "status": current.get("status"),
        "active_epoch": current.get("active_epoch") or current.get("epoch"),
        "active_epochs": current.get("active_epochs") or current.get("epochs"),
        "active_step": current.get("active_step"),
        "active_steps": current.get("active_steps"),
        "last_checkpoint_epoch": current.get("epoch"),
        "last_checkpoint_epochs": current.get("epochs"),
        "best_metric": current.get("best_metric"),
    }


def _current_run(progress: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in progress:
        if row.get("status") == "running_or_partial":
            return row
    for row in progress:
        if row.get("status") in {"partial_ci_missing", "missing"}:
            return row
    return None


def _fresh_current_run(
    progress: list[dict[str, Any]],
    dashboard: dict[str, Any],
    active_training_watch: dict[str, Any],
    live_trend: dict[str, Any],
    queue: dict[str, Any],
) -> dict[str, Any] | None:
    current = _current_run(progress) or {}
    active_rows = dashboard.get("active_rows", []) if isinstance(dashboard, dict) else []
    if active_rows:
        current = {**current, **active_rows[0]}
    queue_active = _queue_active_rows(queue)
    has_queue_active = bool(queue_active)
    if queue_active:
        row = queue_active[0]
        _merge_active_progress(current, row)
        for key in ["active_log", "active_log_mtime"]:
            if row.get(key) is not None:
                current[key] = row.get(key)
        current["name"] = row.get("name") or current.get("name")
        current["status"] = row.get("status") or current.get("status")
    live = dashboard.get("live_training", {}) if isinstance(dashboard, dict) else {}
    if not live and isinstance(live_trend, dict):
        latest = live_trend.get("latest_completed_epoch") or {}
        val = latest.get("val") or {}
        live = {
            "run": live_trend.get("run"),
            "active_progress": live_trend.get("active_progress"),
            "latest_completed_epoch": latest.get("epoch"),
            "latest_val_loss": val.get("loss"),
        }
    current_name = current.get("name")
    live_name = live.get("run")
    if current_name and live_name and str(current_name) != str(live_name):
        live = {}
    active = live.get("active_progress") or {}
    if active:
        _merge_active_progress(
            current,
            {
                "active_epoch": active.get("epoch"),
                "active_epochs": active.get("epochs"),
                "active_step": active.get("step"),
                "active_steps": active.get("steps"),
            },
        )
    latest_epoch = live.get("latest_completed_epoch")
    if latest_epoch is not None:
        current["epoch"] = latest_epoch
    latest_loss = live.get("latest_val_loss")
    if latest_loss is not None:
        current["best_metric"] = min(
            _num(current.get("best_metric")) or float(latest_loss),
            float(latest_loss),
        )
    watch_active = active_training_watch.get("active") if isinstance(active_training_watch, dict) else {}
    if isinstance(watch_active, dict) and watch_active:
        watch_name = watch_active.get("name")
        if (not has_queue_active) and (
            not current.get("name") or not watch_name or str(current.get("name")) == str(watch_name)
        ):
            current["name"] = current.get("name") or watch_name
            current["status"] = current.get("status") or watch_active.get("status")
            for target, source in [
                ("active_epoch", "epoch"),
                ("active_epochs", "epochs"),
                ("active_step", "step"),
                ("active_steps", "steps"),
            ]:
                if watch_active.get(source) is not None:
                    current[target] = watch_active.get(source)
    _merge_training_state(current, queue)
    return current or None


def _queue_active_rows(queue: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(queue, dict):
        return []
    active = queue.get("active_rows", []) if isinstance(queue.get("active_rows"), list) else []
    if active:
        return active
    rows = queue.get("queue_order", []) if isinstance(queue.get("queue_order"), list) else []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            row.get("active_epoch") is not None
            or row.get("active_step") is not None
            or row.get("status") in {"running", "partial", "running_or_partial"}
        )
    ]


def _merge_training_state(current: dict[str, Any], queue: dict[str, Any]) -> None:
    name = current.get("name")
    if not name:
        return
    candidates: list[Path] = []
    for key in ("path", "output_dir"):
        value = current.get(key)
        if value:
            candidates.append(Path(str(value)) / "training_state.json")
    root = queue.get("root") if isinstance(queue, dict) else None
    if root:
        candidates.append(Path(str(root)) / str(name) / "training_state.json")
    for path in candidates:
        if path.parent.name != str(name):
            continue
        state = _load_json(path)
        if not isinstance(state, dict):
            continue
        if state.get("epoch") is not None:
            current["epoch"] = state.get("epoch")
        if state.get("epochs") is not None:
            current["epochs"] = state.get("epochs")
        if current.get("active_epoch") is None and state.get("epoch") is not None:
            current["active_epoch"] = state.get("epoch")
        if current.get("active_epochs") is None and state.get("epochs") is not None:
            current["active_epochs"] = state.get("epochs")
        if state.get("best_metric") is not None:
            current["best_metric"] = state.get("best_metric")
        if state.get("stale_epochs") is not None:
            current["stale_epochs"] = state.get("stale_epochs")
        return


def _merge_active_progress(current: dict[str, Any], candidate: dict[str, Any]) -> None:
    cand_epoch = _as_int(candidate.get("active_epoch"))
    cand_step = _as_int(candidate.get("active_step"))
    cur_epoch = _as_int(current.get("active_epoch"))
    cur_step = _as_int(current.get("active_step"))
    should_update = False
    if cand_epoch is not None and cur_epoch is None:
        should_update = True
    elif cand_epoch is not None and cur_epoch is not None:
        should_update = cand_epoch > cur_epoch or (cand_epoch == cur_epoch and (cand_step or 0) >= (cur_step or 0))
    elif cand_step is not None and cur_step is None:
        should_update = True
    elif cand_step is not None and cur_step is not None:
        should_update = cand_step >= cur_step
    if not should_update:
        return
    for key in ["active_epoch", "active_epochs", "active_step", "active_steps"]:
        if candidate.get(key) is not None:
            current[key] = candidate.get(key)


def _active_progress_line(current: dict[str, Any]) -> str:
    epoch = current.get("active_epoch")
    epochs = current.get("active_epochs")
    step = current.get("active_step")
    steps = current.get("active_steps")
    if epoch is None:
        return "- Active log progress: `-`"
    if step is not None and steps is not None:
        return f"- Active log progress: `epoch {epoch}/{epochs}, step {step}/{steps}`"
    return f"- Active log progress: `epoch {epoch}/{epochs}`"


def _collect_audit_blockers(summary: dict[str, Any]) -> list[str]:
    blockers: set[str] = set()
    for section in ["ablation", "lodo", "single_dataset", "fair_baselines"]:
        for row in summary.get(section, []):
            verdict = row.get("audit_verdict")
            if verdict and verdict != "candidate_ready_for_paper_table":
                blockers.add(f"{row.get('method')}: audit verdict {verdict}")
            if row.get("status") == "partial_ci_missing":
                blockers.add(f"{row.get('method')}: missing bootstrap CI")
    return sorted(blockers)


def _gate_status(rows: list[dict[str, Any]], required_methods: list[str]) -> str:
    by_method = {row.get("method"): row for row in rows}
    complete = [name for name in required_methods if by_method.get(name, {}).get("status") == "complete"]
    return f"{len(complete)}/{len(required_methods)} complete"


def _complete_count(rows: list[dict[str, Any]]) -> str:
    complete = sum(1 for row in rows if row.get("status") == "complete")
    return f"{complete}/{len(rows)}" if rows else "0/0"


def _shortcut_status(rows: list[dict[str, Any]]) -> str:
    vals = [
        row.get("dataset_id_balanced_accuracy")
        for row in rows
        if row.get("dataset_id_balanced_accuracy") is not None
    ]
    if not vals:
        return "missing diagnostics"
    best = min(float(v) for v in vals)
    return f"best dataset-ID bal acc {best:.4f}; lower is better"


def _interval_status(rows: list[dict[str, Any]]) -> str:
    vals = [
        (row.get("method"), row.get("raw_interval_coverage"), row.get("calibrated_width"))
        for row in rows
        if row.get("raw_interval_coverage") is not None
    ]
    if not vals:
        return "missing interval evidence"
    best = max(vals, key=lambda item: float(item[1]))
    return f"best raw coverage {best[0]}={float(best[1]):.4f}; width tracked separately"


def _conditional_calibration_status(rows: list[dict[str, Any]]) -> str:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return "missing conditional calibration evidence"
    keys = [
        "dataset_conditional_calibrated_coverage",
        "dataset_core_conditional_calibrated_coverage",
        "risk_conditional_calibrated_coverage",
    ]
    available = [
        row for row in complete
        if all(row.get(key) is not None for key in keys)
    ]
    if not available:
        return "conditional calibration fields missing"
    best_core = max(
        available,
        key=lambda row: float(row.get("dataset_core_conditional_calibrated_coverage", 0.0)),
    )
    return (
        f"{len(available)}/{len(complete)} completed rows include conditional calibration; "
        f"best dataset::core coverage {best_core.get('method')}="
        f"{float(best_core.get('dataset_core_conditional_calibrated_coverage')):.4f}"
    )


def _next_actions(completeness: dict[str, Any], current: dict[str, Any] | None) -> list[str]:
    actions = []
    if current and current.get("status") == "running_or_partial":
        actions.append(f"Let `{current['name']}` finish; do not launch competing GPU evaluation.")
    for req in completeness.get("requirements", []):
        if req.get("status") == "complete":
            continue
        missing = req.get("missing", [])
        if req.get("name") == "p0_ablation_complete":
            actions.append(f"Finish P0 ablation runs: {', '.join(missing[:6])}.")
        elif req.get("name") == "lodo_complete":
            actions.append("Finish the remaining LODO row; use the completed held-out RoadSaW failure as the key shortcut/wetness stress-test evidence.")
        elif req.get("name") == "fair_single_dataset_complete":
            actions.append("Run single-dataset FAF and matched ConvNeXt baselines, then paired bootstrap comparisons.")
        elif req.get("name") == "candidate_path_complete":
            actions.append("Run P1 candidates after core P0/LODO evidence to choose the final robust method.")
        elif req.get("name") == "summary_tables_complete":
            actions.append("Regenerate summary tables and goal audit after new runs finish.")
    if not actions:
        actions.append("All hard requirements appear complete; rerun final audit and prepare paper tables.")
    return actions


def _roadsaw_protocol_details(report: dict[str, Any]) -> str:
    splits = report.get("splits", {}) if isinstance(report.get("splits"), dict) else {}
    parts = []
    for split in ["train", "val", "test"]:
        summary = splits.get(split, {}) if isinstance(splits.get(split), dict) else {}
        datasets = summary.get("datasets", {})
        parts.append(f"{split}: {_compact_dict(datasets)}")
    return "; ".join(parts) if parts else "-"


def _compact_dict(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}:{val}" for key, val in value.items())


def _quality_mondrian_supported(report: dict[str, Any]) -> list[str]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = []
    for row in rows:
        decision = str(row.get("summary_decision", ""))
        if decision in {"keep_for_interval_calibration", "prefer_safety_checkpoint"}:
            supported.append(f"{row.get('run')}::{row.get('probe')}")
    return supported


def _asymmetric_mondrian_supported(report: dict[str, Any]) -> list[str]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = []
    for row in rows:
        if row.get("summary_decision") == "keep_for_interval_width_reduction":
            supported.append(f"{row.get('run')}::{row.get('probe')}")
    return supported


def _region_mixture_supported(report: dict[str, Any]) -> list[str]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = []
    for row in rows:
        if row.get("summary_decision") == "keep_for_segmentation_style_interval_calibration":
            supported.append(f"{row.get('run')}::{row.get('probe')}")
    return supported


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_latest_dashboard(summary_dir: Path) -> dict[str, Any]:
    candidates = [
        summary_dir / "experiment_status_dashboard.json",
        summary_dir / "experiment_dashboard.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {}
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    return _load_json(newest) or {}


def _load_active_live_trend(
    summary_dir: Path,
    dashboard: dict[str, Any],
    active_training_watch: dict[str, Any],
    queue: dict[str, Any],
) -> dict[str, Any]:
    candidates: list[str] = []
    watch_active = active_training_watch.get("active") if isinstance(active_training_watch, dict) else {}
    if isinstance(watch_active, dict) and watch_active.get("name"):
        candidates.append(str(watch_active["name"]))
    if isinstance(dashboard, dict):
        for row in dashboard.get("active_rows", []) or []:
            if row.get("name"):
                candidates.append(str(row["name"]))
    if isinstance(queue, dict):
        for row in queue.get("active_rows", []) or []:
            if row.get("name"):
                candidates.append(str(row["name"]))
    candidates.append("v5_full_faf")

    seen: set[str] = set()
    for run_name in candidates:
        if run_name in seen:
            continue
        seen.add(run_name)
        trend = _load_json(summary_dir / f"{run_name}_live_training_trend.json")
        if isinstance(trend, dict) and trend:
            return trend
    return {}


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def _fmt_raw(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
