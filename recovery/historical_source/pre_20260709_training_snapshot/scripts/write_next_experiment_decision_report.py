from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "next_experiment_decision_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "next_experiment_decision_report.json",
    )
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    dashboard = _load_dashboard(summary_dir)
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    runtime = _load_json(summary_dir / "runtime_guard_report.json") or {}
    handoff = _load_json(summary_dir / "handoff_health_report.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    claim_ledger = _load_json(summary_dir / "claim_evidence_ledger.json") or {}
    candidate_matrix = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    interval_sources = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    input_style = _load_json(summary_dir / "input_canonicalization_style_audit.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}

    active = _active_run(dashboard, queue, watch)
    readiness = dashboard.get("readiness") or {}
    progress_counts = dashboard.get("progress_counts") or {}
    group_status = dashboard.get("group_status") or {}
    blockers = list(readiness.get("blocking_gates") or [])
    warnings = list(readiness.get("warning_gates") or [])

    module_actions = _module_actions(module_retention)
    failure_signals = list(candidate_matrix.get("current_failure_signals") or [])
    failure_signals.extend(_lodo_failure_signals(lodo))
    candidate_rows = candidate_matrix.get("rows") or []
    queue_order = queue.get("queue_order") or []
    action_plan = _action_plan(active, queue_order, candidate_rows, group_status, blockers)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "scope": "weak-supervised camera-only visual friction-affordance estimation from public road-condition labels",
        "claim_boundary": (
            "RSCD/RoadSaW/RoadSC labels support weak visual friction-affordance intervals; "
            "they are not synchronized measured tire-road friction coefficients."
        ),
        "current_execution": {
            "active_run": active,
            "progress_counts": progress_counts,
            "runtime_verdict": runtime.get("verdict"),
            "handoff_verdict": handoff.get("verdict"),
            "roadsaw_priority_watcher_processes": _get_dashboard_value(
                dashboard, "roadsaw_priority_watcher_processes"
            ),
        },
        "readiness": {
            "verdict": readiness.get("verdict"),
            "num_blocks": readiness.get("num_blocks"),
            "num_warnings": readiness.get("num_warnings"),
            "blockers": blockers,
            "warnings": warnings,
        },
        "module_actions": module_actions,
        "failure_signals": failure_signals,
        "claim_status": _claim_status(claim_ledger),
        "friction_interval_source_status": {
            "verdict": interval_sources.get("verdict"),
            "source_groups": interval_sources.get("source_groups") or interval_sources.get("sources"),
            "num_public_anchors": _num_interval_anchors(interval_sources),
        },
        "lodo_evidence": _compact_lodo_evidence(lodo),
        "input_canonicalization_evidence": _compact_input_canonicalization_style(input_style),
        "action_plan": action_plan,
        "decision_rules": _decision_rules(),
        "fair_comparison_policy": _fair_comparison_policy(),
        "do_not_claim_yet": _do_not_claim_yet(claim_ledger, lodo),
        "safe_runtime_policy": _safe_runtime_policy(active),
        "final_route_current_hypothesis": _final_route_hypothesis(final_selection, module_actions, lodo),
    }


def render_markdown(report: dict[str, Any]) -> str:
    current = report["current_execution"]
    active = current.get("active_run") or {}
    readiness = report["readiness"]
    lines = [
        "# Next Experiment Decision Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Scope: {report['scope']}.",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Current Execution",
        "",
        f"- Readiness: `{readiness.get('verdict')}` ({readiness.get('num_blocks')} blocks, {readiness.get('num_warnings')} warnings).",
        f"- Progress: `{json.dumps(current.get('progress_counts'), ensure_ascii=False)}`.",
        f"- Runtime guard: `{current.get('runtime_verdict')}`; RoadSaW handoff: `{current.get('handoff_verdict')}`.",
    ]
    if active:
        lines.append(
            "- Active run: `{name}` `{status}` epoch `{epoch}/{epochs}` step `{step}/{steps}`.".format(
                name=active.get("name"),
                status=active.get("status"),
                epoch=active.get("active_epoch") or active.get("epoch") or "-",
                epochs=active.get("active_epochs") or active.get("epochs") or "-",
                step=active.get("active_step") or "-",
                steps=active.get("active_steps") or "-",
            )
        )
    else:
        lines.append("- Active run: none detected.")
    lines.append("")

    lines.extend(["## What Is Still Missing", ""])
    lines.append("| Gate | Why It Matters | Missing Evidence | Unlocks |")
    lines.append("|---|---|---|---|")
    for item in report["readiness"].get("blockers", []):
        lines.append(
            "| `{name}` | {message} | {missing} | {unlock} |".format(
                name=item.get("name"),
                message=_clean(item.get("message")),
                missing=_fmt_list(item.get("missing")),
                unlock=_gate_unlock(item.get("name")),
            )
        )
    lines.append("")

    lodo = report.get("lodo_evidence") or {}
    if lodo:
        lines.extend(["## LODO Evidence So Far", ""])
        lines.append(f"- LODO verdict: `{lodo.get('verdict')}`.")
        lines.append("| Held-out | Status | friction F1 | risk F1 | low recall | raw cov | calib cov | width | Diagnosis |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in lodo.get("rows", []):
            lines.append(
                "| {held_out} | `{status}` | {friction} | {risk} | {low} | {raw} | {calib} | {width} | {diag} |".format(
                    held_out=row.get("held_out"),
                    status=row.get("status"),
                    friction=_fmt_pct(row.get("friction_f1")),
                    risk=_fmt_pct(row.get("risk_f1")),
                    low=_fmt_pct(row.get("low_friction_recall")),
                    raw=_fmt_pct(row.get("raw_coverage")),
                    calib=_fmt_pct(row.get("calibrated_coverage")),
                    width=_fmt_num(row.get("calibrated_width")),
                    diag=_clean(row.get("diagnosis")),
                )
            )
        lines.append("")
        for item in lodo.get("failure_interpretation", []):
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(["## Main Problems To Fix", ""])
    lines.append("| Signal | Evidence | Candidate Response |")
    lines.append("|---|---|---|")
    for item in report.get("failure_signals", []):
        lines.append(
            "| `{signal}` | {evidence} | {response} |".format(
                signal=item.get("signal"),
                evidence=_clean(item.get("evidence")),
                response=_clean(item.get("candidate_response")),
            )
        )
    lines.append("")

    input_style = report.get("input_canonicalization_evidence") or {}
    if input_style:
        lines.extend(["## Input Canonicalization Evidence", ""])
        lines.append(
            "- Best diagnostic preprocessing: `{run}` with style-gap `{score}` (`{relative}` of baseline `{base}`).".format(
                run=input_style.get("best_run"),
                score=_fmt_num(input_style.get("best_style_gap_score")),
                relative=_fmt_pct(input_style.get("best_style_gap_relative")),
                base=input_style.get("baseline_run"),
            )
        )
        lines.append("| Run | resize | GrayWorld | style gap | rel. | saturation span | channel-spread span | Decision use |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in input_style.get("rows", []):
            lines.append(
                "| {run} | {resize} | {gray} | {score} | {relative} | {sat} | {channel} | {use} |".format(
                    run=row.get("run"),
                    resize=row.get("resize_mode"),
                    gray=_fmt_num(row.get("gray_world_alpha")),
                    score=_fmt_num(row.get("style_gap_score")),
                    relative=_fmt_pct(row.get("style_gap_relative")),
                    sat=_fmt_num(row.get("saturation_span")),
                    channel=_fmt_num(row.get("channel_mean_spread_span")),
                    use=_clean(row.get("decision_use")),
                )
            )
        lines.append("")

    lines.extend(["## Module Decisions", ""])
    lines.append("| Module | Current Decision | Reason | Next Evidence |")
    lines.append("|---|---|---|---|")
    for item in report.get("module_actions", []):
        lines.append(
            "| {module} | `{decision}` | {reason} | {evidence} |".format(
                module=item.get("module"),
                decision=item.get("decision"),
                reason=_clean(item.get("reason")),
                evidence=_fmt_list(item.get("next_required_evidence")),
            )
        )
    lines.append("")

    lines.extend(["## Priority Action Plan", ""])
    lines.append("| Priority | Action | Runs | Success Rule | Failure Response |")
    lines.append("|---:|---|---|---|---|")
    for index, item in enumerate(report.get("action_plan", []), start=1):
        lines.append(
            "| {idx} | {action} | {runs} | {success} | {failure} |".format(
                idx=index,
                action=_clean(item.get("action")),
                runs=_fmt_list(item.get("runs")),
                success=_clean(item.get("success_rule")),
                failure=_clean(item.get("failure_response")),
            )
        )
    lines.append("")

    lines.extend(["## Decision Rules", ""])
    for item in report.get("decision_rules", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## Fair Comparison Policy", ""])
    for item in report.get("fair_comparison_policy", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## Claims Not Allowed Yet", ""])
    for item in report.get("do_not_claim_yet", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.extend(["## Safe Runtime Policy", ""])
    for item in report.get("safe_runtime_policy", []):
        lines.append(f"- {item}")
    lines.append("")

    hypothesis = report.get("final_route_current_hypothesis") or {}
    lines.extend(["## Current Final-Route Hypothesis", ""])
    lines.append(hypothesis.get("summary", "-"))
    lines.append("")
    for item in hypothesis.get("requirements", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _action_plan(
    active: dict[str, Any],
    queue_order: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    group_status: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_run = {row.get("run"): row for row in candidate_rows if isinstance(row, dict)}
    statuses = {row.get("name"): row.get("status") for row in queue_order if isinstance(row, dict)}
    blocker_names = {item.get("name") for item in blockers}
    actions: list[dict[str, Any]] = []

    if active:
        actions.append(
            {
                "action": "Let the active training finish; do not launch a duplicate GPU worker.",
                "runs": [active.get("name")],
                "success_rule": "Current run reaches postprocessing artifacts without OOM or stale logs.",
                "failure_response": "Use the existing recovery command only after there is no active train or queue process.",
            }
        )

    if statuses.get("lodo_roadsaw_full_faf") != "complete":
        row = by_run.get("lodo_roadsaw_full_faf", {})
        actions.append(
            {
                "action": "Run held-out RoadSaW LODO as the first cross-dataset stress test.",
                "runs": ["lodo_roadsaw_full_faf"],
                "success_rule": row.get("success_criteria")
                or "Held-out RoadSaW risk/friction F1 and low-friction recall are usable.",
                "failure_response": row.get("failure_action")
                or "Prioritize wet-state and shortcut-mitigation candidates before making OOD claims.",
            }
        )

    lodo_missing = _missing_runs(group_status, "lodo")
    if lodo_missing:
        actions.append(
            {
                "action": "Complete the remaining LODO suite.",
                "runs": lodo_missing,
                "success_rule": "Each public dataset is tested as unseen, with no train/val leakage from the held-out dataset.",
                "failure_response": "Use failing held-out domains to narrow the paper claim and choose the final lean route.",
            }
        )

    fair_missing = _missing_runs(group_status, "single_dataset_faf") + _missing_runs(
        group_status, "single_dataset_baselines"
    )
    if fair_missing:
        actions.append(
            {
                "action": "Run matched single-dataset FAF and ConvNeXt baselines.",
                "runs": fair_missing,
                "success_rule": "Same split, same labels, same metrics, same backbone budget, paired bootstrap deltas.",
                "failure_response": "If FAF does not beat ConvNeXt, position FAF around uncertainty, interpretability, and cross-domain safety.",
            }
        )

    candidate_missing = _missing_runs(group_status, "p1_candidates")
    if candidate_missing:
        actions.append(
            {
                "action": "Run candidate innovations that directly answer current failure signals.",
                "runs": candidate_missing,
                "success_rule": "Reduce dataset-ID shortcut, improve RoadSaW wetness, or improve conditional coverage without hurting safety metrics.",
                "failure_response": "Drop candidates that only improve pooled accuracy while hurting held-out/worst-domain or interval quality.",
            }
        )

    final_missing = _missing_runs(group_status, "final_method_lodo") + _missing_runs(
        group_status, "final_method_single_dataset"
    )
    if final_missing or "final_method_complete" in blocker_names:
        actions.append(
            {
                "action": "Freeze and run the final lean road-ROI safety route after candidate evidence exists.",
                "runs": final_missing,
                "success_rule": "Final method beats or complements ConvNeXt and preserves held-out RoadSaW/low-friction/coverage evidence.",
                "failure_response": "Do not freeze the final architecture; return to PhysicsTexture plus targeted evidence/interval fixes.",
            }
        )

    return actions


def _module_actions(module_retention: dict[str, Any]) -> list[dict[str, Any]]:
    rows = module_retention.get("rows") or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = row.get("current_decision") or row.get("decision")
        module = row.get("module")
        if not module:
            continue
        out.append(
            {
                "module": module,
                "decision": decision or "pending",
                "reason": row.get("evidence_summary") or "Evidence pending.",
                "next_required_evidence": row.get("next_required_evidence") or row.get("rescue_or_confirmation_tests") or [],
            }
        )
    return out


def _compact_lodo_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = []
    failures = []
    for row in report.get("rows") or []:
        if not isinstance(row, dict):
            continue
        compact = {
            "held_out": row.get("held_out"),
            "status": row.get("status"),
            "friction_f1": row.get("friction_f1"),
            "risk_f1": row.get("risk_f1"),
            "low_friction_recall": row.get("low_friction_recall"),
            "raw_coverage": row.get("raw_coverage"),
            "calibrated_coverage": row.get("calibrated_coverage"),
            "calibrated_width": row.get("calibrated_width"),
            "diagnosis": _lodo_row_diagnosis(row),
        }
        rows.append(compact)
        if str(row.get("status")) == "complete" and compact["diagnosis"] != "usable":
            failures.append(compact)
    interpretation = []
    for row in failures:
        interpretation.append(
            "{held_out} held-out transfer is {diagnosis}; treat it as evidence that current Full FAF still learns dataset-specific style/label priors.".format(
                held_out=row.get("held_out"),
                diagnosis=row.get("diagnosis"),
            )
        )
    if failures:
        interpretation.append(
            "Prioritize input canonicalization, condition-aware alignment, domain adapters, and lean PhysicsTexture+EvidenceField routes before broad OOD claims."
        )
    return {
        "verdict": report.get("verdict"),
        "rows": rows,
        "failure_interpretation": interpretation,
    }


def _lodo_row_diagnosis(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    if status != "complete":
        return status or "pending"
    risk = _to_float(row.get("risk_f1"))
    friction = _to_float(row.get("friction_f1"))
    raw_cov = _to_float(row.get("raw_coverage"))
    calib_cov = _to_float(row.get("calibrated_coverage"))
    if (risk is not None and risk < 0.20) or (friction is not None and friction < 0.20):
        return "severe transfer failure"
    if calib_cov is not None and calib_cov < 0.70:
        return "interval transfer failure"
    if raw_cov is not None and raw_cov < 0.35:
        return "raw interval undercoverage"
    return "usable"


def _lodo_failure_signals(report: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for row in report.get("rows") or []:
        if not isinstance(row, dict) or row.get("status") != "complete":
            continue
        diagnosis = _lodo_row_diagnosis(row)
        if diagnosis == "usable":
            continue
        held_out = str(row.get("held_out") or "held-out")
        out.append(
            {
                "signal": f"lodo_{held_out.lower()}_{diagnosis.replace(' ', '_')}",
                "evidence": (
                    f"{held_out} LODO: friction F1 {_fmt_pct(row.get('friction_f1'))}, "
                    f"risk F1 {_fmt_pct(row.get('risk_f1'))}, "
                    f"calibrated coverage {_fmt_pct(row.get('calibrated_coverage'))}."
                ),
                "candidate_response": (
                    "Use this as a hard OOD failure signal; prioritize v15/v16 input canonicalization, "
                    "v11 adapters/condition-aware alignment, and final lean road-ROI safety rows."
                ),
            }
        )
    return out


def _compact_input_canonicalization_style(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("configs"):
        return {}
    rows: list[dict[str, Any]] = []
    for item in report.get("configs") or []:
        transform = item.get("transform") or {}
        cross = item.get("cross_dataset_signals") or {}
        relative = item.get("relative_to_first_config") or {}
        run = item.get("run")
        style_gap = _signal_value(cross, "style_gap_score")
        rel = _nested(relative, "style_gap_score", "relative")
        rows.append(
            {
                "run": run,
                "resize_mode": transform.get("resize_mode"),
                "gray_world_alpha": transform.get("gray_world_alpha"),
                "style_gap_score": style_gap,
                "style_gap_relative": rel,
                "saturation_span": _signal_span(cross, "saturation_span"),
                "channel_mean_spread_span": _signal_span(cross, "channel_mean_spread_span"),
                "decision_use": _canonicalization_decision_use(run, style_gap, rel),
            }
        )
    valid = [row for row in rows if isinstance(row.get("style_gap_score"), (int, float))]
    best = min(valid, key=lambda row: float(row["style_gap_score"])) if valid else {}
    return {
        "generated_at": report.get("generated_at"),
        "baseline_run": rows[0].get("run") if rows else None,
        "best_run": best.get("run"),
        "best_style_gap_score": best.get("style_gap_score"),
        "best_style_gap_relative": best.get("style_gap_relative"),
        "rows": rows,
    }


def _canonicalization_decision_use(run: Any, style_gap: Any, relative: Any) -> str:
    run_text = str(run or "")
    if "v16" in run_text and isinstance(relative, (int, float)) and relative < 0.95:
        return "Strong preprocessing candidate; retain only if LODO/task metrics do not regress."
    if "v15" in run_text:
        return "Geometry-only control; useful to separate crop effects from color constancy."
    if isinstance(relative, (int, float)) and relative >= 1.0:
        return "Baseline/control."
    if isinstance(style_gap, (int, float)):
        return "Diagnostic candidate."
    return "Pending."


def _signal_value(cross: dict[str, Any], key: str) -> Any:
    value = cross.get(key) if isinstance(cross, dict) else None
    if isinstance(value, dict):
        return value.get("span")
    return value


def _signal_span(cross: dict[str, Any], key: str) -> Any:
    value = cross.get(key) if isinstance(cross, dict) else None
    if isinstance(value, dict):
        return value.get("span")
    return None


def _nested(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _claim_status(claim_ledger: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_verdict": claim_ledger.get("readiness_verdict"),
        "status_counts": claim_ledger.get("status_counts") or {},
        "blocking_gates": claim_ledger.get("blocking_gates") or [],
        "rules": claim_ledger.get("claim_rules") or [],
    }


def _do_not_claim_yet(claim_ledger: dict[str, Any], lodo_report: dict[str, Any]) -> list[str]:
    rows = claim_ledger.get("claim_rows") or []
    claims = []
    lodo_verdict = lodo_report.get("verdict") if isinstance(lodo_report, dict) else None
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status in {"not_supported", "not_supported_yet", "partial"}:
            claim = str(row.get("claim") or "")
            wording = row.get("allowed_wording")
            if "generalizes across public road-condition datasets" in claim and lodo_verdict:
                if str(lodo_verdict) == "generalization_failure_needs_algorithm_update":
                    wording = (
                        "LODO is complete and shows severe transfer failure; present current Full FAF as a failure analysis "
                        "and require P1/final candidates before any cross-dataset generalization claim."
                    )
                elif str(lodo_verdict).startswith("supports"):
                    wording = "LODO supports only the cautious claim stated in the LODO report; keep single-dataset and baseline claims separate."
            claims.append(f"{claim}: {wording}")
    return claims


def _decision_rules() -> list[str]:
    return [
        "Keep a module only if it improves safety/generalization evidence: risk F1, low-friction recall, held-out or worst-dataset F1, and interval quality.",
        "Do not keep a module merely because it improves pooled accuracy if it worsens held-out RoadSaW, worst-domain F1, or raw/calibrated interval behavior.",
        "Treat FrictionSet, DG losses, and Full fusion as provisional remove-or-merge unless later LODO/fair evidence rescues them.",
        "Treat PhysicsTexture as the current strongest P0 component; verify it under LODO and single-dataset ConvNeXt comparisons.",
        "Use EvidenceField as an interpretability and road-region grounding route only if quantitative evidence maps and safety metrics remain acceptable.",
        "Treat v15/v16 as input-canonicalization tests: v15 isolates road-centric geometry, v16 adds color constancy; keep them only if style-gap reduction transfers to LODO/task metrics.",
        "Do not let color constancy erase wetness cues; reject v16 if RoadSaW wetness F1, severe misorder, or low-friction recall regress materially.",
        "Prefer a lean final method over a large fusion method unless the large method wins under the predeclared multi-metric score.",
    ]


def _fair_comparison_policy() -> list[str]:
    return [
        "Main fair baseline is the locally matched ConvNeXt run on the same public split and weak-label mapping.",
        "External published numbers are context only unless split, label space, preprocessing, and metric definition match.",
        "Report paired bootstrap confidence intervals and deltas, not only point estimates.",
        "Separate single-dataset performance from LODO cross-dataset generalization; these support different claims.",
    ]


def _safe_runtime_policy(active: dict[str, Any]) -> list[str]:
    if active:
        return [
            "Do not start another GPU training process while the active run is healthy.",
            "Let the current queued run finish and postprocess before forcing queue changes.",
            "Do not launch a duplicate training worker while the queue orchestrator is alive.",
            "If all train/queue processes disappear, resume with run_paper_protocol_direct.py --phase all --postprocess-each.",
        ]
    return [
        "No active run detected; resume the queued protocol with the conda Python and --postprocess-each.",
        "Refresh dashboard, runtime guard, and this decision report after launching or resuming.",
    ]


def _final_route_hypothesis(
    final_selection: dict[str, Any],
    module_actions: list[dict[str, Any]],
    lodo_report: dict[str, Any],
) -> dict[str, Any]:
    keep = [row["module"] for row in module_actions if row.get("decision") == "provisional_keep"]
    remove = [
        row["module"]
        for row in module_actions
        if row.get("decision") == "provisional_remove_or_merge"
    ]
    lodo_complete = (
        isinstance(lodo_report, dict)
        and len([row for row in lodo_report.get("rows", []) if row.get("status") == "complete"]) >= 3
    )
    first_requirement = (
        "Use the completed LODO failures as the main domain-gap evidence; do not claim cross-dataset generalization for the current Full FAF."
        if lodo_complete
        else "Finish LODO, especially held-out RoadSaW, before any cross-dataset generalization claim."
    )
    return {
        "summary": (
            "Current evidence favors a lean route built around PhysicsTexture plus carefully constrained "
            "EvidenceField/road-ROI/interval-safety components, not the current Full fusion."
        ),
        "selection_verdict": final_selection.get("verdict"),
        "provisional_keep": keep,
        "provisional_remove_or_merge": remove,
        "requirements": [
            first_requirement,
            "Finish matched FAF vs ConvNeXt single-dataset comparisons before any performance claim against literature-style baselines.",
            "Run v6-v16 to determine whether shortcut, wetness, input/color-canonicalization, and interval fixes actually solve the observed failures.",
            "Only freeze the final architecture after the final lean LODO and final single-dataset rows are complete.",
        ],
    }


def _active_run(
    dashboard: dict[str, Any],
    queue: dict[str, Any],
    watch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in queue.get("queue_order", []):
        if isinstance(row, dict) and row.get("status") in {"running_or_partial", "partial_ci_missing"}:
            return row
    active_rows = dashboard.get("active_rows") or []
    if active_rows and isinstance(active_rows[0], dict):
        candidates.append(active_rows[0])
    watch_active = (watch or {}).get("active") or {}
    if watch_active.get("name"):
        candidates.append({
            "name": watch_active.get("name"),
            "status": watch_active.get("status"),
            "phase": watch_active.get("phase"),
            "active_epoch": watch_active.get("epoch"),
            "active_epochs": watch_active.get("epochs"),
            "active_step": watch_active.get("step"),
            "active_steps": watch_active.get("steps"),
        })
    candidates = [row for row in candidates if row.get("name")]
    if not candidates:
        return {}
    return max(candidates, key=_active_score)


def _active_score(row: dict[str, Any]) -> tuple[int, int, int]:
    epoch = _safe_int(row.get("active_epoch") or row.get("epoch"))
    step = _safe_int(row.get("active_step") or row.get("step"))
    phase = str(row.get("active_phase") or row.get("phase") or "")
    phase_rank = 1 if phase == "eval" else 0
    return (epoch, phase_rank, step)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _missing_runs(group_status: dict[str, Any], key: str) -> list[str]:
    group = group_status.get(key) or {}
    missing = group.get("missing_runs") or []
    return [str(item) for item in missing]


def _gate_unlock(name: str | None) -> str:
    mapping = {
        "lodo_complete": "Cross-dataset generalization evidence.",
        "heldout_roadsaw_missing": "Hardest wetness-domain stress-test result.",
        "fair_single_dataset_complete": "Fair same-split comparison against ConvNeXt.",
        "fair_single_dataset_missing": "Baseline-comparable performance table.",
        "final_method_complete": "Evidence-backed final architecture.",
        "final_heldout_roadsaw_missing": "Final method OOD stress-test evidence.",
        "final_fair_single_dataset_missing": "Final method fair baseline comparison.",
        "full_vs_global_risk_f1": "Proof that fusion improves safety instead of hurting it.",
        "full_vs_global_low_friction_recall": "Proof that low-friction safety is preserved.",
        "full_vs_global_worst_dataset_f1": "Proof that fusion does not overfit the pooled distribution.",
    }
    return mapping.get(str(name), "A required paper claim or safety check.")


def _get_dashboard_value(dashboard: dict[str, Any], key: str) -> Any:
    if key in dashboard:
        return dashboard[key]
    handoff = dashboard.get("handoff_health") or {}
    if key in handoff:
        return handoff[key]
    return None


def _num_interval_anchors(interval_sources: dict[str, Any]) -> int | None:
    for key in ("rows", "anchors", "source_anchors"):
        value = interval_sources.get(key)
        if isinstance(value, list):
            return len(value)
    count = interval_sources.get("num_anchors")
    return int(count) if isinstance(count, int) else None


def _fmt_list(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(f"`{item}`" for item in value) if value else "-"
    return f"`{value}`"


def _fmt_num(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "-"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{100.0 * float(value):.2f}%"
    return "-"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("\n", " ").replace("|", "/")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_dashboard(summary_dir: Path) -> dict[str, Any]:
    candidates = [
        summary_dir / "experiment_status_dashboard.json",
        summary_dir / "experiment_dashboard.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {}
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    return _load_json(newest) or {}


if __name__ == "__main__":
    main()
