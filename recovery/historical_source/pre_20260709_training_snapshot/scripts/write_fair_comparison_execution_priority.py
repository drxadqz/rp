from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "fair_comparison_execution_priority.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "fair_comparison_execution_priority.json"
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
ACTIVE_STATUSES = {"active", "running", "in_progress", "running_or_partial"}


FAIR_ROWS = [
    "single_roadsaw_full_faf",
    "single_rscd_full_faf",
    "single_roadsc_full_faf",
    "baseline_single_roadsaw_global_convnext",
    "baseline_single_rscd_global_convnext",
    "baseline_single_roadsc_global_convnext",
]

RSCD_EXTERNAL_ROWS = ["rscd27_surface_classification_formal"]

CANDIDATE_ROWS = [
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
    "v13_lean_physics_evidence",
    "v14_lean_road_roi_safety",
    "v15_lean_bottom_square_style_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
]

FINAL_ROWS = [
    "final_lodo_roadsaw_lean_road_roi_safety",
    "final_lodo_rscd_lean_road_roi_safety",
    "final_lodo_roadsc_lean_road_roi_safety",
    "final_single_roadsaw_lean_road_roi_safety",
    "final_single_rscd_lean_road_roi_safety",
    "final_single_roadsc_lean_road_roi_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write an execution-priority audit for fair comparisons, CV-transfer "
            "candidate screening, RSCD external checks, and final-method claims."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir, args.log_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path, log_dir: Path = DEFAULT_LOG_DIR) -> dict[str, Any]:
    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}
    next_queue = _load_json(summary_dir / "next_queue_readiness_report.json") or {}
    candidate_hypothesis = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    candidate_pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    rscd_external = _load_json(summary_dir / "rscd_external_comparison_readiness.json") or {}
    topvenue_gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}

    run_status = _collect_run_status(artifact, next_queue, candidate_hypothesis)
    _refresh_active_status_from_logs(run_status, log_dir)
    active_runs = [
        row
        for row in run_status.values()
        if row.get("progress_status") in ACTIVE_STATUSES
        or row.get("active_epoch") is not None
    ]

    stages = [
        _stage(
            name="finish_matched_single_dataset_fairness",
            priority=1,
            rows=FAIR_ROWS,
            run_status=run_status,
            purpose=(
                "Lock same-split FAF vs ConvNeXt evidence before any SOTA-style or "
                "final-method claim."
            ),
            claim_unlocked=(
                "Fair RSCD/RoadSaW/RoadSC public-data comparison and paired bootstrap deltas."
            ),
        ),
        _stage(
            name="run_rscd27_class_label_external_check",
            priority=2,
            rows=RSCD_EXTERNAL_ROWS,
            run_status=run_status,
            purpose=(
                "Provide a secondary RSCD class-label comparison only if the implemented "
                "runner produces local results under the documented split."
            ),
            claim_unlocked="RSCD-27 context result; no RSCD SOTA-style claim before result exists.",
            external_status=rscd_external.get("verdict"),
        ),
        _stage(
            name="screen_cv_transfer_candidates",
            priority=3,
            rows=CANDIDATE_ROWS,
            run_status=run_status,
            purpose=(
                "Evaluate segmentation-style evidence, style/domain shortcut mitigation, "
                "weak consistency, and interval-safety candidates with predeclared pruning rules."
            ),
            claim_unlocked="Module-retention evidence for the final architecture.",
        ),
        _stage(
            name="freeze_and_run_final_lean_method",
            priority=4,
            rows=FINAL_ROWS,
            run_status=run_status,
            purpose=(
                "Run the final lean road-ROI safety method only after fair baselines and "
                "candidate pruning identify which modules survive."
            ),
            claim_unlocked="Final method comparison, final LODO stress evidence, and paper tables.",
        ),
    ]

    claim_locks = _claim_locks(stages, rscd_external, topvenue_gate)
    transfer_routes = _transfer_routes(candidate_hypothesis)
    violations = _sequence_violations(stages)
    next_actions = _next_actions(stages, active_runs, rscd_external, candidate_pruning)
    verdict = _verdict(stages, violations)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "active_runs": active_runs,
        "stages": stages,
        "claim_locks": claim_locks,
        "cv_transfer_routes": transfer_routes,
        "sequence_violations": violations,
        "candidate_pruning_verdict": candidate_pruning.get("verdict"),
        "next_actions": next_actions,
        "policy": [
            "Fair same-split public-dataset comparisons come before SOTA-style claims.",
            "RSCD-27 class-label results are context only unless the local protocol exactly matches the cited benchmark.",
            "LODO failures are reported as stress evidence, not hidden or reframed as success.",
            "CV-transfer modules are retained only when they improve risk/F1/safety coverage or add audited interpretability without harming low-friction recall.",
            "Weak friction intervals remain public-label-derived friction-affordance intervals, not measured tire-road friction coefficients.",
        ],
    }


def _collect_run_status(
    artifact: dict[str, Any],
    next_queue: dict[str, Any],
    candidate_hypothesis: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for row in artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []:
        name = row.get("name")
        if not name:
            continue
        status[name] = {
            "run": name,
            "scope": row.get("scope"),
            "progress_status": row.get("progress_status"),
            "contract_status": row.get("contract_status"),
            "epoch": row.get("epoch"),
            "epochs": row.get("epochs"),
            "active_epoch": row.get("active_epoch"),
            "active_steps": row.get("active_steps"),
            "missing_required_artifacts": row.get("missing_required_artifacts", []),
            "next_action": row.get("next_action"),
        }
    for row in next_queue.get("rows", []) if isinstance(next_queue.get("rows"), list) else []:
        run = row.get("run") or row.get("name")
        if not run:
            continue
        status.setdefault(run, {"run": run})
        for key in [
            "progress_status",
            "contract_status",
            "protocol_ready",
            "gpu_ready",
            "module_ready",
            "claim_use",
            "cv_transfer",
            "role",
        ]:
            if key in row and row.get(key) not in {None, ""}:
                status[run][key] = row.get(key)
    for row in candidate_hypothesis.get("rows", []) if isinstance(candidate_hypothesis.get("rows"), list) else []:
        run = row.get("run")
        if not run:
            continue
        status.setdefault(run, {"run": run})
        for key in [
            "phase",
            "hypothesis",
            "addresses",
            "success_criteria",
            "failure_action",
            "retention_rule",
            "claim_unlocked",
            "available_metrics",
        ]:
            if key in row:
                status[run][key] = row.get(key)
    return status


def _stage(
    *,
    name: str,
    priority: int,
    rows: list[str],
    run_status: dict[str, dict[str, Any]],
    purpose: str,
    claim_unlocked: str,
    external_status: str | None = None,
) -> dict[str, Any]:
    items = [run_status.get(run, {"run": run, "progress_status": "missing"}) for run in rows]
    complete = [
        row["run"]
        for row in items
        if row.get("contract_status") == "complete" or row.get("progress_status") == "complete"
    ]
    active = [
        row["run"]
        for row in items
        if row.get("progress_status") in ACTIVE_STATUSES
        or row.get("active_epoch") is not None
    ]
    missing = [row["run"] for row in items if row["run"] not in set(complete) | set(active)]
    if len(complete) == len(rows):
        status = "complete"
    elif active:
        status = "in_progress"
    elif complete:
        status = "partial"
    else:
        status = "pending"
    if external_status and external_status != "ready":
        status = "protocol_ready_results_pending" if status == "pending" else status
    return {
        "priority": priority,
        "name": name,
        "status": status,
        "purpose": purpose,
        "claim_unlocked": claim_unlocked,
        "runs": rows,
        "complete": complete,
        "active": active,
        "missing_or_pending": missing,
        "external_status": external_status,
    }


def _claim_locks(
    stages: list[dict[str, Any]],
    rscd_external: dict[str, Any],
    topvenue_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_by_name = {stage["name"]: stage for stage in stages}
    fair = stage_by_name["finish_matched_single_dataset_fairness"]
    final = stage_by_name["freeze_and_run_final_lean_method"]
    candidates = stage_by_name["screen_cv_transfer_candidates"]
    return [
        {
            "claim": "same_split_public_dataset_advantage",
            "status": "allowed" if fair["status"] == "complete" else "locked",
            "required_evidence": "All six single-dataset FAF/ConvNeXt rows complete plus fair pairwise report.",
            "current_gap": fair["missing_or_pending"],
        },
        {
            "claim": "rscd_sota_style_context",
            "status": "locked"
            if rscd_external.get("formal_status") != "complete"
            else "allowed_with_protocol_caveat",
            "required_evidence": "Local RSCD-27 class-label result and exact protocol mapping.",
            "current_gap": rscd_external.get("sota_claim", "result missing or protocol not checked"),
        },
        {
            "claim": "final_method_topvenue_ready",
            "status": "allowed" if final["status"] == "complete" else "locked",
            "required_evidence": "Final LODO and final single-dataset comparisons complete.",
            "current_gap": final["missing_or_pending"],
        },
        {
            "claim": "module_retention_or_pruning",
            "status": "allowed" if candidates["status"] == "complete" else "locked",
            "required_evidence": "v6-v25 candidate metrics plus pruning report.",
            "current_gap": candidates["missing_or_pending"],
        },
        {
            "claim": "measured_tire_road_friction_coefficient",
            "status": "disallowed",
            "required_evidence": "Synchronized measured friction/dynamics labels, which the public visual datasets do not provide.",
            "current_gap": "Use visual friction-affordance intervals instead.",
        },
        {
            "claim": "readiness_gate",
            "status": topvenue_gate.get("verdict", "unknown"),
            "required_evidence": "Top-venue gate has no block items.",
            "current_gap": topvenue_gate.get("num_blocks"),
        },
    ]


def _refresh_active_status_from_logs(
    run_status: dict[str, dict[str, Any]],
    log_dir: Path,
) -> None:
    if not log_dir.exists():
        return
    for run, row in run_status.items():
        if row.get("progress_status") not in ACTIVE_STATUSES and row.get("active_epoch") is None:
            continue
        latest_out = _latest_log(log_dir, f"{run}_*.out.log")
        latest_err = _latest_log(log_dir, f"{run}_*.err.log")
        parsed = _parse_live_log(latest_out, latest_err)
        if not parsed:
            continue
        row.update({key: value for key, value in parsed.items() if value is not None})
        row["live_log_refreshed"] = True


def _latest_log(log_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(log_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_live_log(out_log: Path | None, err_log: Path | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if out_log and out_log.exists():
        text = _tail_text(out_log, max_chars=60000)
        epoch_matches = list(re.finditer(r"Epoch\s+(\d+)\s*/\s*(\d+)", text))
        if epoch_matches:
            last = epoch_matches[-1]
            parsed["active_epoch"] = int(last.group(1))
            parsed["active_epochs"] = int(last.group(2))
        step_matches = list(re.finditer(r"train step\s+(\d+)\s*/\s*(\d+)", text))
        if step_matches:
            last = step_matches[-1]
            parsed["active_step"] = int(last.group(1))
            parsed["active_steps"] = int(last.group(2))
        parsed["out_log"] = str(out_log)
    if err_log and err_log.exists():
        text = _tail_text(err_log, max_chars=30000)
        tqdm_matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)\s*\[", text))
        if tqdm_matches:
            last = tqdm_matches[-1]
            parsed["active_step"] = int(last.group(1))
            parsed["active_steps"] = int(last.group(2))
        parsed["err_log"] = str(err_log)
    return parsed


def _tail_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _transfer_routes(candidate_hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = candidate_hypothesis.get("rows", []) if isinstance(candidate_hypothesis.get("rows"), list) else []
    lookup = {row.get("run"): row for row in rows if isinstance(row, dict)}
    specs = [
        (
            "semantic_segmentation_local_evidence",
            ["v14_lean_road_roi_safety", "v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety"],
            "Transfer mask-classification thinking into weak local evidence tokens and region-mixture pooling.",
            "Keep only if RoadSaW wet/white slices, low-friction recall, or audited attention quality improve.",
        ),
        (
            "semi_supervised_segmentation_consistency",
            ["v10_full_faf_consistency", "v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety"],
            "Make logits, intervals, and evidence attention stable under weak/strong visual perturbations.",
            "Drop or weaken if low-friction recall falls or hard wet/snow states are over-smoothed.",
        ),
        (
            "domain_adaptive_segmentation_shortcut_control",
            ["v6_full_faf_fourier", "v7_full_faf_fourier_dann", "v18_lean_mixstyle_quality_safety"],
            "Use Fourier/style statistics and adapter/adversarial ideas to reduce dataset style shortcuts.",
            "Keep only state-conditioned or lightweight pieces; prune generic DANN if safety metrics regress.",
        ),
        (
            "material_texture_physical_vision",
            ["v17_lean_quality_physics_safety", "v21_lean_quality_uncertainty_safety"],
            "Treat wet glare, low texture, near-white snow, and roughness as friction-interval uncertainty evidence.",
            "Retain if coverage-width tradeoff improves without turning intervals into uninformative wide boxes.",
        ),
        (
            "foundation_dense_teacher",
            [],
            "Use DINOv2/SAM/CLIPSeg-style dense features only as offline teachers or strong baselines after fair rows.",
            "Do not claim a method gain until same-split baselines and pseudo-mask audits support it.",
        ),
    ]
    routes = []
    for name, runs, transfer, promotion_rule in specs:
        routes.append(
            {
                "route": name,
                "runs": runs,
                "transfer": transfer,
                "promotion_rule": promotion_rule,
                "evidence_status": [
                    {
                        "run": run,
                        "progress_status": lookup.get(run, {}).get("progress_status", "missing"),
                        "metrics_available": bool(lookup.get(run, {}).get("available_metrics")),
                    }
                    for run in runs
                ],
            }
        )
    return routes


def _sequence_violations(stages: list[dict[str, Any]]) -> list[str]:
    by = {stage["name"]: stage for stage in stages}
    violations: list[str] = []
    fair = by["finish_matched_single_dataset_fairness"]
    candidates = by["screen_cv_transfer_candidates"]
    final = by["freeze_and_run_final_lean_method"]
    if fair["status"] != "complete" and final["complete"]:
        violations.append("final rows completed before the matched single-dataset fair-comparison block")
    if candidates["status"] == "pending" and final["complete"]:
        violations.append("final method completed before candidate pruning evidence exists")
    return violations


def _next_actions(
    stages: list[dict[str, Any]],
    active_runs: list[dict[str, Any]],
    rscd_external: dict[str, Any],
    candidate_pruning: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if active_runs:
        runs = ", ".join(row["run"] for row in active_runs)
        actions.append(f"Let the active GPU run finish before launching a competing job: {runs}.")
    for stage in stages:
        if stage["status"] != "complete":
            if stage["active"]:
                actions.append(
                    f"Priority {stage['priority']}: monitor active run(s) {', '.join(stage['active'])}; "
                    f"then finish {', '.join(stage['missing_or_pending']) or 'remaining postprocess'}."
                )
            else:
                actions.append(
                    f"Priority {stage['priority']}: run {', '.join(stage['missing_or_pending'])}."
                )
            break
    if rscd_external.get("formal_status") != "complete":
        actions.append("Keep RSCD-27 as a secondary external check until the formal local result exists.")
    if candidate_pruning.get("verdict") == "ready_pending_candidate_metrics":
        actions.append("After fair rows, use v6-v25 metrics to prune FrictionSet/DG/full-fusion style modules aggressively.")
    return actions


def _verdict(stages: list[dict[str, Any]], violations: list[str]) -> str:
    if violations:
        return "sequence_violation"
    first_incomplete = next((stage for stage in stages if stage["status"] != "complete"), None)
    if first_incomplete is None:
        return "execution_chain_complete"
    if first_incomplete["active"]:
        return "waiting_for_active_fair_or_formal_run"
    return "ready_for_next_priority_stage"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fair Comparison Execution Priority",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Current Active Runs",
        "",
    ]
    if report["active_runs"]:
        for row in report["active_runs"]:
            lines.append(
                f"- `{row['run']}`: progress=`{row.get('progress_status')}`, "
                f"epoch=`{row.get('active_epoch') or row.get('epoch')}`, next=`{row.get('next_action')}`"
            )
    else:
        lines.append("- None recorded in the artifact contract.")
    lines.extend(
        [
            "",
            "## Execution Stages",
            "",
            "| Priority | Stage | Status | Complete | Active | Missing/Pending | Claim unlocked |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for stage in report["stages"]:
        lines.append(
            "| {priority} | {name} | {status} | {complete} | {active} | {missing} | {claim} |".format(
                priority=stage["priority"],
                name=stage["name"],
                status=stage["status"],
                complete=_join(stage["complete"]),
                active=_join(stage["active"]),
                missing=_join(stage["missing_or_pending"]),
                claim=stage["claim_unlocked"],
            )
        )
    lines.extend(["", "## Claim Locks", ""])
    for lock in report["claim_locks"]:
        lines.append(
            f"- `{lock['claim']}`: `{lock['status']}`. Required: {lock['required_evidence']} "
            f"Gap: {_compact(lock['current_gap'])}"
        )
    lines.extend(
        [
            "",
            "## CV Transfer Routes",
            "",
            "| Route | Runs | Transfer | Promotion/drop rule |",
            "|---|---|---|---|",
        ]
    )
    for route in report["cv_transfer_routes"]:
        lines.append(
            f"| {route['route']} | {_join(route['runs'])} | {route['transfer']} | {route['promotion_rule']} |"
        )
    lines.extend(["", "## Policy", ""])
    lines.extend(f"- {item}" for item in report["policy"])
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(report["next_actions"], start=1))
    if report["sequence_violations"]:
        lines.extend(["", "## Sequence Violations", ""])
        lines.extend(f"- {item}" for item in report["sequence_violations"])
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _join(items: Any) -> str:
    if not items:
        return "-"
    if isinstance(items, list):
        return ", ".join(str(item) for item in items)
    return str(items)


def _compact(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return _join(value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


if __name__ == "__main__":
    main()
