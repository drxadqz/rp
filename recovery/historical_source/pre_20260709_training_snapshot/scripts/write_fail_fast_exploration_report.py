from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
FAST_SCREEN_ANCHOR = "v1_physics_texture"

FULL_STACK_CANDIDATES = [
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
]

LEAN_CANDIDATES = [
    "v14_lean_road_roi_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v15_lean_bottom_square_style_safety",
    "v13_lean_physics_evidence",
]

FAST_PRIORITY = [
    "v14_lean_road_roi_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v15_lean_bottom_square_style_safety",
    "v13_lean_physics_evidence",
    "v9_full_faf_roadsaw_hard_sampling",
    "v12_full_faf_roi_interval_safety",
    "v6_full_faf_fourier",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v8_full_faf_fourier_roadprior",
    "v7_full_faf_fourier_dann",
]

LEAN_FIRST_WAVE = [
    FAST_SCREEN_ANCHOR,
    "v14_lean_road_roi_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v15_lean_bottom_square_style_safety",
    "v13_lean_physics_evidence",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "fail_fast_exploration_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "fail_fast_exploration_report.json",
    )
    args = parser.parse_args()

    report = build_report(
        summary_dir=args.summary_dir,
        python=args.python,
        root=args.root,
        log_dir=args.log_dir,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(*, summary_dir: Path, python: Path, root: Path, log_dir: Path) -> dict[str, Any]:
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    fast = _load_json(summary_dir / "fast_screen_status_report.json") or {}
    promotion = _load_json(summary_dir / "fast_to_formal_promotion_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    guard = _load_json(summary_dir / "gpu_scheduling_guard_report.json") or {}

    p0_rows = {row.get("method"): row for row in p0.get("rows", []) if isinstance(row, dict)}
    physics = p0_rows.get("+ PhysicsTexture") or {}
    full = p0_rows.get("Full model") or {}
    global_only = p0_rows.get("Global-only") or {}
    friction_set = p0_rows.get("+ FrictionSet") or {}
    dg = p0_rows.get("+ DG losses") or {}
    evidence = p0_rows.get("+ EvidenceField aux") or {}

    fast_rows = fast.get("rows", []) if isinstance(fast.get("rows"), list) else []
    fast_status = {str(row.get("source_run") or ""): row for row in fast_rows if isinstance(row, dict)}
    completed_fast = [row for row in fast_rows if row.get("status") == "complete"]

    kill_rows = [
        _module_decision(
            "Full fusion / v5_full_faf as final route",
            "kill_as_final_route",
            [
                _delta_reason("risk F1", full, evidence, "risk_macro_f1", bad_below=-0.02),
                _delta_reason("low-friction recall", full, evidence, "low_friction_recall", bad_below=-0.02),
                _delta_reason("worst dataset F1", full, physics, "worst_dataset_f1", bad_below=-0.03),
                _delta_reason("calibrated width", full, evidence, "calibrated_width", bad_above=0.04),
            ],
            "Do not use the full stack as the paper method unless a later lean/fail-fast screen explicitly rescues one component.",
        ),
        _module_decision(
            "DG losses / v3_dg_losses",
            "kill_current_form",
            [
                _delta_reason("risk F1", dg, friction_set, "risk_macro_f1", bad_below=-0.02),
                _delta_reason("friction F1", dg, friction_set, "friction_macro_f1", bad_below=-0.02),
                _delta_reason("low-friction recall", dg, friction_set, "low_friction_recall", bad_below=-0.02),
                _delta_reason("calibrated width", dg, friction_set, "calibrated_width", bad_above=0.04),
            ],
            "Keep only condition-aware alignment variants that pass fast screen; remove plain DG from final design.",
        ),
        _module_decision(
            "FrictionSet / v2_friction_set as an independent branch",
            "hold_only_as_interval_component",
            [
                _delta_reason("worst dataset F1", friction_set, physics, "worst_dataset_f1", bad_below=-0.03),
                _delta_reason("calibrated coverage", friction_set, physics, "calibrated_coverage"),
            ],
            "Do not keep FrictionSet as a standalone module; allow only a narrow interval-calibration rescue if width and worst-domain metrics stay safe.",
        ),
    ]

    keep_rows = [
        _module_decision(
            "PhysicsTexture",
            "protect",
            [
                _delta_reason("friction F1", physics, global_only, "friction_macro_f1"),
                _delta_reason("risk F1", physics, global_only, "risk_macro_f1"),
                _delta_reason("low-friction recall", physics, global_only, "low_friction_recall"),
                _delta_reason("worst dataset F1", physics, global_only, "worst_dataset_f1"),
            ],
            "Use this as the lean core unless matched ConvNeXt baselines prove the gain is not meaningful.",
        ),
        _module_decision(
            "EvidenceField",
            "keep_if_roi_evidence_passes",
            [
                _delta_reason("risk F1", evidence, physics, "risk_macro_f1", bad_below=-0.02),
                _delta_reason("low-friction recall", evidence, physics, "low_friction_recall", bad_below=-0.02),
                _delta_reason("calibrated width", evidence, physics, "calibrated_width", bad_above=0.04),
            ],
            "Keep for interpretability only with road/bottom ROI evidence and no large safety regression.",
        ),
    ]

    candidate_rows = [
        _candidate_row(source, fast_status.get(source), lodo=lodo, shortcut=shortcut)
        for source in FAST_PRIORITY
    ]
    formal_policy = _formal_policy(completed_fast, candidate_rows, fast, promotion, python, root, log_dir)

    lodo_rows = lodo.get("rows", []) if isinstance(lodo.get("rows"), list) else []
    roadsaw = (lodo.get("roadsaw_readout") or {}).get("heldout_metrics") or {}
    shortcut_counts = {
        "complete": shortcut.get("num_complete"),
        "high_shortcut": shortcut.get("num_high_shortcut"),
        "threshold": shortcut.get("threshold"),
    }
    readiness = {
        "verdict": gate.get("verdict"),
        "blocks": gate.get("num_blocks") or gate.get("blocks"),
        "warnings": gate.get("num_warnings") or gate.get("warnings"),
    }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": _verdict(fast, guard),
        "claim_boundary": (
            "This is a fail-fast direction-control report. It may decide what to screen, skip, "
            "or promote, but publication claims still require full protocol, matched baselines, "
            "LODO, calibration, and bootstrap evidence."
        ),
        "current_gpu_guard": {
            "verdict": guard.get("verdict"),
            "manual_launch_allowed": guard.get("manual_launch_allowed"),
        },
        "readiness": readiness,
        "p0_summary": {
            "physics_vs_global": _metrics_delta(physics, global_only),
            "full_vs_physics": _metrics_delta(full, physics),
            "dg_vs_friction_set": _metrics_delta(dg, friction_set),
            "friction_set_vs_physics": _metrics_delta(friction_set, physics),
        },
        "lodo_summary": {
            "verdict": lodo.get("verdict"),
            "rows": lodo_rows,
            "heldout_roadsaw": roadsaw,
        },
        "shortcut_summary": shortcut_counts,
        "kill_or_downgrade": kill_rows,
        "protect_or_conditional_keep": keep_rows,
        "candidate_priority": candidate_rows,
        "formal_policy": formal_policy,
        "fast_screen_counts": fast.get("counts", {}),
        "strict_rules": _strict_rules(),
        "immediate_next_actions": _next_actions(fast, guard, formal_policy),
    }


def _module_decision(
    item: str,
    decision: str,
    evidence: list[str],
    action: str,
) -> dict[str, Any]:
    return {
        "item": item,
        "decision": decision,
        "evidence": [line for line in evidence if line],
        "action": action,
    }


def _candidate_row(source: str, row: dict[str, Any] | None, *, lodo: dict[str, Any], shortcut: dict[str, Any]) -> dict[str, Any]:
    role = _candidate_role(source)
    status = (row or {}).get("status", "missing")
    screen_score = (row or {}).get("screen_score")
    decision = "fast_screen_required"
    reason = "No completed fast-screen row yet."
    if status == "complete":
        decision, reason = _completed_fast_decision(row or {})
    elif source in FULL_STACK_CANDIDATES:
        decision = "screen_only_before_formal"
        reason = "This candidate inherits the unstable full stack; do not spend a full formal run before proxy evidence."
    elif source in LEAN_CANDIDATES:
        decision = "priority_fast_screen"
        reason = "Lean candidate matches current P0 evidence and can test shortcut/ROI/quality hypotheses quickly."
    return {
        "source_run": source,
        "role": role,
        "status": status,
        "decision": decision,
        "reason": reason,
        "screen_score": screen_score,
        "metrics": {
            "risk_f1": (row or {}).get("risk_f1"),
            "friction_f1": (row or {}).get("friction_f1"),
            "low_friction_recall": (row or {}).get("low_friction_recall"),
            "calibrated_coverage": (row or {}).get("calibrated_coverage"),
            "calibrated_width": (row or {}).get("calibrated_width"),
            "worst_dataset_f1": (row or {}).get("worst_dataset_f1"),
            "dataset_id_bal_acc": (row or {}).get("dataset_id_bal_acc"),
        },
    }


def _completed_fast_decision(row: dict[str, Any]) -> tuple[str, str]:
    risk = _num(row.get("risk_f1"))
    low = _num(row.get("low_friction_recall"))
    cov = _num(row.get("calibrated_coverage"))
    width = _num(row.get("calibrated_width"))
    worst = _num(row.get("worst_dataset_f1"))
    shortcut = _num(row.get("dataset_id_bal_acc"), default=0.85)
    if cov < 0.88:
        return "kill_or_rework", "Screen calibrated coverage is below 88%."
    if width > 0.68:
        return "kill_or_rework", "Screen interval width is too broad for a useful interval claim."
    if risk < 0.86 or low < 0.85 or worst < 0.50:
        return "kill_or_rework", "Screen safety/generalization metric is below the current minimum bar."
    if shortcut > 0.95 and row.get("source_run") in FULL_STACK_CANDIDATES:
        return "hold", "Metrics may be usable, but dataset shortcut remains high for a full-stack route."
    return "promote_candidate", "Screen metrics pass the minimum fail-fast bar."


def _candidate_role(source: str) -> str:
    if source == "v14_lean_road_roi_safety":
        return "main lean ROI-safety candidate"
    if source == "v17_lean_quality_physics_safety":
        return "wet/near-white quality-aware candidate"
    if source == "v18_lean_mixstyle_quality_safety":
        return "feature-statistics style-mixing shortcut probe"
    if source == "v19_lean_state_contrast_quality_safety":
        return "cross-dataset same-state contrastive alignment probe"
    if source == "v20_lean_interval_order_quality_safety":
        return "weak friction-interval order consistency probe"
    if source == "v21_lean_quality_uncertainty_safety":
        return "visual-quality uncertainty interval-safety probe"
    if source == "v22_lean_quality_order_contrast_safety":
        return "ambiguity-aware interval-order plus state-contrast probe"
    if source == "v23_lean_region_mixture_evidence_safety":
        return "segmentation-style region-mixture evidence probe"
    if source == "v24_lean_multi_query_region_evidence_safety":
        return "mask-query-style multi-region friction evidence probe"
    if source == "v25_lean_masked_query_consistency_safety":
        return "MIC-style masked multi-query evidence consistency probe"
    if source == "v16_lean_bottom_square_color_constancy_safety":
        return "input canonicalization plus color constancy"
    if source == "v15_lean_bottom_square_style_safety":
        return "bottom-square input canonicalization"
    if source == "v13_lean_physics_evidence":
        return "lean pruning sanity check"
    if source == "v9_full_faf_roadsaw_hard_sampling":
        return "RoadSaW wetness hard-sampling rescue"
    if source == "v12_full_faf_roi_interval_safety":
        return "interval coverage-width rescue"
    if source == "v6_full_faf_fourier":
        return "style-shortcut probe"
    if source == "v7_full_faf_fourier_dann":
        return "adversarial shortcut probe"
    if source == "v11_full_faf_domain_adapter":
        return "style adapter probe"
    if source == "v10_full_faf_consistency":
        return "prediction/evidence consistency probe"
    if source == "v8_full_faf_fourier_roadprior":
        return "road-prior evidence probe"
    return "candidate"


def _formal_policy(
    completed_fast: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    fast: dict[str, Any],
    promotion: dict[str, Any],
    python: Path,
    root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    counts = fast.get("counts") or {}
    anchor_done = any(str(row.get("source_run") or "") == FAST_SCREEN_ANCHOR for row in completed_fast)
    promoted_by_relative_gate = [
        str(row.get("source_run") or "")
        for row in promotion.get("promoted", [])
        if isinstance(row, dict) and str(row.get("source_run") or "")
    ]
    promoted_by_absolute_gate = [
        row["source_run"]
        for row in candidate_rows
        if row.get("decision") == "promote_candidate"
    ]
    promoted = promoted_by_relative_gate or promoted_by_absolute_gate
    if not counts.get("complete", 0) or not anchor_done:
        promoted = []
    promoted = promoted[:2]
    held_until_screen = [
        row["source_run"]
        for row in candidate_rows
        if row.get("decision") in {"screen_only_before_formal", "fast_screen_required", "priority_fast_screen"}
        and row["source_run"] not in set(promoted)
    ]
    verdict = "screen_required_no_formal_candidates"
    if counts.get("complete", 0) > 0 and not anchor_done:
        verdict = "screen_anchor_required"
    elif counts.get("complete", 0) > 0 and promoted:
        verdict = "promote_only_screen_passers"
    elif counts.get("complete", 0) > 0:
        verdict = "redesign_no_formal_candidate"
    return {
        "verdict": verdict,
        "fast_screen_anchor": FAST_SCREEN_ANCHOR,
        "promoted_or_fallback": promoted,
        "fallback_disabled": True,
        "promotion_report_verdict": promotion.get("verdict"),
        "fast_screen_first_wave": LEAN_FIRST_WAVE,
        "held_until_screen": held_until_screen,
        "full_stack_held_until_screen": FULL_STACK_CANDIDATES,
        "fast_screen_command": (
            f"{python} scripts\\run_fast_screen_protocol.py --scope candidates "
            "--lean-first-wave "
            "--bootstrap-samples 100 --dataset-diagnostic-samples 2000 "
            "--evidence-map-samples 12 --evidence-audit-samples 1000"
        ),
        "formal_promoted_command": (
            f"{python} scripts\\run_paper_protocol_direct.py --phase candidates "
            f"--only {' '.join(promoted)} --python {python} --root {root} "
            f"--log-dir {log_dir} --postprocess-each"
        )
        if promoted
        else None,
        "resume_without_full_stack_command": (
            f"{python} scripts\\run_paper_protocol_direct.py --phase candidates "
            f"--exclude {' '.join(FULL_STACK_CANDIDATES)} --python {python} --root {root} "
            f"--log-dir {log_dir} --postprocess-each"
        ),
    }


def _strict_rules() -> list[str]:
    return [
        "Kill a module in its current form if it drops risk F1 or low-friction recall by at least 2 points without a compensating worst-domain or interval gain.",
        "Kill or downgrade a route if calibrated coverage is below 88% or calibrated width grows by more than 0.04-0.05 without solving a safety-critical cell.",
        "Do not run a full formal full-stack candidate before fast-screen evidence, because P0 shows full fusion and current DG are unstable.",
        "Do not run any formal candidate before the fast-screen anchor and candidate rows exist.",
        "Promote at most one or two candidates from fast screen to expensive formal runs, and promote none if every candidate is only equal or worse than the anchor.",
        "Treat dataset-ID balanced accuracy above 85% as shortcut risk; a high-score candidate must either lower shortcut or improve held-out/worst-domain safety enough to justify it.",
        "Matched ConvNeXt baselines and final lean rows remain mandatory before any superiority or top-venue claim.",
    ]


def _next_actions(fast: dict[str, Any], guard: dict[str, Any], formal_policy: dict[str, Any]) -> list[str]:
    busy = str(guard.get("verdict", "")).lower() == "busy" or guard.get("manual_launch_allowed") is False
    counts = fast.get("counts") or {}
    actions: list[str] = []
    if busy:
        actions.append("Do not launch a new GPU job now; the scheduling guard is busy.")
    if counts.get("complete", 0) == 0:
        actions.append("When the GPU guard is idle, run the candidate fast-screen queue before spending full formal runs on v6-v25.")
    actions.append("Keep fair single-dataset FAF and matched ConvNeXt baselines; they are not optional because they control reviewer fairness.")
    actions.append("Use the fail-fast report to exclude full-stack formal candidates unless their fast-screen row passes the promotion gate.")
    actions.append("If no screen row passes, do not spend formal training on fallback candidates; freeze that branch and design the next candidate from the observed failure mode.")
    if formal_policy.get("formal_promoted_command"):
        actions.append("After fast-screen completion, run only promoted candidates with the formal command in this report.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fail-Fast Exploration Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Current Guard",
        "",
        f"- GPU guard: `{report['current_gpu_guard'].get('verdict')}`.",
        f"- Manual launch allowed: `{report['current_gpu_guard'].get('manual_launch_allowed')}`.",
        f"- Top-venue readiness: `{report['readiness'].get('verdict')}`.",
        "",
        "## Kill Or Downgrade Now",
        "",
        "| Item | Decision | Evidence | Action |",
        "|---|---|---|---|",
    ]
    for row in report["kill_or_downgrade"]:
        lines.append(
            f"| {row['item']} | `{row['decision']}` | {_join_evidence(row['evidence'])} | {row['action']} |"
        )
    lines.extend(["", "## Protect Or Conditional Keep", "", "| Item | Decision | Evidence | Action |", "|---|---|---|---|"])
    for row in report["protect_or_conditional_keep"]:
        lines.append(
            f"| {row['item']} | `{row['decision']}` | {_join_evidence(row['evidence'])} | {row['action']} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Fast-Screen Priority",
            "",
            "| Priority | Candidate | Role | Status | Decision | Reason |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(report["candidate_priority"], start=1):
        lines.append(
            f"| {idx} | `{row['source_run']}` | {row['role']} | `{row['status']}` | `{row['decision']}` | {row['reason']} |"
        )

    policy = report["formal_policy"]
    lines.extend(
        [
            "",
            "## Execution Policy",
            "",
            f"- Policy verdict: `{policy.get('verdict')}`.",
            f"- Selected formal candidates: {_code_list(policy.get('promoted_or_fallback'))}.",
            f"- Fast-screen first wave: {_code_list(policy.get('fast_screen_first_wave'))}.",
            f"- Held until screen evidence: {_code_list(policy.get('held_until_screen'))}.",
            f"- Full-stack routes held until screen evidence: {_code_list(policy.get('full_stack_held_until_screen'))}.",
            f"- Formal fallback disabled: `{policy.get('fallback_disabled')}`.",
            "",
            "Fast-screen command, to run only when the GPU guard is idle:",
            "",
            f"`{policy.get('fast_screen_command')}`",
            "",
        ]
    )
    if policy.get("formal_promoted_command"):
        lines.extend(["Formal promoted-candidate command after screen evidence:", "", f"`{policy['formal_promoted_command']}`", ""])
    lines.extend(
        [
            "Resume-candidates command if we intentionally skip full-stack candidates:",
            "",
            f"`{policy.get('resume_without_full_stack_command')}`",
            "",
            "## Strict Rules",
            "",
        ]
    )
    for item in report["strict_rules"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Immediate Next Actions", ""])
    for item in report["immediate_next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _delta_reason(
    label: str,
    current: dict[str, Any],
    reference: dict[str, Any],
    key: str,
    *,
    bad_below: float | None = None,
    bad_above: float | None = None,
) -> str:
    if not current or not reference:
        return ""
    cur = _num(current.get(key))
    ref = _num(reference.get(key))
    delta = cur - ref
    flag = ""
    if bad_below is not None and delta <= bad_below:
        flag = " FAIL"
    if bad_above is not None and delta >= bad_above:
        flag = " FAIL"
    return f"{label}: {100.0 * delta:+.2f} pp{flag}" if "width" not in label.lower() else f"{label}: {delta:+.4f}{flag}"


def _metrics_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    keys = [
        "friction_macro_f1",
        "risk_macro_f1",
        "low_friction_recall",
        "calibrated_coverage",
        "calibrated_width",
        "worst_dataset_f1",
        "dataset_id_balanced_accuracy",
    ]
    if not current or not reference:
        return {key: None for key in keys}
    return {key: _num(current.get(key)) - _num(reference.get(key)) for key in keys}


def _verdict(fast: dict[str, Any], guard: dict[str, Any]) -> str:
    counts = fast.get("counts") or {}
    if guard.get("manual_launch_allowed") is False:
        return "gpu_busy_screen_when_idle"
    if counts.get("complete", 0) == 0:
        return "run_fast_screen_first"
    return "use_screen_to_prune_formal_runs"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _join_evidence(items: list[str]) -> str:
    return "<br>".join(items) if items else "-"


def _code_list(items: Any) -> str:
    if not items:
        return "-"
    return ", ".join(f"`{item}`" for item in items)


if __name__ == "__main__":
    main()
