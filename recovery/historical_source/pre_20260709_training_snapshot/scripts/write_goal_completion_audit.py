from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUMMARY = Path("reports/paper_protocol_summary")
ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT_JSON = SUMMARY / "goal_completion_audit.json"
OUT_MD = SUMMARY / "goal_completion_audit.md"


def main() -> None:
    checks = [
        _check_literature(),
        _check_dataset_integrity(),
        _check_friction_interval_sources(),
        _check_rscd_formal(),
        _check_candidate_pipeline(),
        _check_direct_visual_friction(),
        _check_claim_readiness(),
    ]
    report = {
        "claim_boundary": (
            "Audit against the active project goal. This file intentionally distinguishes "
            "completed evidence from pending or insufficient evidence."
        ),
        "overall_status": _overall_status(checks),
        "checks": checks,
        "blocking_evidence": [c for c in checks if c["status"] in {"missing", "incomplete", "insufficient"}],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(OUT_MD)


def _exists(path: str | Path) -> bool:
    return Path(path).exists()


def _load_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _check_literature() -> dict[str, Any]:
    path = SUMMARY / "rscd_paper_deep_reading_sota_table.md"
    strict = SUMMARY / "rscd_strict_literature_audit_20260626.md"
    ok = _exists(path) and _exists(strict)
    return {
        "requirement": "Read RSCD-related papers, metrics, and current SOTA.",
        "status": "complete" if ok else "missing",
        "evidence": [str(p) for p in [path, strict] if _exists(p)],
        "note": (
            "Includes RSCD task/metrics, RoadFormer/RoadMamba SOTA table, strict RoadFormer-L target, and source clarification. "
            "Still treat external SOTA as contextual unless protocol-matched."
            if ok
            else "Missing RSCD SOTA report or strict literature audit."
        ),
    }


def _check_dataset_integrity() -> dict[str, Any]:
    path = SUMMARY / "dataset_integrity_view_audit.md"
    route = SUMMARY / "dataset_route_decision_rscd_vs_roadsaw_roadsc_20260625.md"
    ok = _exists(path) and _exists(route)
    return {
        "requirement": "Confirm dataset completeness and decide RSCD vs RoadSaW/RoadSC route.",
        "status": "complete" if ok else "incomplete",
        "evidence": [str(p) for p in [path, route] if _exists(p)],
        "note": (
            "Local audit supports RSCD as primary benchmark and RoadSaW/RoadSC as stress/OOD datasets."
            if ok
            else "Need both integrity audit and route decision report."
        ),
    }


def _check_friction_interval_sources() -> dict[str, Any]:
    path = SUMMARY / "friction_interval_source_audit.md"
    alt = SUMMARY / "friction_interval_source_audit" / "friction_interval_source_audit.md"
    direct = SUMMARY / "direct_visual_friction_benchmark_feasibility.md"
    evidence = [str(p) for p in [path, alt, direct] if _exists(p)]
    ok = bool(evidence)
    return {
        "requirement": "Ground weak friction intervals in public references and direct-friction literature.",
        "status": "complete" if ok else "missing",
        "evidence": evidence,
        "note": (
            "Weak visual friction-affordance intervals are documented; direct measured-friction papers are method references, not matched numeric baselines."
            if ok
            else "Need interval-source audit and direct-friction comparability report."
        ),
    }


def _check_rscd_formal() -> dict[str, Any]:
    formal_files = sorted(ROOT.glob("formal_*/evaluate_test.json")) if ROOT.exists() else []
    trend = _load_json(SUMMARY / "rscd_training_trend_report.json") or {}
    pretraining_audit = SUMMARY / "rscd_pretraining_protocol_audit.md"
    formal_slice = SUMMARY / "rscd_formal_class_slice_comparison.md"
    hard_class = SUMMARY / "rscd_formal_hard_class_diagnosis.md"
    sota_gap = SUMMARY / "rscd_sota_gap_diagnosis.md"
    live_sota = SUMMARY / "rscd_live_sota_audit_20260626.md"
    status = "complete" if formal_files else "incomplete"
    note = "Formal test files are available." if formal_files else "Formal jobs are still running; final test evidence is missing."
    return {
        "requirement": "Run matched RSCD formal experiments and produce final test results.",
        "status": status,
        "evidence": (
            [str(p) for p in formal_files]
            + ([str(SUMMARY / "rscd_training_trend_report.md")] if trend else [])
            + ([str(pretraining_audit)] if _exists(pretraining_audit) else [])
            + ([str(formal_slice)] if _exists(formal_slice) else [])
            + ([str(hard_class)] if _exists(hard_class) else [])
            + ([str(sota_gap)] if _exists(sota_gap) else [])
            + ([str(live_sota)] if _exists(live_sota) else [])
        ),
        "note": note,
    }


def _check_candidate_pipeline() -> dict[str, Any]:
    paths = [
        SUMMARY / "rscd_fast_promotion_decision.md",
        SUMMARY / "rscd_hard_condition_promotion_decision.md",
        SUMMARY / "rscd_residual_adapter_promotion_decision.md",
        SUMMARY / "rscd_texture_film_promotion_decision.md",
        SUMMARY / "rscd_wavelet_formal_warning.md",
        SUMMARY / "rscd_formal_validation_diagnosis.md",
        SUMMARY / "rscd_decision_dashboard.md",
        SUMMARY / "rscd_final_method_selection.md",
        SUMMARY / "experiment_queue_health_report.md",
        SUMMARY / "human_vision_physics_attention_plan.md",
        SUMMARY / "foundation_physics_texture_route_20260626.md",
        SUMMARY / "rscd_patch_geometry_quality_cue_audit_20260626.md",
        SUMMARY / "rscd_patch_region_cue_statistics.md",
        SUMMARY / "rscd_patch_quality_region_decision.md",
        Path("scripts/run_high_priority_texture_candidates_after_formal.ps1"),
        Path("scripts/run_high_priority_formal_promotion_after_fast.ps1"),
        Path("scripts/run_physics_attention_rscd_fast_after_priority.ps1"),
        Path("scripts/run_physics_attention_formal_promotion_after_fast.ps1"),
        Path("scripts/run_texture_film_formal_promotion_after_fast.ps1"),
        Path("scripts/run_wavelet_texture_rscd_fast_after_queue.ps1"),
        Path("scripts/run_foundation_rscd_fast_after_queue.ps1"),
    ]
    evidence = [str(p) for p in paths if _exists(p)]
    fast_outputs = sorted(ROOT.glob("fast_*/*evaluate_test.json")) if ROOT.exists() else []
    status = "incomplete"
    if evidence and fast_outputs:
        status = "partial"
    promoted_patterns = [
        "formal_*residual*/evaluate_test.json",
        "formal_*directional*/evaluate_test.json",
        "formal_*hard_condition*/evaluate_test.json",
        "formal_*film*/evaluate_test.json",
        "formal_*wavelet*/evaluate_test.json",
    ]
    if any(any(ROOT.glob(pattern)) for pattern in promoted_patterns):
        status = "complete"
    return {
        "requirement": "Explore innovative candidates, fast-screen, promote only if useful, and prune weak modules.",
        "status": status,
        "evidence": evidence + [str(p) for p in fast_outputs],
        "note": (
            "Candidate gates exist, but queued candidates and formal promoted rows are still pending."
            if status != "complete"
            else "Candidate promotion produced formal test evidence."
        ),
    }


def _check_direct_visual_friction() -> dict[str, Any]:
    report = SUMMARY / "direct_visual_friction_benchmark_feasibility.md"
    final = SUMMARY / "direct_visual_friction_report.md"
    watcher = Path("scripts/run_direct_visual_friction_after_rscd_queue.ps1")
    evidence = [str(p) for p in [report, final, watcher] if _exists(p)]
    complete_report = _load_json(SUMMARY / "direct_visual_friction_report.json")
    direct_status = complete_report.get("status") if complete_report else None
    if isinstance(direct_status, dict):
        is_complete = direct_status.get("status") == "complete"
    else:
        is_complete = direct_status == "complete"
    status = "complete" if is_complete else "incomplete"
    return {
        "requirement": "Create a separate direct visual friction-affordance route if measured-friction comparison is not matched.",
        "status": status,
        "evidence": evidence,
        "note": (
            "Direct route is queued and methodologically separated, but final direct-route results are pending."
            if status != "complete"
            else "Direct route results are complete."
        ),
    }


def _check_claim_readiness() -> dict[str, Any]:
    formal = _load_json(SUMMARY / "rscd_formal_result_summary.json") or {}
    dashboard = _load_json(SUMMARY / "rscd_decision_dashboard.json") or {}
    formal_ready = bool(formal.get("local_rows"))
    decision = (dashboard.get("current_decision") or {}).get("status")
    ready = formal_ready and decision == "formal_results_available"
    return {
        "requirement": "Have enough evidence for a final top-tier claim and module-pruning decision.",
        "status": "complete" if ready else "insufficient",
        "evidence": [str(p) for p in [SUMMARY / "rscd_formal_result_summary.md", SUMMARY / "rscd_decision_dashboard.md"] if _exists(p)],
        "note": (
            "Final claim can be drafted from formal evidence."
            if ready
            else "Do not mark goal complete: final formal test, promoted-candidate outcomes, and pruning decisions are not fully proven."
        ),
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    if all(c["status"] == "complete" for c in checks):
        return "complete"
    if any(c["status"] in {"missing", "insufficient"} for c in checks):
        return "incomplete"
    return "partial"


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Goal Completion Audit",
        "",
        report["claim_boundary"],
        "",
        f"Overall status: `{report['overall_status']}`",
        "",
        "| requirement | status | note |",
        "|---|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['requirement']} | `{check['status']}` | {check['note']} |")
    lines += ["", "## Evidence", ""]
    for check in report["checks"]:
        lines.append(f"### {check['requirement']}")
        if check["evidence"]:
            lines.extend(f"- `{item}`" for item in check["evidence"])
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
