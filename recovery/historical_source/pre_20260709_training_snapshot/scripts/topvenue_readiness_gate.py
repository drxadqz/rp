from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "topvenue_readiness_gate.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "topvenue_readiness_gate.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    config_audit = _load_json(summary_dir / "protocol_config_audit.json") or {}
    roadsaw_lodo_protocol = _load_json(summary_dir / "roadsaw_lodo_protocol_audit.json") or {}
    fair_comparison_protocol = _load_json(summary_dir / "fair_comparison_protocol_audit.json") or {}
    gpu_protocol_audit = _load_json(summary_dir / "gpu_protocol_audit.json") or {}
    dataset_inventory = _load_json(summary_dir / "dataset_inventory_report.json") or {}
    dataset_view_source = _load_text(summary_dir / "dataset_view_source_evidence_report.md")
    interval_source_audit = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    interval_claim_matrix = _load_json(summary_dir / "friction_interval_claim_matrix.json") or {}
    interval_quality = _load_json(summary_dir / "interval_quality_report.json") or {}
    wetness_state = _load_json(summary_dir / "wetness_state_report.json") or {}
    dataset_shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    external_benchmark = _load_json(summary_dir / "external_benchmark_report.json") or {}
    direct_friction_public_benchmark = (
        _load_json(summary_dir / "direct_friction_public_benchmark_audit.json") or {}
    )
    rscd_external_comparison = _load_json(summary_dir / "rscd_external_comparison_readiness.json") or {}
    open_source_plan = _load_json(summary_dir / "open_source_reproducibility_plan.json") or {}
    innovation_roadmap = _load_json(summary_dir / "topvenue_innovation_roadmap.json") or {}
    cv_transfer_decision = _load_json(summary_dir / "cv_transfer_decision_report.json") or {}
    pseudo_segmentation_mask_audit = (
        _load_json(summary_dir / "pseudo_segmentation_masks" / "pseudo_segmentation_mask_audit.json")
        or {}
    )
    external_segmentation_mask_audit = (
        _load_json(summary_dir / "external_segmentation_masks" / "external_segmentation_mask_audit.json")
        or {}
    )
    external_road_mask_cache = _latest_json(summary_dir / "external_road_mask_cache") or _latest_json(
        summary_dir / "external_road_mask_cache_smoke"
    ) or {}
    segmentation_transfer_config = _load_json(summary_dir / "segmentation_transfer_config_audit.json") or {}
    segmentation_transfer_queue = _load_json(summary_dir / "segmentation_transfer_queue_status.json") or {}
    next_queue_readiness = _load_json(summary_dir / "next_queue_readiness_report.json") or {}
    fair_execution_priority = _load_json(summary_dir / "fair_comparison_execution_priority.json") or {}
    cv_transfer_experiment_protocol = _load_json(summary_dir / "cv_transfer_experiment_protocol.json") or {}
    wet_slippery_mechanism = _load_json(summary_dir / "wet_slippery_failure_mechanism_report.json") or {}
    cv_transfer_candidate_priority = _load_json(summary_dir / "cv_transfer_candidate_priority_report.json") or {}
    cv_transfer_retention_decision = _load_json(summary_dir / "cv_transfer_retention_decision_report.json") or {}
    final_freeze_audit = _load_json(summary_dir / "final_freeze_audit.json") or {}
    candidate_hypothesis = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    candidate_coverage = _load_json(summary_dir / "candidate_implementation_coverage_audit.json") or {}
    candidate_pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    online_source_refresh = _load_json(summary_dir / "online_source_refresh_report.json") or {}
    latest_visual_friction_web_check = _load_text(
        summary_dir / "latest_visual_friction_web_check_20260624.md"
    )
    p0_claim = _load_json(summary_dir / "p0_claim_report.json") or {}
    safety_selection = _load_json(summary_dir / "safety_selection_report.json") or {}
    checkpoint_policy = _load_json(summary_dir / "checkpoint_policy_report.json") or {}
    evidence_failure = _load_json(summary_dir / "evidence_failure_report.json") or {}
    algorithm_audit = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    config_to_code_trace = _load_json(summary_dir / "config_to_code_trace_report.json") or {}
    mask_aware_consistency = _load_json(summary_dir / "mask_aware_consistency_smoke.json") or {}
    wet_optical_quality = _load_json(summary_dir / "wet_optical_quality_cues_smoke.json") or {}
    goal_evidence = _load_json(summary_dir / "goal_evidence_audit.json") or {}
    objective_completion = _load_json(summary_dir / "objective_completion_audit.json") or {}
    artifact_contract = _load_json(summary_dir / "artifact_contract_report.json") or {}
    lodo_report = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    claim_ledger = _load_json(summary_dir / "claim_evidence_ledger.json") or {}
    quality_mondrian = _load_json(summary_dir / "quality_mondrian_summary.json") or {}
    asymmetric_mondrian = _load_json(summary_dir / "asymmetric_mondrian_summary.json") or {}
    region_mixture = _load_json(summary_dir / "region_mixture_summary.json") or {}
    checkpoint_divergence = _load_json(summary_dir / "checkpoint_divergence_report.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    module_decisions = _load_module_decisions(summary_dir / "module_decisions.csv")

    gates: list[dict[str, Any]] = []
    _gate_protocol_config(gates, config_audit)
    _gate_roadsaw_lodo_protocol(gates, roadsaw_lodo_protocol)
    _gate_fair_comparison_protocol(gates, fair_comparison_protocol)
    _gate_gpu_protocol(gates, gpu_protocol_audit)
    _gate_dataset_inventory(gates, dataset_inventory)
    _gate_dataset_view_source_evidence(gates, dataset_view_source)
    _gate_friction_interval_sources(gates, interval_source_audit)
    _gate_friction_interval_claim_matrix(gates, interval_claim_matrix)
    _gate_external_benchmark_report(gates, external_benchmark)
    _gate_direct_friction_public_benchmark(gates, direct_friction_public_benchmark)
    _gate_rscd_external_comparison(gates, rscd_external_comparison)
    _gate_open_source_reproducibility_plan(gates, open_source_plan)
    _gate_topvenue_innovation_roadmap(gates, innovation_roadmap)
    _gate_cv_transfer_decision_report(gates, cv_transfer_decision)
    _gate_pseudo_segmentation_mask_audit(gates, pseudo_segmentation_mask_audit)
    _gate_external_segmentation_mask_audit(gates, external_segmentation_mask_audit)
    _gate_external_road_mask_cache(gates, external_road_mask_cache)
    _gate_segmentation_transfer_config(gates, segmentation_transfer_config)
    _gate_segmentation_transfer_queue(gates, segmentation_transfer_queue)
    _gate_next_queue_readiness(gates, next_queue_readiness)
    _gate_fair_execution_priority(gates, fair_execution_priority)
    _gate_cv_transfer_experiment_protocol(gates, cv_transfer_experiment_protocol)
    _gate_wet_slippery_failure_mechanism(gates, wet_slippery_mechanism)
    _gate_cv_transfer_candidate_priority(gates, cv_transfer_candidate_priority)
    _gate_cv_transfer_retention_decision(gates, cv_transfer_retention_decision)
    _gate_final_freeze_audit(gates, final_freeze_audit)
    _gate_candidate_hypothesis_matrix(gates, candidate_hypothesis)
    _gate_candidate_implementation_coverage(gates, candidate_coverage)
    _gate_candidate_pruning_report(gates, candidate_pruning)
    _gate_online_source_refresh(gates, online_source_refresh)
    _gate_latest_visual_friction_web_check(gates, latest_visual_friction_web_check)
    _gate_p0_claim_report(gates, p0_claim)
    _gate_completeness(gates, completeness, artifact_contract)
    _gate_core_ablation(gates, summary)
    _gate_lodo(gates, summary, lodo_report)
    _gate_fair_single_dataset(gates, summary)
    _gate_final_method(gates, summary)
    _gate_shortcut_and_intervals(gates, summary)
    _gate_dataset_shortcut_report(gates, dataset_shortcut)
    _gate_conditional_interval_report(gates, interval_quality)
    _gate_wetness_state_report(gates, wetness_state)
    _gate_safety_selection(gates, safety_selection)
    _gate_checkpoint_policy(gates, checkpoint_policy)
    _gate_evidence(gates, summary)
    _gate_evidence_failure_report(gates, evidence_failure)
    _gate_config_to_code_trace(gates, config_to_code_trace)
    _gate_mask_aware_consistency_smoke(gates, mask_aware_consistency)
    _gate_wet_optical_quality_cues_smoke(gates, wet_optical_quality)
    _gate_goal_evidence_audit(gates, goal_evidence)
    _gate_objective_completion_audit(gates, objective_completion)
    _gate_artifact_contract(gates, artifact_contract)
    _gate_claim_evidence_ledger(gates, claim_ledger)
    _gate_quality_mondrian_summary(gates, quality_mondrian)
    _gate_asymmetric_mondrian_summary(gates, asymmetric_mondrian)
    _gate_region_mixture_summary(gates, region_mixture)
    _gate_checkpoint_divergence(gates, checkpoint_divergence)
    _gate_p1_style_augmentation_candidates(gates, algorithm_audit)
    _gate_p3_interval_candidates(gates, algorithm_audit)
    _gate_p1_wetness_ordinal_candidates(gates, algorithm_audit)
    _gate_p1_conditional_alignment_candidates(gates, algorithm_audit)
    _gate_final_method_selection_report(gates, final_selection)
    _gate_module_retention_report(gates, module_retention)
    _gate_module_decisions(gates, module_decisions)

    blocks = [gate for gate in gates if gate["level"] == "block"]
    warns = [gate for gate in gates if gate["level"] == "warn"]
    passes = [gate for gate in gates if gate["level"] == "pass"]
    if blocks:
        verdict = "not_ready"
    elif warns:
        verdict = "promising_but_needs_caution"
    else:
        verdict = "ready_for_strict_paper_claims"
    return {
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "num_blocks": len(blocks),
        "num_warnings": len(warns),
        "num_pass": len(passes),
        "gates": gates,
        "recommended_next_actions": _recommended_next_actions(gates, summary),
    }


def _gate_protocol_config(gates: list[dict[str, Any]], config_audit: dict[str, Any]) -> None:
    verdict = config_audit.get("verdict")
    if verdict == "pass":
        _add(gates, "pass", "protocol_config_audit", "Protocol configs pass split, LODO, and fair-baseline checks.")
    elif verdict:
        _add(gates, "block", "protocol_config_audit", f"Protocol config audit is `{verdict}`.")
    else:
        _add(gates, "block", "protocol_config_audit_missing", "Missing protocol_config_audit.json.")


def _gate_roadsaw_lodo_protocol(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    verdict = report.get("verdict")
    splits = report.get("splits", {}) if isinstance(report.get("splits"), dict) else {}
    train = splits.get("train", {}) if isinstance(splits.get("train"), dict) else {}
    val = splits.get("val", {}) if isinstance(splits.get("val"), dict) else {}
    test = splits.get("test", {}) if isinstance(splits.get("test"), dict) else {}
    details = {
        "train_datasets": train.get("datasets"),
        "val_datasets": val.get("datasets"),
        "test_datasets": test.get("datasets"),
        "train_rows": train.get("num_rows"),
        "val_rows": val.get("num_rows"),
        "test_rows": test.get("num_rows"),
    }
    if verdict == "pass":
        _add(
            gates,
            "pass",
            "roadsaw_lodo_protocol_audit",
            "Held-out RoadSaW LODO protocol excludes RoadSaW from train/validation and tests only on RoadSaW.",
            **details,
        )
    elif verdict == "pass_with_warnings":
        _add(
            gates,
            "warn",
            "roadsaw_lodo_protocol_audit",
            "Held-out RoadSaW LODO protocol passes leakage checks but has warnings to inspect.",
            warnings=[
                item.get("name")
                for item in report.get("checks", [])
                if item.get("level") == "warn"
            ],
            **details,
        )
    elif verdict:
        _add(
            gates,
            "block",
            "roadsaw_lodo_protocol_audit",
            f"Held-out RoadSaW LODO protocol audit is `{verdict}`.",
            failures=[
                item.get("name")
                for item in report.get("checks", [])
                if item.get("level") == "block"
            ],
            **details,
        )
    else:
        _add(
            gates,
            "block",
            "roadsaw_lodo_protocol_audit_missing",
            "Missing roadsaw_lodo_protocol_audit.json; held-out RoadSaW protocol is not independently audited.",
        )


def _gate_fair_comparison_protocol(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    verdict = report.get("verdict")
    details = {
        "pairs": report.get("num_pairs"),
        "blocks": report.get("num_blocks"),
        "warnings": report.get("num_warnings"),
        "strict_single_pairs_pass": sum(
            1
            for row in report.get("rows", [])
            if row.get("scope") == "single_dataset_full_faf_vs_convnext"
            and row.get("status") == "pass"
        )
        if isinstance(report.get("rows"), list)
        else None,
    }
    if verdict in {"pass", "pass_with_warnings"} and int(report.get("num_blocks", 0) or 0) == 0:
        _add(
            gates,
            "pass",
            "fair_comparison_protocol_audit",
            "FAF and matched ConvNeXt single-dataset comparisons share split, labels, backbone, training budget, effective batch, and evaluation protocol.",
            **details,
        )
    elif verdict:
        failures = [
            item.get("name")
            for item in report.get("checks", [])
            if item.get("level") == "block"
        ]
        _add(
            gates,
            "block",
            "fair_comparison_protocol_audit",
            f"Fair comparison protocol audit is `{verdict}`; matched baseline claims are not yet defensible.",
            failures=failures[:8],
            **details,
        )
    else:
        _add(
            gates,
            "block",
            "fair_comparison_protocol_audit_missing",
            "Missing fair_comparison_protocol_audit.json; matched FAF vs ConvNeXt claims are not independently audited.",
        )


def _gate_gpu_protocol(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    verdict = report.get("verdict")
    failures = report.get("failures", [])
    warnings = report.get("warnings", [])
    torch_info = report.get("torch") or {}
    details = {
        "configs": report.get("num_configs"),
        "cuda_available": torch_info.get("cuda_available"),
        "device": torch_info.get("device_name"),
        "queue_python_exists": report.get("queue_python_exists"),
        "failures": len(failures),
        "warnings": len(warnings),
    }
    if verdict == "pass":
        _add(
            gates,
            "pass",
            "gpu_protocol_audit",
            "Conda/CUDA/GPU protocol audit passes for all queued paper configs.",
            **details,
        )
    elif verdict:
        _add(
            gates,
            "block",
            "gpu_protocol_audit",
            f"GPU protocol audit is `{verdict}`; fix runtime/config failures before treating results as final.",
            **details,
        )
    else:
        _add(
            gates,
            "block",
            "gpu_protocol_audit_missing",
            "Missing gpu_protocol_audit.json.",
        )


def _gate_dataset_inventory(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "block",
            "dataset_inventory_missing",
            "Missing dataset_inventory_report.json; local data availability and manifest row counts are not audited.",
        )
        return
    checks = report.get("checks", []) if isinstance(report.get("checks"), list) else []
    blocks = [item.get("name") for item in checks if item.get("level") == "block"]
    warnings = [item.get("name") for item in checks if item.get("level") == "warn"]
    datasets = report.get("datasets", []) if isinstance(report.get("datasets"), list) else []
    manifests = report.get("manifests", []) if isinstance(report.get("manifests"), list) else []
    rows = sum(int(item.get("rows") or 0) for item in manifests)
    details = {
        "datasets": [item.get("dataset") for item in datasets if item.get("exists")],
        "manifests": len(manifests),
        "rows": rows,
        "blocks": blocks,
        "warnings": warnings,
    }
    if blocks:
        _add(
            gates,
            "block",
            "dataset_inventory",
            "Dataset inventory has blocking local-data, manifest, or weak-friction-interval issues.",
            **details,
        )
        return
    level = "warn" if warnings else "pass"
    _add(
        gates,
        level,
        "dataset_inventory",
        "Local RSCD/RoadSaW/RoadSC paths, manifests, weak friction intervals, and disk space are audited.",
        **details,
    )


def _gate_dataset_view_source_evidence(gates: list[dict[str, Any]], text: str | None) -> None:
    if not text:
        _add(
            gates,
            "warn",
            "dataset_view_source_evidence_missing",
            "Dataset view/source evidence report is missing; RSCD view, RoadSaW near-white samples, and pooling boundaries are not documented.",
        )
        return
    lower = text.lower()
    checks = {
        "rscd_not_left_right_wheel_claim": (
            "rscd should **not** be described" in lower
            and "left-wheel" in lower
            and "road-surface image patches" in lower
        ),
        "roadsaw_near_white_kept_as_quality_slice": (
            "near-white" in lower
            and "keep roadsaw near-white samples" in lower
            and "wet-road" in lower
        ),
        "naive_pooling_boundary": (
            "naive pooling" in lower
            and "dataset identity" in lower
            and "not a valid top-venue claim" in lower
        ),
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        _add(
            gates,
            "warn",
            "dataset_view_source_evidence",
            "Dataset view/source report exists but is missing one or more reviewer-facing boundary decisions.",
            missing=missing,
        )
        return
    _add(
        gates,
        "pass",
        "dataset_view_source_evidence",
        "Dataset view/source report documents RSCD view limits, RoadSaW near-white handling, and the no-naive-pooling boundary.",
        checks=checks,
    )


def _gate_friction_interval_sources(gates: list[dict[str, Any]], interval_source_audit: dict[str, Any]) -> None:
    verdict = interval_source_audit.get("verdict")
    if verdict == "pass":
        _add(
            gates,
            "pass",
            "friction_interval_source_audit",
            "Weak friction intervals cover public road-condition/TRFC reference anchors.",
        )
    elif verdict:
        _add(
            gates,
            "block",
            "friction_interval_source_audit",
            f"Friction interval source audit is `{verdict}`.",
        )
    else:
        _add(
            gates,
            "block",
            "friction_interval_source_audit_missing",
            "Missing friction_interval_source_audit.json.",
        )


def _gate_friction_interval_claim_matrix(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "friction_interval_claim_matrix_missing",
            "Friction interval claim matrix is missing; reviewer-facing interval target and comparison boundaries are not machine-checked.",
        )
        return
    blockers = report.get("blockers", []) if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings", []) if isinstance(report.get("warnings"), list) else []
    dataset_rows = report.get("dataset_claim_rows", []) if isinstance(report.get("dataset_claim_rows"), list) else []
    state_rows = report.get("state_interval_rows", []) if isinstance(report.get("state_interval_rows"), list) else []
    details = {
        "verdict": report.get("verdict"),
        "datasets": len(dataset_rows),
        "states": len(state_rows),
        "warnings": warnings[:8],
        "linked_reports": report.get("linked_reports"),
    }
    if blockers:
        _add(
            gates,
            "block",
            "friction_interval_claim_matrix",
            "Friction interval claim matrix has blockers; weak interval targets or fair-comparison boundaries are not defensible.",
            blockers=blockers[:8],
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "friction_interval_claim_matrix",
        "Friction interval claim matrix documents dataset claim cards, weak interval anchors, unsafe claims, and fair-comparison levels.",
        **details,
    )


def _gate_external_benchmark_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("public_sources"):
        _add(
            gates,
            "warn",
            "external_benchmark_report_missing",
            "External benchmark/fair-comparison report is missing.",
        )
        return
    sources = report.get("public_sources", [])
    comparability = report.get("comparability_matrix", [])
    source_names = {str(item.get("name", "")).lower() for item in sources}
    aligned_datasets = {str(item.get("dataset", "")).lower() for item in report.get("dataset_alignment", [])}
    levels = {str(item.get("comparison_level", "")) for item in comparability}
    required_sources = {
        "rscd": any("rscd" in name for name in source_names),
        "roadsaw": any("roadsaw" in name for name in source_names),
        "roadsc": any("roadsc" in name for name in source_names),
    }
    required_alignment = {
        "rscd": "rscd" in aligned_datasets,
        "roadsaw": "roadsaw" in aligned_datasets,
        "roadsc": "roadsc" in aligned_datasets,
    }
    missing_sources = [name for name, ok in required_sources.items() if not ok]
    missing_alignment = [name for name, ok in required_alignment.items() if not ok]
    if missing_sources or missing_alignment or not comparability:
        _add(
            gates,
            "warn",
            "external_benchmark_dataset_coverage",
            "External benchmark report is present but does not fully cover the required public datasets.",
            missing_sources=missing_sources,
            missing_alignment=missing_alignment,
            comparability_rows=len(comparability),
        )
        return
    if "primary_numeric_baseline" not in levels or "context_or_reimplementation_target" not in levels:
        _add(
            gates,
            "warn",
            "external_benchmark_comparability_matrix",
            "External benchmark report lacks explicit primary-baseline/context-only comparability levels.",
            levels=sorted(levels),
        )
        return
    _add(
        gates,
        "pass",
        "external_benchmark_report",
        "External benchmark and fair-comparison report exists with public dataset/source alignment and explicit comparability levels.",
        sources=len(sources),
        comparability_rows=len(comparability),
        datasets=sorted(required_sources),
    )


def _gate_direct_friction_public_benchmark(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("public_sources"):
        _add(
            gates,
            "warn",
            "direct_friction_public_benchmark_audit_missing",
            "Missing direct-friction public-benchmark audit; direct measured-friction claims are not independently bounded.",
        )
        return

    sources = report.get("public_sources", [])
    route_decisions = report.get("route_decisions", [])
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    direct_sources = int(counts.get("direct_context_sources", 0) or 0)
    fair_proxy_sources = int(counts.get("fair_proxy_sources", 0) or 0)
    decisions = {str(row.get("decision", "")) for row in route_decisions if isinstance(row, dict)}
    missing_expected_decisions = [
        decision
        for decision in [
            "discard_as_current_main_numeric_claim",
            "discard_as_main_claim",
            "keep_as_current_main_route",
        ]
        if decision not in decisions
    ]
    if direct_sources < 2 or fair_proxy_sources < 3 or missing_expected_decisions:
        _add(
            gates,
            "warn",
            "direct_friction_public_benchmark_audit_incomplete",
            "Direct-friction benchmark audit exists but does not yet fully separate direct measured-friction references from proxy-label fair comparisons.",
            sources=len(sources),
            direct_context_sources=direct_sources,
            fair_proxy_sources=fair_proxy_sources,
            missing_expected_decisions=missing_expected_decisions,
        )
        return

    verdict = report.get("verdict")
    if verdict == "strict_proxy_route_required":
        _add(
            gates,
            "pass",
            "direct_friction_public_benchmark_audit",
            "Direct image-to-measured-friction papers are bounded as context-only, while RSCD/RoadSaW/RoadSC remain the fair same-split proxy-label route.",
            sources=len(sources),
            direct_context_sources=direct_sources,
            fair_proxy_sources=fair_proxy_sources,
            missing_single_dataset_rows=counts.get("missing_single_dataset_rows"),
        )
    else:
        _add(
            gates,
            "warn",
            "direct_friction_public_benchmark_audit_verdict",
            f"Direct-friction benchmark audit verdict is `{verdict}`; inspect before making public comparison claims.",
            sources=len(sources),
            direct_context_sources=direct_sources,
            fair_proxy_sources=fair_proxy_sources,
        )


def _gate_rscd_external_comparison(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "rscd_external_comparison_readiness_missing",
            "RSCD-27 external comparison readiness report is missing; RSCD SOTA-style claims are not independently bounded.",
        )
        return
    verdict = str(report.get("verdict", "missing"))
    manifests = report.get("manifests", {}) if isinstance(report.get("manifests"), dict) else {}
    runner = report.get("runner", {}) if isinstance(report.get("runner"), dict) else {}
    formal = (report.get("results", {}) or {}).get("formal", {}) if isinstance(report.get("results"), dict) else {}
    details = {
        "verdict": verdict,
        "runner_exists": runner.get("exists"),
        "train_rows": (manifests.get("train") or {}).get("rows"),
        "val_rows": (manifests.get("val") or {}).get("rows"),
        "test_rows": (manifests.get("test") or {}).get("rows"),
        "classes": (manifests.get("train") or {}).get("classes"),
        "formal_status": formal.get("status"),
        "formal_top1": formal.get("top1"),
        "formal_macro_f1": formal.get("macro_f1"),
        "sota_claim": (report.get("decision") or {}).get("sota_claim")
        if isinstance(report.get("decision"), dict)
        else None,
    }
    if verdict == "formal_result_ready_for_local_rscd_context":
        _add(
            gates,
            "pass",
            "rscd_external_comparison_readiness",
            "RSCD-27 class-label protocol has a local result and can be discussed as secondary RSCD-style context under the documented split/metric boundary.",
            **details,
        )
        return
    if verdict == "protocol_ready_results_pending":
        _add(
            gates,
            "warn",
            "rscd_external_comparison_readiness",
            "RSCD-27 class-label protocol is implemented and manifests are ready, but no local RSCD-27 result exists yet; no RSCD SOTA-style claim is allowed.",
            **details,
        )
        return
    _add(
        gates,
        "block",
        "rscd_external_comparison_readiness",
        "RSCD-27 external comparison protocol is incomplete.",
        **details,
    )


def _gate_open_source_reproducibility_plan(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("rows", []) if isinstance(report, dict) else []
    if not rows:
        _add(
            gates,
            "warn",
            "open_source_reproducibility_plan_missing",
            "Open-source/GitHub reproducibility plan is missing.",
        )
        return
    _add(
        gates,
        "pass",
        "open_source_reproducibility_plan",
        "Open-source/GitHub references are mapped to protocol, baseline, candidate, or future-only roles with explicit claim limits.",
        sources=len(rows),
        implemented_or_configured=report.get("num_implemented_or_configured"),
        future_only=report.get("num_future_only"),
    )


def _gate_topvenue_innovation_roadmap(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("pattern_rows") or not report.get("source_rows"):
        _add(
            gates,
            "warn",
            "topvenue_innovation_roadmap_missing",
            "Top-venue innovation roadmap is missing or incomplete.",
        )
        return
    _add(
        gates,
        "pass",
        "topvenue_innovation_roadmap",
        "Top-venue innovation patterns are mapped to configured runs, evidence gates, and strict claim rules.",
        patterns=len(report.get("pattern_rows", [])),
        sources=len(report.get("source_rows", [])),
        pending_decisions=len(report.get("next_decisions", [])),
    )


def _gate_cv_transfer_decision_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("rows", []) if isinstance(report, dict) else []
    rules = report.get("reviewer_rules", []) if isinstance(report, dict) else []
    if not rows:
        _add(
            gates,
            "warn",
            "cv_transfer_decision_report_missing",
            "CV subfield transfer decision report is missing; segmentation/foundation/depth/domain-transfer routes are not centrally triaged.",
        )
        return

    implemented = [
        row
        for row in rows
        if "implemented" in str(row.get("claim_status", ""))
        or "configured" in str(row.get("claim_status", ""))
    ]
    future_only = [
        row
        for row in rows
        if str(row.get("claim_status", "")) == "future_only"
        or str(row.get("status", "")) == "demoted"
    ]
    if not implemented or len(rules) < 4:
        _add(
            gates,
            "warn",
            "cv_transfer_decision_report_incomplete",
            "CV transfer report exists but lacks enough implemented/configured routes or reviewer rules.",
            routes=len(rows),
            implemented_or_configured=len(implemented),
            future_or_demoted=len(future_only),
            rules=len(rules),
        )
        return
    _add(
        gates,
        "pass",
        "cv_transfer_decision_report",
        "CV subfield transfer routes are triaged into implemented/configured, future-only, and demoted paths with promotion/drop rules.",
        routes=len(rows),
        implemented_or_configured=len(implemented),
        future_or_demoted=len(future_only),
        verdict=report.get("verdict"),
    )


def _gate_pseudo_segmentation_mask_audit(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("dataset_rows", []) if isinstance(report, dict) else []
    claim_boundary = report.get("claim_boundary") if isinstance(report, dict) else None
    if not rows:
        _add(
            gates,
            "warn",
            "pseudo_segmentation_mask_audit_missing",
            "Pseudo-segmentation mask feasibility audit is missing; external SAM/Mask2Former routes are not sample-checked.",
        )
        return
    if not claim_boundary:
        _add(
            gates,
            "warn",
            "pseudo_segmentation_mask_audit_incomplete",
            "Pseudo-segmentation mask audit exists but lacks the pixel-label claim boundary.",
            datasets=len(rows),
            samples=report.get("samples_total"),
        )
        return
    _add(
        gates,
        "pass",
        "pseudo_segmentation_mask_audit",
        "Pseudo-segmentation/ROI masks are sample-audited before any heavy external mask preprocessing is allowed.",
        datasets=len(rows),
        samples=report.get("samples_total"),
        verdict=report.get("verdict"),
    )


def _gate_external_segmentation_mask_audit(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "external_segmentation_mask_audit_missing",
            "External segmentation backend audit is missing; CLIPSeg/SAM/SegFormer feasibility is not recorded.",
        )
        return
    deps = report.get("dependency_status", {}) if isinstance(report.get("dependency_status"), dict) else {}
    rows = report.get("dataset_rows", []) if isinstance(report.get("dataset_rows"), list) else []
    backend = report.get("backend")
    verdict = report.get("verdict")
    if not rows:
        _add(
            gates,
            "warn",
            "external_segmentation_mask_dependency_audit",
            "External segmentation backend dependencies are audited, but no sample masks have been generated yet.",
            backend=backend,
            verdict=verdict,
            transformers=deps.get("transformers"),
            segment_anything=deps.get("segment_anything"),
        )
        return
    _add(
        gates,
        "pass",
        "external_segmentation_mask_audit",
        "External/optional segmentation mask evaluation pipeline has sample-mask evidence before heavy preprocessing is allowed.",
        backend=backend,
        verdict=verdict,
        datasets=len(rows),
        samples=report.get("samples_total"),
        transformers=deps.get("transformers"),
        segment_anything=deps.get("segment_anything"),
    )


def _gate_external_road_mask_cache(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "external_road_mask_cache_missing",
            "External road-mask cache bridge is missing; optional mask-supervised EvidenceField ablations are not yet reproducible.",
        )
        return
    manifest_reports = report.get("manifest_reports", []) if isinstance(report.get("manifest_reports"), list) else []
    cached_rows = sum(int(row.get("rows", 0) or 0) for row in manifest_reports if isinstance(row, dict))
    contract = report.get("training_contract", {}) if isinstance(report.get("training_contract"), dict) else {}
    required = contract.get("data_config_required", {}) if isinstance(contract.get("data_config_required"), dict) else {}
    has_contract = (
        contract.get("manifest_column") == "road_mask_path"
        and required.get("load_road_masks") is True
        and required.get("road_mask_pretransformed") is True
        and float(required.get("augmentation.horizontal_flip_p", 1.0)) == 0.0
        and required.get("augmentation.random_resized_crop") is False
    )
    if cached_rows <= 0 or not has_contract:
        _add(
            gates,
            "warn",
            "external_road_mask_cache_incomplete",
            "External road-mask cache exists but lacks cached rows or a complete training-alignment contract.",
            cached_rows=cached_rows,
            has_contract=has_contract,
            verdict=report.get("verdict"),
        )
        return
    _add(
        gates,
        "pass",
        "external_road_mask_cache",
        "External road-mask pseudo-label cache bridge is smoke-tested with manifest paths and an aligned training contract.",
        cached_rows=cached_rows,
        verdict=report.get("verdict"),
        cache_root=report.get("cache_root"),
    )


def _gate_segmentation_transfer_config(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "segmentation_transfer_config_audit_missing",
            "Mask-supervised segmentation-transfer config audit is missing; optional road-mask EvidenceField ablations are not yet smoke-tested.",
        )
        return
    batch = report.get("batch_report", {}) if isinstance(report.get("batch_report"), dict) else {}
    pseudo_loss = float(batch.get("loss_evidence_attention_pseudo_road", 0.0) or 0.0)
    if report.get("verdict") == "pass" and batch.get("batch_has_road_mask") and pseudo_loss > 0:
        _add(
            gates,
            "pass",
            "segmentation_transfer_config_audit",
            "Mask-supervised EvidenceField candidate config is smoke-tested through manifest, dataloader, and pseudo-road attention loss.",
            config=report.get("config"),
            pseudo_loss=pseudo_loss,
            road_mass=batch.get("attention_pseudo_road_mass"),
        )
        return
    _add(
        gates,
        "warn",
        "segmentation_transfer_config_audit",
        "Mask-supervised EvidenceField config audit exists but did not prove the pseudo-road loss path.",
        verdict=report.get("verdict"),
        batch_has_road_mask=batch.get("batch_has_road_mask"),
        pseudo_loss=pseudo_loss,
    )


def _gate_segmentation_transfer_queue(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "segmentation_transfer_queue_status_missing",
            "Background CLIPSeg/SAM audit queue status is missing; resource isolation is not documented.",
        )
        return
    after = report.get("after_queue", {}) if isinstance(report.get("after_queue"), dict) else {}
    promotion = report.get("promotion", {}) if isinstance(report.get("promotion"), dict) else {}
    wait_pids = after.get("wait_pids") if isinstance(after.get("wait_pids"), list) else []
    device = after.get("device")
    process_rows = report.get("process_rows", []) if isinstance(report.get("process_rows"), list) else []
    waits_for_queue = 28496 in [int(pid) for pid in wait_pids if str(pid).isdigit()]
    cpu_only = str(device).lower() == "cpu"
    if waits_for_queue and cpu_only:
        _add(
            gates,
            "pass",
            "segmentation_transfer_queue_status",
            "External segmentation audit automation waits for the full formal queue and is configured CPU-only.",
            backend=after.get("backend"),
            wait_pids=wait_pids,
            promotion_wait_pid=promotion.get("wait_pid"),
            processes=len(process_rows),
        )
        return
    _add(
        gates,
        "warn",
        "segmentation_transfer_queue_status",
        "External segmentation audit automation exists but its queue wait/device contract should be checked.",
        backend=after.get("backend"),
        wait_pids=wait_pids,
        device=device,
    )


def _gate_next_queue_readiness(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "next_queue_readiness_missing",
            "Next-queue readiness report is missing; upcoming baselines/candidates lack a central run-readiness and pruning-policy audit.",
        )
        return
    blocks = report.get("blocks", []) if isinstance(report.get("blocks"), list) else []
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    queue_counts = report.get("queue_counts", {}) if isinstance(report.get("queue_counts"), dict) else {}
    not_ready = [
        row.get("run")
        for row in rows
        if not (
            row.get("protocol_ready")
            and row.get("gpu_ready")
            and row.get("module_ready")
        )
    ]
    details = {
        "verdict": report.get("verdict"),
        "rows": len(rows),
        "queue_complete": queue_counts.get("complete"),
        "queue_missing": queue_counts.get("missing"),
        "protocol_verdict": report.get("protocol_verdict"),
        "gpu_verdict": report.get("gpu_verdict"),
        "cv_route_verdict": report.get("cv_route_verdict"),
        "not_ready": not_ready[:8],
        "blocks": blocks[:8],
    }
    if blocks:
        _add(
            gates,
            "block",
            "next_queue_readiness",
            "Next-queue readiness report has blocking protocol, GPU, or module-audit gaps.",
            **details,
        )
        return
    if not_ready:
        _add(
            gates,
            "warn",
            "next_queue_readiness",
            "Some upcoming runs are not fully backed by protocol, GPU, and module-readiness evidence.",
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "next_queue_readiness",
        "Upcoming fair baselines and CV-transfer candidates have protocol, GPU-envelope, module, and pruning-policy readiness evidence.",
        **details,
    )


def _gate_fair_execution_priority(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("stages"):
        _add(
            gates,
            "warn",
            "fair_comparison_execution_priority_missing",
            "Fair-comparison execution priority audit is missing; claim order is not machine-checked.",
        )
        return
    violations = report.get("sequence_violations", [])
    if violations:
        _add(
            gates,
            "block",
            "fair_comparison_execution_priority",
            "Execution-priority audit found claim-order violations.",
            verdict=report.get("verdict"),
            violations=violations,
        )
        return
    stages = report.get("stages", [])
    first_pending = next((stage for stage in stages if stage.get("status") != "complete"), None)
    claim_locks = report.get("claim_locks", [])
    locked = [
        row.get("claim")
        for row in claim_locks
        if row.get("status") in {"locked", "disallowed"}
    ]
    _add(
        gates,
        "pass",
        "fair_comparison_execution_priority",
        "Execution-priority audit locks fair comparisons, RSCD external context, CV-transfer screening, and final-method claims in the correct order.",
        verdict=report.get("verdict"),
        first_pending=first_pending.get("name") if isinstance(first_pending, dict) else None,
        locked_claims=locked,
        active_runs=[row.get("run") for row in report.get("active_runs", [])],
    )


def _gate_cv_transfer_experiment_protocol(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "cv_transfer_experiment_protocol_missing",
            "CV-transfer experiment protocol is missing; semantic-segmentation transfer routes are not checked as executable experiments.",
        )
        return
    blocks = report.get("blocks", []) if isinstance(report.get("blocks"), list) else []
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    details = {
        "verdict": report.get("verdict"),
        "routes": counts.get("routes"),
        "implementation_ready": counts.get("implementation_ready"),
        "metric_pending": counts.get("metric_pending"),
        "future_only": counts.get("future_only"),
        "blocks": blocks,
    }
    if blocks:
        _add(
            gates,
            "block",
            "cv_transfer_experiment_protocol",
            "Some CV-transfer routes lack required source, config, or smoke-test evidence.",
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "cv_transfer_experiment_protocol",
        "Semantic-segmentation, consistency, domain-adaptation, material-vision, and teacher-model transfer routes have executable protocols and drop rules.",
        **details,
    )


def _gate_wet_slippery_failure_mechanism(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("mechanisms"):
        _add(
            gates,
            "warn",
            "wet_slippery_failure_mechanism_report_missing",
            "Wet/slippery failure mechanisms are not mapped to CV-subfield transfer routes and pruning metrics.",
        )
        return
    blocks = report.get("blocks", []) if isinstance(report.get("blocks"), list) else []
    mechanisms = report.get("mechanisms", []) if isinstance(report.get("mechanisms"), list) else []
    missing_rules = [
        row.get("mechanism")
        for row in mechanisms
        if not row.get("promotion_rule") or not row.get("drop_rule")
    ]
    segmentation_routes = [
        row.get("mechanism")
        for row in mechanisms
        if "semantic segmentation" in str(row.get("cv_subfield_transfer", ""))
    ]
    details = {
        "verdict": report.get("verdict"),
        "mechanisms": len(mechanisms),
        "blocks": blocks,
        "missing_rules": missing_rules,
        "segmentation_style_mechanisms": segmentation_routes,
    }
    if blocks or missing_rules:
        _add(
            gates,
            "block",
            "wet_slippery_failure_mechanism_report",
            "Some wet/slippery CV-transfer mechanisms lack candidate configs, metrics, or pruning rules.",
            **details,
        )
        return
    level = "pass" if len(mechanisms) >= 6 and segmentation_routes else "warn"
    message = (
        "Wet/slippery failure mechanisms are mapped to semantic-segmentation-style local evidence, consistency, domain-shift, material-vision, interval, and teacher routes."
        if level == "pass"
        else "Wet/slippery mechanism map exists but is thin; expand it before using it as the candidate-pruning spine."
    )
    _add(gates, level, "wet_slippery_failure_mechanism_report", message, **details)


def _gate_cv_transfer_candidate_priority(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "cv_transfer_candidate_priority_report_missing",
            "CV-transfer candidate priority report is missing; fast exploration and pruning order are not machine-checked.",
        )
        return
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    missing_rules = [
        row.get("name")
        for row in rows
        if not row.get("keep_rule") or not row.get("drop_rule") or not row.get("rapid_prune_trigger")
    ]
    priorities = [row.get("priority") for row in rows]
    has_fair_first = any(row.get("priority") == 0 and row.get("name") == "fair_comparison_before_claims" for row in rows)
    details = {
        "verdict": report.get("verdict"),
        "rows": len(rows),
        "priorities": priorities,
        "first_incomplete": report.get("first_incomplete"),
        "active_runs": report.get("active_runs"),
        "missing_rules": missing_rules,
    }
    if missing_rules or not has_fair_first:
        _add(
            gates,
            "block",
            "cv_transfer_candidate_priority_report",
            "CV-transfer candidate priority report lacks fair-first ordering or pruning rules.",
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "cv_transfer_candidate_priority_report",
        "CV-transfer candidates are ordered by fair-claim prerequisites, segmentation-style local evidence, material uncertainty, consistency, domain-shift control, and teacher routes.",
        **details,
    )


def _gate_cv_transfer_retention_decision(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "cv_transfer_retention_decision_report_missing",
            "CV-transfer retain/prune decision report is missing; completed candidates will not be routed into keep/merge/prune decisions.",
        )
        return
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    pruned_routes = [
        row.get("group")
        for row in rows
        if row.get("decision") == "prune_route"
    ]
    available_decisions = [
        row.get("group")
        for row in rows
        if row.get("decision") in {"keep_route", "rescue_route", "merge_only", "prune_route"}
    ]
    pending = [
        row.get("group")
        for row in rows
        if row.get("decision") in {"pending_metrics", "pending_mixed", "wait_for_fair_baselines"}
    ]
    details = {
        "verdict": report.get("verdict"),
        "rows": len(rows),
        "available_decisions": available_decisions,
        "pending": pending,
        "pruned_routes": pruned_routes,
    }
    if pruned_routes:
        _add(
            gates,
            "block",
            "cv_transfer_retention_decision_report",
            "One or more CV-transfer routes have completed metrics that require pruning or rework before finalizing the method.",
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "cv_transfer_retention_decision_report",
        "CV-transfer routes have result-driven keep/rescue/merge/prune decision slots, with pending groups explicitly held as hypotheses.",
        **details,
    )


def _gate_final_freeze_audit(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("dependencies"):
        _add(
            gates,
            "warn",
            "final_freeze_audit_missing",
            "Final-freeze audit is missing; the final method could be claimed before fair/candidate evidence is ready.",
        )
        return
    risky = report.get("final_risky_modules", [])
    if risky:
        _add(
            gates,
            "block",
            "final_freeze_audit",
            "Final configs still contain modules that current evidence says should be removed or rescued first.",
            verdict=report.get("verdict"),
            risky=risky,
        )
        return
    _add(
        gates,
        "pass",
        "final_freeze_audit",
        "Final-freeze audit keeps the final method provisional until fair baselines, CV-transfer candidates, pruning, and final runs are complete.",
        verdict=report.get("verdict"),
        blocking_dependencies=report.get("blocking_dependencies"),
        final_rows=len(report.get("final_rows", [])),
    )


def _gate_candidate_hypothesis_matrix(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "candidate_hypothesis_matrix_missing",
            "Candidate experiment hypothesis matrix is missing; v6-v16/final runs lack a central success/failure rule audit.",
        )
        return
    coverage = report.get("coverage", {}) if isinstance(report.get("coverage"), dict) else {}
    missing_specs = coverage.get("missing_specs", []) if isinstance(coverage.get("missing_specs"), list) else []
    incomplete = coverage.get("incomplete_fields", []) if isinstance(coverage.get("incomplete_fields"), list) else []
    final_runs = coverage.get("final_runs", []) if isinstance(coverage.get("final_runs"), list) else []
    candidate_runs = coverage.get("candidate_runs", []) if isinstance(coverage.get("candidate_runs"), list) else []
    lodo_runs = coverage.get("lodo_runs", []) if isinstance(coverage.get("lodo_runs"), list) else []
    fair_runs = coverage.get("fair_baseline_runs", []) if isinstance(coverage.get("fair_baseline_runs"), list) else []
    if missing_specs or incomplete:
        _add(
            gates,
            "warn",
            "candidate_hypothesis_matrix",
            "Candidate hypothesis matrix exists but some runs lack hypothesis, success, failure, or retention fields.",
            rows=report.get("num_rows"),
            missing_specs=missing_specs,
            incomplete_fields=incomplete[:8],
        )
        return
    _add(
        gates,
        "pass",
        "candidate_hypothesis_matrix",
        "Candidate and final runs have predeclared hypotheses, success criteria, failure actions, and module-retention rules.",
        rows=report.get("num_rows"),
        candidates=len(candidate_runs),
        final_runs=len(final_runs),
        lodo_runs=len(lodo_runs),
        fair_runs=len(fair_runs),
    )


def _gate_candidate_implementation_coverage(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("rows", []) if isinstance(report, dict) else []
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    if not rows:
        _add(
            gates,
            "warn",
            "candidate_implementation_coverage_missing",
            "Candidate implementation coverage audit is missing; innovation routes are not centrally tied to code/config evidence.",
        )
        return
    source_gaps = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("source_gap_modules")
    ]
    if source_gaps:
        _add(
            gates,
            "block",
            "candidate_implementation_coverage",
            "Some candidate innovation routes have source/config trace gaps.",
            routes=[row.get("name") for row in source_gaps[:8]],
            source_gap=counts.get("source_gap"),
        )
        return
    _add(
        gates,
        "pass",
        "candidate_implementation_coverage",
        "Candidate innovation routes are tied to source/config evidence; remaining gaps are metric evidence rather than implementation reachability.",
        routes=counts.get("routes"),
        pending_or_partial=counts.get("pending_or_partial"),
        evidence_complete=counts.get("evidence_complete"),
    )


def _gate_candidate_pruning_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "candidate_pruning_report_missing",
            "Candidate pruning report is missing; queued CV-transfer modules lack an automatic keep/prune/rescue audit.",
        )
        return
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    policy = report.get("policy", []) if isinstance(report.get("policy"), list) else []
    verdict = report.get("verdict")
    if not policy:
        _add(
            gates,
            "warn",
            "candidate_pruning_report",
            "Candidate pruning rows exist but the keep/prune policy is missing.",
            verdict=verdict,
            rows=counts.get("rows"),
        )
        return
    pruned = int(counts.get("complete_pruned", 0) or 0)
    kept = int(counts.get("complete_keep", 0) or 0)
    rescued = int(counts.get("complete_rescue", 0) or 0)
    neutral = int(counts.get("complete_neutral", 0) or 0)
    pending = int(counts.get("pending", 0) or 0)
    message = (
        "Candidate pruning audit has predeclared keep/prune/rescue rules; "
        "completed CV-transfer runs will be retained only with metric or interpretability evidence."
    )
    _add(
        gates,
        "pass",
        "candidate_pruning_report",
        message,
        verdict=verdict,
        rows=counts.get("rows"),
        pending=pending,
        keep=kept,
        prune_or_rework=pruned,
        rescue=rescued,
        neutral_or_merge=neutral,
        reference=report.get("reference", {}).get("run") if isinstance(report.get("reference"), dict) else None,
    )


def _gate_online_source_refresh(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("source_rows", []) if isinstance(report, dict) else []
    rules = report.get("decision_rules", []) if isinstance(report, dict) else []
    if len(rows) >= 10 and len(rules) >= 5:
        _add(
            gates,
            "pass",
            "online_source_refresh_report",
            "Online/public source refresh maps datasets, method inspirations, friction anchors, and reviewer decision rules to the protocol.",
            sources=len(rows),
            rules=len(rules),
        )
        return
    if rows or rules:
        _add(
            gates,
            "warn",
            "online_source_refresh_report_incomplete",
            "Online/public source refresh exists but is incomplete.",
            sources=len(rows),
            rules=len(rules),
        )
        return
    _add(
        gates,
        "warn",
        "online_source_refresh_report_missing",
        "Missing online_source_refresh_report.json.",
    )


def _gate_latest_visual_friction_web_check(gates: list[dict[str, Any]], text: str | None) -> None:
    if not text:
        _add(
            gates,
            "warn",
            "latest_visual_friction_web_check_missing",
            "Missing latest visual-friction web-check report; current direct-friction comparability boundary is not freshly recorded.",
        )
        return
    required = ["SIWNet", "WCamNet", "ROAD Camera-IMU", "RoadSaW", "RoadSC", "RSCD", "ConvNeXt"]
    present = [item for item in required if item in text]
    if len(present) == len(required) and "Verdict" in text and "Decision" in text:
        _add(
            gates,
            "pass",
            "latest_visual_friction_web_check",
            "Latest visual-friction web check records why direct measured-friction papers are context-only and matched ConvNeXt remains the fair numeric route.",
            sources=len(present),
        )
        return
    _add(
        gates,
        "warn",
        "latest_visual_friction_web_check_incomplete",
        "Latest visual-friction web check exists but is missing required source coverage or decision wording.",
        present=present,
        required=required,
    )


def _gate_module_retention_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("rows", []) if isinstance(report, dict) else []
    if not rows:
        _add(
            gates,
            "warn",
            "module_retention_report_missing",
            "Missing module_retention_report.json; final module pruning rules are not independently summarized.",
        )
        return
    decisions = {str(row.get("current_decision")) for row in rows}
    if "provisional_remove_or_merge" in decisions or "pending" in decisions:
        _add(
            gates,
            "warn",
            "module_retention_report",
            "Module retention report exists and identifies provisional removals/pending modules before final architecture freeze.",
            verdict=report.get("verdict"),
            rows=len(rows),
            decisions=sorted(decisions),
        )
        return
    _add(
        gates,
        "pass",
        "module_retention_report",
        "Module retention report exists and all module decisions are final-keep candidates under the hard rule.",
        verdict=report.get("verdict"),
        rows=len(rows),
        decisions=sorted(decisions),
    )


def _gate_p0_claim_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "p0_claim_report_missing",
            "P0 claim report with CI-aware module deltas is missing.",
        )
        return
    level = "pass" if report.get("core_status") == "complete" else "warn"
    _add(
        gates,
        level,
        "p0_claim_report",
        "P0 claim report exists with bootstrap-CI row metrics and conservative adjacent module deltas.",
        status=report.get("core_status"),
    )


def _gate_completeness(
    gates: list[dict[str, Any]],
    completeness: dict[str, Any],
    artifact_contract: dict[str, Any],
) -> None:
    requirements = {item.get("name"): item for item in completeness.get("requirements", [])}
    artifact_groups = artifact_contract.get("hard_status", {}) if isinstance(artifact_contract, dict) else {}
    artifact_map = {
        "p0_ablation_complete": "p0_ablation",
        "lodo_complete": "lodo",
        "fair_single_dataset_complete": "single_dataset_fair",
        "final_method_complete": "final_method",
    }
    names = [
        "p0_ablation_complete",
        "lodo_complete",
        "fair_single_dataset_complete",
        "final_method_complete",
        "summary_tables_complete",
    ]
    for name in names:
        artifact_group = artifact_groups.get(artifact_map.get(name, ""))
        if isinstance(artifact_group, dict):
            description = (requirements.get(name) or {}).get("description", name)
            if artifact_group.get("complete"):
                _add(
                    gates,
                    "pass",
                    name,
                    description,
                    source="artifact_contract",
                    complete_runs=artifact_group.get("num_complete"),
                    runs=artifact_group.get("num_runs"),
                )
            else:
                _add(
                    gates,
                    "block",
                    name,
                    _incomplete_requirement_message(name, description),
                    source="artifact_contract",
                    missing=artifact_group.get("missing", []),
                    complete_runs=artifact_group.get("num_complete"),
                    runs=artifact_group.get("num_runs"),
                )
            continue
        item = requirements.get(name)
        if not item:
            _add(gates, "block", f"{name}_missing", f"Completeness requirement `{name}` is missing.")
            continue
        if item.get("status") == "complete":
            _add(gates, "pass", name, item.get("description", name))
        else:
            _add(
                gates,
                "block",
                name,
                _incomplete_requirement_message(name, item.get("description", name)),
                missing=item.get("missing", []),
            )
    candidate = requirements.get("candidate_path_complete")
    if not candidate:
        _add(gates, "warn", "candidate_path_complete_missing", "Candidate-path completeness entry is missing.")
    elif candidate.get("status") == "complete":
        _add(gates, "pass", "candidate_path_complete", candidate.get("description", "Candidate path complete."))
    else:
        _add(
            gates,
            "warn",
            "candidate_path_complete",
            "P1 robustness/domain-shift candidates are not yet complete.",
            missing=candidate.get("missing", []),
        )


def _incomplete_requirement_message(name: str, description: str) -> str:
    custom = {
        "fair_single_dataset_complete": (
            "Single-dataset FAF and global ConvNeXt baselines are not yet complete for fair comparisons."
        ),
        "final_method_complete": (
            "Final lean road-ROI safety method evidence is not yet complete."
        ),
        "p0_ablation_complete": "Core P0 ablation evidence is not yet complete.",
        "lodo_complete": "LODO evidence is not yet complete.",
        "summary_tables_complete": "Required summary tables are not yet complete.",
    }
    return custom.get(name, f"Requirement is incomplete: {description}")


def _gate_core_ablation(gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    core = {row.get("method"): row for row in summary.get("core_ablation", [])}
    baseline = core.get("Global-only")
    full = core.get("Full model")
    if not baseline or baseline.get("status") != "complete":
        _add(gates, "block", "core_baseline_missing", "Global-only baseline is not complete.")
        return
    if not full or full.get("status") != "complete":
        _add(gates, "block", "full_model_missing", "Full model row is not complete, so P0 cannot support the main claim.")
        return

    checks = [
        ("full_vs_global_risk_f1", "risk_f1", 0.005, "Full model should improve risk F1 by at least 0.5 point."),
        (
            "full_vs_global_low_friction_recall",
            "low_friction_recall",
            -0.005,
            "Full model should preserve low-friction recall within 0.5 point or improve it.",
        ),
        (
            "full_vs_global_worst_dataset_f1",
            "worst_dataset_f1",
            0.005,
            "Full model should improve worst-dataset F1 by at least 0.5 point.",
        ),
    ]
    for name, key, threshold, message in checks:
        delta = _delta(full, baseline, key)
        if delta is None:
            _add(gates, "block", name, f"Cannot compute `{key}` delta for full model.")
        elif delta >= threshold:
            _add(gates, "pass", name, message, delta=delta)
        else:
            _add(gates, "block", name, message, delta=delta, required_delta=threshold)


def _gate_lodo(gates: list[dict[str, Any]], summary: dict[str, Any], lodo_report: dict[str, Any]) -> None:
    summary_rows = {row.get("method"): row for row in summary.get("lodo", [])}
    summary_roadsaw = summary_rows.get("held-out RoadSaW")
    if isinstance(lodo_report, dict) and lodo_report.get("rows"):
        rows = {f"held-out {row.get('held_out')}": row for row in lodo_report.get("rows", [])}
        roadsaw = rows.get("held-out RoadSaW")
        risk_key = "risk_f1"
    else:
        rows = summary_rows
        roadsaw = rows.get("held-out RoadSaW")
        risk_key = "risk_macro_f1"
    if not roadsaw or roadsaw.get("status") != "complete":
        _add(gates, "block", "heldout_roadsaw_missing", "Held-out RoadSaW LODO result is missing.")
        return
    risk = roadsaw.get(risk_key)
    low_source = summary_roadsaw if summary_roadsaw and summary_roadsaw.get("status") == "complete" else roadsaw
    low = low_source.get("low_friction_recall")
    low_applicable = low_source.get("low_friction_recall_applicable")
    low_positive_count = _num(low_source.get("low_friction_positive_count"))
    if risk is not None and float(risk) >= 0.55:
        _add(gates, "pass", "heldout_roadsaw_risk_f1", "Held-out RoadSaW risk F1 clears the minimum generalization gate.", value=risk)
    else:
        _add(gates, "block", "heldout_roadsaw_risk_f1", "Held-out RoadSaW risk F1 is below the minimum generalization gate.", value=risk, threshold=0.55)
    if low_applicable is False or low_positive_count == 0:
        _add(
            gates,
            "pass",
            "heldout_roadsaw_low_recall_not_applicable",
            "Held-out RoadSaW has no high/very_high risk positives, so low-friction recall is not a valid gate.",
            num_positive=low_positive_count,
        )
    elif low is None:
        _add(gates, "warn", "heldout_roadsaw_low_recall_missing", "Held-out RoadSaW low-friction recall is unavailable.")
    elif float(low) >= 0.80:
        _add(gates, "pass", "heldout_roadsaw_low_recall", "Held-out RoadSaW low-friction recall is acceptable.", value=low)
    else:
        _add(gates, "block", "heldout_roadsaw_low_recall", "Held-out RoadSaW low-friction recall is too low for a safety claim.", value=low, threshold=0.80)


def _gate_fair_single_dataset(gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    deltas = summary.get("fair_single_dataset_deltas", [])
    complete = [row for row in deltas if row.get("status") == "complete"]
    if len(complete) < 3:
        _add(gates, "block", "fair_single_dataset_missing", "Matched single-dataset FAF vs ConvNeXt comparisons are incomplete.")
        return
    positive = [
        row
        for row in complete
        if _num(row.get("delta_risk_macro_f1")) is not None
        and _num(row.get("delta_risk_macro_f1")) >= 0.005
    ]
    if positive:
        _add(gates, "pass", "fair_single_dataset_advantage", "At least one matched public-dataset comparison improves risk F1.", datasets=[row.get("dataset") for row in positive])
    else:
        _add(gates, "block", "fair_single_dataset_advantage", "No matched single-dataset comparison shows a risk-F1 advantage.")


def _gate_final_method(gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    final_lodo = {row.get("method"): row for row in summary.get("final_lodo", [])}
    roadsaw = final_lodo.get("final held-out RoadSaW")
    if not roadsaw or roadsaw.get("status") != "complete":
        _add(gates, "block", "final_heldout_roadsaw_missing", "Final lean method has no held-out RoadSaW LODO result.")
    else:
        risk = _num(roadsaw.get("risk_macro_f1"))
        if risk is not None and risk >= 0.55:
            _add(gates, "pass", "final_heldout_roadsaw_risk_f1", "Final lean method clears held-out RoadSaW risk-F1 gate.", value=risk)
        else:
            _add(gates, "block", "final_heldout_roadsaw_risk_f1", "Final lean method fails held-out RoadSaW risk-F1 gate.", value=risk, threshold=0.55)

    final_deltas = summary.get("final_fair_single_dataset_deltas", [])
    complete = [row for row in final_deltas if row.get("status") == "complete"]
    if len(complete) < 3:
        _add(gates, "block", "final_fair_single_dataset_missing", "Final lean method vs ConvNeXt single-dataset comparisons are incomplete.")
        return
    positive = [
        row
        for row in complete
        if _num(row.get("delta_risk_macro_f1")) is not None
        and _num(row.get("delta_risk_macro_f1")) >= 0.005
    ]
    if positive:
        _add(gates, "pass", "final_fair_single_dataset_advantage", "Final lean method improves risk F1 on at least one matched public dataset.", datasets=[row.get("dataset") for row in positive])
    else:
        _add(gates, "block", "final_fair_single_dataset_advantage", "Final lean method shows no matched single-dataset risk-F1 advantage.")


def _gate_shortcut_and_intervals(gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = [row for row in summary.get("ablation", []) if row.get("status") == "complete"]
    if not rows:
        _add(gates, "block", "no_completed_rows_for_quality", "No completed rows are available for shortcut/interval checks.")
        return
    best = max(rows, key=lambda row: _num(row.get("risk_macro_f1")) or -1.0)
    raw_cov = _num(best.get("raw_interval_coverage"))
    cal_cov = _num(best.get("calibrated_coverage"))
    cal_width = _num(best.get("calibrated_width"))
    domain_acc = _num(best.get("dataset_id_balanced_accuracy"))

    if raw_cov is not None and raw_cov >= 0.70:
        _add(gates, "pass", "raw_interval_coverage", "Best completed row has acceptable raw interval coverage.", run=best.get("method"), value=raw_cov)
    else:
        _add(gates, "warn", "raw_interval_coverage", "Raw interval coverage remains weak; rely on P3 candidates and conformal calibration.", run=best.get("method"), value=raw_cov, threshold=0.70)

    if cal_cov is not None and 0.88 <= cal_cov <= 0.94 and cal_width is not None and cal_width <= 0.65:
        _add(gates, "pass", "calibrated_interval_quality", "Calibrated interval coverage/width are in a usable band.", run=best.get("method"), coverage=cal_cov, width=cal_width)
    else:
        _add(gates, "block", "calibrated_interval_quality", "Calibrated intervals are missing or outside the usable coverage/width band.", run=best.get("method"), coverage=cal_cov, width=cal_width)

    if domain_acc is not None and domain_acc <= 0.85:
        _add(gates, "pass", "dataset_shortcut", "Dataset-ID probe is low enough for a strong shortcut-resistance claim.", run=best.get("method"), value=domain_acc)
    else:
        _add(gates, "warn", "dataset_shortcut", "Dataset-ID probe remains high; claim shortcut mitigation only after P1 candidates improve it.", run=best.get("method"), value=domain_acc, threshold=0.85)


def _gate_conditional_interval_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "conditional_interval_report_missing",
            "Conditional interval quality report is missing; P3 interval failures are not cell-audited.",
        )
        return
    watch_count = int(report.get("num_watchlist_items", 0) or 0)
    if watch_count <= 0:
        _add(
            gates,
            "pass",
            "conditional_interval_report",
            "Conditional interval report exists and no group crossed the configured undercoverage watch thresholds.",
            runs=report.get("num_runs"),
            cells=report.get("num_cells"),
        )
        return
    top = []
    for item in report.get("watchlist", [])[:5]:
        top.append(
            {
                "run": item.get("run"),
                "scope": item.get("scope"),
                "group": item.get("group_label"),
                "reason": item.get("reason"),
                "coverage": item.get("raw_coverage")
                if item.get("reason") == "raw_undercoverage"
                else item.get("calibrated_coverage"),
            }
        )
    _add(
        gates,
        "warn",
        "conditional_interval_report",
        "Conditional interval report exists and identifies undercovered cells for P3/final tuning.",
        watchlist_items=watch_count,
        top=top,
    )


def _gate_dataset_shortcut_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "dataset_shortcut_report_missing",
            "Dataset shortcut aggregate report is missing.",
        )
        return
    level = "pass" if report.get("verdict") == "pass" else "warn"
    _add(
        gates,
        level,
        "dataset_shortcut_report",
        "Dataset-ID shortcut aggregate report exists for overall/risk/core-state probes.",
        verdict=report.get("verdict"),
        complete=report.get("num_complete"),
        high_shortcut=report.get("num_high_shortcut"),
        threshold=report.get("threshold"),
    )


def _gate_wetness_state_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "wetness_state_report_missing",
            "RoadSaW wetness-state diagnostic report is missing.",
        )
        return
    _add(
        gates,
        "pass",
        "wetness_state_report",
        "RoadSaW wetness-state diagnostics exist for damp/wet/very_wet failure tracking.",
        complete=report.get("num_complete"),
        watchlist=report.get("num_watchlist"),
    )


def _gate_safety_selection(gates: list[dict[str, Any]], safety_selection: dict[str, Any]) -> None:
    verdict = safety_selection.get("verdict")
    rows = safety_selection.get("rows", []) if isinstance(safety_selection, dict) else []
    complete = [row for row in rows if row.get("status") == "complete"]
    pending = [row for row in rows if row.get("status") != "complete"]
    if not rows:
        _add(gates, "warn", "safety_selection_missing", "Safety-selected checkpoint report is missing.")
        return
    if complete:
        helpful = [
            row
            for row in complete
            if (_num(row.get("delta_low_friction_recall")) or 0.0) > 0
            or (_num(row.get("delta_raw_interval_coverage")) or 0.0) > 0
        ]
        if helpful:
            _add(
                gates,
                "pass",
                "safety_selection_supplement",
                "Safety-selected checkpoint has supplemental safety gains.",
                runs=[row.get("method") for row in helpful],
            )
        else:
            _add(
                gates,
                "warn",
                "safety_selection_no_gain",
                "Safety-selected checkpoints are evaluated but do not improve low recall or raw coverage.",
                verdict=verdict,
            )
    elif pending:
        _add(
            gates,
            "warn",
            "safety_selection_pending",
            "Safety-selected checkpoint exists but supplemental evaluation is pending.",
            pending=[row.get("method") for row in pending],
        )


def _gate_checkpoint_policy(gates: list[dict[str, Any]], checkpoint_policy: dict[str, Any]) -> None:
    policy = checkpoint_policy.get("policy", {}) if isinstance(checkpoint_policy, dict) else {}
    audit_rules = checkpoint_policy.get("audit_rules", []) if isinstance(checkpoint_policy, dict) else []
    required = {
        "main_ablation_table",
        "supplemental_safety_analysis",
        "final_method_selection",
        "claim_boundary",
    }
    missing = sorted(required - set(policy))
    if not policy:
        _add(
            gates,
            "warn",
            "checkpoint_policy_missing",
            "Checkpoint selection policy report is missing; safety checkpoints could be confused with main-table checkpoints.",
        )
        return
    if missing:
        _add(
            gates,
            "warn",
            "checkpoint_policy_incomplete",
            "Checkpoint selection policy exists but does not cover every required decision boundary.",
            missing=missing,
        )
        return
    _add(
        gates,
        "pass",
        "checkpoint_policy",
        "Checkpoint selection policy is explicit: main rows use loss-selected checkpoints, safety checkpoints are supplemental, and final candidates use predeclared selection.",
        policy_keys=sorted(policy),
        audit_rules=len(audit_rules),
    )


def _gate_evidence(gates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    rows = [
        row
        for row in summary.get("ablation", [])
        if row.get("status") == "complete"
        and _run_name(row).startswith(
            ("v4_", "v5_", "v6_", "v7_", "v8_", "v9_", "v10_", "v11_", "v12_", "v13_", "v14_", "v15_", "v16_")
        )
    ]
    rows.extend(
        row
        for row in summary.get("final_lodo", []) + summary.get("final_single_dataset", [])
        if row.get("status") == "complete"
    )
    evidence_rows = [
        row
        for row in rows
        if "evidence" in _run_name(row) or "faf" in _run_name(row) or "lean" in _run_name(row)
    ]
    if not evidence_rows:
        _add(gates, "block", "evidence_rows_missing", "No completed EvidenceField/FAF row is available.")
        return
    missing = []
    for row in evidence_rows:
        out = Path(str(row.get("output_dir", "")))
        if not (out / "evidence_maps").exists() or not (out / "evidence_field_audit.json").exists():
            missing.append(_run_name(row))
    if missing:
        _add(gates, "block", "evidence_interpretability_artifacts", "Completed EvidenceField rows lack evidence maps or attention audit artifacts.", missing=missing)
    else:
        _add(gates, "pass", "evidence_interpretability_artifacts", "EvidenceField rows include maps and attention audit artifacts.")


def _gate_evidence_failure_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("runs"):
        _add(
            gates,
            "warn",
            "evidence_failure_report_missing",
            "EvidenceField success/failure analysis report is missing.",
        )
        return
    _add(
        gates,
        "pass",
        "evidence_failure_report",
        "EvidenceField success/failure report exists for quantitative interpretability and figure selection.",
        runs=report.get("num_evidence_runs"),
        examples=len(report.get("examples", [])),
    )


def _gate_config_to_code_trace(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("rows"):
        _add(
            gates,
            "warn",
            "config_to_code_trace_missing",
            "Config-to-code trace report is missing; configured innovation knobs are not implementation-traced.",
        )
        return
    if report.get("num_blocks"):
        failures = [
            row.get("name")
            for row in report.get("rows", [])
            if row.get("num_configured_runs", 0) > 0 and not row.get("source_ok")
        ]
        _add(
            gates,
            "block",
            "config_to_code_trace",
            "At least one configured innovation module lacks a source-code trace.",
            failures=failures,
            blocks=report.get("num_blocks"),
        )
        return
    _add(
        gates,
        "pass",
        "config_to_code_trace",
        "Configured innovation knobs are backed by source-code traces.",
        rows=report.get("num_rows"),
        warnings=report.get("num_warnings"),
    )


def _gate_mask_aware_consistency_smoke(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "mask_aware_consistency_smoke_missing",
            "Mask-aware consistency smoke report is missing; the segmentation-transfer consistency path is not independently smoke-tested.",
        )
        return
    logs = report.get("logs", {}) if isinstance(report.get("logs"), dict) else {}
    required = [
        "loss_aug_consistency",
        "loss_aug_consistency_attention",
        "aug_consistency_attention_mask_mean",
    ]
    missing = [key for key in required if logs.get(key) is None]
    if report.get("status") == "ok" and not missing:
        _add(
            gates,
            "pass",
            "mask_aware_consistency_smoke",
            "Segmentation-style mask-aware attention consistency is executable and logs the masked attention loss.",
            config=report.get("config"),
            attention_loss=logs.get("loss_aug_consistency_attention"),
            mask_mean=logs.get("aug_consistency_attention_mask_mean"),
        )
        return
    _add(
        gates,
        "warn",
        "mask_aware_consistency_smoke",
        "Mask-aware consistency smoke test did not prove the masked attention path.",
        status=report.get("status"),
        missing=missing,
        error=report.get("error"),
    )


def _gate_wet_optical_quality_cues_smoke(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "wet_optical_quality_cues_smoke_missing",
            "Wet optical quality cues smoke report is missing; the wet-road physical-vision path is not independently smoke-tested.",
        )
        return
    checks = report.get("checks", []) if isinstance(report.get("checks"), list) else []
    failed = [row.get("name") for row in checks if not row.get("pass")]
    quality = next((row for row in checks if row.get("name") == "quality_stats_expanded"), {})
    if report.get("status") == "ok" and not failed:
        _add(
            gates,
            "pass",
            "wet_optical_quality_cues_smoke",
            "PhysicsTexture wet optical quality cues are wired, finite, and expand the quality-stat descriptor.",
            base_num_stats=quality.get("base_num_stats"),
            quality_num_stats=quality.get("quality_num_stats"),
        )
        return
    _add(
        gates,
        "block",
        "wet_optical_quality_cues_smoke",
        "Wet optical quality cue smoke test failed.",
        status=report.get("status"),
        failed=failed,
    )


def _gate_goal_evidence_audit(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "goal_evidence_audit_missing",
            "Goal-evidence audit JSON is missing; objective-level evidence tracking is not machine-checkable.",
        )
        return
    requirements = report.get("requirements", []) if isinstance(report.get("requirements"), list) else []
    sections = report.get("sections", {}) if isinstance(report.get("sections"), dict) else {}
    protocol = report.get("protocol_evidence", {}) if isinstance(report.get("protocol_evidence"), dict) else {}
    missing_protocol = [
        name
        for name, item in protocol.items()
        if isinstance(item, dict) and str(item.get("verdict", "missing")) == "missing"
    ]
    if not requirements or not sections:
        _add(
            gates,
            "warn",
            "goal_evidence_audit_incomplete",
            "Goal-evidence audit exists but lacks hard requirements or section counts.",
            requirements=len(requirements),
            sections=sorted(sections),
        )
        return
    if missing_protocol:
        _add(
            gates,
            "warn",
            "goal_evidence_audit_protocol_missing",
            "Goal-evidence audit exists but some protocol-evidence entries are missing.",
            missing_protocol=missing_protocol,
            incomplete_requirements=report.get("incomplete_requirements"),
        )
        return
    _add(
        gates,
        "pass",
        "goal_evidence_audit",
        "Objective-level evidence audit is present and tracks hard requirements, protocol evidence, and next actions.",
        requirements=len(requirements),
        incomplete_requirements=report.get("num_incomplete_requirements"),
        current_run=(report.get("current_execution") or {}).get("name"),
    )


def _gate_objective_completion_audit(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    rows = report.get("requirements", []) if isinstance(report, dict) else []
    counts = report.get("counts", {}) if isinstance(report, dict) else {}
    disallowed = report.get("disallowed_claims", []) if isinstance(report, dict) else []
    if not rows:
        _add(
            gates,
            "warn",
            "objective_completion_audit_missing",
            "Strict objective completion audit is missing; the full user objective is not mapped to current evidence.",
        )
        return
    if len(rows) < 15 or not disallowed:
        _add(
            gates,
            "warn",
            "objective_completion_audit",
            "Objective completion audit exists but does not yet cover enough requirements or claim boundaries.",
            rows=len(rows),
            counts=counts,
            disallowed_claims=len(disallowed),
        )
        return
    _add(
        gates,
        "pass",
        "objective_completion_audit",
        "Strict objective completion audit maps the full objective to completed, partial, configured, and incomplete evidence.",
        rows=len(rows),
        counts=counts,
        disallowed_claims=len(disallowed),
    )


def _gate_artifact_contract(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "artifact_contract_missing",
            "Artifact contract report is missing; per-run evidence completeness is not centrally audited.",
        )
        return
    invalid = int(report.get("num_invalid_complete_like", 0) or 0)
    stale = int(report.get("num_stale_rows", 0) or 0)
    details = {
        "runs": report.get("num_runs"),
        "contract_complete": report.get("num_contract_complete"),
        "contract_incomplete": report.get("num_contract_incomplete"),
        "invalid_complete_like": invalid,
        "stale_rows": stale,
        "verdict": report.get("verdict"),
    }
    if invalid or stale or report.get("verdict") == "block":
        _add(
            gates,
            "block",
            "artifact_contract",
            "Some complete-looking rows have missing or stale evidence artifacts; paper-table claims are not safe.",
            **details,
        )
        return
    _add(
        gates,
        "pass",
        "artifact_contract",
        "Per-run artifact contract is present and no complete-looking rows have stale or missing required evidence.",
        **details,
    )


def _gate_claim_evidence_ledger(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "claim_evidence_ledger_missing",
            "Claim-evidence ledger is missing; paper claims are not centrally mapped to required evidence.",
        )
        return
    rows = report.get("claim_rows", []) if isinstance(report.get("claim_rows"), list) else []
    counts = report.get("status_counts", {}) if isinstance(report.get("status_counts"), dict) else {}
    if not rows:
        _add(
            gates,
            "warn",
            "claim_evidence_ledger_empty",
            "Claim-evidence ledger exists but has no claim rows.",
        )
        return
    _add(
        gates,
        "pass",
        "claim_evidence_ledger",
        "Claim-evidence ledger maps paper-level claims to supported, partial, or not-yet-supported evidence states.",
        claims=len(rows),
        supported=counts.get("supported", 0),
        partial=counts.get("partial", 0),
        not_supported=counts.get("not_supported_yet", 0),
    )


def _gate_quality_mondrian_summary(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "quality_mondrian_summary_missing",
            "Quality-Mondrian post-hoc interval calibration summary is missing.",
        )
        return
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = [
        row
        for row in rows
        if row.get("summary_decision")
        in {"keep_for_interval_calibration", "prefer_safety_checkpoint"}
    ]
    roadsaw_full = [
        row
        for row in supported
        if row.get("run") == "single_roadsaw_full_faf"
        and row.get("probe") == "quality_mondrian_full_cpu"
    ]
    level = "pass" if roadsaw_full else "warn"
    message = (
        "Quality-Mondrian post-hoc calibration has supported interval evidence on RoadSaW/RoadSC."
        if roadsaw_full
        else "Quality-Mondrian post-hoc calibration summary exists but lacks full RoadSaW support."
    )
    _add(
        gates,
        level,
        "quality_mondrian_summary",
        message,
        verdict=report.get("verdict"),
        rows=len(rows),
        supported=[f"{row.get('run')}::{row.get('probe')}" for row in supported],
        claim_boundary=report.get("claim_boundary"),
    )


def _gate_asymmetric_mondrian_summary(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "asymmetric_mondrian_summary_missing",
            "Asymmetric Mondrian post-hoc interval-width summary is missing.",
        )
        return
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = [
        row
        for row in rows
        if row.get("summary_decision") == "keep_for_interval_width_reduction"
    ]
    level = "pass" if supported else "warn"
    message = (
        "Asymmetric Mondrian post-hoc calibration has supported coverage-width evidence."
        if supported
        else "Asymmetric Mondrian summary exists but has no clear coverage-width improvement."
    )
    _add(
        gates,
        level,
        "asymmetric_mondrian_summary",
        message,
        verdict=report.get("verdict"),
        rows=len(rows),
        supported=[f"{row.get('run')}::{row.get('probe')}" for row in supported],
        claim_boundary=report.get("claim_boundary"),
    )


def _gate_region_mixture_summary(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "region_mixture_summary_missing",
            "Segmentation-style region-mixture calibration summary is missing.",
        )
        return
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    supported = [
        row
        for row in rows
        if row.get("summary_decision") == "keep_for_segmentation_style_interval_calibration"
    ]
    level = "pass" if supported else "warn"
    message = (
        "Segmentation-style region-mixture calibration has supported interval evidence."
        if supported
        else "Region-mixture summary exists but lacks a clear coverage-width-worst-slice gain."
    )
    _add(
        gates,
        level,
        "region_mixture_summary",
        message,
        verdict=report.get("verdict"),
        rows=len(rows),
        supported=[f"{row.get('run')}::{row.get('probe')}" for row in supported],
        claim_boundary=report.get("claim_boundary"),
    )


def _gate_checkpoint_divergence(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        _add(
            gates,
            "warn",
            "checkpoint_divergence_report_missing",
            "Checkpoint divergence report is missing; loss-vs-interval selection risk is not audited.",
        )
        return
    verdict = str(report.get("verdict", "missing"))
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    diverged = [row for row in rows if row.get("status") == "diverged_use_safety_for_interval_claims"]
    level = "warn" if diverged else "pass"
    message = (
        "Some loss-selected checkpoints diverge from interval-safety checkpoints; use safety analysis for interval claims."
        if diverged
        else "Checkpoint divergence report shows no major loss-vs-interval conflict."
    )
    _add(
        gates,
        level,
        "checkpoint_divergence_report",
        message,
        verdict=verdict,
        rows=len(rows),
        diverged=[row.get("run") for row in diverged],
    )


def _gate_p1_style_augmentation_candidates(gates: list[dict[str, Any]], algorithm_audit: dict[str, Any]) -> None:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    if not rows:
        _add(gates, "warn", "p1_style_augmentation_audit_missing", "Algorithm module audit is missing; cannot verify style-augmentation candidates.")
        return

    base_aug_required = [
        row
        for row in rows
        if str(row.get("run", "")).startswith(("v", "lodo_", "single_", "baseline_single_", "final_"))
    ]
    missing_base = [
        row.get("run")
        for row in base_aug_required
        if not row.get("modules", {}).get("photometric_jitter")
        or not row.get("modules", {}).get("blur_aug")
        or not row.get("modules", {}).get("random_erasing")
    ]

    fourier_required_prefixes = (
        "v6_full_faf_fourier",
        "v7_full_faf_fourier_dann",
        "v8_full_faf_fourier_roadprior",
        "v9_full_faf_roadsaw_hard_sampling",
        "v10_full_faf_consistency",
        "v11_full_faf_domain_adapter",
        "v12_full_faf_roi_interval_safety",
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
        "final_lodo_",
        "final_single_",
    )
    fourier_required = [
        row
        for row in rows
        if any(str(row.get("run", "")).startswith(prefix) for prefix in fourier_required_prefixes)
    ]
    missing_fourier = [
        row.get("run")
        for row in fourier_required
        if not row.get("modules", {}).get("fourier_style_jitter")
    ]

    if base_aug_required and fourier_required and not missing_base and not missing_fourier:
        _add(
            gates,
            "pass",
            "p1_style_augmentation_candidate_readiness",
            "P1/final shortcut-mitigation candidates expose photometric/blur/erasing and Fourier style augmentation in the audit.",
            fourier_runs=[row.get("run") for row in fourier_required],
        )
    else:
        _add(
            gates,
            "warn",
            "p1_style_augmentation_candidate_readiness",
            "Some P1/final shortcut-mitigation candidates lack auditable photometric/blur/erasing or Fourier style augmentation.",
            missing_base_aug=missing_base,
            missing_fourier=missing_fourier,
        )


def _gate_module_decisions(gates: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    if not decisions:
        _add(gates, "warn", "module_decisions_missing", "Module decision table is missing.")
        return
    bad = [
        row
        for row in decisions
        if row.get("decision") in {"rework_or_remove", "remove_or_rework"}
        and row.get("module") in {"FrictionSet", "DG losses", "EvidenceField aux", "Full fusion"}
    ]
    if bad:
        _add(gates, "warn", "modules_need_rework", "Some modules currently need rework/removal before finalizing the method.", modules=[row.get("module") for row in bad])
    else:
        _add(gates, "pass", "module_decisions", "No completed core module is currently flagged for removal.")


def _gate_p3_interval_candidates(gates: list[dict[str, Any]], algorithm_audit: dict[str, Any]) -> None:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    if not rows:
        _add(gates, "warn", "p3_interval_candidate_audit_missing", "Algorithm module audit is missing; cannot verify P3 interval candidates.")
        return
    required_prefixes = (
        "v12_full_faf_roi_interval_safety",
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
        "final_lodo_",
        "final_single_",
    )
    required = [
        row
        for row in rows
        if any(str(row.get("run", "")).startswith(prefix) for prefix in required_prefixes)
    ]
    missing = [
        row.get("run")
        for row in required
        if not row.get("modules", {}).get("coverage_aware_training")
        or not row.get("modules", {}).get("safety_weighted_coverage")
    ]
    if required and not missing:
        _add(
            gates,
            "pass",
            "p3_interval_candidate_readiness",
            "P3/final interval candidates include coverage-aware and safety-weighted coverage losses.",
            runs=[row.get("run") for row in required],
        )
    else:
        _add(
            gates,
            "warn",
            "p3_interval_candidate_readiness",
            "Some P3/final interval candidates lack coverage-aware or safety-weighted coverage losses.",
            missing=missing,
        )


def _gate_p1_wetness_ordinal_candidates(gates: list[dict[str, Any]], algorithm_audit: dict[str, Any]) -> None:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    if not rows:
        _add(gates, "warn", "p1_wetness_ordinal_audit_missing", "Algorithm module audit is missing; cannot verify wet-state ordinal candidates.")
        return
    required_prefixes = (
        "v9_full_faf_roadsaw_hard_sampling",
        "v10_full_faf_consistency",
        "v11_full_faf_domain_adapter",
        "v12_full_faf_roi_interval_safety",
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
        "final_lodo_",
        "final_single_",
    )
    required = [
        row
        for row in rows
        if any(str(row.get("run", "")).startswith(prefix) for prefix in required_prefixes)
    ]
    missing = [
        row.get("run")
        for row in required
        if not row.get("modules", {}).get("wetness_ordinal_loss")
    ]
    if required and not missing:
        _add(
            gates,
            "pass",
            "p1_wetness_ordinal_candidate_readiness",
            "RoadSaW wet-state candidates include ordinal wetness supervision.",
            runs=[row.get("run") for row in required],
        )
    else:
        _add(
            gates,
            "warn",
            "p1_wetness_ordinal_candidate_readiness",
            "Some RoadSaW/final candidates lack ordinal wetness supervision.",
            missing=missing,
        )


def _gate_p1_conditional_alignment_candidates(gates: list[dict[str, Any]], algorithm_audit: dict[str, Any]) -> None:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    if not rows:
        _add(gates, "warn", "p1_conditional_alignment_audit_missing", "Algorithm module audit is missing; cannot verify conditional alignment candidates.")
        return
    required_runs = {
        "v10_full_faf_consistency",
        "v11_full_faf_domain_adapter",
        "v12_full_faf_roi_interval_safety",
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
        "final_lodo_rscd_lean_road_roi_safety",
        "final_lodo_roadsaw_lean_road_roi_safety",
        "final_lodo_roadsc_lean_road_roi_safety",
        "final_single_rscd_lean_road_roi_safety",
        "final_single_roadsaw_lean_road_roi_safety",
        "final_single_roadsc_lean_road_roi_safety",
    }
    required = [row for row in rows if row.get("run") in required_runs]
    missing = [
        row.get("run")
        for row in required
        if not row.get("modules", {}).get("semantic_conditional_alignment")
    ]
    if len(required) == len(required_runs) and not missing:
        _add(
            gates,
            "pass",
            "p1_conditional_alignment_candidate_readiness",
            "P1/final domain-shift candidates include state-conditioned semantic feature alignment.",
            runs=[row.get("run") for row in required],
        )
    else:
        _add(
            gates,
            "warn",
            "p1_conditional_alignment_candidate_readiness",
            "Some P1/final domain-shift candidates lack state-conditioned semantic feature alignment.",
            missing=missing or sorted(required_runs - {str(row.get("run")) for row in required}),
        )


def _gate_final_method_selection_report(gates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report.get("selection_rule"):
        _add(
            gates,
            "warn",
            "final_method_selection_report_missing",
            "Final method selection report is missing; module pruning/final-route evidence is not centrally audited.",
        )
        return
    verdict = report.get("verdict")
    level = "pass" if verdict == "ready_to_select_final_method" else "warn"
    _add(
        gates,
        level,
        "final_method_selection_report",
        "Final method selection report exists with multi-metric ranking, module decisions, and risk register.",
        verdict=verdict,
        risks=len(report.get("risk_register", [])),
        top_completed=[
            row.get("method")
            for row in report.get("provisional_top_completed", [])[:3]
        ],
    )


def _recommended_next_actions(gates: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    names = {gate["name"] for gate in gates if gate["level"] in {"block", "warn"}}
    actions: list[str] = []
    if "p0_ablation_complete" in names or "full_model_missing" in names:
        actions.append("Finish P0 v4/v5, then refresh summary, module decisions, and top-venue audit.")
    if "lodo_complete" in names or "heldout_roadsaw_missing" in names:
        actions.append("Finish the remaining LODO row; treat the completed held-out RoadSaW failure as the key shortcut/wetness stress-test evidence.")
    if "candidate_path_complete" in names or "dataset_shortcut" in names:
        actions.append("Run v6-v25; prioritize Fourier style jitter, input/color canonicalization, quality-aware wet-road cues, MixStyle feature-statistics mixing, state contrast, weak interval-order consistency, visual-quality uncertainty weighting, ambiguity-aware interval ordering, segmentation-style region-mixture/multi-query evidence, MIC-style masked consistency, condition-aware CORAL, wet-state sampling, consistency, ROI interval safety, and lean Physics+Evidence pruning.")
    if "candidate_hypothesis_matrix_missing" in names or "candidate_hypothesis_matrix" in names:
        actions.append("Refresh the candidate hypothesis matrix so every queued candidate has a predeclared success rule and removal/merge rule.")
    if "fair_comparison_execution_priority_missing" in names or "fair_comparison_execution_priority" in names:
        actions.append("Refresh the execution-priority audit so fair comparisons, RSCD context checks, CV-transfer candidates, and final-method claims stay in the correct order.")
    if "cv_transfer_experiment_protocol_missing" in names or "cv_transfer_experiment_protocol" in names:
        actions.append("Refresh the CV-transfer experiment protocol so each semantic-segmentation-inspired route has source/config/smoke evidence plus a promotion/drop rule.")
    if "wet_slippery_failure_mechanism_report_missing" in names or "wet_slippery_failure_mechanism_report" in names:
        actions.append("Refresh the wet/slippery mechanism map so semantic segmentation, material vision, domain-shift, consistency, interval, and teacher routes each have promotion/drop metrics.")
    if "cv_transfer_candidate_priority_report_missing" in names or "cv_transfer_candidate_priority_report" in names:
        actions.append("Refresh the CV-transfer candidate priority report so fair comparisons, local evidence, material uncertainty, consistency, shortcut control, and teacher routes stay in the intended pruning order.")
    if "cv_transfer_retention_decision_report_missing" in names or "cv_transfer_retention_decision_report" in names:
        actions.append("Refresh the CV-transfer retention decision report so completed candidate metrics are routed into keep, rescue, merge-only, or prune decisions before final-method freeze.")
    if "final_freeze_audit_missing" in names or "final_freeze_audit" in names:
        actions.append("Refresh the final-freeze audit so the final architecture cannot be claimed before fair baselines, candidate metrics, pruning, and final runs are complete.")
    if "p1_style_augmentation_candidate_readiness" in names:
        actions.append("Make shortcut-mitigation augmentation explicit in P1/final configs before rerunning the algorithm audit.")
    if "p1_wetness_ordinal_candidate_readiness" in names:
        actions.append("Enable ordinal wetness supervision for RoadSaW/final candidates before rerunning the algorithm audit.")
    if "p1_conditional_alignment_candidate_readiness" in names:
        actions.append("Enable wetness-conditioned feature alignment for P1 domain-shift candidates before rerunning the algorithm audit.")
    if "fair_single_dataset_missing" in names:
        actions.append("Run matched single-dataset FAF and ConvNeXt baselines for RSCD, RoadSaW, and RoadSC.")
    if "rscd_external_comparison_readiness" in names or "rscd_external_comparison_readiness_missing" in names:
        actions.append("Run the RSCD-27 class-label fast check and formal ConvNeXt baseline before making any RoadFormer/RSCD SOTA-style numeric comparison.")
    if "final_method_complete" in names or "final_heldout_roadsaw_missing" in names or "final_fair_single_dataset_missing" in names:
        actions.append("Run final lean road-ROI safety LODO and matched single-dataset comparisons before making the final method claim.")
    if "raw_interval_coverage" in names:
        actions.append("Use P3 interval candidates to improve raw coverage while tracking calibrated width.")
    if "evidence_rows_missing" in names or "evidence_interpretability_artifacts" in names:
        actions.append("Export evidence maps and quantitative attention audits for completed EvidenceField/FAF runs.")
    if not actions:
        actions.append("Prepare paper tables, bootstrap confidence intervals, failure cases, and final module-pruning decision.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Top-Venue Readiness Gate",
        "",
        f"Summary dir: `{report['summary_dir']}`",
        f"Verdict: `{report['verdict']}`",
        f"Blocks: {report['num_blocks']}; warnings: {report['num_warnings']}",
        "",
        "## Gates",
        "",
        "| Level | Gate | Message | Details |",
        "|---|---|---|---|",
    ]
    for gate in report["gates"]:
        details = {k: v for k, v in gate.items() if k not in {"level", "name", "message"}}
        lines.append(
            "| {level} | {name} | {message} | {details} |".format(
                level=gate["level"],
                name=gate["name"],
                message=gate["message"],
                details=_compact(details),
            )
        )
    lines.extend(["", "## Recommended Next Actions", ""])
    for idx, action in enumerate(report["recommended_next_actions"], start=1):
        lines.append(f"{idx}. {action}")
    lines.append("")
    return "\n".join(lines)


def _add(gates: list[dict[str, Any]], level: str, name: str, message: str, **details: Any) -> None:
    gates.append({"level": level, "name": name, "message": message, **details})


def _delta(cur: dict[str, Any], prev: dict[str, Any], key: str) -> float | None:
    cur_val = _num(cur.get(key))
    prev_val = _num(prev.get(key))
    if cur_val is None or prev_val is None:
        return None
    return cur_val - prev_val


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_name(row: dict[str, Any]) -> str:
    return Path(str(row.get("output_dir", ""))).name.lower()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_json(path: Path) -> Any:
    if not path.exists():
        return None
    files = sorted(path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for item in files:
        try:
            return _load_json(item)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_module_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compact(value: Any) -> str:
    if not value:
        return "-"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > 180:
        return text[:177] + "..."
    return text


if __name__ == "__main__":
    main()
