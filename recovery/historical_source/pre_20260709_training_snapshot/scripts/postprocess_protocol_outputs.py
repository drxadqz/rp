from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    manifest_args = _existing_manifest_args(
        [
            "data/manifests_full/rscd_prepared_train.csv",
            "data/manifests_full/rscd_prepared_val.csv",
            "data/manifests_full/rscd_prepared_test.csv",
            "data/manifests_full/roadsaw_train.csv",
            "data/manifests_full/roadsaw_val.csv",
            "data/manifests_full/roadsaw_test.csv",
            "data/manifests_full/roadsc_train.csv",
            "data/manifests_full/roadsc_val.csv",
            "data/manifests_full/roadsc_test.csv",
        ]
    )
    if manifest_args:
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_manifest_labels.py",
                *manifest_args,
                "--out-json",
                str(args.summary_dir / "manifest_label_audit.json"),
                "--out-md",
                str(args.summary_dir / "manifest_label_audit.md"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_dataset_inventory_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "dataset_inventory_report.md"),
            "--out-json",
            str(args.summary_dir / "dataset_inventory_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_dataset_image_style.py",
            "--out-md",
            str(args.summary_dir / "dataset_image_style_audit.md"),
            "--out-json",
            str(args.summary_dir / "dataset_image_style_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_input_canonicalization_style.py",
            "--out-md",
            str(args.summary_dir / "input_canonicalization_style_audit.md"),
            "--out-json",
            str(args.summary_dir / "input_canonicalization_style_audit.json"),
        ],
        check=True,
    )
    ensure_rule_baselines(args.summary_dir)
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_friction_interval_sources.py",
            "--out-json",
            str(args.summary_dir / "friction_interval_source_audit.json"),
            "--out-md",
            str(args.summary_dir / "friction_interval_source_audit.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_friction_interval_claim_matrix.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "friction_interval_claim_matrix.md"),
            "--out-json",
            str(args.summary_dir / "friction_interval_claim_matrix.json"),
        ],
        check=True,
    )

    for run_dir in sorted(path for path in args.root.glob("*") if path.is_dir()):
        detailed = run_dir / "detailed_test.json"
        ensure_history_from_queue_log(run_dir)
        ensure_training_history_summary(run_dir, args.summary_dir)
        if not detailed.exists():
            continue
        ensure_confusion(detailed, run_dir, "friction")
        ensure_confusion(detailed, run_dir, "risk")
        text = detailed.read_text(encoding="utf-8")
        if '"roadsaw"' in text:
            ensure_confusion(detailed, run_dir, "friction", dataset="roadsaw")
            ensure_confusion(detailed, run_dir, "risk", dataset="roadsaw")

    subprocess.run(
        [
            sys.executable,
            "scripts/backfill_bootstrap_metrics.py",
            "--root",
            str(args.root),
            "--num-bootstrap",
            "500",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/backfill_safety_checkpoint_metrics.py",
            "--root",
            str(args.root),
            "--num-bootstrap",
            "300",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/slim_best_checkpoints.py",
            "--root",
            str(args.root),
            "--apply",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_paper_protocol.py",
            "--root",
            str(args.root),
            "--out-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_required_result_tables.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_algorithm_modules.py",
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--out-md",
            str(args.summary_dir / "algorithm_module_audit.md"),
            "--out-json",
            str(args.summary_dir / "algorithm_module_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_config_to_code_trace_report.py",
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--source-root",
            ".",
            "--out-md",
            str(args.summary_dir / "config_to_code_trace_report.md"),
            "--out-json",
            str(args.summary_dir / "config_to_code_trace_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/check_mask_aware_consistency.py",
            "--config",
            "configs/experiments/paper_protocol/v23_lean_region_mixture_evidence_safety.yaml",
            "--out-md",
            str(args.summary_dir / "mask_aware_consistency_smoke.md"),
            "--out-json",
            str(args.summary_dir / "mask_aware_consistency_smoke.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/check_wet_optical_quality_cues.py",
            "--out-md",
            str(args.summary_dir / "wet_optical_quality_cues_smoke.md"),
            "--out-json",
            str(args.summary_dir / "wet_optical_quality_cues_smoke.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_protocol_configs.py",
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--out-md",
            str(args.summary_dir / "protocol_config_audit.md"),
            "--out-json",
            str(args.summary_dir / "protocol_config_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_roadsaw_lodo_protocol.py",
            "--config",
            "configs/experiments/paper_protocol/lodo_roadsaw_full_faf.yaml",
            "--out-md",
            str(args.summary_dir / "roadsaw_lodo_protocol_audit.md"),
            "--out-json",
            str(args.summary_dir / "roadsaw_lodo_protocol_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_fair_comparison_protocol.py",
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--out-md",
            str(args.summary_dir / "fair_comparison_protocol_audit.md"),
            "--out-json",
            str(args.summary_dir / "fair_comparison_protocol_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/backfill_fair_pairwise_comparisons.py",
            "--root",
            str(args.root),
            "--out-dir",
            str(args.summary_dir / "fair_pairwise"),
            "--num-bootstrap",
            "500",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_fair_comparison_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "fair_comparison_report.md"),
            "--out-json",
            str(args.summary_dir / "fair_comparison_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_p0_claim_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "p0_claim_report.md"),
            "--out-json",
            str(args.summary_dir / "p0_claim_report.json"),
            "--out-csv",
            str(args.summary_dir / "p0_claim_deltas.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_paper_p0_ablation_table.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "paper_p0_ablation_table.md"),
            "--out-csv",
            str(args.summary_dir / "paper_p0_ablation_table.csv"),
            "--out-tex",
            str(args.summary_dir / "paper_p0_ablation_table.tex"),
            "--out-json",
            str(args.summary_dir / "paper_p0_ablation_table.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_lodo_generalization_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "lodo_generalization_report.md"),
            "--out-json",
            str(args.summary_dir / "lodo_generalization_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_external_benchmark_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "external_benchmark_report.md"),
            "--out-json",
            str(args.summary_dir / "external_benchmark_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_direct_friction_public_benchmark_audit.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "direct_friction_public_benchmark_audit.md"),
            "--out-json",
            str(args.summary_dir / "direct_friction_public_benchmark_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_rscd_external_comparison_readiness.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "rscd_external_comparison_readiness.md"),
            "--out-json",
            str(args.summary_dir / "rscd_external_comparison_readiness.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_open_source_reproducibility_plan.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "open_source_reproducibility_plan.md"),
            "--out-json",
            str(args.summary_dir / "open_source_reproducibility_plan.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_safety_selection_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "safety_selection_report.md"),
            "--out-json",
            str(args.summary_dir / "safety_selection_report.json"),
        ],
        check=True,
    )
    ensure_live_training_trend("v5_full_faf", args.summary_dir)
    subprocess.run(
        [
            sys.executable,
            "scripts/write_live_training_diagnosis.py",
            "--run",
            "v5_full_faf",
            "--summary-dir",
            str(args.summary_dir),
            "--out-json",
            str(args.summary_dir / "v5_full_faf_training_diagnosis.json"),
            "--out-md",
            str(args.summary_dir / "v5_full_faf_training_diagnosis.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_checkpoint_policy_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "checkpoint_policy_report.md"),
            "--out-json",
            str(args.summary_dir / "checkpoint_policy_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_evidence_failure_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "evidence_failure_report.md"),
            "--out-json",
            str(args.summary_dir / "evidence_failure_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_interval_quality_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "interval_quality_report.md"),
            "--out-json",
            str(args.summary_dir / "interval_quality_report.json"),
            "--out-csv",
            str(args.summary_dir / "interval_quality_cells.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_quality_mondrian_summary.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "quality_mondrian_summary.md"),
            "--out-json",
            str(args.summary_dir / "quality_mondrian_summary.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_asymmetric_mondrian_summary.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "asymmetric_mondrian_summary.md"),
            "--out-json",
            str(args.summary_dir / "asymmetric_mondrian_summary.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_region_mixture_summary.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "region_mixture_summary.md"),
            "--out-json",
            str(args.summary_dir / "region_mixture_summary.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_checkpoint_divergence_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "checkpoint_divergence_report.md"),
            "--out-json",
            str(args.summary_dir / "checkpoint_divergence_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_wetness_state_report.py",
            "--root",
            str(args.root),
            "--out-md",
            str(args.summary_dir / "wetness_state_report.md"),
            "--out-json",
            str(args.summary_dir / "wetness_state_report.json"),
            "--out-csv",
            str(args.summary_dir / "wetness_state_report.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_dataset_shortcut_report.py",
            "--root",
            str(args.root),
            "--out-md",
            str(args.summary_dir / "dataset_shortcut_report.md"),
            "--out-json",
            str(args.summary_dir / "dataset_shortcut_report.json"),
            "--out-csv",
            str(args.summary_dir / "dataset_shortcut_report.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_quality_domain_diagnostic_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "quality_domain_diagnostic_report.md"),
            "--out-json",
            str(args.summary_dir / "quality_domain_diagnostic_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/decide_modules.py",
            "--summary-json",
            str(args.summary_dir / "paper_protocol_summary.json"),
            "--out-md",
            str(args.summary_dir / "module_decisions.md"),
            "--out-csv",
            str(args.summary_dir / "module_decisions.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_final_method_selection_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "final_method_selection_report.md"),
            "--out-json",
            str(args.summary_dir / "final_method_selection_report.json"),
            "--out-csv",
            str(args.summary_dir / "final_method_selection_scores.csv"),
        ],
        check=True,
    )
    ensure_live_training_diagnostics(args.root, args.summary_dir)
    subprocess.run(
        [
            sys.executable,
            "scripts/write_module_retention_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "module_retention_report.md"),
            "--out-json",
            str(args.summary_dir / "module_retention_report.json"),
            "--out-csv",
            str(args.summary_dir / "module_retention_report.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_topvenue_innovation_roadmap.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "topvenue_innovation_roadmap.md"),
            "--out-json",
            str(args.summary_dir / "topvenue_innovation_roadmap.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_candidate_hypothesis_matrix.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "candidate_hypothesis_matrix.md"),
            "--out-json",
            str(args.summary_dir / "candidate_hypothesis_matrix.json"),
            "--out-csv",
            str(args.summary_dir / "candidate_hypothesis_matrix.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_candidate_implementation_coverage_audit.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "candidate_implementation_coverage_audit.md"),
            "--out-json",
            str(args.summary_dir / "candidate_implementation_coverage_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_candidate_pruning_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "candidate_pruning_report.md"),
            "--out-json",
            str(args.summary_dir / "candidate_pruning_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_online_source_refresh_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "online_source_refresh_report.md"),
            "--out-json",
            str(args.summary_dir / "online_source_refresh_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_queue_recovery_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-md",
            str(args.summary_dir / "queue_recovery_report.md"),
            "--out-json",
            str(args.summary_dir / "queue_recovery_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_next_queue_readiness_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "next_queue_readiness_report.md"),
            "--out-json",
            str(args.summary_dir / "next_queue_readiness_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_fair_comparison_execution_priority.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "fair_comparison_execution_priority.md"),
            "--out-json",
            str(args.summary_dir / "fair_comparison_execution_priority.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_cv_transfer_experiment_protocol.py",
            "--summary-dir",
            str(args.summary_dir),
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--source-root",
            ".",
            "--out-md",
            str(args.summary_dir / "cv_transfer_experiment_protocol.md"),
            "--out-json",
            str(args.summary_dir / "cv_transfer_experiment_protocol.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_wet_slippery_failure_mechanism_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "wet_slippery_failure_mechanism_report.md"),
            "--out-json",
            str(args.summary_dir / "wet_slippery_failure_mechanism_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_cv_transfer_candidate_priority_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "cv_transfer_candidate_priority_report.md"),
            "--out-json",
            str(args.summary_dir / "cv_transfer_candidate_priority_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_cv_transfer_retention_decision_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "cv_transfer_retention_decision_report.md"),
            "--out-json",
            str(args.summary_dir / "cv_transfer_retention_decision_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_final_freeze_audit.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "final_freeze_audit.md"),
            "--out-json",
            str(args.summary_dir / "final_freeze_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_handoff_health_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-md",
            str(args.summary_dir / "handoff_health_report.md"),
            "--out-json",
            str(args.summary_dir / "handoff_health_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_runtime_guard_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-md",
            str(args.summary_dir / "runtime_guard_report.md"),
            "--out-json",
            str(args.summary_dir / "runtime_guard_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_paper_protocol.py",
            "--root",
            str(args.root),
            "--out-dir",
            str(args.summary_dir / "audit"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/check_protocol_completeness.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_artifact_contract_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--config-dir",
            "configs/experiments/paper_protocol",
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-md",
            str(args.summary_dir / "artifact_contract_report.md"),
            "--out-json",
            str(args.summary_dir / "artifact_contract_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_active_live_training_reports.py",
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-json",
            str(args.summary_dir / "active_live_training_reports.json"),
            "--out-md",
            str(args.summary_dir / "active_live_training_reports.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_goal_evidence_audit.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out",
            str(args.summary_dir / "goal_evidence_audit.md"),
            "--out-json",
            str(args.summary_dir / "goal_evidence_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_objective_completion_audit.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "objective_completion_audit.md"),
            "--out-json",
            str(args.summary_dir / "objective_completion_audit.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_reviewer_action_matrix.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "reviewer_action_matrix.md"),
            "--out-json",
            str(args.summary_dir / "reviewer_action_matrix.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_reviewer_evidence_checklist.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "reviewer_evidence_checklist.md"),
            "--out-json",
            str(args.summary_dir / "reviewer_evidence_checklist.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_claim_evidence_ledger.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "claim_evidence_ledger.md"),
            "--out-json",
            str(args.summary_dir / "claim_evidence_ledger.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_latex_paper_tables.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-tex",
            str(args.summary_dir / "paper_tables.tex"),
            "--out-md",
            str(args.summary_dir / "paper_tables_index.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/topvenue_readiness_gate.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "topvenue_readiness_gate.md"),
            "--out-json",
            str(args.summary_dir / "topvenue_readiness_gate.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_current_algorithm_gap_analysis.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "current_algorithm_gap_analysis.md"),
            "--out-json",
            str(args.summary_dir / "current_algorithm_gap_analysis.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_required_result_tables.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_latex_paper_tables.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-tex",
            str(args.summary_dir / "paper_tables.tex"),
            "--out-md",
            str(args.summary_dir / "paper_tables_index.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_fast_screen_status_report.py",
            "--config-dir",
            "configs/experiments/fast_screen",
            "--log-dir",
            "outputs/fast_screen_queue",
            "--out-md",
            str(args.summary_dir / "fast_screen_status_report.md"),
            "--out-json",
            str(args.summary_dir / "fast_screen_status_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_fast_to_formal_promotion_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "fast_to_formal_promotion_report.md"),
            "--out-json",
            str(args.summary_dir / "fast_to_formal_promotion_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_fail_fast_exploration_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "fail_fast_exploration_report.md"),
            "--out-json",
            str(args.summary_dir / "fail_fast_exploration_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_active_training_watch_report.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-md",
            str(args.summary_dir / "active_training_watch_report.md"),
            "--out-json",
            str(args.summary_dir / "active_training_watch_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_experiment_dashboard.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-json",
            str(args.summary_dir / "experiment_status_dashboard.json"),
            "--out-md",
            str(args.summary_dir / "experiment_status_dashboard.md"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_online_source_refresh_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "online_source_refresh_report.md"),
            "--out-json",
            str(args.summary_dir / "online_source_refresh_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_reviewer_action_matrix.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "reviewer_action_matrix.md"),
            "--out-json",
            str(args.summary_dir / "reviewer_action_matrix.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_reviewer_evidence_checklist.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "reviewer_evidence_checklist.md"),
            "--out-json",
            str(args.summary_dir / "reviewer_evidence_checklist.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_next_experiment_decision_report.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "next_experiment_decision_report.md"),
            "--out-json",
            str(args.summary_dir / "next_experiment_decision_report.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/topvenue_readiness_gate.py",
            "--summary-dir",
            str(args.summary_dir),
            "--out-md",
            str(args.summary_dir / "topvenue_readiness_gate.md"),
            "--out-json",
            str(args.summary_dir / "topvenue_readiness_gate.json"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_experiment_dashboard.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
            "--log-dir",
            "outputs/paper_protocol_queue",
            "--out-json",
            str(args.summary_dir / "experiment_status_dashboard.json"),
            "--out-md",
            str(args.summary_dir / "experiment_status_dashboard.md"),
        ],
        check=True,
    )
    shutil.copyfile(
        args.summary_dir / "experiment_status_dashboard.json",
        args.summary_dir / "experiment_dashboard.json",
    )
    shutil.copyfile(
        args.summary_dir / "experiment_status_dashboard.md",
        args.summary_dir / "experiment_dashboard.md",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_followup_watcher_report.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_current_remaining_reports.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_live_research_route_update.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_live_route_update_automation.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        check=True,
    )


def ensure_confusion(detailed: Path, run_dir: Path, task: str, dataset: str | None = None) -> None:
    suffix = f"_{dataset}" if dataset else "_overall"
    out_csv = run_dir / f"confusion_{task}{suffix}.csv"
    out_md = run_dir / f"confusion_{task}{suffix}.md"
    if out_csv.exists() and out_md.exists():
        return
    cmd = [
        sys.executable,
        "scripts/summarize_confusions.py",
        "--detailed",
        str(detailed),
        "--task",
        task,
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
    ]
    if dataset:
        cmd.extend(["--dataset", dataset])
    subprocess.run(cmd, check=True)


def ensure_history_from_queue_log(run_dir: Path) -> None:
    history = run_dir / "metrics_history.json"
    log_dir = Path("outputs/paper_protocol_queue")
    if not log_dir.exists():
        return
    candidates = sorted(
        log_dir.glob(f"{run_dir.name}_*.out.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        ensure_history_from_tensorboard(run_dir)
        return
    existing_count = _history_epoch_count(history)
    for log_path in candidates:
        if not _log_looks_like_training_history(log_path):
            continue
        if history.exists() and history.stat().st_mtime >= log_path.stat().st_mtime and existing_count > 0:
            break
        with tempfile.TemporaryDirectory(prefix="history_extract_", dir=run_dir) as tmp:
            tmp_dir = Path(tmp)
            tmp_json = tmp_dir / "metrics_history.json"
            tmp_csv = tmp_dir / "metrics_history.csv"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/extract_training_history_from_log.py",
                    "--log",
                    str(log_path),
                    "--out-json",
                    str(tmp_json),
                    "--out-csv",
                    str(tmp_csv),
                ],
                check=True,
            )
            parsed_count = _history_epoch_count(tmp_json)
            if parsed_count <= 0:
                continue
            if existing_count > parsed_count:
                return
            tmp_json.replace(history)
            tmp_csv.replace(run_dir / "metrics_history.csv")
            existing_count = parsed_count
            break
    ensure_history_from_tensorboard(run_dir, min_existing_epochs=existing_count)


def ensure_history_from_tensorboard(run_dir: Path, min_existing_epochs: int | None = None) -> None:
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return
    history = run_dir / "metrics_history.json"
    existing_count = _history_epoch_count(history) if min_existing_epochs is None else min_existing_epochs
    with tempfile.TemporaryDirectory(prefix="tb_history_extract_", dir=run_dir) as tmp:
        tmp_dir = Path(tmp)
        tmp_json = tmp_dir / "metrics_history.json"
        tmp_csv = tmp_dir / "metrics_history.csv"
        try:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recover_training_history_from_tensorboard.py",
                    "--tb-dir",
                    str(tb_dir),
                    "--run-dir",
                    str(run_dir),
                    "--out-json",
                    str(tmp_json),
                    "--out-csv",
                    str(tmp_csv),
                ],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        parsed_count = _history_epoch_count(tmp_json)
        if parsed_count <= existing_count:
            return
        tmp_json.replace(history)
        tmp_csv.replace(run_dir / "metrics_history.csv")


def _history_epoch_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, list):
        return 0
    return sum(1 for row in payload if isinstance(row, dict) and row.get("epoch") is not None)


def _log_looks_like_training_history(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "Epoch " in text and ("train:" in text or "val  :" in text or "val:" in text)


def ensure_training_history_summary(run_dir: Path, summary_dir: Path) -> None:
    history = run_dir / "metrics_history.json"
    if not history.exists():
        return
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_training_history.py",
            "--history",
            str(history),
            "--run-name",
            run_dir.name,
            "--out-md",
            str(summary_dir / f"{run_dir.name}_training_history.md"),
            "--out-json",
            str(summary_dir / f"{run_dir.name}_training_history.json"),
        ],
        check=True,
    )


def ensure_live_training_trend(run_name: str, summary_dir: Path) -> None:
    log_dir = Path("outputs/paper_protocol_queue")
    if not log_dir.exists():
        return
    candidates = sorted(
        log_dir.glob(f"{run_name}_*.out.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return
    log_path = candidates[0]
    cmd = [
        sys.executable,
        "scripts/extract_training_log_metrics.py",
        "--log",
        str(log_path),
        "--run",
        run_name,
        "--out-json",
        str(summary_dir / f"{run_name}_live_training_trend.json"),
        "--out-md",
        str(summary_dir / f"{run_name}_live_training_trend.md"),
    ]
    err_path = log_path.with_name(log_path.name[:-8] + ".err.log")
    if err_path.exists():
        cmd.extend(["--err-log", str(err_path)])
    subprocess.run(cmd, check=True)


def ensure_live_training_diagnostics(root: Path, summary_dir: Path) -> None:
    run_names = {"v5_full_faf"}
    if root.exists():
        run_names.update(path.name for path in root.iterdir() if path.is_dir())
    for run_name in sorted(run_names):
        ensure_live_training_trend(run_name, summary_dir)
        trend = summary_dir / f"{run_name}_live_training_trend.json"
        if not trend.exists():
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/write_live_training_diagnosis.py",
                "--run",
                run_name,
                "--summary-dir",
                str(summary_dir),
                "--out-json",
                str(summary_dir / f"{run_name}_training_diagnosis.json"),
                "--out-md",
                str(summary_dir / f"{run_name}_training_diagnosis.md"),
            ],
            check=True,
        )


def ensure_rule_baselines(summary_dir: Path) -> None:
    specs = [
        (
            "rscd",
            ["data/manifests_full/rscd_prepared_train.csv"],
            ["data/manifests_full/rscd_prepared_test.csv"],
        ),
        (
            "roadsaw",
            ["data/manifests_full/roadsaw_train.csv"],
            ["data/manifests_full/roadsaw_test.csv"],
        ),
        (
            "roadsc",
            ["data/manifests_full/roadsc_train.csv"],
            ["data/manifests_full/roadsc_test.csv"],
        ),
    ]
    for dataset, train_manifests, eval_manifests in specs:
        out = summary_dir / f"rule_baseline_{dataset}_test.json"
        cmd = [sys.executable, "scripts/rule_baseline.py"]
        for manifest in train_manifests:
            if not Path(manifest).exists():
                return
            cmd.extend(["--train-manifest", manifest])
        for manifest in eval_manifests:
            if not Path(manifest).exists():
                return
            cmd.extend(["--eval-manifest", manifest])
        cmd.extend(["--out", str(out)])
        subprocess.run(cmd, check=True)


def _existing_manifest_args(paths: list[str]) -> list[str]:
    args: list[str] = []
    for item in paths:
        path = Path(item)
        if path.exists():
            args.extend(["--manifest", str(path)])
    return args


if __name__ == "__main__":
    main()
