from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")

RUN_GROUPS = {
    "p0_ablation": [
        "v0_global_only",
        "v1_physics_texture",
        "v2_friction_set",
        "v3_dg_losses",
        "v4_evidence_aux",
        "v5_full_faf",
    ],
    "p1_candidates": [
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
    ],
    "lodo": [
        "lodo_roadsaw_full_faf",
        "lodo_rscd_full_faf",
        "lodo_roadsc_full_faf",
    ],
    "single_dataset_faf": [
        "single_roadsaw_full_faf",
        "single_rscd_full_faf",
        "single_roadsc_full_faf",
    ],
    "single_dataset_baselines": [
        "baseline_single_roadsaw_global_convnext",
        "baseline_single_rscd_global_convnext",
        "baseline_single_roadsc_global_convnext",
    ],
    "final_method_lodo": [
        "final_lodo_roadsaw_lean_road_roi_safety",
        "final_lodo_rscd_lean_road_roi_safety",
        "final_lodo_roadsc_lean_road_roi_safety",
    ],
    "final_method_single_dataset": [
        "final_single_roadsaw_lean_road_roi_safety",
        "final_single_rscd_lean_road_roi_safety",
        "final_single_roadsc_lean_road_roi_safety",
    ],
}

CORE_ARTIFACTS = [
    "best.pt",
    "evaluate_test.json",
    "detailed_test.json",
    "interval_calibration_90.json",
    "bootstrap_metrics.json",
    "topvenue_result_audit.json",
]

SUMMARY_ARTIFACTS = [
    "ablation_table.csv",
    "core_ablation_table.csv",
    "core_ablation_table.md",
    "lodo_table.csv",
    "single_dataset_table.csv",
    "fair_baseline_table.csv",
    "fair_single_dataset_deltas.csv",
    "final_lodo_table.csv",
    "final_single_dataset_table.csv",
    "final_fair_single_dataset_deltas.csv",
    "rule_baseline_table.csv",
    "module_recommendations.csv",
    "final_method_selection_report.md",
    "final_method_selection_report.json",
    "final_method_selection_scores.csv",
    "queue_recovery_report.md",
    "queue_recovery_report.json",
    "dataset_breakdown_table.csv",
    "class_f1_breakdown.csv",
    "paper_protocol_summary.json",
    "paper_protocol_summary.md",
    "open_source_reproducibility_plan.md",
    "open_source_reproducibility_plan.json",
    "paper_tables.tex",
    "paper_tables_index.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY / "protocol_completeness.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY / "protocol_completeness.md")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(root: Path, summary_dir: Path) -> dict[str, Any]:
    groups = {name: [_inspect_run(root, run) for run in runs] for name, runs in RUN_GROUPS.items()}
    group_status = {name: _group_status(rows) for name, rows in groups.items()}
    artifact_contract = _load_json(summary_dir / "artifact_contract_report.json") or {}
    group_status = _merge_artifact_contract_status(group_status, artifact_contract)
    summary_artifacts = {
        name: {
            "exists": (summary_dir / name).exists(),
            "path": str(summary_dir / name),
            "bytes": (summary_dir / name).stat().st_size if (summary_dir / name).exists() else 0,
        }
        for name in SUMMARY_ARTIFACTS
    }
    requirements = [
        _requirement(
            "p0_ablation_complete",
            "All core ablation runs v0-v5 have best checkpoint, test metrics, calibration, and audit.",
            group_status["p0_ablation"]["complete"],
            group_status["p0_ablation"]["missing_runs"],
        ),
        _requirement(
            "lodo_complete",
            "Leave-one-dataset-out runs are complete, especially held-out RoadSaW.",
            group_status["lodo"]["complete"],
            group_status["lodo"]["missing_runs"],
        ),
        _requirement(
            "candidate_path_complete",
            "P1 candidate robustness and lean-final runs v6-v25 are complete.",
            group_status["p1_candidates"]["complete"],
            group_status["p1_candidates"]["missing_runs"],
        ),
        _requirement(
            "fair_single_dataset_complete",
            "Single-dataset FAF and global ConvNeXt baselines are complete for fair comparisons.",
            group_status["single_dataset_faf"]["complete"] and group_status["single_dataset_baselines"]["complete"],
            group_status["single_dataset_faf"]["missing_runs"] + group_status["single_dataset_baselines"]["missing_runs"],
        ),
        _requirement(
            "final_method_complete",
            "Final lean road-ROI safety method has LODO and matched single-dataset evidence.",
            group_status["final_method_lodo"]["complete"] and group_status["final_method_single_dataset"]["complete"],
            group_status["final_method_lodo"]["missing_runs"] + group_status["final_method_single_dataset"]["missing_runs"],
        ),
        _requirement(
            "summary_tables_complete",
            "Summary, module decision, dataset breakdown, and class breakdown tables exist.",
            all(item["exists"] and item["bytes"] > 0 for item in summary_artifacts.values()),
            [name for name, item in summary_artifacts.items() if not item["exists"] or item["bytes"] <= 0],
        ),
    ]
    return {
        "root": str(root),
        "summary_dir": str(summary_dir),
        "overall_complete": all(item["status"] == "complete" for item in requirements),
        "requirements": requirements,
        "groups": groups,
        "group_status": group_status,
        "summary_artifacts": summary_artifacts,
    }


def _merge_artifact_contract_status(
    group_status: dict[str, dict[str, Any]],
    artifact_contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hard = artifact_contract.get("hard_status", {}) if isinstance(artifact_contract, dict) else {}
    if not isinstance(hard, dict):
        return group_status

    mapping = {
        "p0_ablation": ["p0_ablation"],
        "lodo": ["lodo"],
        "single_dataset_faf": ["single_dataset_fair"],
        "single_dataset_baselines": ["single_dataset_fair"],
        "p1_candidates": ["p1_candidates"],
        "final_method_lodo": ["final_method"],
        "final_method_single_dataset": ["final_method"],
    }
    merged = {name: dict(status) for name, status in group_status.items()}
    for group_name, contract_names in mapping.items():
        status = merged.get(group_name)
        if not status:
            continue
        for contract_name in contract_names:
            row = hard.get(contract_name)
            if not isinstance(row, dict):
                continue
            missing = [str(item) for item in row.get("missing", []) or []]
            if group_name == "single_dataset_faf":
                missing = [name for name in missing if name.startswith("single_")]
            elif group_name == "single_dataset_baselines":
                missing = [name for name in missing if name.startswith("baseline_single_")]
            elif group_name == "final_method_lodo":
                missing = [name for name in missing if name.startswith("final_lodo_")]
            elif group_name == "final_method_single_dataset":
                missing = [name for name in missing if name.startswith("final_single_")]
            elif group_name == "p1_candidates":
                missing = [name for name in missing if name.startswith("v")]
            elif group_name == "lodo":
                missing = [name for name in missing if name.startswith("lodo_")]
            elif group_name == "p0_ablation":
                missing = [name for name in missing if name.startswith(("v0_", "v1_", "v2_", "v3_", "v4_", "v5_"))]

            local_missing = [str(name) for name in status.get("missing_runs", []) or []]
            if local_missing:
                missing = sorted(set(missing) | set(local_missing))

            if row.get("complete") is True and not missing:
                status["complete"] = True
                status["num_complete"] = status["num_runs"]
                status["missing_runs"] = []
                status["source"] = "artifact_contract"
            elif missing:
                status["complete"] = False
                status["missing_runs"] = missing
                status["num_complete"] = max(0, status["num_runs"] - len(missing))
                status["source"] = "artifact_contract"
            break
    return merged


def _inspect_run(root: Path, name: str) -> dict[str, Any]:
    run_dir = root / name
    artifacts = {artifact: (run_dir / artifact).exists() for artifact in CORE_ARTIFACTS}
    config = _load_json(run_dir / "config.json")
    scope = _scope_for(name)
    if scope not in {"single_dataset", "single_dataset_baseline"}:
        artifacts["dataset_id_diagnostic.json"] = (run_dir / "dataset_id_diagnostic.json").exists()
    if _uses_evidence(config):
        artifacts["evidence_maps"] = (run_dir / "evidence_maps").exists()
        artifacts["evidence_field_audit.json"] = (run_dir / "evidence_field_audit.json").exists()
        artifacts["evidence_field_audit.md"] = (run_dir / "evidence_field_audit.md").exists()
    stale_reason = _stale_result_reason(run_dir)
    if stale_reason:
        artifacts["fresh_evaluation_artifacts"] = False
    missing = [artifact for artifact, exists in artifacts.items() if not exists]
    state = _load_json(run_dir / "training_state.json")
    return {
        "name": name,
        "path": str(run_dir),
        "scope": scope,
        "status": "complete" if not missing else ("partial" if run_dir.exists() else "missing"),
        "epoch": state.get("epoch") if isinstance(state, dict) else None,
        "epochs": state.get("epochs") if isinstance(state, dict) else None,
        "stale_reason": stale_reason,
        "missing_artifacts": missing,
        "artifacts": artifacts,
    }


def _stale_result_reason(run_dir: Path) -> str | None:
    detailed = run_dir / "detailed_test.json"
    if not detailed.exists():
        return None
    detailed_mtime = detailed.stat().st_mtime
    best = run_dir / "best.pt"
    if best.exists() and best.stat().st_mtime > detailed_mtime + 1:
        return "best_checkpoint_newer_than_detailed_test"
    last = run_dir / "last.pt"
    state = _load_json(run_dir / "training_state.json")
    if last.exists() and isinstance(state, dict):
        epoch = state.get("epoch")
        epochs = state.get("epochs")
        if epoch is None or epochs is None or int(epoch) < int(epochs):
            return "training_or_resume_checkpoint_present_after_detailed_test"
    return None


def _group_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [row["name"] for row in rows if row["status"] != "complete"]
    return {
        "complete": len(missing) == 0,
        "num_runs": len(rows),
        "num_complete": sum(1 for row in rows if row["status"] == "complete"),
        "missing_runs": missing,
    }


def _requirement(name: str, description: str, passed: bool, missing: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "status": "complete" if passed else "incomplete",
        "missing": missing,
    }


def _scope_for(name: str) -> str:
    if name.startswith("baseline_single_"):
        return "single_dataset_baseline"
    if name.startswith("single_") or name.startswith("final_single_"):
        return "single_dataset"
    if name.startswith("lodo_") or name.startswith("final_lodo_"):
        return "lodo"
    if name.startswith("v"):
        return "multi_dataset"
    return "unknown"


def _uses_evidence(config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get("model", {}).get("use_evidence_field", False))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Protocol Completeness", "", f"Root: `{report['root']}`", ""]
    lines.append(f"Overall complete: `{report['overall_complete']}`")
    lines.append("")
    lines.append("| Requirement | Status | Missing |")
    lines.append("|---|---|---|")
    for req in report["requirements"]:
        missing = ", ".join(req["missing"]) if req["missing"] else "-"
        lines.append(f"| {req['name']} | {req['status']} | {missing} |")
    lines.append("")
    lines.append("## Run Groups")
    lines.append("")
    lines.append("| Group | Complete | Runs | Complete Runs | Missing Runs |")
    lines.append("|---|---:|---:|---:|---|")
    for name, item in report["group_status"].items():
        missing = ", ".join(item["missing_runs"]) if item["missing_runs"] else "-"
        lines.append(
            f"| {name} | {item['complete']} | {item['num_runs']} | {item['num_complete']} | {missing} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
