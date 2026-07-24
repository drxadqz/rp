from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "candidate_implementation_coverage_audit.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "candidate_implementation_coverage_audit.json"


MODULE_TRACE_ALIASES = {
    "physics_texture": "physics_texture_branch",
    "physics_quality_cues": "physics_quality_cues",
    "photometric_jitter": "photometric_style_augmentation",
    "grayscale_aug": "photometric_style_augmentation",
    "blur_aug": "photometric_style_augmentation",
    "random_erasing": "photometric_style_augmentation",
    "fourier_style_jitter": "fourier_low_frequency_style_jitter",
    "bottom_square_input_canonicalization": "bottom_square_input_canonicalization",
    "gray_world_color_constancy": "gray_world_color_constancy",
    "dann": "domain_adversarial_training",
    "semantic_conditional_alignment": "condition_aware_coral_alignment",
    "risk_conditional_coral": "condition_aware_coral_alignment",
    "wetness_conditional_coral": "condition_aware_coral_alignment",
    "state_contrastive_alignment": "state_contrastive_alignment",
    "interval_order_consistency": "interval_order_consistency",
    "domain_adapter": "domain_specific_adapter",
    "domain_adapter_regularized": "domain_specific_adapter",
    "feature_mixstyle": "feature_mixstyle_shortcut_probe",
    "condition_hard_sampling": "condition_hard_sampling",
    "dataset_scoped_sampling": "condition_hard_sampling",
    "evidence_field": "evidence_field",
    "evidence_final_mix": "evidence_field",
    "road_likelihood_prior": "road_likelihood_prior",
    "region_mixture_evidence": "region_mixture_evidence",
    "multi_query_evidence": "multi_query_evidence",
    "query_disagreement_uncertainty": "query_disagreement_uncertainty",
    "pseudo_road_mask_supervision": "pseudo_road_mask_attention",
    "roi_attention_constraint": "bottom_roi_attention_constraint",
    "weak_view_consistency": "weak_view_consistency",
    "mask_aware_consistency": "mask_aware_consistency",
    "coverage_aware_training": "coverage_aware_interval_training",
    "safety_weighted_coverage": "safety_weighted_coverage",
    "visual_quality_weighted_coverage": "visual_quality_weighted_coverage",
    "wetness_ordinal_loss": "wetness_ordinal_supervision",
    "friction_set": "friction_set_interval_expansion",
    "dg_losses": "condition_aware_coral_alignment",
}


ROUTES: list[dict[str, Any]] = [
    {
        "name": "P0 physics texture signal",
        "phase": "P0",
        "topvenue_pattern": "Use a domain prior only if the same protocol shows measurable benefit.",
        "modules": ["physics_texture"],
        "runs": ["v1_physics_texture"],
        "decision_rule": "Keep as a core module if it improves risk F1, low-friction recall, and worst-dataset F1 over Global-only.",
        "next_if_pending": "Already complete; use as the current strongest evidence-supported module.",
    },
    {
        "name": "P0 FrictionSet interval expansion",
        "phase": "P0/P3",
        "topvenue_pattern": "Latent set/mixture heads must improve calibration without hiding errors by widening intervals.",
        "modules": ["friction_set"],
        "runs": ["v2_friction_set"],
        "decision_rule": "Keep or merge only if calibrated coverage-width and worst-dataset F1 improve under later candidates.",
        "next_if_pending": "Treat as provisional; current P0 evidence does not justify a standalone final module.",
    },
    {
        "name": "P0 generic DG losses",
        "phase": "P0/P1",
        "topvenue_pattern": "Robust optimization must help the actual OOD metric, not only look theoretically motivated.",
        "modules": ["dg_losses", "semantic_conditional_alignment", "risk_conditional_coral"],
        "runs": ["v3_dg_losses"],
        "decision_rule": "Remove generic DG losses unless condition-aware variants improve LODO or shortcut probes.",
        "next_if_pending": "Use v10-v16 to test condition-aware alignment rather than retaining generic DG by default.",
    },
    {
        "name": "P0 EvidenceField auxiliary branch",
        "phase": "P0/P2",
        "topvenue_pattern": "An interpretable local evidence branch needs both metric value and grounded attention evidence.",
        "modules": ["evidence_field", "evidence_final_mix"],
        "runs": ["v4_evidence_aux"],
        "decision_rule": "Keep only if ROI/consistency candidates improve evidence grounding or safety metrics.",
        "next_if_pending": "Use v8/v10/v12/v14-v24 to decide whether EvidenceField becomes final or supplemental.",
    },
    {
        "name": "P1 style and input canonicalization",
        "phase": "P1",
        "topvenue_pattern": "Reduce dataset shortcut by destroying camera/style cues while preserving road texture and wetness evidence.",
        "modules": [
            "photometric_jitter",
            "grayscale_aug",
            "blur_aug",
            "random_erasing",
            "fourier_style_jitter",
            "bottom_square_input_canonicalization",
            "gray_world_color_constancy",
            "feature_mixstyle",
            "interval_order_consistency",
            "visual_quality_weighted_coverage",
        ],
        "runs": [
            "v6_full_faf_fourier",
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
        ],
        "decision_rule": "Promote only if dataset-ID balanced accuracy drops or LODO/worst-dataset metrics improve without damaging wet-state cues.",
        "next_if_pending": "Run fast-screen or formal v6/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 after current GPU queue frees.",
    },
    {
        "name": "P1 conditional domain alignment and adapters",
        "phase": "P1",
        "topvenue_pattern": "Align domains conditionally so true road-state differences are not collapsed into one averaged feature space.",
        "modules": [
            "semantic_conditional_alignment",
            "risk_conditional_coral",
            "wetness_conditional_coral",
            "state_contrastive_alignment",
            "interval_order_consistency",
            "dann",
            "domain_adapter",
            "domain_adapter_regularized",
            "condition_hard_sampling",
            "dataset_scoped_sampling",
        ],
        "runs": [
            "v7_full_faf_fourier_dann",
            "v9_full_faf_roadsaw_hard_sampling",
            "v10_full_faf_consistency",
            "v11_full_faf_domain_adapter",
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
        ],
        "decision_rule": "Keep DANN/adapters/sampling only with lower shortcut and stable low-friction recall; otherwise prune them.",
        "next_if_pending": "Run v7/v9/v10/v11 and compare against v6 and v1/v5.",
    },
    {
        "name": "P2 road-grounded EvidenceField",
        "phase": "P2",
        "topvenue_pattern": "Turn interpretability into a falsifiable local-evidence hypothesis with ROI, pseudo-road masks, and consistency.",
        "modules": [
            "evidence_field",
            "road_likelihood_prior",
            "region_mixture_evidence",
            "pseudo_road_mask_supervision",
            "roi_attention_constraint",
            "weak_view_consistency",
            "mask_aware_consistency",
        ],
        "runs": [
            "v8_full_faf_fourier_roadprior",
            "v10_full_faf_consistency",
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
        ],
        "decision_rule": "Keep if attention moves to plausible road/contact evidence and safety metrics do not regress.",
        "next_if_pending": "Use evidence failure reports and attention mass metrics after candidate runs finish.",
    },
    {
        "name": "P2 segmentation-transfer region mixture and mask queries",
        "phase": "P2/P3",
        "topvenue_pattern": "Transfer segmentation reasoning by modeling local visual regions, material mixtures, and mask-query evidence without requiring pixel labels.",
        "modules": [
            "evidence_field",
            "region_mixture_evidence",
            "multi_query_evidence",
            "query_disagreement_uncertainty",
            "road_likelihood_prior",
            "coverage_aware_training",
            "visual_quality_weighted_coverage",
        ],
        "runs": [
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
        ],
        "decision_rule": "Promote only if region-mixture or multi-query evidence improves quality-slice or conditional interval coverage at bounded width.",
        "next_if_pending": "Run v23 then v24 fast-screen; keep v24 only if it beats v23 on wet/snow/low-texture slices or calibrated coverage-width tradeoff.",
    },
    {
        "name": "P3 safety interval quality",
        "phase": "P3",
        "topvenue_pattern": "Interval methods must report coverage and width together, especially for high-risk/wet/snow cells.",
        "modules": [
            "coverage_aware_training",
            "safety_weighted_coverage",
            "visual_quality_weighted_coverage",
            "wetness_ordinal_loss",
            "friction_set",
            "interval_order_consistency",
        ],
        "runs": [
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
        ],
        "decision_rule": "Promote only if conditional undercoverage drops without excessive calibrated-width inflation.",
        "next_if_pending": "Run conditional interval reports for all P3 candidates and compare coverage-width tradeoffs.",
    },
    {
        "name": "P1/P3 wet-road quality physics",
        "phase": "P1/P3",
        "topvenue_pattern": "Borrow physics of wet reflection, water films, texture loss, and snow brightness to build testable visual cues.",
        "modules": [
            "physics_texture",
            "physics_quality_cues",
            "gray_world_color_constancy",
            "bottom_square_input_canonicalization",
            "coverage_aware_training",
            "safety_weighted_coverage",
            "visual_quality_weighted_coverage",
        ],
        "runs": [
            "v17_lean_quality_physics_safety",
            "v18_lean_mixstyle_quality_safety",
            "v19_lean_state_contrast_quality_safety",
            "v20_lean_interval_order_quality_safety",
            "v21_lean_quality_uncertainty_safety",
            "v22_lean_quality_order_contrast_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
        ],
        "decision_rule": "Keep only if near-white/wet/low-texture slices improve without increasing dataset shortcut.",
        "next_if_pending": "v17/v18/v19/v20/v21/v22/v23/v24 are queued for fast-screen after the current GPU chain; inspect quality-slice and shortcut deltas before promoting.",
    },
    {
        "name": "Fair single-dataset comparisons",
        "phase": "Fair baseline",
        "topvenue_pattern": "Top-venue claims need same-split, same-backbone, same-label baselines before external-number comparisons.",
        "modules": ["physics_texture", "evidence_field", "coverage_aware_training"],
        "runs": [
            "single_roadsaw_full_faf",
            "single_rscd_full_faf",
            "single_roadsc_full_faf",
            "baseline_single_roadsaw_global_convnext",
            "baseline_single_rscd_global_convnext",
            "baseline_single_roadsc_global_convnext",
        ],
        "decision_rule": "Use paired bootstrap deltas; do not claim SOTA if ConvNeXt wins under the same protocol.",
        "next_if_pending": "Run these rows before making any numerical superiority claim.",
    },
    {
        "name": "Final lean method evidence",
        "phase": "Final",
        "topvenue_pattern": "Freeze a lean method only after ablations, candidate evidence, LODO, and fair baselines agree.",
        "modules": [
            "physics_texture",
            "semantic_conditional_alignment",
            "evidence_field",
            "road_likelihood_prior",
            "pseudo_road_mask_supervision",
            "roi_attention_constraint",
            "weak_view_consistency",
            "mask_aware_consistency",
            "coverage_aware_training",
            "safety_weighted_coverage",
            "wetness_ordinal_loss",
        ],
        "runs": [
            "final_lodo_roadsaw_lean_road_roi_safety",
            "final_lodo_rscd_lean_road_roi_safety",
            "final_lodo_roadsc_lean_road_roi_safety",
            "final_single_roadsaw_lean_road_roi_safety",
            "final_single_rscd_lean_road_roi_safety",
            "final_single_roadsc_lean_road_roi_safety",
        ],
        "decision_rule": "Final method is claimable only with completed final LODO and final same-split ConvNeXt comparisons.",
        "next_if_pending": "Wait for candidate ranking; then run the final queue serially.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    trace = _load_json(summary_dir / "config_to_code_trace_report.json") or {}
    algorithm = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    candidate = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}

    trace_rows = {row.get("name"): row for row in trace.get("rows", []) if isinstance(row, dict)}
    algorithm_sources = algorithm.get("source_implementation", {}) if isinstance(algorithm, dict) else {}
    algorithm_rows = algorithm.get("rows", []) if isinstance(algorithm.get("rows"), list) else []
    candidate_rows = {row.get("run"): row for row in candidate.get("rows", []) if isinstance(row, dict)}
    p0_by_run = _p0_by_run(summary)

    rows = [
        _route_row(
            route,
            trace_rows=trace_rows,
            algorithm_sources=algorithm_sources,
            algorithm_rows=algorithm_rows,
            candidate_rows=candidate_rows,
            p0_by_run=p0_by_run,
            summary=summary,
        )
        for route in ROUTES
    ]
    source_gaps = [row for row in rows if row["source_gap_modules"]]
    pending = [row for row in rows if row["status"] in {"configured_waiting_for_results", "partial_results"}]
    complete = [row for row in rows if row["status"] == "evidence_complete"]
    verdict = "block" if source_gaps else "pass_with_pending_results" if pending else "pass"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "counts": {
            "routes": len(rows),
            "evidence_complete": len(complete),
            "pending_or_partial": len(pending),
            "source_gap": len(source_gaps),
        },
        "rows": rows,
        "current_failure_context": {
            "dataset_shortcut_verdict": shortcut.get("verdict"),
            "high_shortcut_rows": shortcut.get("num_high_shortcut"),
            "interval_watchlist_items": len(interval.get("watchlist", [])) if isinstance(interval.get("watchlist"), list) else None,
        },
        "claim_rules": [
            "Implementation coverage only proves that a candidate can run; metric claims still require completed formal runs.",
            "A module is kept only if it improves a same-split paired delta, LODO transfer, low-friction recall, coverage-width tradeoff, or evidence grounding.",
            "External paper numbers are not used as direct wins unless labels, splits, preprocessing, and metrics match.",
            "Weak public road-condition labels support friction-affordance intervals, not measured tire-road friction regression.",
        ],
        "next_order": _next_order(rows),
    }


def _route_row(
    route: dict[str, Any],
    *,
    trace_rows: dict[str, dict[str, Any]],
    algorithm_sources: dict[str, Any],
    algorithm_rows: list[dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    p0_by_run: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    module_rows = [
        _module_status(module, trace_rows, algorithm_sources, algorithm_rows)
        for module in route["modules"]
    ]
    source_gap_modules = [
        row["module"]
        for row in module_rows
        if row["source_status"] not in {"trace_pass", "algorithm_source_pass"}
    ]
    run_rows = [_run_status(run, candidate_rows, p0_by_run, summary) for run in route["runs"]]
    complete = [row for row in run_rows if row["status"] == "complete"]
    missing = [row for row in run_rows if row["status"] != "complete"]
    if source_gap_modules:
        status = "implementation_gap"
    elif not missing:
        status = "evidence_complete"
    elif complete:
        status = "partial_results"
    else:
        status = "configured_waiting_for_results"
    return {
        "name": route["name"],
        "phase": route["phase"],
        "status": status,
        "topvenue_pattern": route["topvenue_pattern"],
        "modules": route["modules"],
        "module_trace": module_rows,
        "source_gap_modules": source_gap_modules,
        "runs": run_rows,
        "completed_runs": [row["run"] for row in complete],
        "missing_runs": [row["run"] for row in missing],
        "current_evidence": _current_evidence(route, run_rows, p0_by_run, summary),
        "decision_rule": route["decision_rule"],
        "next_action": _route_next_action(route, status, missing),
    }


def _module_status(
    module: str,
    trace_rows: dict[str, dict[str, Any]],
    algorithm_sources: dict[str, Any],
    algorithm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_name = MODULE_TRACE_ALIASES.get(module, module)
    trace = trace_rows.get(trace_name)
    configured_by_algorithm = [
        row.get("run")
        for row in algorithm_rows
        if isinstance(row.get("modules"), dict) and row["modules"].get(module)
    ]
    if trace and trace.get("source_ok") and int(trace.get("num_configured_runs", 0) or 0) > 0:
        return {
            "module": module,
            "trace_name": trace_name,
            "source_status": "trace_pass",
            "configured_runs": trace.get("configured_runs", []),
            "source_files": trace.get("existing_source_files", []),
        }
    if module in algorithm_sources and configured_by_algorithm:
        return {
            "module": module,
            "trace_name": trace_name,
            "source_status": "algorithm_source_pass",
            "configured_runs": configured_by_algorithm,
            "source_files": algorithm_sources.get(module, []),
        }
    if trace:
        return {
            "module": module,
            "trace_name": trace_name,
            "source_status": "trace_incomplete",
            "configured_runs": trace.get("configured_runs", []),
            "missing": trace.get("missing_tokens", []) + trace.get("missing_source_files", []),
        }
    return {
        "module": module,
        "trace_name": trace_name,
        "source_status": "missing_source_trace",
        "configured_runs": configured_by_algorithm,
        "source_files": algorithm_sources.get(module, []),
    }


def _run_status(
    run: str,
    candidate_rows: dict[str, dict[str, Any]],
    p0_by_run: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    if run in p0_by_run:
        row = p0_by_run[run]
        return {
            "run": run,
            "status": row.get("status", "complete"),
            "source": "p0",
            "risk_f1": row.get("risk_macro_f1") or row.get("risk_f1"),
            "friction_f1": row.get("friction_macro_f1") or row.get("friction_f1"),
            "low_friction_recall": row.get("low_friction_recall"),
            "worst_dataset_f1": row.get("worst_dataset_f1"),
        }
    if run in candidate_rows:
        row = candidate_rows[run]
        progress = row.get("progress_status", "missing")
        return {
            "run": run,
            "status": progress,
            "source": "candidate_hypothesis_matrix",
            "contract_status": row.get("contract_status"),
            "next_action": row.get("next_action"),
        }
    for key in ["lodo", "single_dataset", "fair_baselines", "final_lodo", "final_single_dataset"]:
        for row in summary.get(key, []) if isinstance(summary.get(key), list) else []:
            if Path(str(row.get("output_dir", ""))).name == run:
                return {
                    "run": run,
                    "status": row.get("status", "missing"),
                    "source": key,
                    "risk_f1": row.get("risk_macro_f1"),
                    "friction_f1": row.get("friction_macro_f1"),
                    "low_friction_recall": row.get("low_friction_recall"),
                    "calibrated_coverage": row.get("calibrated_coverage"),
                }
    return {"run": run, "status": "missing", "source": "not_found"}


def _current_evidence(
    route: dict[str, Any],
    run_rows: list[dict[str, Any]],
    p0_by_run: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    if route["phase"].startswith("P0"):
        global_row = p0_by_run.get("v0_global_only", {})
        parts = []
        for run in route["runs"]:
            row = p0_by_run.get(run)
            if not row:
                continue
            parts.append(_p0_delta_sentence(run, row, global_row))
        return " ".join(parts) if parts else "P0 evidence not found."
    if route["name"].startswith("Fair"):
        return "Fair rows are configured but not complete; same-split paired deltas are still missing."
    if route["phase"] == "Final":
        return "Final rows are configured but must wait for candidate ranking and full formal evaluation."
    if any(row["status"] == "complete" for row in run_rows):
        return "Some required rows are complete, but the full route still lacks enough evidence for a final claim."
    lodo = summary.get("lodo", []) if isinstance(summary.get("lodo"), list) else []
    roadsaw = next((row for row in lodo if Path(str(row.get("output_dir", ""))).name == "lodo_roadsaw_full_faf"), None)
    if roadsaw and route["phase"].startswith("P1"):
        return (
            "Current motivation is strong: held-out RoadSaW full-FAF risk F1 is "
            f"{_pct(roadsaw.get('risk_macro_f1'))}, so shortcut/wetness fixes are necessary."
        )
    return "Configured and source-backed, but formal metric evidence is pending."


def _p0_delta_sentence(run: str, row: dict[str, Any], global_row: dict[str, Any]) -> str:
    risk_delta = _delta_pp(row.get("risk_macro_f1"), global_row.get("risk_macro_f1"))
    low_delta = _delta_pp(row.get("low_friction_recall"), global_row.get("low_friction_recall"))
    worst_delta = _delta_pp(row.get("worst_dataset_f1"), global_row.get("worst_dataset_f1"))
    return (
        f"{run}: risk F1 {_pct(row.get('risk_macro_f1'))} ({risk_delta}), "
        f"low-friction recall {_pct(row.get('low_friction_recall'))} ({low_delta}), "
        f"worst-dataset F1 {_pct(row.get('worst_dataset_f1'))} ({worst_delta}) vs Global-only."
    )


def _route_next_action(route: dict[str, Any], status: str, missing: list[dict[str, Any]]) -> str:
    if status == "implementation_gap":
        return "Fix source/config trace gaps before using this route as paper evidence."
    if status == "evidence_complete":
        return "Use completed metrics for retention/pruning decisions; do not rerun unless protocol changes."
    if missing:
        first = ", ".join(row["run"] for row in missing[:4])
        extra = "" if len(missing) <= 4 else f" (+{len(missing) - 4} more)"
        return f"{route['next_if_pending']} Missing: {first}{extra}."
    return route["next_if_pending"]


def _p0_by_run(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key in ["ablation", "core_ablation"]:
        for row in summary.get(key, []) if isinstance(summary.get(key), list) else []:
            run = Path(str(row.get("output_dir", ""))).name
            if run:
                rows[run] = row
    return rows


def _next_order(rows: list[dict[str, Any]]) -> list[str]:
    order = []
    for target in [
        "P1/P3 wet-road quality physics",
        "P1 style and input canonicalization",
        "P1 conditional domain alignment and adapters",
        "P2 road-grounded EvidenceField",
        "P3 safety interval quality",
        "Fair single-dataset comparisons",
        "Final lean method evidence",
    ]:
        row = next((item for item in rows if item["name"] == target), None)
        if row and row["status"] != "evidence_complete":
            order.append(row["next_action"])
    return order


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Implementation Coverage Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Verdict: `{report['verdict']}`",
        "",
        "This audit links each proposed innovation route to code/config evidence, current metric evidence, and the next experiment needed before a paper claim.",
        "",
        "## Counts",
        "",
        f"- Routes: `{report['counts']['routes']}`",
        f"- Evidence complete: `{report['counts']['evidence_complete']}`",
        f"- Pending or partial: `{report['counts']['pending_or_partial']}`",
        f"- Source gaps: `{report['counts']['source_gap']}`",
        "",
        "## Route Table",
        "",
        "| Route | Phase | Status | Source gaps | Completed runs | Missing runs | Current evidence | Next action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {name} | {phase} | `{status}` | {gaps} | {complete} | {missing} | {evidence} | {next_action} |".format(
                name=row["name"],
                phase=row["phase"],
                status=row["status"],
                gaps=_join(row["source_gap_modules"]),
                complete=_join(row["completed_runs"]),
                missing=_join(row["missing_runs"]),
                evidence=_escape(row["current_evidence"]),
                next_action=_escape(row["next_action"]),
            )
        )
    lines.extend(["", "## Module Trace Detail", ""])
    for row in report["rows"]:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append(f"- Pattern: {row['topvenue_pattern']}")
        lines.append(f"- Decision rule: {row['decision_rule']}")
        lines.append("")
        lines.append("| Module | Trace name | Source status | Configured runs |")
        lines.append("|---|---|---|---:|")
        for module in row["module_trace"]:
            lines.append(
                "| {module} | `{trace}` | `{status}` | {count} |".format(
                    module=module["module"],
                    trace=module["trace_name"],
                    status=module["source_status"],
                    count=len(module.get("configured_runs", []) or []),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Current Failure Context",
            "",
            f"- Dataset shortcut verdict: `{report['current_failure_context'].get('dataset_shortcut_verdict')}`.",
            f"- High-shortcut completed rows: `{report['current_failure_context'].get('high_shortcut_rows')}`.",
            f"- Interval watchlist items: `{report['current_failure_context'].get('interval_watchlist_items')}`.",
            "",
            "## Claim Rules",
            "",
        ]
    )
    for rule in report["claim_rules"]:
        lines.append(f"- {rule}")
    lines.extend(["", "## Next Order", ""])
    for item in report["next_order"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _join(items: list[str]) -> str:
    return ", ".join(items) if items else "-"


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _delta_pp(value: Any, base: Any) -> str:
    try:
        return f"{(float(value) - float(base)) * 100:+.2f} pp"
    except (TypeError, ValueError):
        return "+-.-- pp"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
