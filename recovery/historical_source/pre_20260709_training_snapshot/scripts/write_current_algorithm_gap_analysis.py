from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "current_algorithm_gap_analysis.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "current_algorithm_gap_analysis.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    external = _load_json(summary_dir / "external_benchmark_report.json") or {}
    roadmap = _load_json(summary_dir / "topvenue_innovation_roadmap.json") or {}
    checkpoint = _load_json(summary_dir / "checkpoint_policy_report.json") or {}
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    active_live = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    algorithm_audit = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    module_decisions = _load_csv(summary_dir / "module_decisions.csv")

    rows = p0.get("rows") or summary.get("core_ablation", [])
    completed = [row for row in rows if row.get("status") == "complete"]
    pending = [row for row in rows if row.get("status") != "complete"]
    blocks = [item for item in gate.get("gates", []) if item.get("level") == "block"]
    warnings = [item for item in gate.get("gates", []) if item.get("level") == "warn"]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "readiness": {
            "verdict": gate.get("verdict"),
            "num_blocks": gate.get("num_blocks"),
            "num_warnings": gate.get("num_warnings"),
            "blocks": blocks,
            "warnings": warnings,
        },
        "p0_status": p0.get("core_status"),
        "p0_rows": rows,
        "completed_p0": len(completed),
        "pending_p0": [row.get("method") for row in pending],
        "module_decisions": module_decisions,
        "candidate_implementation": _candidate_implementation_status(algorithm_audit),
        "key_failures": _key_failures(shortcut, wetness, interval, final_selection, checkpoint),
        "fair_comparison_status": _fair_comparison_status(external),
        "queue_status": {
            "complete": queue.get("num_complete"),
            "partial": queue.get("num_partial"),
            "missing": queue.get("num_missing"),
            "next_incomplete": (queue.get("next_incomplete") or {}).get("name"),
            "active_rows": _active_rows(queue, active_live),
        },
        "research_decisions": _research_decisions(module_decisions, gate, roadmap),
        "hard_claim_rules": _hard_claim_rules(external, checkpoint),
    }


def _active_rows(queue: dict[str, Any], active_live: dict[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("active_rows", []) if isinstance(queue.get("active_rows"), list) else []
    active = active_live.get("active", {}) if isinstance(active_live, dict) else {}
    if not active.get("name"):
        return rows

    overlay = {
        "name": active.get("name"),
        "active_epoch": active.get("epoch"),
        "active_epochs": active.get("epochs"),
        "active_step": active.get("step"),
        "active_steps": active.get("steps"),
        "phase": active.get("phase"),
        "eval_step": active.get("eval_step"),
        "eval_steps": active.get("eval_steps"),
        "eval_tqdm_eta": active.get("eval_tqdm_eta") or active.get("eta") or active.get("tqdm_eta"),
        "eval_tqdm_rate": active.get("eval_tqdm_rate") or active.get("rate") or active.get("tqdm_rate"),
        "status": "running_or_partial",
    }
    for row in rows:
        if row.get("name") == active.get("name"):
            return [{**row, **{k: v for k, v in overlay.items() if v is not None}}]
    return [overlay]


def _candidate_implementation_status(algorithm_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    by_run = {row.get("run"): row for row in rows}
    groups = [
        {
            "phase": "P1 shortcut/domain robustness",
            "runs": ["v6_full_faf_fourier", "v7_full_faf_fourier_dann", "v11_full_faf_domain_adapter", "v15_lean_bottom_square_style_safety", "v16_lean_bottom_square_color_constancy_safety", "v17_lean_quality_physics_safety", "v18_lean_mixstyle_quality_safety", "v19_lean_state_contrast_quality_safety", "v20_lean_interval_order_quality_safety", "v21_lean_quality_uncertainty_safety", "v22_lean_quality_order_contrast_safety", "v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety"],
            "required_modules": [
                "fourier_style_jitter",
                "bottom_square_input_canonicalization",
                "dann",
                "semantic_conditional_alignment",
                "wetness_conditional_coral",
                "condition_hard_sampling",
                "weak_view_consistency",
                "masked_image_consistency",
                "domain_adapter",
                "multi_query_evidence",
            ],
            "limitation": "Effect is unproven until dataset-ID probe, LODO, and RoadSaW wetness metrics are available.",
        },
        {
            "phase": "P2 evidence grounding",
            "runs": ["v8_full_faf_fourier_roadprior", "v10_full_faf_consistency", "v12_full_faf_roi_interval_safety", "v14_lean_road_roi_safety", "v15_lean_bottom_square_style_safety", "v16_lean_bottom_square_color_constancy_safety", "v17_lean_quality_physics_safety", "v18_lean_mixstyle_quality_safety", "v19_lean_state_contrast_quality_safety", "v20_lean_interval_order_quality_safety", "v21_lean_quality_uncertainty_safety", "v22_lean_quality_order_contrast_safety", "v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety"],
            "required_modules": [
                "evidence_field",
                "road_likelihood_prior",
                "region_mixture_evidence",
                "pseudo_road_mask_supervision",
                "roi_attention_constraint",
                "weak_view_consistency",
                "masked_image_consistency",
                "multi_query_evidence",
                "query_disagreement_uncertainty",
            ],
            "limitation": "Pseudo-road supervision is currently heuristic/built-in unless external segmentation masks are generated later.",
        },
        {
            "phase": "P3 interval/safety quality",
            "runs": ["v12_full_faf_roi_interval_safety", "v14_lean_road_roi_safety", "v15_lean_bottom_square_style_safety", "v16_lean_bottom_square_color_constancy_safety", "v17_lean_quality_physics_safety", "v18_lean_mixstyle_quality_safety", "v19_lean_state_contrast_quality_safety", "v20_lean_interval_order_quality_safety", "v21_lean_quality_uncertainty_safety", "v22_lean_quality_order_contrast_safety", "v23_lean_region_mixture_evidence_safety", "v24_lean_multi_query_region_evidence_safety", "v25_lean_masked_query_consistency_safety", "final_lodo_roadsaw_lean_road_roi_safety"],
            "required_modules": [
                "coverage_aware_training",
                "safety_weighted_coverage",
                "wetness_ordinal_loss",
                "roi_attention_constraint",
                "query_disagreement_uncertainty",
            ],
            "limitation": "Must be judged by conditional coverage and width together, not pooled calibrated coverage alone.",
        },
        {
            "phase": "Final lean method",
            "runs": [
                "final_lodo_rscd_lean_road_roi_safety",
                "final_lodo_roadsaw_lean_road_roi_safety",
                "final_lodo_roadsc_lean_road_roi_safety",
                "final_single_rscd_lean_road_roi_safety",
                "final_single_roadsaw_lean_road_roi_safety",
                "final_single_roadsc_lean_road_roi_safety",
            ],
            "required_modules": [
                "physics_texture",
                "evidence_field",
                "fourier_style_jitter",
                "road_likelihood_prior",
                "pseudo_road_mask_supervision",
                "weak_view_consistency",
                "roi_attention_constraint",
                "coverage_aware_training",
                "safety_weighted_coverage",
                "semantic_conditional_alignment",
                "wetness_ordinal_loss",
            ],
            "limitation": "Final method is only publishable after matched single-dataset baselines and LODO rows are complete; semantic conditional alignment is kept as a small state-conditioned term, not the unstable full DG stack.",
        },
    ]
    out: list[dict[str, Any]] = []
    for group in groups:
        run_rows = [by_run.get(run) for run in group["runs"] if by_run.get(run)]
        module_hits = {
            module: [row.get("run") for row in run_rows if row.get("modules", {}).get(module)]
            for module in group["required_modules"]
        }
        missing_modules = [module for module, hits in module_hits.items() if not hits]
        out.append(
            {
                "phase": group["phase"],
                "runs": group["runs"],
                "present_runs": [row.get("run") for row in run_rows],
                "required_modules": group["required_modules"],
                "module_hits": module_hits,
                "missing_modules": missing_modules,
                "status": "implemented_or_configured" if run_rows and not missing_modules else "partial",
                "limitation": group["limitation"],
            }
        )
    return out


def _key_failures(
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    interval: dict[str, Any],
    final_selection: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if shortcut.get("num_high_shortcut"):
        failures.append(
            {
                "issue": "dataset_shortcut_high",
                "evidence": f"{shortcut.get('num_high_shortcut')}/{shortcut.get('num_complete')} completed rows exceed the dataset-ID threshold.",
                "next_test": "Prioritize v6/v7/v11/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 and judge them by dataset-ID drop plus risk F1, low recall, quality-slice robustness, and worst-dataset F1.",
            }
        )
    if wetness.get("num_watchlist"):
        failures.append(
            {
                "issue": "roadsaw_wetness_weak",
                "evidence": f"{wetness.get('num_watchlist')} completed rows are on the RoadSaW wetness watchlist.",
                "next_test": "Prioritize ordinal wetness supervision, wet-state hard sampling, region-mixture evidence, and very_wet failure review.",
            }
        )
    if interval.get("num_watchlist_items"):
        failures.append(
            {
                "issue": "conditional_interval_undercoverage",
                "evidence": f"{interval.get('num_watchlist_items')} dataset/state/risk cells are undercovered.",
                "next_test": "Prioritize v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 and final interval-safety rows; report coverage together with width.",
            }
        )
    if final_selection.get("verdict") != "ready_to_select_final_method":
        failures.append(
            {
                "issue": "final_method_not_selected",
                "evidence": f"Final selection verdict is {final_selection.get('verdict')}.",
                "next_test": "Finish P0, LODO, candidate rows, matched baselines, and final lean rows before selecting the final method.",
            }
        )
    live = checkpoint.get("live_v5", {}) if isinstance(checkpoint, dict) else {}
    if live.get("validation_degradation_flag"):
        failures.append(
            {
                "issue": "full_model_validation_degradation",
                "evidence": "v5 validation loss is worse than the best validation-loss epoch.",
                "next_test": "Use selected checkpoint policy and compare against lean v13/v14 instead of assuming the full stack is best.",
            }
        )
    return failures


def _fair_comparison_status(external: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in external.get("dataset_alignment", []) if isinstance(external, dict) else []:
        rows.append(
            {
                "dataset": item.get("dataset"),
                "fair_unit": item.get("fair_unit"),
                "status": item.get("claim_status"),
                "primary_metrics": item.get("primary_metrics"),
            }
        )
    return rows


def _research_decisions(
    module_decisions: list[dict[str, Any]],
    gate: dict[str, Any],
    roadmap: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in module_decisions:
        module = row.get("module")
        decision = row.get("decision")
        if decision in {"keep", "rework_or_remove"}:
            decisions.append(
                {
                    "item": module,
                    "decision": decision,
                    "reason": row.get("reason"),
                    "metric_signal": {
                        "delta_friction_macro_f1": row.get("delta_friction_macro_f1"),
                        "delta_risk_macro_f1": row.get("delta_risk_macro_f1"),
                        "delta_low_friction_recall": row.get("delta_low_friction_recall"),
                        "delta_worst_dataset_risk_f1": row.get("delta_worst_dataset_risk_f1"),
                    },
                }
            )
    for item in roadmap.get("next_decisions", []) if isinstance(roadmap, dict) else []:
        decisions.append(
            {
                "item": item.get("decision"),
                "decision": "next_priority",
                "reason": item.get("trigger"),
                "action": item.get("action"),
            }
        )
    block_names = [item.get("name") for item in gate.get("gates", []) if item.get("level") == "block"]
    if block_names:
        decisions.append(
            {
                "item": "readiness_blocks",
                "decision": "must_close_before_claim",
                "reason": ", ".join(str(item) for item in block_names),
            }
        )
    return decisions


def _hard_claim_rules(external: dict[str, Any], checkpoint: dict[str, Any]) -> list[str]:
    rules = []
    if external.get("strict_comparison_rule"):
        rules.append(external["strict_comparison_rule"])
    policy = checkpoint.get("policy", {}) if isinstance(checkpoint, dict) else {}
    for key in [
        "main_ablation_table",
        "supplemental_safety_analysis",
        "final_method_selection",
        "claim_boundary",
    ]:
        if policy.get(key):
            rules.append(policy[key])
    rules.extend(
        [
            "Do not describe public road-condition labels as measured tire-road friction coefficients.",
            "Do not keep a module in the final method unless it earns a safety, generalization, interval, or interpretability gain.",
            "Do not claim cross-dataset generalization unless completed LODO evidence supports it; current LODO is failure analysis, not an OOD success result.",
        ]
    )
    return rules


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Current Algorithm Gap Analysis",
        "",
        f"Generated at: {report['generated_at']}",
        f"Readiness: `{report['readiness']['verdict']}` ({report['readiness']['num_blocks']} blocks, {report['readiness']['num_warnings']} warnings)",
        "",
        "## P0 Evidence",
        "",
        f"- P0 status: `{report['p0_status']}`; completed `{report['completed_p0']}` rows.",
        f"- Pending P0 rows: {', '.join(f'`{item}`' for item in report['pending_p0']) or '-'}.",
        "",
        "| Method | Status | friction F1 | risk F1 | low recall | calibrated coverage | worst dataset F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["p0_rows"]:
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {worst} |".format(
                method=row.get("method"),
                status=row.get("status"),
                friction=_fmt_metric(row, "friction_f1", "friction_macro_f1"),
                risk=_fmt_metric(row, "risk_f1", "risk_macro_f1"),
                low=_fmt_metric(row, "low_recall", "low_friction_recall"),
                cov=_fmt_metric(row, "calibrated_coverage"),
                worst=_fmt_metric(row, "worst_dataset_f1", "worst_dataset_risk_f1"),
            )
        )

    lines.extend(["", "## Key Failures", ""])
    for item in report["key_failures"]:
        lines.append(f"- `{item['issue']}`: {item['evidence']} Next: {item['next_test']}")

    lines.extend(["", "## Candidate Implementation Coverage", ""])
    lines.append("| Phase | Status | Candidate runs | Missing modules | Limitation |")
    lines.append("|---|---|---|---|---|")
    for item in report["candidate_implementation"]:
        lines.append(
            "| {phase} | `{status}` | {runs} | {missing} | {limitation} |".format(
                phase=item.get("phase"),
                status=item.get("status"),
                runs=", ".join(f"`{run}`" for run in item.get("present_runs", [])) or "-",
                missing=", ".join(f"`{module}`" for module in item.get("missing_modules", [])) or "-",
                limitation=item.get("limitation"),
            )
        )

    lines.extend(["", "## Module Decisions", ""])
    for item in report["research_decisions"]:
        lines.append(
            "- `{item}` -> `{decision}`: {reason}".format(
                item=item.get("item"),
                decision=item.get("decision"),
                reason=item.get("reason") or item.get("action") or "-",
            )
        )

    lines.extend(["", "## Fair Comparison Status", ""])
    lines.append("| Dataset | Fair unit | Status | Primary metrics |")
    lines.append("|---|---|---|---|")
    for row in report["fair_comparison_status"]:
        lines.append(
            "| {dataset} | `{unit}` | `{status}` | {metrics} |".format(
                dataset=row.get("dataset"),
                unit=row.get("fair_unit"),
                status=row.get("status"),
                metrics=row.get("primary_metrics"),
            )
        )

    queue = report["queue_status"]
    lines.extend(["", "## Queue Status", ""])
    lines.append(
        "- Complete `{complete}`, partial `{partial}`, missing `{missing}`; next incomplete `{next}`.".format(
            complete=queue.get("complete"),
            partial=queue.get("partial"),
            missing=queue.get("missing"),
            next=queue.get("next_incomplete"),
        )
    )
    for row in queue.get("active_rows", []):
        message = "- Active `{name}`: epoch `{epoch}/{epochs}`, step `{step}/{steps}`".format(
            name=row.get("name"),
            epoch=row.get("active_epoch") or row.get("epoch"),
            epochs=row.get("active_epochs") or row.get("epochs"),
            step=row.get("active_step"),
            steps=row.get("active_steps"),
        )
        if row.get("phase") == "eval":
            message += ", validation `{step}/{steps}`, ETA `{eta}`, rate `{rate}`".format(
                step=row.get("eval_step") or "-",
                steps=row.get("eval_steps") or "-",
                eta=row.get("eval_tqdm_eta") or "-",
                rate=row.get("eval_tqdm_rate") or "-",
            )
        lines.append(message + ".")

    lines.extend(["", "## Hard Claim Rules", ""])
    for rule in report["hard_claim_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    return "\n".join(lines)


def _fmt_metric(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            try:
                return f"{100.0 * float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)
    return "-"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
