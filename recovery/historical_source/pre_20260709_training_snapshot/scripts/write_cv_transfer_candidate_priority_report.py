from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "cv_transfer_candidate_priority_report.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "cv_transfer_candidate_priority_report.json"


PRIORITY_GROUPS = [
    {
        "priority": 0,
        "name": "fair_comparison_before_claims",
        "cv_source": "reviewer protocol, not an algorithm module",
        "rationale": (
            "Before claiming any CV-transfer module is better, FAF and ConvNeXt must be compared "
            "on the same single-dataset splits and labels."
        ),
        "runs": [
            "single_roadsaw_full_faf",
            "single_rscd_full_faf",
            "single_roadsc_full_faf",
            "baseline_single_roadsaw_global_convnext",
            "baseline_single_rscd_global_convnext",
            "baseline_single_roadsc_global_convnext",
        ],
        "primary_metrics": [
            "paired risk F1 delta",
            "paired friction F1 delta",
            "low-friction recall",
            "calibrated coverage",
            "calibrated width",
        ],
        "keep_rule": "Unlock method claims only after all six fair rows are complete and pairwise deltas are computed.",
        "drop_rule": "No algorithm claim is allowed while the matching ConvNeXt row is missing.",
    },
    {
        "priority": 1,
        "name": "segmentation_style_local_evidence",
        "cv_source": "semantic segmentation, weakly supervised segmentation, mask classification",
        "rationale": (
            "This is the most on-topic transfer: friction depends on local road-material evidence, "
            "not only a global image vector."
        ),
        "runs": [
            "v14_lean_road_roi_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "primary_metrics": [
            "RoadSaW wet/near-white F1",
            "attention_pseudo_road_mass",
            "risk F1",
            "low-friction recall",
            "calibrated coverage-width",
        ],
        "keep_rule": "Keep if local/multi-query evidence improves hard wet/snow slices, query/attention quality, or coverage-width without fair-baseline regression.",
        "drop_rule": "If attention improves but task/safety metrics regress, keep visualizations only and remove the training loss.",
    },
    {
        "priority": 2,
        "name": "material_physics_quality_uncertainty",
        "cv_source": "material recognition, physical vision, visual quality estimation",
        "rationale": (
            "Wet glare, thin water film, black ice, and snow are ambiguous visual states; the model "
            "should express that as calibrated interval uncertainty."
        ),
        "runs": [
            "v17_lean_quality_physics_safety",
            "v21_lean_quality_uncertainty_safety",
            "v22_lean_quality_order_contrast_safety",
        ],
        "primary_metrics": [
            "near-white conditional coverage",
            "low-texture conditional coverage",
            "specular conditional coverage",
            "calibrated width",
            "low-friction recall",
        ],
        "keep_rule": "Keep if hard visual-quality cells gain coverage with bounded width.",
        "drop_rule": "Drop or weaken if it only widens intervals or learns brightness shortcuts.",
    },
    {
        "priority": 3,
        "name": "mask_aware_consistency",
        "cv_source": "semi-supervised semantic segmentation consistency",
        "rationale": (
            "Wet/snow cues are appearance-sensitive; weak/strong consistency can stabilize predictions, "
            "but the mask-aware version avoids over-constraining background."
        ),
        "runs": [
            "v10_full_faf_consistency",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "primary_metrics": [
            "weak/strong interval stability",
            "RoadSaW wetness macro F1",
            "risk F1",
            "low-friction recall",
        ],
        "keep_rule": "Keep if interval and logit stability improve while wet/snow recall is preserved.",
        "drop_rule": "Remove if it over-smooths hard wet/snow cases or lowers low-friction recall.",
    },
    {
        "priority": 4,
        "name": "domain_adaptive_shortcut_control",
        "cv_source": "domain-adaptive segmentation and domain generalization",
        "rationale": (
            "The completed evidence shows high dataset-ID shortcut, so style-control is necessary; "
            "generic adversarial losses are risky and must be pruned aggressively."
        ),
        "runs": [
            "v6_full_faf_fourier",
            "v7_full_faf_fourier_dann",
            "v11_full_faf_domain_adapter",
            "v15_lean_bottom_square_style_safety",
            "v16_lean_bottom_square_color_constancy_safety",
            "v18_lean_mixstyle_quality_safety",
        ],
        "primary_metrics": [
            "dataset-ID balanced accuracy",
            "worst-dataset F1",
            "risk F1",
            "low-friction recall",
        ],
        "keep_rule": "Keep only if dataset-ID probe drops and safety metrics remain stable.",
        "drop_rule": "Drop DANN/full DG variants immediately if they repeat the P0 DG-loss regressions.",
    },
    {
        "priority": 5,
        "name": "foundation_or_promptable_teacher",
        "cv_source": "SAM, CLIPSeg, DINOv2-style dense teachers",
        "rationale": (
            "Teacher masks/features can help without new labels, but they must be audited and should not "
            "replace fair same-split baselines."
        ),
        "runs": ["smoke_opencv_mask_supervised_evidence", "clipseg_mask_supervised_evidence_screen"],
        "primary_metrics": [
            "mask-quality audit",
            "attention-on-road gain",
            "RoadSaW wet/snow metrics",
            "same-split baseline delta",
        ],
        "keep_rule": "Promote only after mask audit and a bounded candidate run show measurable benefit.",
        "drop_rule": "Do not claim innovation for a larger teacher or unaudited pseudo mask.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prioritize CV-subfield transfer candidates with rapid promotion/pruning rules."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path) -> dict[str, Any]:
    fair = _load_json(summary_dir / "fair_comparison_execution_priority.json") or {}
    mechanism = _load_json(summary_dir / "wet_slippery_failure_mechanism_report.json") or {}
    pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    hypothesis = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    p0 = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}

    completed = _completed_runs(artifact, hypothesis)
    active = _active_runs(fair)
    rows = []
    for spec in PRIORITY_GROUPS:
        rows.append(_priority_row(spec, completed, active, pruning, p0, lodo, mechanism))

    first_incomplete = next((row for row in rows if row["status"] != "complete"), None)
    blocks = [
        "fair_comparison_before_claims"
        for row in rows
        if row["name"] == "fair_comparison_before_claims" and row["status"] != "complete"
    ]
    verdict = "waiting_for_fair_comparisons_then_cv_candidates" if blocks else "ready_for_cv_candidate_screening"
    if any(row["status"] == "active" for row in rows):
        verdict = "active_queue_running"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "This is an execution-priority report. It ranks CV-transfer experiments and "
            "predeclares pruning rules; it does not claim final method superiority."
        ),
        "rows": rows,
        "blocks": blocks,
        "first_incomplete": first_incomplete["name"] if first_incomplete else None,
        "completed_runs": sorted(completed),
        "active_runs": sorted(active),
        "decision_summary": _decision_summary(rows),
    }


def _priority_row(
    spec: dict[str, Any],
    completed: set[str],
    active: set[str],
    pruning: dict[str, Any],
    p0: dict[str, Any],
    lodo: dict[str, Any],
    mechanism: dict[str, Any],
) -> dict[str, Any]:
    runs = spec["runs"]
    complete = [run for run in runs if run in completed]
    active_runs = [run for run in runs if run in active]
    missing = [run for run in runs if run not in completed and run not in active]
    if missing and active_runs:
        status = "active"
    elif missing:
        status = "pending"
    else:
        status = "complete"

    current_evidence = _current_evidence(spec["name"], p0, lodo, mechanism, pruning)
    return {
        **spec,
        "status": status,
        "complete_runs": complete,
        "active_runs": active_runs,
        "missing_runs": missing,
        "current_evidence": current_evidence,
        "rapid_prune_trigger": _rapid_prune_trigger(spec["name"]),
    }


def _current_evidence(
    name: str,
    p0: dict[str, Any],
    lodo: dict[str, Any],
    mechanism: dict[str, Any],
    pruning: dict[str, Any],
) -> dict[str, Any]:
    if name == "fair_comparison_before_claims":
        return {
            "reason": "Required before algorithm superiority claims.",
            "candidate_pruning_verdict": pruning.get("verdict"),
        }
    if name == "domain_adaptive_shortcut_control":
        return {
            "p0_dg_regression": _p0_delta(p0, "+ DG losses", "+ PhysicsTexture"),
            "mechanism": "dataset_style_shortcut_over_physics",
        }
    if name == "segmentation_style_local_evidence":
        return {
            "p0_evidence_delta": _p0_delta(p0, "+ EvidenceField aux", "+ PhysicsTexture"),
            "mechanism": "global_pooling_misses_contact_patch_evidence",
        }
    if name == "material_physics_quality_uncertainty":
        return {
            "p0_physics_delta": _p0_delta(p0, "+ PhysicsTexture", "Global-only"),
            "mechanism": "water_film_specular_and_thin_ice_uncertainty",
        }
    if name == "mask_aware_consistency":
        return {
            "mechanism": "weak_strong_appearance_instability",
            "wetness_watchlist": _nested(mechanism, ["evidence_snapshot", "wetness_watchlist_count"]),
        }
    if name == "foundation_or_promptable_teacher":
        return {
            "mechanism": "foundation_dense_teacher_without_new_labels",
            "claim_limit": "teacher_or_baseline_until_audited",
        }
    if name == "weak_interval_undercoverage":
        return {"lodo": lodo.get("verdict")}
    return {}


def _p0_delta(p0: dict[str, Any], current: str, reference: str) -> dict[str, Any]:
    rows = p0.get("rows", []) if isinstance(p0.get("rows"), list) else []
    cur = _row_by_method(rows, current)
    ref = _row_by_method(rows, reference)
    keys = ["risk_macro_f1", "friction_macro_f1", "low_friction_recall", "worst_dataset_f1", "raw_interval_coverage"]
    return {key: _delta(cur, ref, key) for key in keys}


def _rapid_prune_trigger(name: str) -> str:
    triggers = {
        "fair_comparison_before_claims": "Stop paper-level claims until all matched FAF and ConvNeXt rows exist.",
        "segmentation_style_local_evidence": "Prune if RoadSaW/RoadSC hard-slice metrics or risk F1 fall while attention-only metrics improve.",
        "material_physics_quality_uncertainty": "Prune or weaken if calibrated width grows without hard-cell coverage gain.",
        "mask_aware_consistency": "Prune if low-friction recall drops or wet/snow states become over-smoothed.",
        "domain_adaptive_shortcut_control": "Prune if risk F1 or low-friction recall regresses more than the P0 DG-loss warning band.",
        "foundation_or_promptable_teacher": "Do not promote without mask-quality audit and same-split benefit.",
    }
    return triggers.get(name, "Prune if the predeclared primary metrics do not improve.")


def _completed_runs(artifact: dict[str, Any], hypothesis: dict[str, Any]) -> set[str]:
    completed: set[str] = set()
    for report in (artifact, hypothesis):
        rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
        for row in rows:
            run = row.get("run")
            if not run:
                continue
            status = row.get("contract_status") or row.get("progress_status") or row.get("status")
            if status == "complete":
                completed.add(str(run))
    return completed


def _active_runs(fair: dict[str, Any]) -> set[str]:
    rows = fair.get("active_runs", []) if isinstance(fair.get("active_runs"), list) else []
    return {str(row.get("run")) for row in rows if row.get("run")}


def _decision_summary(rows: list[dict[str, Any]]) -> list[str]:
    summary: list[str] = []
    for row in rows:
        if row["status"] == "complete":
            continue
        summary.append(
            f"Priority {row['priority']}: {row['name']} -> {row['status']}; missing {', '.join(row['missing_runs']) or '-'}."
        )
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CV Transfer Candidate Priority Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Priority Table",
        "",
        "| Priority | Candidate group | CV source | Status | Runs | Primary metrics | Keep rule | Rapid prune trigger |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {priority} | {name} | {source} | {status} | {runs} | {metrics} | {keep} | {prune} |".format(
                priority=row["priority"],
                name=row["name"],
                source=row["cv_source"],
                status=row["status"],
                runs=_join(row["runs"]),
                metrics=_join(row["primary_metrics"]),
                keep=row["keep_rule"],
                prune=row["rapid_prune_trigger"],
            )
        )
    lines.extend(["", "## Evidence Notes", ""])
    for row in report["rows"]:
        lines.append(f"### Priority {row['priority']}: {row['name']}")
        lines.append(f"- Rationale: {row['rationale']}")
        lines.append(f"- Complete: {_join(row['complete_runs'])}")
        lines.append(f"- Active: {_join(row['active_runs'])}")
        lines.append(f"- Missing: {_join(row['missing_runs'])}")
        lines.append(f"- Current evidence: {_format(row['current_evidence'])}")
        lines.append(f"- Drop rule: {row['drop_rule']}")
        lines.append("")
    lines.extend(["## Current Decision Summary", ""])
    if report["decision_summary"]:
        lines.extend(f"- {item}" for item in report["decision_summary"])
    else:
        lines.append("- All listed groups are complete.")
    lines.append("")
    return "\n".join(lines)


def _row_by_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


def _delta(cur: dict[str, Any], ref: dict[str, Any], key: str) -> float | None:
    cur_val = _num(cur.get(key))
    ref_val = _num(ref.get(key))
    if cur_val is None or ref_val is None:
        return None
    return cur_val - ref_val


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(payload: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _join(items: list[Any]) -> str:
    cleaned = [str(item) for item in items if item]
    return ", ".join(cleaned) if cleaned else "-"


def _format(value: Any) -> str:
    if not value:
        return "-"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > 500:
        return text[:497] + "..."
    return text


if __name__ == "__main__":
    main()
