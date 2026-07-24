from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "wet_slippery_failure_mechanism_report.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "wet_slippery_failure_mechanism_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map wet/slippery visual failure mechanisms to executable CV-transfer routes."
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
    quality = _load_json(summary_dir / "quality_domain_diagnostic_report.json") or {}
    dataset_style = _load_json(summary_dir / "dataset_image_style_audit.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    intervals = _load_json(summary_dir / "interval_quality_report.json") or {}
    evidence = _load_json(summary_dir / "evidence_failure_report.json") or {}
    p0 = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    cv_transfer = _load_json(summary_dir / "cv_transfer_experiment_protocol.json") or {}
    candidate_pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    wet_optical = _load_json(summary_dir / "wet_optical_quality_cues_smoke.json") or {}

    evidence_snapshot = _evidence_snapshot(
        quality=quality,
        dataset_style=dataset_style,
        wetness=wetness,
        shortcut=shortcut,
        intervals=intervals,
        evidence=evidence,
        p0=p0,
        lodo=lodo,
        wet_optical=wet_optical,
    )
    mechanisms = _mechanism_rows(evidence_snapshot, cv_transfer)
    blocks = [
        row["mechanism"]
        for row in mechanisms
        if not row["candidate_configs"] or not row["success_metrics"]
    ]
    verdict = "mechanism_protocol_ready_waiting_for_candidate_metrics"
    if blocks:
        verdict = "mechanism_mapping_gaps"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "This report maps computer-vision subfield ideas to public-label visual "
            "friction-affordance interval estimation. It does not claim synchronized "
            "measured tire-road friction coefficients."
        ),
        "evidence_snapshot": evidence_snapshot,
        "mechanisms": mechanisms,
        "blocks": blocks,
        "counts": {
            "mechanisms": len(mechanisms),
            "segmentation_style_mechanisms": sum(
                1 for row in mechanisms if "semantic segmentation" in row["cv_subfield_transfer"]
            ),
            "candidate_metric_pending": sum(1 for row in mechanisms if row["metric_status"] == "pending"),
            "future_only": sum(1 for row in mechanisms if row["metric_status"] == "future_only"),
        },
        "decision_policy": [
            "Treat semantic segmentation as weak local evidence and mask-like pooling, not as a need for pixel labels.",
            "Promote a route only if same-split FAF-vs-ConvNeXt or candidate ablations improve safety metrics, interval coverage-width, or audited attention quality.",
            "Prune generic domain-adversarial or full-fusion modules if they repeat the P0 safety regressions.",
            "Use SAM/CLIPSeg/DINOv2-style models as offline teachers or baselines until mask/teacher quality is audited.",
            "Keep the claim wording as visual friction-affordance intervals derived from public road-condition labels.",
        ],
        "candidate_pruning_dependency": {
            "verdict": candidate_pruning.get("verdict"),
            "pending": _nested(candidate_pruning, ["counts", "pending"]),
        },
    }


def _evidence_snapshot(
    *,
    quality: dict[str, Any],
    dataset_style: dict[str, Any],
    wetness: dict[str, Any],
    shortcut: dict[str, Any],
    intervals: dict[str, Any],
    evidence: dict[str, Any],
    p0: dict[str, Any],
    lodo: dict[str, Any],
    wet_optical: dict[str, Any],
) -> dict[str, Any]:
    quality_summary = quality.get("quality_summary", {}) if isinstance(quality, dict) else {}
    by_dataset = quality_summary.get("by_dataset", {}) if isinstance(quality_summary, dict) else {}
    roadsaw_q = by_dataset.get("roadsaw", {}) if isinstance(by_dataset, dict) else {}
    roadsc_q = by_dataset.get("roadsc", {}) if isinstance(by_dataset, dict) else {}

    p0_rows = p0.get("rows", []) if isinstance(p0.get("rows"), list) else []
    p0_global = _row_by_method(p0_rows, "Global-only")
    p0_physics = _row_by_method(p0_rows, "+ PhysicsTexture")
    p0_dg = _row_by_method(p0_rows, "+ DG losses")
    p0_full = _row_by_method(p0_rows, "Full model")

    shortcut_rows = shortcut.get("rows", []) if isinstance(shortcut.get("rows"), list) else []
    shortcut_flagged = [row for row in shortcut_rows if row.get("shortcut_flag")]

    evidence_runs = evidence.get("runs", []) if isinstance(evidence.get("runs"), list) else []
    lodo_rows = lodo.get("rows", []) if isinstance(lodo.get("rows"), list) else []

    return {
        "roadsaw_near_white_rate": _num(roadsaw_q.get("near_white_rate")),
        "roadsaw_low_texture_rate": _num(roadsaw_q.get("low_texture_rate")),
        "roadsc_low_texture_rate": _num(roadsc_q.get("low_texture_rate")),
        "dataset_style_dimensions": _dataset_dimensions(dataset_style, quality_summary),
        "wetness_watchlist_count": len(wetness.get("watchlist", []))
        if isinstance(wetness.get("watchlist"), list)
        else None,
        "interval_watchlist_count": intervals.get("num_watchlist_items"),
        "shortcut_flagged_completed_rows": len(shortcut_flagged),
        "shortcut_probe_threshold": shortcut.get("threshold"),
        "p0_physics_delta_vs_global": {
            "friction_macro_f1": _delta(p0_physics, p0_global, "friction_macro_f1"),
            "risk_macro_f1": _delta(p0_physics, p0_global, "risk_macro_f1"),
            "low_friction_recall": _delta(p0_physics, p0_global, "low_friction_recall"),
            "worst_dataset_f1": _delta(p0_physics, p0_global, "worst_dataset_f1"),
            "raw_interval_coverage": _delta(p0_physics, p0_global, "raw_interval_coverage"),
        },
        "p0_dg_delta_vs_physics": {
            "risk_macro_f1": _delta(p0_dg, p0_physics, "risk_macro_f1"),
            "low_friction_recall": _delta(p0_dg, p0_physics, "low_friction_recall"),
            "worst_dataset_f1": _delta(p0_dg, p0_physics, "worst_dataset_f1"),
        },
        "p0_full_delta_vs_physics": {
            "risk_macro_f1": _delta(p0_full, p0_physics, "risk_macro_f1"),
            "low_friction_recall": _delta(p0_full, p0_physics, "low_friction_recall"),
            "worst_dataset_f1": _delta(p0_full, p0_physics, "worst_dataset_f1"),
        },
        "lodo_summary": [
            {
                "held_out": row.get("held_out") or row.get("run"),
                "risk_f1": row.get("risk_f1", row.get("risk_macro_f1")),
                "friction_f1": row.get("friction_f1", row.get("friction_macro_f1")),
                "low_friction_recall": row.get("low_friction_recall"),
                "raw_coverage": row.get("raw_coverage"),
                "calibrated_coverage": row.get("calibrated_coverage"),
                "calibrated_width": row.get("calibrated_width"),
            }
            for row in lodo_rows
            if row.get("held_out") or str(row.get("run", "")).startswith("lodo_")
        ],
        "evidence_attention_snapshot": [
            {
                "run": row.get("run"),
                "risk_accuracy_sampled": row.get("risk_accuracy_sampled"),
                "raw_interval_coverage_sampled": row.get("raw_interval_coverage_sampled"),
                "failure_minus_success_road_likelihood": row.get("failure_minus_success_road_likelihood"),
                "failure_minus_success_bottom_mass": row.get("failure_minus_success_bottom_mass"),
            }
            for row in evidence_runs
        ],
        "wet_optical_quality_cues": {
            "status": wet_optical.get("status"),
            "quality_num_stats": _quality_num_stats(wet_optical),
        },
    }


def _mechanism_rows(snapshot: dict[str, Any], cv_transfer: dict[str, Any]) -> list[dict[str, Any]]:
    route_status = _route_status(cv_transfer)
    return [
        {
            "mechanism": "wet_snow_near_white_visual_aliasing",
            "cv_subfield_transfer": "semantic segmentation, mask classification, and material recognition",
            "why_it_matters": (
                "Very wet asphalt, glare, slush, and snow can all erase texture and push RGB statistics "
                "toward smooth bright regions, so a global classifier can confuse wet risk with snow/ice risk."
            ),
            "local_evidence": _compact_evidence(
                snapshot,
                ["roadsaw_near_white_rate", "roadsaw_low_texture_rate", "roadsc_low_texture_rate"],
            ),
            "candidate_configs": [
                "v17_lean_quality_physics_safety",
                "v21_lean_quality_uncertainty_safety",
                "v22_lean_quality_order_contrast_safety",
                "v23_lean_region_mixture_evidence_safety",
                "v24_lean_multi_query_region_evidence_safety",
            ],
            "success_metrics": [
                "RoadSaW wet/very_wet F1",
                "RoadSC fresh_snow/partial_snow calibrated coverage",
                "low-friction recall",
                "calibrated coverage-width tradeoff",
            ],
            "promotion_rule": "Keep if wet/snow slices improve without simply widening every interval.",
            "drop_rule": "If it mainly learns brightness or overpredicts snow for wet asphalt, keep only the audited quality flags.",
            "metric_status": "pending",
            "linked_routes": [
                route_status.get("material_texture_physical_vision"),
                route_status.get("semantic_segmentation_local_evidence"),
            ],
        },
        {
            "mechanism": "water_film_specular_and_thin_ice_uncertainty",
            "cv_subfield_transfer": "physical vision, image quality estimation, and uncertainty calibration",
            "why_it_matters": (
                "A water film or black ice can be visually smooth and low-texture, so the correct output is often a "
                "wider low-friction interval rather than an overconfident class."
            ),
            "local_evidence": _compact_evidence(snapshot, ["wet_optical_quality_cues", "interval_watchlist_count"]),
            "candidate_configs": [
                "v17_lean_quality_physics_safety",
                "v21_lean_quality_uncertainty_safety",
                "v22_lean_quality_order_contrast_safety",
            ],
            "success_metrics": [
                "near-white/low-texture/specular conditional coverage",
                "calibrated width",
                "low-friction recall",
                "risk F1",
            ],
            "promotion_rule": "Keep if hard visual-quality cells gain coverage with bounded width.",
            "drop_rule": "Prune quality weighting if it inflates all intervals or damages risk F1.",
            "metric_status": "pending",
            "linked_routes": [route_status.get("material_texture_physical_vision")],
        },
        {
            "mechanism": "dataset_style_shortcut_over_physics",
            "cv_subfield_transfer": "domain-adaptive semantic segmentation and domain generalization",
            "why_it_matters": (
                "RSCD patches, RoadSaW square wetness crops, and RoadSC snow crops have distinct image styles; "
                "models can solve dataset identity instead of road friction evidence."
            ),
            "local_evidence": _compact_evidence(
                snapshot,
                ["dataset_style_dimensions", "shortcut_flagged_completed_rows", "shortcut_probe_threshold"],
            ),
            "candidate_configs": [
                "v6_full_faf_fourier",
                "v7_full_faf_fourier_dann",
                "v11_full_faf_domain_adapter",
                "v15_lean_bottom_square_style_safety",
                "v16_lean_bottom_square_color_constancy_safety",
                "v18_lean_mixstyle_quality_safety",
            ],
            "success_metrics": [
                "dataset-ID balanced accuracy",
                "matched single-dataset FAF-vs-ConvNeXt deltas",
                "worst-dataset F1",
                "low-friction recall",
            ],
            "promotion_rule": "Keep style control only when shortcut probes fall and safety metrics stay stable.",
            "drop_rule": "Drop generic adversarial alignment if it repeats the P0 DG-loss safety regression.",
            "metric_status": "pending",
            "linked_routes": [route_status.get("domain_adaptive_segmentation_shortcut_control")],
        },
        {
            "mechanism": "global_pooling_misses_contact_patch_evidence",
            "cv_subfield_transfer": "semantic segmentation, weakly supervised segmentation, and multiple-instance learning",
            "why_it_matters": (
                "Friction is decided by local road material under the likely contact region; full-image global pooling "
                "can dilute glare, snow edges, wet patches, or rough texture with irrelevant background."
            ),
            "local_evidence": _compact_evidence(
                snapshot,
                ["p0_physics_delta_vs_global", "evidence_attention_snapshot"],
            ),
            "candidate_configs": [
                "v14_lean_road_roi_safety",
                "v23_lean_region_mixture_evidence_safety",
                "v24_lean_multi_query_region_evidence_safety",
                "smoke_opencv_mask_supervised_evidence",
            ],
            "success_metrics": [
                "attention_pseudo_road_mass",
                "RoadSaW wet/near-white F1",
                "risk F1",
                "evidence failure-map pass rate",
            ],
            "promotion_rule": "Keep if local evidence improves hard slices or attention quality without hurting fair baselines.",
            "drop_rule": "If attention maps look plausible but metrics regress, keep maps for analysis and remove the loss path.",
            "metric_status": "pending",
            "linked_routes": [
                route_status.get("semantic_segmentation_local_evidence"),
                route_status.get("promptable_or_open_vocabulary_mask_teacher"),
            ],
        },
        {
            "mechanism": "weak_strong_appearance_instability",
            "cv_subfield_transfer": "semi-supervised semantic segmentation consistency",
            "why_it_matters": (
                "Wetness and snow cues are sensitive to color, brightness, blur, and crop; consistency should be "
                "enforced mostly on road-like support so the model does not chase background changes."
            ),
            "local_evidence": _compact_evidence(snapshot, ["wetness_watchlist_count"]),
            "candidate_configs": [
                "v10_full_faf_consistency",
                "v23_lean_region_mixture_evidence_safety",
                "v24_lean_multi_query_region_evidence_safety",
            ],
            "success_metrics": [
                "weak/strong interval stability",
                "RoadSaW wetness macro F1",
                "low-friction recall",
                "risk F1",
            ],
            "promotion_rule": "Keep mask-aware consistency if it stabilizes intervals while preserving hard wet/snow recall.",
            "drop_rule": "Reduce or remove if it over-smooths the exact ambiguous cases the paper must handle.",
            "metric_status": "pending",
            "linked_routes": [route_status.get("mask_aware_weak_strong_consistency")],
        },
        {
            "mechanism": "weak_interval_undercoverage",
            "cv_subfield_transfer": "uncertainty estimation, conformal prediction, and risk-sensitive learning",
            "why_it_matters": (
                "Public labels provide friction-affordance intervals rather than measured mu. The scientific target is "
                "therefore calibrated coverage with useful width, especially on low-friction slices."
            ),
            "local_evidence": _compact_evidence(
                snapshot,
                ["interval_watchlist_count", "p0_full_delta_vs_physics", "lodo_summary"],
            ),
            "candidate_configs": [
                "v12_full_faf_roi_interval_safety",
                "v21_lean_quality_uncertainty_safety",
                "v22_lean_quality_order_contrast_safety",
                "v23_lean_region_mixture_evidence_safety",
                "v24_lean_multi_query_region_evidence_safety",
            ],
            "success_metrics": [
                "raw interval coverage",
                "calibrated coverage",
                "calibrated width",
                "worst conditional cell coverage",
            ],
            "promotion_rule": "Keep if coverage improves on hard cells with small or justified width increases.",
            "drop_rule": "Reject any route that only wins by making all intervals uninformatively wide.",
            "metric_status": "pending",
            "linked_routes": [route_status.get("material_texture_physical_vision")],
        },
        {
            "mechanism": "foundation_dense_teacher_without_new_labels",
            "cv_subfield_transfer": "self-supervised dense representation and promptable/open-vocabulary segmentation",
            "why_it_matters": (
                "DINOv2/SAM/CLIPSeg-style teachers can provide road/material masks or dense texture tokens without "
                "new manual data collection, but they must not become an unfair hidden dependency."
            ),
            "local_evidence": _compact_evidence(snapshot, ["dataset_style_dimensions"]),
            "candidate_configs": ["offline_teacher_after_fair_baselines"],
            "success_metrics": [
                "teacher mask-quality audit",
                "attention-on-road gain",
                "fair ConvNeXt comparison after same-split controls",
            ],
            "promotion_rule": "Use as teacher/baseline only after fair same-split rows and mask audits pass.",
            "drop_rule": "Do not claim innovation for a larger backbone or unaudited pseudo mask.",
            "metric_status": "future_only",
            "linked_routes": [
                route_status.get("promptable_or_open_vocabulary_mask_teacher"),
                route_status.get("foundation_dense_teacher"),
            ],
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Wet/Slippery Failure Mechanism and CV Transfer Map",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Evidence Snapshot",
        "",
    ]
    for key, value in report["evidence_snapshot"].items():
        lines.append(f"- `{key}`: {_format_value(value)}")
    lines.extend(
        [
            "",
            "## Mechanism Map",
            "",
            "| Mechanism | CV subfield transfer | Candidate configs | Success metrics | Promotion/drop rule |",
            "|---|---|---|---|---|",
        ]
    )
    for row in report["mechanisms"]:
        lines.append(
            "| {mechanism} | {subfield} | {configs} | {metrics} | {promote} / {drop} |".format(
                mechanism=row["mechanism"],
                subfield=row["cv_subfield_transfer"],
                configs=_join(row["candidate_configs"]),
                metrics=_join(row["success_metrics"]),
                promote=row["promotion_rule"],
                drop=row["drop_rule"],
            )
        )
    lines.extend(["", "## Details", ""])
    for row in report["mechanisms"]:
        lines.append(f"### {row['mechanism']}")
        lines.append(f"- Why it matters: {row['why_it_matters']}")
        lines.append(f"- Local evidence: {_format_value(row['local_evidence'])}")
        lines.append(f"- Linked routes: {_format_value(row['linked_routes'])}")
        lines.append(f"- Metric status: `{row['metric_status']}`")
        lines.append("")
    lines.extend(["## Decision Policy", ""])
    lines.extend(f"- {item}" for item in report["decision_policy"])
    lines.append("")
    return "\n".join(lines)


def _route_status(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    return {
        str(row.get("route")): {
            "route": row.get("route"),
            "implementation_ready": row.get("implementation_ready"),
            "metric_status": row.get("metric_status"),
            "configs": row.get("configs", []),
        }
        for row in rows
        if row.get("route")
    }


def _compact_evidence(snapshot: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: snapshot.get(key) for key in keys}


def _dataset_dimensions(dataset_style: dict[str, Any], quality_summary: dict[str, Any]) -> dict[str, Any]:
    if isinstance(dataset_style.get("dimension_top"), dict):
        return dataset_style.get("dimension_top")
    overall = quality_summary.get("overall", {}) if isinstance(quality_summary, dict) else {}
    by_dataset = quality_summary.get("by_dataset", {}) if isinstance(quality_summary, dict) else {}
    return {
        "overall": overall.get("dimension_top"),
        "by_dataset": {
            name: value.get("dimension_top")
            for name, value in by_dataset.items()
            if isinstance(value, dict) and value.get("dimension_top")
        }
        if isinstance(by_dataset, dict)
        else {},
    }


def _row_by_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in rows:
        if row.get("method") == method:
            return row
    return {}


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


def _quality_num_stats(payload: dict[str, Any]) -> Any:
    checks = payload.get("checks", []) if isinstance(payload.get("checks"), list) else []
    for row in checks:
        if isinstance(row, dict) and row.get("name") == "quality_stats_expanded":
            return row.get("quality_num_stats")
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


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > 420:
        return text[:417] + "..."
    return text


if __name__ == "__main__":
    main()
