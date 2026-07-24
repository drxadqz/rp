from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "cv_transfer_experiment_protocol.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "cv_transfer_experiment_protocol.json"


ROUTES = [
    {
        "route": "semantic_segmentation_local_evidence",
        "cv_subfield": "semantic segmentation and mask classification",
        "paper_anchors": ["Mask2Former", "Mask DINO", "SegFormer"],
        "configs": [
            "v14_lean_road_roi_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "required_source_tokens": {
            "src/friction_affordance/models/evidence_field.py": [
                "LocalFrictionEvidenceField",
                "road_likelihood",
                "region_mixture_signal",
                "attention_queries",
                "query_disagreement",
            ],
            "src/friction_affordance/losses.py": [
                "attention_region_mass_losses",
                "attention_soft_mask_mass_loss",
                "attention_query_diversity_loss",
            ],
        },
        "required_config_paths": [
            "model.use_evidence_field",
            "model.evidence_road_likelihood_prior_strength",
            "loss.evidence_attention_region_weight",
            "loss.evidence_attention_pseudo_road_weight",
            "model.evidence_num_queries",
            "loss.evidence_query_diversity_weight",
        ],
        "data_fit": "Uses image-level RSCD/RoadSaW/RoadSC labels; no pixel-level mask labels are required.",
        "promotion_rule": (
            "Keep if RoadSaW wet/near-white slices, low-friction recall, query/attention diagnostics, "
            "or conditional coverage-width improve without hurting matched single-dataset risk F1."
        ),
        "drop_rule": "If attention maps do not localize road evidence or safety metrics regress, keep only as visualization or prune.",
    },
    {
        "route": "mask_aware_weak_strong_consistency",
        "cv_subfield": "semi-supervised semantic segmentation consistency",
        "paper_anchors": ["UniMatch", "MIC-style masked consistency"],
        "configs": [
            "v10_full_faf_consistency",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "required_source_tokens": {
            "src/friction_affordance/losses.py": [
                "prediction_consistency_loss",
                "_consistency_attention_mask",
                "aug_consistency_attention_mask",
            ],
            "src/friction_affordance/engine.py": [
                "aug_consistency_attention_mask",
                "prediction_consistency_loss",
            ],
        },
        "required_config_paths": [
            "loss.aug_consistency_weight",
            "loss.aug_consistency_mask_ratio",
            "loss.aug_consistency_attention_weight",
            "loss.aug_consistency_attention_mask",
        ],
        "data_fit": "Needs only existing images; augmentations generate weak/strong views.",
        "promotion_rule": (
            "Keep if prediction and interval stability improve while low-friction recall and wet/snow hard states do not collapse."
        ),
        "drop_rule": "If it oversmooths hard wet/snow states, reduce attention weight or remove from final configs.",
    },
    {
        "route": "domain_adaptive_segmentation_shortcut_control",
        "cv_subfield": "domain-adaptive semantic segmentation and domain generalization",
        "paper_anchors": ["DAFormer", "HRDA", "FDA", "MixStyle", "DomainBed"],
        "configs": [
            "v6_full_faf_fourier",
            "v7_full_faf_fourier_dann",
            "v11_full_faf_domain_adapter",
            "v18_lean_mixstyle_quality_safety",
        ],
        "required_source_tokens": {
            "src/friction_affordance/transforms.py": ["fourier_low_freq_jitter", "gray_world"],
            "src/friction_affordance/models/friction_affordance.py": ["MixStyle", "domain_adapter"],
            "src/friction_affordance/losses.py": ["conditional_coral_loss", "state_contrastive_loss"],
        },
        "required_config_paths": [
            "data.augmentation.fourier_low_freq_jitter_p",
            "loss.risk_conditional_coral_weight",
            "model.num_domains",
        ],
        "data_fit": "Uses dataset IDs and shared risk/friction labels to reduce style shortcut without mixing incompatible states.",
        "promotion_rule": (
            "Keep only if dataset-ID probe falls and matched risk F1/low-friction recall remain safe."
        ),
        "drop_rule": "Generic DANN/DG losses are removed or merged if they repeat the P0 safety regressions.",
    },
    {
        "route": "material_texture_physical_vision",
        "cv_subfield": "material recognition, physical vision, and visual quality estimation",
        "paper_anchors": ["physics-inspired texture cues", "wet/glare/snow ambiguity modeling"],
        "configs": ["v17_lean_quality_physics_safety", "v21_lean_quality_uncertainty_safety"],
        "required_source_tokens": {
            "src/friction_affordance/models/texture.py": [
                "PhysicsTextureBranch",
                "white_hi",
                "specular",
                "low_texture",
                "smooth_bright",
                "smooth_dark",
                "mirror_candidate",
                "thin_water",
            ],
            "src/friction_affordance/losses.py": [
                "coverage_near_white_weight",
                "coverage_low_texture_weight",
                "coverage_specular_weight",
            ],
        },
        "required_config_paths": [
            "model.physics_quality_cues",
            "loss.coverage_near_white_weight",
            "loss.coverage_low_texture_weight",
            "loss.coverage_specular_weight",
        ],
        "data_fit": "Uses raw RGB cues and public wet/snow/dry labels; it estimates uncertainty, not measured friction.",
        "promotion_rule": (
            "Keep if coverage-width tradeoff improves on RoadSaW wetness and RoadSC snow/near-white slices."
        ),
        "drop_rule": "If it only widens intervals or learns brightness shortcuts, keep PhysicsTexture but remove quality weighting.",
    },
    {
        "route": "promptable_or_open_vocabulary_mask_teacher",
        "cv_subfield": "promptable and open-vocabulary segmentation",
        "paper_anchors": ["SAM", "SAM 2", "CLIPSeg", "ODISE"],
        "configs": ["smoke_opencv_mask_supervised_evidence"],
        "required_source_tokens": {
            "scripts/audit_external_segmentation_masks.py": ["backend", "mask", "road"],
            "scripts/audit_segmentation_transfer_config.py": ["road_mask", "pseudo_road_loss_active"],
            "src/friction_affordance/datasets/manifest.py": ["load_road_masks", "road_mask_path", "road_mask"],
        },
        "required_config_paths": [
            "data.load_road_masks",
            "loss.evidence_attention_pseudo_road_weight",
            "loss.evidence_pseudo_road_min_mass",
        ],
        "data_fit": "External masks are offline pseudo-labels only; public image labels remain the supervised target.",
        "promotion_rule": (
            "Promote only after mask-quality audit and a bounded ablation show better attention-on-road or wet/snow metrics."
        ),
        "drop_rule": "If pseudo masks segment glare/background instead of road contact evidence, keep the audit and do not train with them.",
    },
    {
        "route": "foundation_dense_teacher",
        "cv_subfield": "self-supervised dense visual representation",
        "paper_anchors": ["DINOv2", "MAE-style dense token probes"],
        "configs": [],
        "required_source_tokens": {},
        "required_config_paths": [],
        "data_fit": "Feasible as a teacher/baseline after fair rows; not needed for current GPU queue.",
        "promotion_rule": (
            "Use only if dense-token distillation improves local material evidence beyond ConvNeXt baselines."
        ),
        "drop_rule": "Do not claim innovation for simply swapping in a larger foundation backbone.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the executable protocol for CV-subfield transfer experiments."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir, args.config_dir, args.source_root)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path, config_dir: Path, source_root: Path) -> dict[str, Any]:
    fair_priority = _load_json(summary_dir / "fair_comparison_execution_priority.json") or {}
    mask_smoke = _load_json(summary_dir / "mask_aware_consistency_smoke.json") or {}
    wet_optical_smoke = _load_json(summary_dir / "wet_optical_quality_cues_smoke.json") or {}
    segmentation_config = _load_json(summary_dir / "segmentation_transfer_config_audit.json") or {}
    candidate_pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    algorithm_audit = _load_json(summary_dir / "algorithm_module_audit.json") or {}

    rows = []
    blocks: list[str] = []
    for spec in ROUTES:
        row = _route_row(
            spec,
            config_dir,
            source_root,
            mask_smoke,
            wet_optical_smoke,
            segmentation_config,
            algorithm_audit,
        )
        rows.append(row)
        if row["route"] != "foundation_dense_teacher" and not row["implementation_ready"]:
            blocks.append(row["route"])

    fair_stage = _first_incomplete_stage(fair_priority)
    verdict = "protocol_ready_waiting_for_metrics"
    if blocks:
        verdict = "implementation_gaps"
    elif fair_stage and fair_stage.get("name") == "finish_matched_single_dataset_fairness":
        verdict = "protocol_ready_waiting_for_fair_comparisons"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "config_dir": str(config_dir),
        "verdict": verdict,
        "claim_boundary": (
            "These routes transfer CV methods into public-label visual friction-affordance interval estimation. "
            "They do not create measured tire-road friction labels."
        ),
        "rows": rows,
        "blocks": blocks,
        "counts": {
            "routes": len(rows),
            "implementation_ready": sum(1 for row in rows if row["implementation_ready"]),
            "metric_pending": sum(1 for row in rows if row["metric_status"] == "pending"),
            "future_only": sum(1 for row in rows if row["metric_status"] == "future_only"),
        },
        "fair_dependency": {
            "verdict": fair_priority.get("verdict"),
            "first_incomplete_stage": fair_stage,
        },
        "candidate_pruning_dependency": {
            "verdict": candidate_pruning.get("verdict"),
            "policy_ready": bool(candidate_pruning.get("policy")),
        },
        "decision_policy": [
            "Run matched single-dataset ConvNeXt baselines before claiming algorithm advantage.",
            "Promote a CV-transfer module only with paired metrics, calibration-width evidence, or quantitative attention evidence.",
            "Prune or merge FrictionSet/DG/full-fusion modules if candidate metrics repeat P0 regressions.",
            "Treat external masks/foundation models as offline teachers or baselines until audited on local samples.",
        ],
    }


def _route_row(
    spec: dict[str, Any],
    config_dir: Path,
    source_root: Path,
    mask_smoke: dict[str, Any],
    wet_optical_smoke: dict[str, Any],
    segmentation_config: dict[str, Any],
    algorithm_audit: dict[str, Any],
) -> dict[str, Any]:
    config_checks = [_config_check(name, spec["required_config_paths"], config_dir) for name in spec["configs"]]
    source_checks = _source_checks(spec["required_source_tokens"], source_root)
    smoke_checks = _smoke_checks(spec["route"], mask_smoke, wet_optical_smoke, segmentation_config)
    implementation_ready = (
        all(check["ok"] for check in source_checks)
        and (not config_checks or any(check["ok"] for check in config_checks))
        and all(check["ok"] for check in smoke_checks)
    )
    metric_status = "future_only" if spec["route"] == "foundation_dense_teacher" else "pending"
    if _route_has_completed_metrics(spec["configs"], algorithm_audit):
        metric_status = "partial_metrics_available"
    return {
        "route": spec["route"],
        "cv_subfield": spec["cv_subfield"],
        "paper_anchors": spec["paper_anchors"],
        "configs": spec["configs"],
        "config_checks": config_checks,
        "source_checks": source_checks,
        "smoke_checks": smoke_checks,
        "implementation_ready": implementation_ready,
        "metric_status": metric_status,
        "data_fit": spec["data_fit"],
        "promotion_rule": spec["promotion_rule"],
        "drop_rule": spec["drop_rule"],
    }


def _config_check(name: str, required_paths: list[str], config_dir: Path) -> dict[str, Any]:
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        alt = Path("configs/experiments/segmentation_transfer") / f"{name}.yaml"
        path = alt if alt.exists() else path
    if not path.exists():
        return {"config": name, "path": str(path), "ok": False, "missing": ["config_file"]}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [item for item in required_paths if not _truthy_path(cfg, item)]
    return {"config": name, "path": str(path), "ok": not missing, "missing": missing}


def _truthy_path(cfg: dict[str, Any], dotted: str) -> bool:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    if isinstance(cur, bool):
        return cur
    if isinstance(cur, (int, float)):
        return cur > 0
    if isinstance(cur, str):
        return cur.lower() not in {"", "none", "false", "0"}
    return cur is not None


def _source_checks(required: dict[str, list[str]], source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for rel, tokens in required.items():
        path = source_root / rel
        if not path.exists():
            rows.append({"path": rel, "ok": False, "missing": tokens})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [token for token in tokens if token not in text]
        rows.append({"path": rel, "ok": not missing, "missing": missing})
    return rows


def _smoke_checks(
    route: str,
    mask_smoke: dict[str, Any],
    wet_optical_smoke: dict[str, Any],
    segmentation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if route == "mask_aware_weak_strong_consistency":
        return [
            {
                "name": "mask_aware_consistency_smoke",
                "ok": mask_smoke.get("status") in {"ok", "pass"} or mask_smoke.get("verdict") == "pass",
                "attention_loss": _nested(mask_smoke, ["logs", "loss_aug_consistency_attention"]),
                "mask_mean": _nested(mask_smoke, ["logs", "aug_consistency_attention_mask_mean"]),
            }
        ]
    if route == "promptable_or_open_vocabulary_mask_teacher":
        return [
            {
                "name": "segmentation_transfer_config_audit",
                "ok": segmentation_config.get("verdict") == "pass",
                "pseudo_loss": _nested(segmentation_config, ["batch_report", "loss_evidence_attention_pseudo_road"]),
                "road_mass": _nested(segmentation_config, ["batch_report", "attention_pseudo_road_mass"]),
            }
        ]
    if route == "material_texture_physical_vision":
        return [
            {
                "name": "wet_optical_quality_cues_smoke",
                "ok": wet_optical_smoke.get("status") == "ok",
                "quality_num_stats": _quality_num_stats(wet_optical_smoke),
            }
        ]
    return []


def _route_has_completed_metrics(configs: list[str], algorithm_audit: dict[str, Any]) -> bool:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit.get("rows"), list) else []
    completed = {row.get("run") for row in rows if row.get("progress_status") == "complete"}
    return any(name in completed for name in configs)


def _first_incomplete_stage(report: dict[str, Any]) -> dict[str, Any] | None:
    stages = report.get("stages", []) if isinstance(report.get("stages"), list) else []
    return next((stage for stage in stages if stage.get("status") != "complete"), None)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CV Transfer Experiment Protocol",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Routes",
        "",
        "| Route | CV subfield | Implementation | Metric status | Data fit | Promotion rule | Drop rule |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {route} | {subfield} | {impl} | {metric} | {data_fit} | {promote} | {drop} |".format(
                route=row["route"],
                subfield=row["cv_subfield"],
                impl="ready" if row["implementation_ready"] else "gap",
                metric=row["metric_status"],
                data_fit=row["data_fit"],
                promote=row["promotion_rule"],
                drop=row["drop_rule"],
            )
        )
    lines.extend(["", "## Route Details", ""])
    for row in report["rows"]:
        lines.append(f"### {row['route']}")
        lines.append(f"- Paper anchors: {_join(row['paper_anchors'])}")
        lines.append(f"- Configs: {_join(row['configs'])}")
        lines.append(f"- Source checks: {_check_summary(row['source_checks'])}")
        lines.append(f"- Config checks: {_config_check_summary(row['config_checks'])}")
        lines.append(f"- Smoke checks: {_check_summary(row['smoke_checks'])}")
        lines.append("")
    lines.extend(["## Decision Policy", ""])
    lines.extend(f"- {item}" for item in report["decision_policy"])
    fair = report["fair_dependency"].get("first_incomplete_stage")
    lines.extend(["", "## Current Dependency", ""])
    if fair:
        lines.append(
            f"- First incomplete stage: `{fair.get('name')}` with status `{fair.get('status')}`."
        )
    else:
        lines.append("- Fair execution chain is complete or unavailable.")
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _nested(payload: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _quality_num_stats(payload: dict[str, Any]) -> Any:
    checks = payload.get("checks", []) if isinstance(payload.get("checks"), list) else []
    for row in checks:
        if isinstance(row, dict) and row.get("name") == "quality_stats_expanded":
            return row.get("quality_num_stats")
    return None


def _join(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items) if items else "-"


def _check_summary(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "not required"
    failed = [check for check in checks if not check.get("ok")]
    if not failed:
        return "pass"
    labels = [str(check.get("config") or check.get("path") or check.get("name")) for check in failed]
    return "missing " + ", ".join(labels)


def _config_check_summary(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "not required"
    passed = [check for check in checks if check.get("ok")]
    if len(passed) == len(checks):
        return "pass"
    if passed:
        return "pass for " + ", ".join(str(check.get("config")) for check in passed)
    labels = [str(check.get("config")) for check in checks]
    return "missing " + ", ".join(labels)


if __name__ == "__main__":
    main()
