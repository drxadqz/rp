from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/algorithm_module_audit.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/algorithm_module_audit.json")


RUN_ORDER = [
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
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
    "lodo_roadsaw_full_faf",
    "lodo_rscd_full_faf",
    "lodo_roadsc_full_faf",
    "single_roadsaw_full_faf",
    "single_rscd_full_faf",
    "single_roadsc_full_faf",
    "baseline_single_roadsaw_global_convnext",
    "baseline_single_rscd_global_convnext",
    "baseline_single_roadsc_global_convnext",
    "final_lodo_roadsaw_lean_road_roi_safety",
    "final_lodo_rscd_lean_road_roi_safety",
    "final_lodo_roadsc_lean_road_roi_safety",
    "final_single_roadsaw_lean_road_roi_safety",
    "final_single_rscd_lean_road_roi_safety",
    "final_single_roadsc_lean_road_roi_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    rows = []
    for name in RUN_ORDER:
        cfg_path = args.config_dir / f"{name}.yaml"
        rows.append(inspect_config(name, cfg_path))
    report = {
        "config_dir": str(args.config_dir),
        "rows": rows,
        "source_implementation": source_implementation_map(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def inspect_config(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"run": name, "config": str(path), "status": "missing_config"}
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    loss = cfg.get("loss", {})
    aug = data.get("augmentation", {})
    modules = {
        "physics_texture": bool(model.get("use_physics_branch", False)),
        "physics_quality_cues": bool(model.get("physics_quality_cues", False)),
        "wet_optical_quality_cues": bool(model.get("physics_quality_cues", False)),
        "friction_set": bool(model.get("use_friction_set", False)),
        "dg_losses": any_positive(
            loss,
            [
                "group_dro_weight",
                "group_vrex_weight",
                "feature_coral_weight",
            ],
        ),
        "semantic_conditional_alignment": any_positive(
            loss,
            [
                "risk_conditional_coral_weight",
                "wetness_conditional_coral_weight",
            ],
        ),
        "risk_conditional_coral": positive(loss.get("risk_conditional_coral_weight")),
        "wetness_conditional_coral": positive(loss.get("wetness_conditional_coral_weight")),
        "state_contrastive_alignment": any_positive(
            loss,
            [
                "risk_state_contrastive_weight",
                "friction_state_contrastive_weight",
                "wetness_state_contrastive_weight",
            ],
        ),
        "interval_order_consistency": positive(loss.get("interval_order_weight")),
        "evidence_field": bool(model.get("use_evidence_field", False)),
        "evidence_final_mix": bool(model.get("use_evidence_field", False))
        and (positive(model.get("evidence_interval_mix")) or positive(model.get("evidence_risk_logit_mix"))),
        "photometric_jitter": color_jitter_enabled(aug.get("color_jitter", {})),
        "grayscale_aug": positive(aug.get("random_grayscale_p")),
        "blur_aug": positive(aug.get("gaussian_blur_p")),
        "random_erasing": positive(aug.get("random_erasing_p")),
        "fourier_style_jitter": positive(aug.get("fourier_low_freq_jitter_p")),
        "bottom_square_input_canonicalization": str(aug.get("resize_mode", "")).lower()
        in {"bottom_square", "bottom_center_square", "road_bottom_square"},
        "gray_world_color_constancy": positive(aug.get("gray_world_alpha")),
        "dann": positive(loss.get("domain_weight")) and positive(loss.get("domain_grl_lambda")),
        "road_likelihood_prior": positive(model.get("evidence_road_likelihood_prior_strength")),
        "region_mixture_evidence": bool(model.get("evidence_region_mixture_cues", False))
        or positive(model.get("evidence_region_mixture_expansion")),
        "multi_query_evidence": positive(model.get("evidence_num_queries", 1))
        and int(model.get("evidence_num_queries", 1)) > 1,
        "query_disagreement_uncertainty": positive(model.get("evidence_query_disagreement_expansion")),
        "query_attention_diversity": positive(loss.get("evidence_query_diversity_weight")),
        "pseudo_road_mask_supervision": positive(loss.get("evidence_attention_pseudo_road_weight")),
        "condition_hard_sampling": bool(data.get("balanced_weight_overrides")),
        "dataset_scoped_sampling": any(
            isinstance(item, dict) and isinstance(item.get("where"), dict) and bool(item["where"].get("dataset"))
            for item in data.get("balanced_weight_overrides", [])
        ),
        "weak_view_consistency": positive(loss.get("aug_consistency_weight")),
        "masked_image_consistency": positive(loss.get("aug_consistency_mask_ratio")),
        "mask_aware_consistency": positive(loss.get("aug_consistency_attention_weight"))
        and str(loss.get("aug_consistency_attention_mask", "none")).lower() not in {"", "none", "full", "global"},
        "domain_adapter": bool(model.get("use_domain_adapters", False)),
        "domain_adapter_regularized": positive(loss.get("domain_adapter_weight")),
        "feature_mixstyle": bool(model.get("use_feature_mixstyle", False)),
        "roi_attention_constraint": positive(loss.get("evidence_attention_region_weight")),
        "coverage_aware_training": positive(loss.get("coverage_weight")) or positive(loss.get("coverage_margin")),
        "safety_weighted_coverage": any_positive(
            loss,
            ["coverage_risk_weight", "coverage_wetness_weight", "coverage_snow_weight"],
        ),
        "visual_quality_weighted_coverage": any_positive(
            loss,
            ["coverage_near_white_weight", "coverage_low_texture_weight", "coverage_specular_weight"],
        ),
        "wetness_ordinal_loss": positive(loss.get("wetness_ordinal_weight")),
    }
    return {
        "run": name,
        "config": str(path),
        "status": "ok",
        "note": cfg.get("experiment_note"),
        "modules": modules,
    }


def any_positive(items: dict[str, Any], keys: list[str]) -> bool:
    return any(positive(items.get(key)) for key in keys)


def color_jitter_enabled(items: Any) -> bool:
    if not isinstance(items, dict):
        return False
    return any_positive(items, ["brightness", "contrast", "saturation", "hue"])


def positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def source_implementation_map() -> dict[str, list[str]]:
    return {
        "physics_texture": ["src/friction_affordance/models/texture.py"],
        "physics_quality_cues": ["src/friction_affordance/models/texture.py", "scripts/make_paper_protocol_configs.py"],
        "wet_optical_quality_cues": ["src/friction_affordance/models/texture.py"],
        "friction_set": ["src/friction_affordance/models/friction_set.py", "src/friction_affordance/losses.py"],
        "dg_losses": ["src/friction_affordance/losses.py"],
        "semantic_conditional_alignment": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "risk_conditional_coral": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "wetness_conditional_coral": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "state_contrastive_alignment": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "interval_order_consistency": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "evidence_field": ["src/friction_affordance/models/evidence_field.py", "src/friction_affordance/losses.py"],
        "photometric_jitter": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "grayscale_aug": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "blur_aug": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "random_erasing": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "fourier_style_jitter": ["src/friction_affordance/transforms.py"],
        "bottom_square_input_canonicalization": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "gray_world_color_constancy": ["src/friction_affordance/transforms.py", "scripts/make_paper_protocol_configs.py"],
        "dann": ["src/friction_affordance/models/friction_affordance.py", "src/friction_affordance/losses.py"],
        "road_likelihood_prior": ["src/friction_affordance/models/evidence_field.py"],
        "region_mixture_evidence": ["src/friction_affordance/models/evidence_field.py", "scripts/make_paper_protocol_configs.py"],
        "multi_query_evidence": ["src/friction_affordance/models/evidence_field.py"],
        "query_disagreement_uncertainty": ["src/friction_affordance/models/evidence_field.py"],
        "query_attention_diversity": ["src/friction_affordance/losses.py"],
        "pseudo_road_mask_supervision": ["src/friction_affordance/losses.py", "src/friction_affordance/models/evidence_field.py"],
        "condition_hard_sampling": ["src/friction_affordance/engine.py", "scripts/make_paper_protocol_configs.py"],
        "dataset_scoped_sampling": ["scripts/make_paper_protocol_configs.py"],
        "weak_view_consistency": ["src/friction_affordance/engine.py", "src/friction_affordance/losses.py"],
        "masked_image_consistency": ["src/friction_affordance/engine.py", "scripts/make_paper_protocol_configs.py"],
        "mask_aware_consistency": ["src/friction_affordance/engine.py", "src/friction_affordance/losses.py"],
        "domain_adapter": ["src/friction_affordance/models/friction_affordance.py"],
        "feature_mixstyle": ["src/friction_affordance/models/friction_affordance.py", "scripts/make_paper_protocol_configs.py"],
        "roi_attention_constraint": ["src/friction_affordance/losses.py", "src/friction_affordance/models/evidence_field.py"],
        "coverage_aware_training": ["src/friction_affordance/losses.py", "scripts/calibrate_intervals.py"],
        "safety_weighted_coverage": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "visual_quality_weighted_coverage": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
        "wetness_ordinal_loss": ["src/friction_affordance/losses.py", "scripts/make_paper_protocol_configs.py"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    module_names = [
        "physics_texture",
        "physics_quality_cues",
        "wet_optical_quality_cues",
        "friction_set",
        "dg_losses",
        "semantic_conditional_alignment",
        "risk_conditional_coral",
        "wetness_conditional_coral",
        "state_contrastive_alignment",
        "interval_order_consistency",
        "evidence_field",
        "evidence_final_mix",
        "photometric_jitter",
        "grayscale_aug",
        "blur_aug",
        "random_erasing",
        "fourier_style_jitter",
        "bottom_square_input_canonicalization",
        "gray_world_color_constancy",
        "dann",
        "road_likelihood_prior",
        "region_mixture_evidence",
        "multi_query_evidence",
        "query_disagreement_uncertainty",
        "query_attention_diversity",
        "pseudo_road_mask_supervision",
        "condition_hard_sampling",
        "dataset_scoped_sampling",
        "weak_view_consistency",
        "masked_image_consistency",
        "mask_aware_consistency",
        "domain_adapter",
        "domain_adapter_regularized",
        "feature_mixstyle",
        "roi_attention_constraint",
        "coverage_aware_training",
        "safety_weighted_coverage",
        "visual_quality_weighted_coverage",
        "wetness_ordinal_loss",
    ]
    lines = ["# Algorithm Module Audit", "", f"Config dir: `{report['config_dir']}`", ""]
    lines.append("| Run | " + " | ".join(module_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(module_names)) + "|")
    for row in report["rows"]:
        modules = row.get("modules", {})
        values = ["yes" if modules.get(name) else "-" for name in module_names]
        lines.append(f"| {row['run']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Implementation Files", ""])
    for module, files in report["source_implementation"].items():
        joined = ", ".join(f"`{item}`" for item in files)
        lines.append(f"- `{module}`: {joined}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
