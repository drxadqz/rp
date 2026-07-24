from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path("configs/experiments/paper_protocol")
DEFAULT_SOURCE_ROOT = Path(".")
DEFAULT_OUT_MD = Path("reports/paper_protocol_summary/config_to_code_trace_report.md")
DEFAULT_OUT_JSON = Path("reports/paper_protocol_summary/config_to_code_trace_report.json")


TRACE_ITEMS: list[dict[str, Any]] = [
    {
        "name": "physics_texture_branch",
        "activator_paths": ["model.use_physics_branch"],
        "config_paths": [
            "model.use_physics_branch",
            "model.physics_dim",
            "model.physics_quality_cues",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/texture.py",
        ],
        "source_tokens": ["PhysicsTextureBranch", "use_physics_branch", "num_stats"],
        "claim": "Physics-inspired road texture statistics are a real model branch, not only an experiment label.",
    },
    {
        "name": "physics_quality_cues",
        "activator_paths": ["model.physics_quality_cues"],
        "config_paths": ["model.physics_quality_cues"],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/texture.py",
            "src/friction_affordance/models/evidence_field.py",
        ],
        "source_tokens": ["quality_cues", "specular", "low_texture", "wet_proxy"],
        "claim": "Wet-road quality cues such as specular highlights, dark water, and low texture are implemented for the v17/v18/v19/v20/v21/v22/v23 routes.",
    },
    {
        "name": "photometric_style_augmentation",
        "activator_paths": [
            "data.augmentation.color_jitter.brightness",
            "data.augmentation.color_jitter.contrast",
            "data.augmentation.color_jitter.saturation",
            "data.augmentation.color_jitter.hue",
            "data.augmentation.random_grayscale_p",
            "data.augmentation.gaussian_blur_p",
            "data.augmentation.random_erasing_p",
        ],
        "config_paths": [
            "data.augmentation.color_jitter.brightness",
            "data.augmentation.color_jitter.contrast",
            "data.augmentation.color_jitter.saturation",
            "data.augmentation.color_jitter.hue",
            "data.augmentation.random_grayscale_p",
            "data.augmentation.gaussian_blur_p",
            "data.augmentation.random_erasing_p",
        ],
        "source_files": ["src/friction_affordance/transforms.py"],
        "source_tokens": ["ColorJitter", "RandomGrayscale", "GaussianBlur", "RandomErasing"],
        "claim": "Camera/color/blur/erasing augmentation is executable, not only declared in YAML.",
    },
    {
        "name": "fourier_low_frequency_style_jitter",
        "activator_paths": ["data.augmentation.fourier_low_freq_jitter_p"],
        "config_paths": [
            "data.augmentation.fourier_low_freq_jitter_p",
            "data.augmentation.fourier_beta",
            "data.augmentation.fourier_strength",
        ],
        "source_files": ["src/friction_affordance/transforms.py"],
        "source_tokens": ["FourierLowFrequencyJitter", "fourier_low_freq_jitter_p", "torch.fft"],
        "claim": "Low-frequency amplitude jitter is implemented as a train-time transform.",
    },
    {
        "name": "bottom_square_input_canonicalization",
        "activator_paths": ["data.augmentation.resize_mode"],
        "config_paths": ["data.augmentation.resize_mode"],
        "source_files": ["src/friction_affordance/transforms.py"],
        "source_tokens": ["BottomSquareCropResize", "resize_mode", "bottom_square"],
        "claim": "Bottom-centered square road cropping is implemented as a deterministic input transform.",
    },
    {
        "name": "gray_world_color_constancy",
        "activator_paths": ["data.augmentation.gray_world_alpha"],
        "config_paths": ["data.augmentation.gray_world_alpha"],
        "source_files": ["src/friction_affordance/transforms.py"],
        "source_tokens": ["GrayWorldColorConstancy", "gray_world_alpha", "channel_mean"],
        "claim": "Soft gray-world color constancy is implemented as a deterministic style-canonicalization transform.",
    },
    {
        "name": "condition_hard_sampling",
        "activator_paths": ["data.balanced_weight_overrides"],
        "config_paths": ["data.balanced_weight_overrides"],
        "source_files": ["src/friction_affordance/engine.py"],
        "source_tokens": ["balanced_weight_overrides", "WeightedRandomSampler", "_balanced_sampling_weights"],
        "claim": "RoadSaW damp/wet/very_wet hard sampling reaches the DataLoader sampler.",
    },
    {
        "name": "domain_adversarial_training",
        "activator_paths": ["loss.domain_weight", "loss.domain_grl_lambda"],
        "require_all_activators": True,
        "config_paths": ["loss.domain_weight", "loss.domain_grl_lambda"],
        "source_files": [
            "src/friction_affordance/engine.py",
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/losses.py",
        ],
        "source_tokens": ["grl_lambda", "_grad_reverse", "domain_logits", "domain_weight"],
        "claim": "DANN-style gradient reversal is connected from loss config to model forward.",
    },
    {
        "name": "condition_aware_coral_alignment",
        "activator_paths": [
            "loss.feature_coral_weight",
            "loss.risk_conditional_coral_weight",
            "loss.wetness_conditional_coral_weight",
        ],
        "config_paths": [
            "loss.feature_coral_weight",
            "loss.risk_conditional_coral_weight",
            "loss.wetness_conditional_coral_weight",
            "loss.coral_min_samples_per_domain",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": [
            "feature_coral_loss",
            "risk_conditional_coral_weight",
            "wetness_conditional_coral_weight",
            "condition_idx",
        ],
        "claim": "Domain alignment can be unconditional or conditioned on risk/wetness states.",
    },
    {
        "name": "state_contrastive_alignment",
        "activator_paths": [
            "loss.risk_state_contrastive_weight",
            "loss.friction_state_contrastive_weight",
            "loss.wetness_state_contrastive_weight",
        ],
        "config_paths": [
            "loss.state_contrastive_temperature",
            "loss.state_contrastive_cross_domain_only",
            "loss.risk_state_contrastive_weight",
            "loss.friction_state_contrastive_weight",
            "loss.wetness_state_contrastive_weight",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": [
            "state_contrastive_loss",
            "risk_state_contrastive_weight",
            "friction_state_contrastive_weight",
            "wetness_state_contrastive_weight",
        ],
        "claim": "Cross-dataset samples with the same weak road state can be explicitly aligned while different states remain separable.",
    },
    {
        "name": "interval_order_consistency",
        "activator_paths": ["loss.interval_order_weight"],
        "config_paths": [
            "loss.interval_order_weight",
            "loss.interval_order_margin_scale",
            "loss.interval_order_min_gap",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": [
            "interval_order_consistency_loss",
            "interval_order_weight",
            "interval_order_margin_scale",
            "interval_order_min_gap",
        ],
        "claim": "Non-overlapping weak friction intervals can provide a physical pairwise order signal without measured tire-road friction labels.",
    },
    {
        "name": "domain_specific_adapter",
        "activator_paths": ["model.use_domain_adapters"],
        "config_paths": [
            "model.use_domain_adapters",
            "model.domain_adapter_scale",
            "loss.domain_adapter_weight",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/losses.py",
        ],
        "source_tokens": ["DomainAffineAdapter", "domain_adapter_weight", "domain_adapter_penalty"],
        "claim": "Small per-domain affine adapters are regularized and used in forward.",
    },
    {
        "name": "feature_mixstyle_shortcut_probe",
        "activator_paths": ["model.use_feature_mixstyle"],
        "config_paths": [
            "model.use_feature_mixstyle",
            "model.feature_mixstyle_p",
            "model.feature_mixstyle_alpha",
        ],
        "source_files": ["src/friction_affordance/models/friction_affordance.py"],
        "source_tokens": ["FeatureMixStyle", "use_feature_mixstyle", "feature_mixstyle_alpha"],
        "claim": "Training-only feature-statistics mixing is implemented as a cheap shortcut-mitigation candidate.",
    },
    {
        "name": "evidence_field",
        "activator_paths": ["model.use_evidence_field"],
        "config_paths": [
            "model.use_evidence_field",
            "model.evidence_entropy_expansion",
            "model.evidence_interval_mix",
            "model.evidence_risk_logit_mix",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/evidence_field.py",
            "src/friction_affordance/losses.py",
        ],
        "source_tokens": ["LocalFrictionEvidenceField", "evidence_interval_mix", "evidence_risk_logit_mix"],
        "claim": "Local visual evidence is a model branch with prediction/interval fusion.",
    },
    {
        "name": "road_likelihood_prior",
        "activator_paths": ["model.evidence_road_likelihood_prior_strength"],
        "config_paths": ["model.evidence_road_likelihood_prior_strength"],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/evidence_field.py",
        ],
        "source_tokens": ["evidence_road_likelihood_prior_strength", "road_likelihood_prior_strength", "_road_likelihood"],
        "claim": "Evidence attention can be biased toward heuristic road-likelihood regions.",
    },
    {
        "name": "region_mixture_evidence",
        "activator_paths": ["model.evidence_region_mixture_cues", "model.evidence_region_mixture_expansion"],
        "config_paths": [
            "model.evidence_region_mixture_cues",
            "model.evidence_region_mixture_expansion",
            "model.evidence_region_mixture_kernel_size",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/evidence_field.py",
        ],
        "source_tokens": [
            "evidence_region_mixture_cues",
            "region_mixture_expansion",
            "_region_mixture_cues",
            "region_mixture_signal",
        ],
        "claim": "Segmentation-style local region-mixture cues can expand friction intervals on visually heterogeneous road evidence.",
    },
    {
        "name": "multi_query_evidence",
        "activator_paths": ["model.evidence_num_queries"],
        "config_paths": [
            "model.evidence_num_queries",
            "loss.evidence_query_diversity_weight",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/evidence_field.py",
            "src/friction_affordance/losses.py",
        ],
        "source_tokens": [
            "evidence_num_queries",
            "attention_queries",
            "query_gate",
            "attention_query_diversity_loss",
        ],
        "claim": "Segmentation-style mask-query pooling lets multiple latent local evidence maps represent heterogeneous road regions.",
    },
    {
        "name": "query_disagreement_uncertainty",
        "activator_paths": ["model.evidence_query_disagreement_expansion"],
        "config_paths": [
            "model.evidence_query_disagreement_expansion",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/evidence_field.py",
        ],
        "source_tokens": [
            "evidence_query_disagreement_expansion",
            "query_disagreement_expansion",
            "query_disagreement",
        ],
        "claim": "Disagreement among local evidence queries can conservatively expand weak friction intervals on mixed road appearances.",
    },
    {
        "name": "bottom_roi_attention_constraint",
        "activator_paths": ["loss.evidence_attention_region_weight"],
        "config_paths": [
            "loss.evidence_attention_region_weight",
            "loss.evidence_bottom_mass_target",
            "loss.evidence_center_bottom_mass_target",
            "loss.evidence_top_mass_max",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": ["attention_region_mass_losses", "evidence_bottom_mass_target", "evidence_top_mass_max"],
        "claim": "Evidence attention has explicit bottom-road/center-bottom mass constraints.",
    },
    {
        "name": "pseudo_road_mask_attention",
        "activator_paths": ["loss.evidence_attention_pseudo_road_weight"],
        "config_paths": [
            "loss.evidence_attention_pseudo_road_weight",
            "loss.evidence_pseudo_road_min_mass",
            "loss.evidence_pseudo_road_threshold",
        ],
        "source_files": [
            "src/friction_affordance/losses.py",
            "src/friction_affordance/models/evidence_field.py",
        ],
        "source_tokens": ["attention_soft_mask_mass_loss", "evidence_pseudo_road_min_mass", "road_likelihood"],
        "claim": "Pseudo-road supervision can use dataset masks if present or the built-in soft road-likelihood map.",
    },
    {
        "name": "weak_view_consistency",
        "activator_paths": ["loss.aug_consistency_weight"],
        "config_paths": [
            "loss.aug_consistency_weight",
            "loss.aug_consistency_interval_weight",
            "loss.aug_consistency_attention_weight",
        ],
        "source_files": ["src/friction_affordance/engine.py", "src/friction_affordance/losses.py"],
        "source_tokens": ["_weak_style_perturb_normalized", "prediction_consistency_loss", "aug_consistency_weight"],
        "claim": "A weakly perturbed view is compared against the clean prediction/evidence branch.",
    },
    {
        "name": "mask_aware_consistency",
        "activator_paths": ["loss.aug_consistency_attention_mask"],
        "config_paths": [
            "loss.aug_consistency_attention_weight",
            "loss.aug_consistency_attention_mask",
            "loss.aug_consistency_attention_mask_threshold",
            "loss.aug_consistency_attention_mask_sharpness",
        ],
        "source_files": ["src/friction_affordance/engine.py", "src/friction_affordance/losses.py"],
        "source_tokens": ["_consistency_attention_mask", "aug_consistency_attention_mask", "road_likelihood"],
        "claim": "Attention consistency is restricted to road/ROI evidence support instead of forcing full-image attention agreement.",
    },
    {
        "name": "coverage_aware_interval_training",
        "activator_paths": ["loss.coverage_weight", "loss.coverage_margin"],
        "config_paths": ["loss.coverage_weight", "loss.coverage_margin"],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": ["interval_coverage_loss", "coverage_weight", "coverage_margin"],
        "claim": "Raw interval coverage is directly trained with a coverage violation term.",
    },
    {
        "name": "safety_weighted_coverage",
        "activator_paths": [
            "loss.coverage_risk_weight",
            "loss.coverage_wetness_weight",
            "loss.coverage_snow_weight",
        ],
        "config_paths": [
            "loss.coverage_risk_weight",
            "loss.coverage_wetness_weight",
            "loss.coverage_snow_weight",
            "loss.coverage_weight_max",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": ["build_safety_coverage_weights", "coverage_risk_weight", "coverage_wetness_weight"],
        "claim": "Coverage errors can be upweighted for high-risk/wet/snow states.",
    },
    {
        "name": "visual_quality_weighted_coverage",
        "activator_paths": [
            "loss.coverage_near_white_weight",
            "loss.coverage_low_texture_weight",
            "loss.coverage_specular_weight",
        ],
        "config_paths": [
            "loss.coverage_near_white_weight",
            "loss.coverage_low_texture_weight",
            "loss.coverage_specular_weight",
            "loss.coverage_weight_max",
        ],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": [
            "visual_quality_coverage_weight",
            "coverage_near_white_weight",
            "coverage_low_texture_weight",
            "coverage_specular_weight",
        ],
        "claim": "Coverage errors can be upweighted on near-white, low-texture, or specular visual-quality slices without adding new labels.",
    },
    {
        "name": "wet_optical_quality_cues",
        "activator_paths": ["model.physics_quality_cues"],
        "config_paths": ["model.physics_quality_cues"],
        "source_files": ["src/friction_affordance/models/texture.py"],
        "source_tokens": ["smooth_bright", "smooth_dark", "mirror_candidate", "thin_water"],
        "claim": "PhysicsTexture quality mode includes water-film, mirror-like reflection, and low-texture dark wet/ice proxies for wet-road ambiguity.",
    },
    {
        "name": "wetness_ordinal_supervision",
        "activator_paths": ["loss.wetness_ordinal_weight"],
        "config_paths": ["loss.wetness_ordinal_weight"],
        "source_files": ["src/friction_affordance/losses.py"],
        "source_tokens": ["wetness_ordinal_weight", "ordinal_cdf_emd_loss"],
        "claim": "RoadSaW-sensitive wetness labels are trained with ordinal structure.",
    },
    {
        "name": "friction_set_interval_expansion",
        "activator_paths": ["model.use_friction_set"],
        "config_paths": [
            "model.use_friction_set",
            "model.friction_set_entropy_expansion",
            "model.friction_set_interval_mix",
        ],
        "source_files": [
            "src/friction_affordance/models/friction_affordance.py",
            "src/friction_affordance/models/friction_set.py",
        ],
        "source_tokens": ["FrictionSetHead", "friction_set_entropy_expansion", "friction_set_interval_mix"],
        "claim": "FrictionSet interval expansion is implemented for ablation and possible rescue.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.config_dir, args.source_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(config_dir: Path, source_root: Path) -> dict[str, Any]:
    configs = _load_configs(config_dir)
    rows = [_trace_item(item, configs, source_root) for item in TRACE_ITEMS]
    checks: list[dict[str, Any]] = []
    for row in rows:
        if row["num_configured_runs"] <= 0:
            checks.append(
                {
                    "level": "warn",
                    "name": f"{row['name']}_not_configured",
                    "message": f"{row['name']} is not enabled by any queued config.",
                }
            )
        elif not row["source_ok"]:
            checks.append(
                {
                    "level": "block",
                    "name": f"{row['name']}_source_trace_missing",
                    "message": f"{row['name']} is configured but expected source tokens are missing.",
                    "missing_tokens": row["missing_tokens"],
                    "source_files": row["source_files"],
                }
            )
        else:
            checks.append(
                {
                    "level": "pass",
                    "name": f"{row['name']}_source_trace",
                    "message": f"{row['name']} is configured and backed by source-code references.",
                }
            )
    blocks = [item for item in checks if item["level"] == "block"]
    warns = [item for item in checks if item["level"] == "warn"]
    verdict = "pass" if not blocks else "block"
    if warns and not blocks:
        verdict = "pass_with_warnings"
    return {
        "config_dir": str(config_dir),
        "source_root": str(source_root),
        "verdict": verdict,
        "num_rows": len(rows),
        "num_blocks": len(blocks),
        "num_warnings": len(warns),
        "rows": rows,
        "checks": checks,
        "policy": [
            "Configured innovation modules must have an explicit source-code trace before their results are used as paper evidence.",
            "This report checks implementation reachability only; metric claims still require completed runs, LODO, dataset-ID probes, and matched baselines.",
        ],
    }


def _load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    if not config_dir.exists():
        return configs
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        configs[path.stem] = cfg
    return configs


def _trace_item(item: dict[str, Any], configs: dict[str, dict[str, Any]], source_root: Path) -> dict[str, Any]:
    configured_runs = []
    field_values: dict[str, dict[str, Any]] = {}
    for run, cfg in sorted(configs.items()):
        active_fields: dict[str, Any] = {}
        activator_values = [_get_nested(cfg, field) for field in item.get("activator_paths", item["config_paths"])]
        if item.get("require_all_activators"):
            is_enabled = all(_is_active_value(value) for value in activator_values)
        else:
            is_enabled = any(_is_active_value(value) for value in activator_values)
        if not is_enabled:
            continue
        for field in item["config_paths"]:
            value = _get_nested(cfg, field)
            if value is not None:
                active_fields[field] = value
        configured_runs.append(run)
        field_values[run] = active_fields

    source_text = ""
    existing_files = []
    missing_files = []
    for rel in item["source_files"]:
        path = source_root / rel
        if path.exists():
            existing_files.append(rel)
            source_text += "\n" + path.read_text(encoding="utf-8", errors="ignore")
        else:
            missing_files.append(rel)
    missing_tokens = [token for token in item["source_tokens"] if token not in source_text]
    source_ok = bool(existing_files) and not missing_files and not missing_tokens
    return {
        "name": item["name"],
        "claim": item["claim"],
        "config_paths": item["config_paths"],
        "source_files": item["source_files"],
        "source_tokens": item["source_tokens"],
        "configured_runs": configured_runs,
        "num_configured_runs": len(configured_runs),
        "field_values": field_values,
        "existing_source_files": existing_files,
        "missing_source_files": missing_files,
        "missing_tokens": missing_tokens,
        "source_ok": source_ok,
    }


def _get_nested(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_active_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_is_active_value(item) for item in value.values())
    return bool(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Config-To-Code Trace Report",
        "",
        f"Config dir: `{report['config_dir']}`",
        f"Source root: `{report['source_root']}`",
        f"Verdict: `{report['verdict']}` ({report['num_blocks']} blocks, {report['num_warnings']} warnings)",
        "",
        "This report checks that configured innovation knobs are backed by executable source-code paths.",
        "It does not replace metric evidence from completed experiments.",
        "",
        "## Trace Rows",
        "",
        "| Module | Configured runs | Source trace | Missing source/tokens | Claim |",
        "|---|---:|---|---|---|",
    ]
    for row in report["rows"]:
        missing = []
        missing.extend(row.get("missing_source_files", []))
        missing.extend(row.get("missing_tokens", []))
        lines.append(
            "| {name} | {runs} | {source} | {missing} | {claim} |".format(
                name=row["name"],
                runs=row["num_configured_runs"],
                source="pass" if row["source_ok"] else "missing",
                missing=", ".join(f"`{item}`" for item in missing) or "-",
                claim=row["claim"],
            )
        )
    lines.extend(["", "## Configured Run Map", ""])
    for row in report["rows"]:
        runs = ", ".join(f"`{run}`" for run in row["configured_runs"][:12]) or "-"
        more = len(row["configured_runs"]) - 12
        suffix = f" (+{more} more)" if more > 0 else ""
        lines.append(f"- `{row['name']}`: {runs}{suffix}")
    lines.extend(["", "## Policy", ""])
    for item in report.get("policy", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
