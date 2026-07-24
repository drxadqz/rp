from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "experiments" / "topvenue_v4_evidencefield_rtx3050_stable.yaml"
OUT_ROOT = ROOT / "configs" / "experiments" / "paper_protocol"
RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    configs = {}
    configs.update(make_ablation_configs(base))
    configs.update(make_lodo_configs(base))
    configs.update(make_single_dataset_configs(base))
    configs.update(make_fair_baseline_configs(base))
    configs.update(make_final_method_configs(base))
    for name, cfg in configs.items():
        path = OUT_ROOT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote: {path}")


def make_ablation_configs(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "v0_global_only": make_variant(
            base,
            "v0_global_only",
            note="Paper ablation V0: ConvNeXt global image-level multitask baseline.",
            model={
                "use_physics_branch": False,
                "use_friction_set": False,
                "use_evidence_field": False,
            },
            loss=loss_profile("global"),
        ),
        "v1_physics_texture": make_variant(
            base,
            "v1_physics_texture",
            note="Paper ablation V1: add physics-inspired texture statistics.",
            model={
                "use_physics_branch": True,
                "use_friction_set": False,
                "use_evidence_field": False,
            },
            loss=loss_profile("global"),
        ),
        "v2_friction_set": make_variant(
            base,
            "v2_friction_set",
            note="Paper ablation V2: add latent FrictionSet interval head.",
            model={
                "use_physics_branch": True,
                "use_friction_set": True,
                "use_evidence_field": False,
            },
            loss=loss_profile("friction_set"),
        ),
        "v3_dg_losses": make_variant(
            base,
            "v3_dg_losses",
            note="Paper ablation V3: add Group-DRO, V-REx, and conditional CORAL.",
            model={
                "use_physics_branch": True,
                "use_friction_set": True,
                "use_evidence_field": False,
            },
            loss=loss_profile("dg"),
        ),
        "v4_evidence_aux": evidence_safe_microbatch(
            make_variant(
                base,
                "v4_evidence_aux",
                note="Paper ablation V4: add local evidence-field features and auxiliary supervision without final interval/logit mixing.",
                model={
                    "use_physics_branch": True,
                    "use_friction_set": True,
                    "use_evidence_field": True,
                    "evidence_interval_mix": 0.0,
                    "evidence_risk_logit_mix": 0.0,
                },
                loss=loss_profile("evidence"),
            )
        ),
        "v5_full_faf": evidence_safe_microbatch(
            make_variant(
                base,
                "v5_full_faf",
                note="Paper ablation V5: full Friction Affordance Field model.",
                model={},
                loss=loss_profile("full"),
            )
        ),
        "v6_full_faf_fourier": evidence_safe_microbatch(
            make_variant(
                base,
                "v6_full_faf_fourier",
                note="Candidate improvement: full FAF with Fourier low-frequency style jitter and slightly more conservative intervals.",
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
            )
        ),
        "v7_full_faf_fourier_dann": evidence_safe_microbatch(
            make_variant(
                base,
                "v7_full_faf_fourier_dann",
                note="Candidate improvement: full FAF with Fourier low-frequency style jitter plus domain-adversarial feature learning.",
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                    "domain_weight": 0.05,
                    "domain_grl_lambda": 0.20,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
            )
        ),
        "v8_full_faf_fourier_roadprior": evidence_safe_microbatch(
            make_variant(
                base,
                "v8_full_faf_fourier_roadprior",
                note="Candidate improvement: full FAF with Fourier jitter and pseudo road-likelihood attention prior.",
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                    "evidence_road_likelihood_prior_strength": 0.60,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
            )
        ),
        "v9_full_faf_roadsaw_hard_sampling": evidence_safe_microbatch(
            make_variant(
                base,
                "v9_full_faf_roadsaw_hard_sampling",
                note=(
                    "Candidate improvement: full FAF with Fourier jitter, road prior, condition-aware "
                    "wet/damp/very-wet hard-case resampling, and ordinal wetness supervision."
                ),
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                    "evidence_road_likelihood_prior_strength": 0.60,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                    "wetness_ordinal_weight": 0.08,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v10_full_faf_consistency": evidence_safe_microbatch(
            make_variant(
                base,
                "v10_full_faf_consistency",
                note=(
                    "Candidate improvement: full FAF with Fourier jitter, road prior, condition-aware "
                    "wet-state hard-case resampling, ordinal wetness supervision, and weak-view "
                    "prediction/attention consistency."
                ),
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                    "evidence_road_likelihood_prior_strength": 0.60,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                    "wetness_ordinal_weight": 0.08,
                    "wetness_conditional_coral_weight": 0.02,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.25,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v11_full_faf_domain_adapter": evidence_safe_microbatch(
            make_variant(
                base,
                "v11_full_faf_domain_adapter",
                note="Candidate improvement: v10 plus tiny domain-specific affine adapters on shared features.",
                model={
                    "friction_set_entropy_expansion": 0.16,
                    "evidence_entropy_expansion": 0.10,
                    "evidence_road_likelihood_prior_strength": 0.60,
                    "use_domain_adapters": True,
                    "domain_adapter_scale": 0.15,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.02,
                    "coverage_weight": 1.2,
                    "target_width_weight": 0.4,
                    "wetness_ordinal_weight": 0.08,
                    "wetness_conditional_coral_weight": 0.02,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.25,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                    "domain_adapter_weight": 0.01,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"dataset": "roadsaw", "wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"dataset": "roadsaw", "wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"dataset": "roadsaw", "wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v12_full_faf_roi_interval_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v12_full_faf_roi_interval_safety",
                note=(
                    "Candidate improvement: v10 plus explicit bottom-road evidence ROI constraints "
                    "and more conservative safety-weighted coverage-aware interval training."
                ),
                model={
                    "friction_set_entropy_expansion": 0.18,
                    "evidence_entropy_expansion": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("full"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.6,
                    "coverage_risk_weight": 0.30,
                    "coverage_wetness_weight": 0.20,
                    "coverage_snow_weight": 0.15,
                    "coverage_weight_max": 1.75,
                    "interval_weight": 0.08,
                    "wetness_ordinal_weight": 0.10,
                    "wetness_conditional_coral_weight": 0.025,
                    "endpoint_weight": 1.7,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.04,
                    "evidence_attention_pseudo_road_weight": 0.03,
                    "evidence_pseudo_road_min_mass": 0.72,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.58,
                    "evidence_center_bottom_mass_target": 0.24,
                    "evidence_top_mass_max": 0.42,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v13_lean_physics_evidence": evidence_safe_microbatch(
            make_variant(
                base,
                "v13_lean_physics_evidence",
                note=(
                    "Candidate simplification: remove the unstable FrictionSet and DG-loss stack while "
                    "retaining the physics texture branch, local EvidenceField, ordinal risk, and "
                    "coverage-aware interval supervision."
                ),
                model={
                    "use_physics_branch": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.12,
                    "evidence_interval_mix": 0.18,
                    "evidence_risk_logit_mix": 0.12,
                },
                loss=loss_profile("lean_evidence"),
            )
        ),
        "v14_lean_road_roi_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v14_lean_road_roi_safety",
                note=(
                    "Candidate final route: lean PhysicsTexture+EvidenceField with Fourier style jitter, "
                    "pseudo road-likelihood prior, ordinal wetness supervision, condition-aware wet-state hard sampling, weak-view consistency, "
                    "bottom-road ROI attention constraints, and stronger safety-weighted coverage-aware interval training."
                ),
                model={
                    "use_physics_branch": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.60,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.40,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "fourier_low_freq_jitter_p": 0.30,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.72, 1.28],
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v15_lean_bottom_square_style_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v15_lean_bottom_square_style_safety",
                note=(
                    "Candidate input-canonicalized final route: lean PhysicsTexture+EvidenceField "
                    "with bottom-centered square road crop, stronger style augmentation, pseudo "
                    "road-likelihood prior, ordinal wetness supervision, wet-state hard sampling, "
                    "weak-view consistency, and safety-weighted interval training."
                ),
                model={
                    "use_physics_branch": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.22,
                        "hue": 0.04,
                    },
                    "random_grayscale_p": 0.12,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.68, 1.32],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v16_lean_bottom_square_color_constancy_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v16_lean_bottom_square_color_constancy_safety",
                note=(
                    "Candidate color-canonicalized final route: v15 plus soft Gray-World "
                    "color constancy before Fourier style jitter, designed to suppress "
                    "camera/dataset color cast while preserving road texture and wetness cues."
                ),
                model={
                    "use_physics_branch": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v17_lean_quality_physics_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v17_lean_quality_physics_safety",
                note=(
                    "Candidate quality-aware final route: v16 plus explicit differentiable "
                    "photometric quality and wet-road regional physics cues inside the "
                    "PhysicsTexture branch, designed for RoadSaW near-white wet patches "
                    "and low-texture RoadSC snow patches."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v18_lean_mixstyle_quality_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v18_lean_mixstyle_quality_safety",
                note=(
                    "Candidate style-statistics final route: v17 plus training-only "
                    "grouped Feature MixStyle on shared channel statistics. This is a "
                    "cheap shortcut-mitigation probe: keep it only if dataset-ID leakage "
                    "or worst-domain behavior improves without hurting low-friction recall."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                    "use_feature_mixstyle": True,
                    "feature_mixstyle_p": 0.45,
                    "feature_mixstyle_alpha": 0.20,
                    "feature_mixstyle_groups": 8,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v19_lean_state_contrast_quality_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v19_lean_state_contrast_quality_safety",
                note=(
                    "Candidate state-invariant final route: v17 plus supervised "
                    "cross-dataset state contrastive alignment on shared features. "
                    "This directly tests whether dry/wet/snow/risk semantics can be "
                    "aligned across RSCD, RoadSaW, and RoadSC instead of memorizing "
                    "dataset style. Keep it only if shortcut probes or worst-domain "
                    "metrics improve without losing safety recall or interval coverage."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.008,
                    "wetness_conditional_coral_weight": 0.012,
                    "state_contrastive_temperature": 0.18,
                    "state_contrastive_cross_domain_only": True,
                    "risk_state_contrastive_weight": 0.035,
                    "friction_state_contrastive_weight": 0.020,
                    "wetness_state_contrastive_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v20_lean_interval_order_quality_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v20_lean_interval_order_quality_safety",
                note=(
                    "Candidate weak-interval physics route: v17 plus a pairwise "
                    "non-overlapping interval-order consistency loss. It uses only "
                    "public weak friction interval anchors: when one road-state "
                    "interval is definitely lower-friction than another, the predicted "
                    "mu mean is softly ordered the same way. Keep it only if safety "
                    "recall or worst-dataset behavior improves without widening "
                    "calibrated intervals."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_weight_max": 1.85,
                    "interval_weight": 0.09,
                    "interval_order_weight": 0.08,
                    "interval_order_margin_scale": 0.35,
                    "interval_order_min_gap": 0.02,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v21_lean_quality_uncertainty_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v21_lean_quality_uncertainty_safety",
                note=(
                    "Candidate visual-quality uncertainty route: v17 plus image-derived "
                    "near-white, low-texture, and specular-highlight weights inside the "
                    "coverage loss. This targets RoadSaW wet/overexposed and RoadSC "
                    "low-texture snow cases by making interval under-coverage more costly "
                    "when the camera evidence is ambiguous, without treating visual quality "
                    "as measured tire-road friction."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_near_white_weight": 0.35,
                    "coverage_low_texture_weight": 0.22,
                    "coverage_specular_weight": 0.18,
                    "coverage_weight_max": 2.05,
                    "interval_weight": 0.09,
                    "risk_conditional_coral_weight": 0.01,
                    "wetness_conditional_coral_weight": 0.015,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v22_lean_quality_order_contrast_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v22_lean_quality_order_contrast_safety",
                note=(
                    "Candidate ambiguity-ordered state route: v21 plus a small "
                    "weak-friction interval-order loss and lightweight cross-domain "
                    "state contrast. It tests whether visually ambiguous wet/snow "
                    "cases can stay conservative while preserving road-state ordering "
                    "and avoiding dataset-style memorization. Promote only if it "
                    "improves quality slices or worst-domain behavior without excessive "
                    "calibrated interval widening."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.16,
                    "evidence_interval_mix": 0.22,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_near_white_weight": 0.35,
                    "coverage_low_texture_weight": 0.22,
                    "coverage_specular_weight": 0.18,
                    "coverage_weight_max": 2.05,
                    "interval_weight": 0.09,
                    "interval_order_weight": 0.06,
                    "interval_order_margin_scale": 0.30,
                    "interval_order_min_gap": 0.02,
                    "risk_conditional_coral_weight": 0.008,
                    "wetness_conditional_coral_weight": 0.012,
                    "state_contrastive_temperature": 0.18,
                    "state_contrastive_cross_domain_only": True,
                    "risk_state_contrastive_weight": 0.020,
                    "friction_state_contrastive_weight": 0.012,
                    "wetness_state_contrastive_weight": 0.010,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v23_lean_region_mixture_evidence_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v23_lean_region_mixture_evidence_safety",
                note=(
                    "Candidate segmentation-transfer route: v21-style safety training "
                    "with differentiable region-mixture cues inside the EvidenceField. "
                    "It borrows the semantic-segmentation idea of reasoning over local "
                    "material regions, but uses only public image-level road-state labels: "
                    "local color/texture/state-mixture evidence expands the friction "
                    "interval where road appearance is spatially mixed, reflective, snowy, "
                    "or otherwise visually ambiguous. Promote only if it improves quality "
                    "slices or interval coverage without losing safety recall."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.14,
                    "evidence_interval_mix": 0.26,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                    "evidence_region_mixture_cues": True,
                    "evidence_region_mixture_expansion": 0.05,
                    "evidence_region_mixture_kernel_size": 11,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_near_white_weight": 0.35,
                    "coverage_low_texture_weight": 0.22,
                    "coverage_specular_weight": 0.18,
                    "coverage_weight_max": 2.05,
                    "interval_weight": 0.09,
                    "interval_order_weight": 0.04,
                    "interval_order_margin_scale": 0.25,
                    "interval_order_min_gap": 0.02,
                    "risk_conditional_coral_weight": 0.008,
                    "wetness_conditional_coral_weight": 0.012,
                    "state_contrastive_temperature": 0.18,
                    "state_contrastive_cross_domain_only": True,
                    "risk_state_contrastive_weight": 0.012,
                    "friction_state_contrastive_weight": 0.008,
                    "wetness_state_contrastive_weight": 0.006,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v24_lean_multi_query_region_evidence_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v24_lean_multi_query_region_evidence_safety",
                note=(
                    "Candidate segmentation-transfer route: v23-style region-mixture "
                    "EvidenceField upgraded to multi-query mask-style local evidence. "
                    "It borrows semantic segmentation mask-query reasoning: each query "
                    "can focus on a different local material, wetness, snow, or glare "
                    "region; query disagreement expands the weak friction interval; "
                    "and a small diversity regularizer discourages all queries from "
                    "collapsing to the same patch. It uses only public image-level "
                    "labels and must be promoted only if hard wet/snow/near-white "
                    "slices or calibrated coverage-width improve without hurting "
                    "low-friction recall."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.12,
                    "evidence_interval_mix": 0.26,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                    "evidence_region_mixture_cues": True,
                    "evidence_region_mixture_expansion": 0.04,
                    "evidence_region_mixture_kernel_size": 11,
                    "evidence_num_queries": 4,
                    "evidence_query_disagreement_expansion": 0.08,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_near_white_weight": 0.35,
                    "coverage_low_texture_weight": 0.22,
                    "coverage_specular_weight": 0.18,
                    "coverage_weight_max": 2.05,
                    "interval_weight": 0.09,
                    "interval_order_weight": 0.04,
                    "interval_order_margin_scale": 0.25,
                    "interval_order_min_gap": 0.02,
                    "risk_conditional_coral_weight": 0.008,
                    "wetness_conditional_coral_weight": 0.012,
                    "state_contrastive_temperature": 0.18,
                    "state_contrastive_cross_domain_only": True,
                    "risk_state_contrastive_weight": 0.012,
                    "friction_state_contrastive_weight": 0.008,
                    "wetness_state_contrastive_weight": 0.006,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_query_diversity_weight": 0.012,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.08,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.08,
                    "aug_consistency_noise_std": 0.01,
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
        "v25_lean_masked_query_consistency_safety": evidence_safe_microbatch(
            make_variant(
                base,
                "v25_lean_masked_query_consistency_safety",
                note=(
                    "Candidate semantic-segmentation consistency route: v24 multi-query "
                    "local evidence plus MIC-style masked weak-view consistency. The "
                    "weak view receives camera-style perturbation and a few random "
                    "region masks, then risk logits, friction intervals, and road-limited "
                    "attention must remain consistent with the clean view. This tests "
                    "whether the model is learning stable road material evidence rather "
                    "than memorizing dataset style or a single brittle texture patch."
                ),
                model={
                    "use_physics_branch": True,
                    "physics_quality_cues": True,
                    "use_friction_set": False,
                    "use_evidence_field": True,
                    "evidence_entropy_expansion": 0.12,
                    "evidence_interval_mix": 0.26,
                    "evidence_risk_logit_mix": 0.12,
                    "evidence_road_likelihood_prior_strength": 0.75,
                    "evidence_region_mixture_cues": True,
                    "evidence_region_mixture_expansion": 0.04,
                    "evidence_region_mixture_kernel_size": 11,
                    "evidence_num_queries": 4,
                    "evidence_query_disagreement_expansion": 0.08,
                },
                loss={
                    **loss_profile("lean_evidence"),
                    "coverage_margin": 0.04,
                    "coverage_weight": 1.8,
                    "coverage_risk_weight": 0.35,
                    "coverage_wetness_weight": 0.25,
                    "coverage_snow_weight": 0.20,
                    "coverage_near_white_weight": 0.35,
                    "coverage_low_texture_weight": 0.22,
                    "coverage_specular_weight": 0.18,
                    "coverage_weight_max": 2.05,
                    "interval_weight": 0.09,
                    "interval_order_weight": 0.04,
                    "interval_order_margin_scale": 0.25,
                    "interval_order_min_gap": 0.02,
                    "risk_conditional_coral_weight": 0.008,
                    "wetness_conditional_coral_weight": 0.012,
                    "state_contrastive_temperature": 0.18,
                    "state_contrastive_cross_domain_only": True,
                    "risk_state_contrastive_weight": 0.012,
                    "friction_state_contrastive_weight": 0.008,
                    "wetness_state_contrastive_weight": 0.006,
                    "wetness_ordinal_weight": 0.10,
                    "endpoint_weight": 1.8,
                    "target_width_weight": 0.35,
                    "evidence_attention_region_weight": 0.05,
                    "evidence_attention_pseudo_road_weight": 0.04,
                    "evidence_query_diversity_weight": 0.012,
                    "evidence_pseudo_road_min_mass": 0.74,
                    "evidence_pseudo_road_threshold": 0.35,
                    "evidence_bottom_mass_target": 0.62,
                    "evidence_center_bottom_mass_target": 0.25,
                    "evidence_top_mass_max": 0.35,
                    "aug_consistency_weight": 0.10,
                    "aug_consistency_max_samples": 6,
                    "aug_consistency_strength": 0.10,
                    "aug_consistency_noise_std": 0.012,
                    "aug_consistency_mask_ratio": 0.18,
                    "aug_consistency_mask_block_frac": 0.18,
                    "aug_consistency_mask_max_blocks": 4,
                    "aug_consistency_mask_value": "mean",
                    "aug_consistency_interval_weight": 1.0,
                    "aug_consistency_attention_weight": 0.40,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
                },
                augmentation={
                    "random_resized_crop": False,
                    "resize_mode": "bottom_square",
                    "gray_world_alpha": 0.75,
                    "color_jitter": {
                        "brightness": 0.35,
                        "contrast": 0.35,
                        "saturation": 0.18,
                        "hue": 0.035,
                    },
                    "random_grayscale_p": 0.10,
                    "gaussian_blur_p": 0.14,
                    "fourier_low_freq_jitter_p": 0.35,
                    "fourier_beta": 0.08,
                    "fourier_strength": [0.70, 1.30],
                    "random_erasing_p": 0.04,
                },
                data={
                    "balanced_weight_overrides": [
                        {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                        {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                        {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                    ],
                },
            )
        ),
    }


def make_lodo_configs(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = {
        "rscd": {
            "train": "data/manifests_full/rscd_prepared_train.csv",
            "val": "data/manifests_full/rscd_prepared_val.csv",
            "test": "data/manifests_full/rscd_prepared_test.csv",
        },
        "roadsaw": {
            "train": "data/manifests_full/roadsaw_train.csv",
            "val": "data/manifests_full/roadsaw_val.csv",
            "test": "data/manifests_full/roadsaw_test.csv",
        },
        "roadsc": {
            "train": "data/manifests_full/roadsc_train.csv",
            "val": "data/manifests_full/roadsc_val.csv",
            "test": "data/manifests_full/roadsc_test.csv",
        },
    }
    out = {}
    for held_out, files in datasets.items():
        train_sets = [info["train"] for name, info in datasets.items() if name != held_out]
        val_sets = [info["val"] for name, info in datasets.items() if name != held_out]
        cfg = make_variant(
            base,
            f"lodo_{held_out}_full_faf",
            note=f"Leave-one-dataset-out full FAF: train/val exclude {held_out}; test only {held_out}.",
            model={},
            loss=loss_profile("full"),
        )
        cfg = evidence_safe_microbatch(cfg)
        cfg["data"]["train_manifests"] = train_sets
        cfg["data"]["val_manifests"] = val_sets
        cfg["data"]["test_manifests"] = [files["test"]]
        cfg["data"]["balanced_num_samples_per_epoch"] = 24000
        cfg["model"]["num_domains"] = 2
        out[f"lodo_{held_out}_full_faf"] = cfg
    return out


def make_single_dataset_configs(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {
        "rscd": ("rscd_prepared", "RSCD single-dataset fair comparison."),
        "roadsaw": ("roadsaw", "RoadSaW single-dataset fair comparison."),
        "roadsc": ("roadsc", "RoadSC single-dataset fair comparison."),
    }
    out = {}
    for key, (prefix, note) in specs.items():
        cfg = make_variant(
            base,
            f"single_{key}_full_faf",
            note=note,
            model={},
            loss=loss_profile("full"),
        )
        cfg = evidence_safe_microbatch(cfg)
        apply_single_dataset_split(cfg, key, prefix)
        out[f"single_{key}_full_faf"] = cfg
    return out


def make_fair_baseline_configs(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {
        "rscd": ("rscd_prepared", "RSCD single-dataset ConvNeXt global baseline."),
        "roadsaw": ("roadsaw", "RoadSaW single-dataset ConvNeXt global baseline."),
        "roadsc": ("roadsc", "RoadSC single-dataset ConvNeXt global baseline."),
    }
    out = {}
    for key, (prefix, note) in specs.items():
        cfg = make_variant(
            base,
            f"baseline_single_{key}_global_convnext",
            note=note,
            model={
                "use_physics_branch": False,
                "use_friction_set": False,
                "use_evidence_field": False,
            },
            loss=loss_profile("global"),
        )
        apply_single_dataset_split(cfg, key, prefix)
        out[f"baseline_single_{key}_global_convnext"] = cfg
    return out


def make_final_method_configs(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    datasets = {
        "rscd": {
            "prefix": "rscd_prepared",
            "train": "data/manifests_full/rscd_prepared_train.csv",
            "val": "data/manifests_full/rscd_prepared_val.csv",
            "test": "data/manifests_full/rscd_prepared_test.csv",
        },
        "roadsaw": {
            "prefix": "roadsaw",
            "train": "data/manifests_full/roadsaw_train.csv",
            "val": "data/manifests_full/roadsaw_val.csv",
            "test": "data/manifests_full/roadsaw_test.csv",
        },
        "roadsc": {
            "prefix": "roadsc",
            "train": "data/manifests_full/roadsc_train.csv",
            "val": "data/manifests_full/roadsc_val.csv",
            "test": "data/manifests_full/roadsc_test.csv",
        },
    }
    out: dict[str, dict[str, Any]] = {}
    for held_out, files in datasets.items():
        name = f"final_lodo_{held_out}_lean_road_roi_safety"
        cfg = make_lean_road_roi_safety_variant(
            base,
            name,
            note=f"Final-method LODO: lean road-ROI safety model trained without {held_out} and tested on held-out {held_out}.",
        )
        cfg["data"]["train_manifests"] = [info["train"] for key, info in datasets.items() if key != held_out]
        cfg["data"]["val_manifests"] = [info["val"] for key, info in datasets.items() if key != held_out]
        cfg["data"]["test_manifests"] = [files["test"]]
        cfg["data"]["balanced_num_samples_per_epoch"] = 24000
        cfg["model"]["num_domains"] = 2
        out[name] = cfg

    for key, files in datasets.items():
        name = f"final_single_{key}_lean_road_roi_safety"
        cfg = make_lean_road_roi_safety_variant(
            base,
            name,
            note=f"Final-method single-dataset fair comparison on {key} against the matched global ConvNeXt baseline.",
        )
        apply_single_dataset_split(cfg, key, files["prefix"])
        out[name] = cfg
    return out


def make_lean_road_roi_safety_variant(base: dict[str, Any], name: str, note: str) -> dict[str, Any]:
    return evidence_safe_microbatch(
        make_variant(
            base,
            name,
            note=(
                f"{note} Uses ordinal wetness supervision and safety-weighted coverage-aware "
                "interval supervision for high-risk/wet/snow states."
            ),
            model={
                "use_physics_branch": True,
                "use_friction_set": False,
                "use_evidence_field": True,
                "evidence_entropy_expansion": 0.16,
                "evidence_interval_mix": 0.22,
                "evidence_risk_logit_mix": 0.12,
                "evidence_road_likelihood_prior_strength": 0.75,
            },
            loss={
                **loss_profile("lean_evidence"),
                "coverage_margin": 0.04,
                "coverage_weight": 1.8,
                "coverage_risk_weight": 0.35,
                "coverage_wetness_weight": 0.25,
                "coverage_snow_weight": 0.20,
                "coverage_weight_max": 1.85,
                "interval_weight": 0.09,
                "risk_conditional_coral_weight": 0.01,
                "wetness_conditional_coral_weight": 0.015,
                "wetness_ordinal_weight": 0.10,
                "endpoint_weight": 1.8,
                "target_width_weight": 0.35,
                "evidence_attention_region_weight": 0.05,
                "evidence_attention_pseudo_road_weight": 0.04,
                "evidence_pseudo_road_min_mass": 0.74,
                "evidence_pseudo_road_threshold": 0.35,
                "evidence_bottom_mass_target": 0.60,
                "evidence_center_bottom_mass_target": 0.25,
                "evidence_top_mass_max": 0.40,
                "aug_consistency_weight": 0.08,
                "aug_consistency_max_samples": 6,
                "aug_consistency_strength": 0.08,
                "aug_consistency_noise_std": 0.01,
                "aug_consistency_interval_weight": 1.0,
                "aug_consistency_attention_weight": 0.35,
                    "aug_consistency_attention_mask": "road_likelihood",
                    "aug_consistency_attention_mask_threshold": 0.35,
                    "aug_consistency_attention_mask_sharpness": 12.0,
            },
            augmentation={
                "fourier_low_freq_jitter_p": 0.30,
                "fourier_beta": 0.08,
                "fourier_strength": [0.72, 1.28],
            },
            data={
                "balanced_weight_overrides": [
                    {"where": {"wetness_label": "very_wet"}, "multiplier": 1.8},
                    {"where": {"wetness_label": "wet"}, "multiplier": 1.4},
                    {"where": {"wetness_label": "damp"}, "multiplier": 1.2},
                ],
            },
        )
    )


def make_variant(
    base: dict[str, Any],
    name: str,
    note: str,
    model: dict[str, Any],
    loss: dict[str, Any],
    augmentation: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["output_dir"] = str(RUN_ROOT / name)
    cfg["experiment_note"] = note
    set_full_protocol_data(cfg)
    cfg["model"].update(model)
    cfg["loss"].update(loss)
    if data:
        cfg["data"].update(data)
    if augmentation:
        cfg["data"].setdefault("augmentation", {}).update(augmentation)
    return cfg


def evidence_safe_microbatch(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep effective batch 32 while lowering RTX 3050 activation memory."""
    cfg["data"]["batch_size"] = 8
    cfg.setdefault("optim", {})["grad_accum_steps"] = 4
    cfg["experiment_note"] = (
        f"{cfg.get('experiment_note', '')} Uses batch_size=8 with grad_accum_steps=4 "
        "to match the effective batch of the batch_size=16/grad_accum_steps=2 rows on 4GB VRAM."
    ).strip()
    return cfg


def apply_single_dataset_split(cfg: dict[str, Any], key: str, prefix: str) -> None:
    if key == "rscd":
        cfg["data"]["train_manifests"] = ["data/manifests_full/rscd_prepared_train.csv"]
        cfg["data"]["val_manifests"] = ["data/manifests_full/rscd_prepared_val.csv"]
        cfg["data"]["test_manifests"] = ["data/manifests_full/rscd_prepared_test.csv"]
        cfg["data"]["balanced_num_samples_per_epoch"] = 36000
    else:
        cfg["data"]["train_manifests"] = [f"data/manifests_full/{prefix}_train.csv"]
        cfg["data"]["val_manifests"] = [f"data/manifests_full/{prefix}_val.csv"]
        cfg["data"]["test_manifests"] = [f"data/manifests_full/{prefix}_test.csv"]
        cfg["data"]["balanced_num_samples_per_epoch"] = 12000
    cfg["model"]["num_domains"] = 1


def set_full_protocol_data(cfg: dict[str, Any]) -> None:
    data = cfg["data"]
    optim = cfg.setdefault("optim", {})
    data["num_workers"] = 2
    data["prefetch_factor"] = 2
    data["max_train_samples"] = None
    data["max_val_samples"] = None
    data["max_test_samples"] = None
    data["max_train_samples_per_class"] = None
    data["max_val_samples_per_class"] = None
    data["max_test_samples_per_class"] = None
    data["balanced_sampling"] = True
    data["balanced_dataset_first"] = True
    data["balanced_group_columns"] = ["dataset", "class_label"]
    optim["log_every_steps"] = 100


def loss_profile(name: str) -> dict[str, Any]:
    zero_evidence = {
        "evidence_risk_weight": 0.0,
        "evidence_interval_weight": 0.0,
        "evidence_endpoint_weight": 0.0,
        "evidence_width_weight": 0.0,
        "evidence_target_width_weight": 0.0,
        "evidence_attention_prior_weight": 0.0,
        "evidence_attention_smooth_weight": 0.0,
        "evidence_attention_region_weight": 0.0,
        "evidence_attention_pseudo_road_weight": 0.0,
        "evidence_risk_consistency_weight": 0.0,
        "evidence_interval_consistency_weight": 0.0,
    }
    no_dg = {
        "group_dro_weight": 0.0,
        "group_vrex_weight": 0.0,
        "feature_coral_weight": 0.0,
        "risk_conditional_coral_weight": 0.0,
    }
    if name == "global":
        return {
            "task_weight": 1.0,
            "compatibility_weight": 0.0,
            **no_dg,
            **zero_evidence,
        }
    if name == "friction_set":
        return {
            "task_weight": 0.45,
            "compatibility_weight": 1.0,
            **no_dg,
            **zero_evidence,
        }
    if name == "dg":
        return {
            "task_weight": 0.35,
            "compatibility_weight": 1.0,
            "group_dro_weight": 0.25,
            "group_vrex_weight": 0.08,
            "feature_coral_weight": 0.01,
            "risk_conditional_coral_weight": 0.02,
            **zero_evidence,
        }
    if name == "evidence":
        return {
            "task_weight": 0.32,
            "compatibility_weight": 1.0,
            "group_dro_weight": 0.25,
            "group_vrex_weight": 0.08,
            "feature_coral_weight": 0.01,
            "risk_conditional_coral_weight": 0.02,
            "evidence_risk_weight": 0.20,
            "evidence_interval_weight": 0.05,
            "evidence_endpoint_weight": 0.50,
            "evidence_width_weight": 0.05,
            "evidence_target_width_weight": 0.50,
            "evidence_attention_prior_weight": 0.01,
            "evidence_attention_smooth_weight": 0.01,
            "evidence_risk_consistency_weight": 0.05,
            "evidence_interval_consistency_weight": 0.05,
        }
    if name == "full":
        return loss_profile("evidence")
    if name == "lean_evidence":
        return {
            "task_weight": 0.58,
            "compatibility_weight": 0.0,
            **no_dg,
            "risk_ordinal_weight": 0.20,
            "interval_weight": 0.08,
            "coverage_weight": 1.4,
            "endpoint_weight": 1.6,
            "target_width_weight": 0.40,
            "coverage_margin": 0.02,
            "monotonic_weight": 0.05,
            "risk_mu_monotonic_weight": 0.08,
            "evidence_risk_weight": 0.20,
            "evidence_interval_weight": 0.06,
            "evidence_endpoint_weight": 0.55,
            "evidence_width_weight": 0.04,
            "evidence_target_width_weight": 0.45,
            "evidence_attention_prior_weight": 0.01,
            "evidence_attention_smooth_weight": 0.01,
            "evidence_attention_region_weight": 0.0,
            "evidence_attention_pseudo_road_weight": 0.0,
            "evidence_risk_consistency_weight": 0.05,
            "evidence_interval_consistency_weight": 0.05,
        }
    raise ValueError(f"Unknown loss profile: {name}")


if __name__ == "__main__":
    main()

