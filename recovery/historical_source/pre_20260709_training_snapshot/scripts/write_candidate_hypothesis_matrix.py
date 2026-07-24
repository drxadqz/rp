from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


INTERESTING_MODULES = [
    "physics_texture",
    "physics_quality_cues",
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
    "blur_aug",
    "random_erasing",
    "fourier_style_jitter",
    "bottom_square_input_canonicalization",
    "gray_world_color_constancy",
    "dann",
    "road_likelihood_prior",
    "region_mixture_evidence",
    "pseudo_road_mask_supervision",
    "condition_hard_sampling",
    "dataset_scoped_sampling",
    "weak_view_consistency",
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


EXPERIMENT_SPECS: list[dict[str, Any]] = [
    {
        "run": "lodo_roadsaw_full_faf",
        "phase": "P0.5 LODO",
        "hypothesis": "The full FAF representation transfers to an unseen RoadSaW wetness domain better than pooled-test metrics suggest.",
        "addresses": "Cross-dataset generalization and RoadSaW wetness stress testing.",
        "primary_metrics": ["held-out RoadSaW risk F1", "held-out RoadSaW friction F1", "low-friction recall", "calibrated coverage", "dataset-ID probe"],
        "success_criteria": "Usable held-out RoadSaW risk/friction F1 with low-friction recall preserved, and no evidence leakage from RoadSaW train/val.",
        "failure_action": "Treat RoadSaW wetness as the main failure mode; prioritize v9/v10/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25 and avoid broad OOD claims.",
        "retention_rule": "Full FAF remains only a stress-test reference unless it beats lean final routes on held-out RoadSaW.",
        "claim_unlocked": "Cross-dataset claim boundary for the hardest public wetness dataset.",
    },
    {
        "run": "lodo_rscd_full_faf",
        "phase": "P0.5 LODO",
        "hypothesis": "The full FAF representation transfers to an unseen RSCD road-surface-label domain.",
        "addresses": "Dataset shift between RSCD and the smaller RoadSaW/RoadSC sources.",
        "primary_metrics": ["held-out RSCD friction F1", "risk F1", "low-friction recall", "worst core-state F1"],
        "success_criteria": "Held-out RSCD remains competitive without relying on RSCD collection style.",
        "failure_action": "Use RSCD as a source-dominance warning and strengthen per-domain balancing or adapters.",
        "retention_rule": "Keep mechanisms that improve held-out RSCD without hurting RoadSaW.",
        "claim_unlocked": "Evidence that the method is not only tuned to RSCD-like labels.",
    },
    {
        "run": "lodo_roadsc_full_faf",
        "phase": "P0.5 LODO",
        "hypothesis": "The representation transfers to an unseen snow-coverage domain.",
        "addresses": "Winter-condition transfer and low-friction interval coverage.",
        "primary_metrics": ["held-out RoadSC friction F1", "snow-state recall", "conditional interval coverage", "calibrated width"],
        "success_criteria": "Snow/ice-like states keep reasonable recall with bounded calibrated width.",
        "failure_action": "Make RoadSC-specific conditional coverage a P3 target; do not overstate winter robustness.",
        "retention_rule": "Keep snow-sensitive evidence/interval terms only if they improve RoadSC conditional cells.",
        "claim_unlocked": "Winter-domain generalization evidence.",
    },
    {
        "run": "single_roadsaw_full_faf",
        "phase": "Fair baseline",
        "hypothesis": "FAF adds useful inductive bias over the same RoadSaW public split.",
        "addresses": "Fair same-dataset comparison against ConvNeXt on wetness-heavy labels.",
        "primary_metrics": ["RoadSaW risk F1", "friction F1", "wetness F1", "bootstrap delta vs ConvNeXt"],
        "success_criteria": "Improves or complements the matched ConvNeXt baseline on safety and wetness metrics.",
        "failure_action": "If no gain, keep FAF as multi-dataset/interpretability method only, not a single-dataset SOTA claim.",
        "retention_rule": "Retain RoadSaW-specific modules only with paired-bootstrap support.",
        "claim_unlocked": "Fair RoadSaW public-data comparison.",
    },
    {
        "run": "single_rscd_full_faf",
        "phase": "Fair baseline",
        "hypothesis": "FAF improves RSCD road-surface friction proxy recognition under the same split and labels.",
        "addresses": "Comparable RSCD performance rather than relying on mismatched external numbers.",
        "primary_metrics": ["RSCD friction F1", "risk F1", "bootstrap delta vs ConvNeXt"],
        "success_criteria": "Positive same-split delta over ConvNeXt or clearly better interval/evidence quality at similar F1.",
        "failure_action": "Use ConvNeXt as the final RSCD baseline and reposition FAF around uncertainty/generalization.",
        "retention_rule": "Do not keep extra heads if they do not improve RSCD or cross-dataset evidence.",
        "claim_unlocked": "Fair RSCD public-data comparison.",
    },
    {
        "run": "single_roadsc_full_faf",
        "phase": "Fair baseline",
        "hypothesis": "FAF improves RoadSC snow-coverage friction proxy recognition under the same split and labels.",
        "addresses": "Snow/coverage public-data comparison.",
        "primary_metrics": ["RoadSC friction F1", "snow-state F1", "low-friction recall", "bootstrap delta vs ConvNeXt"],
        "success_criteria": "Improves safety-relevant snow/low-friction metrics over matched ConvNeXt.",
        "failure_action": "Use RoadSC as an interval/failure-analysis case rather than a claimed win.",
        "retention_rule": "Keep only modules that help snow-state recall or interval coverage.",
        "claim_unlocked": "Fair RoadSC public-data comparison.",
    },
    {
        "run": "baseline_single_roadsaw_global_convnext",
        "phase": "Fair baseline",
        "hypothesis": "A modern global ConvNeXt baseline is the minimum fair comparator on RoadSaW.",
        "addresses": "Strong baseline requirement.",
        "primary_metrics": ["same-split risk F1", "friction F1", "wetness F1", "calibrated coverage"],
        "success_criteria": "Provides a clean matched comparator for paired deltas.",
        "failure_action": "If baseline is unstable, fix baseline training before claiming FAF improvement.",
        "retention_rule": "Baseline is always retained for reviewer fairness.",
        "claim_unlocked": "Fair RoadSaW delta.",
    },
    {
        "run": "baseline_single_rscd_global_convnext",
        "phase": "Fair baseline",
        "hypothesis": "A modern global ConvNeXt baseline is the minimum fair comparator on RSCD.",
        "addresses": "Strong baseline requirement.",
        "primary_metrics": ["same-split friction F1", "risk F1", "calibrated coverage"],
        "success_criteria": "Provides a clean matched comparator for paired deltas.",
        "failure_action": "If baseline dominates, prune method complexity and focus on uncertainty/interpretability.",
        "retention_rule": "Baseline is always retained for reviewer fairness.",
        "claim_unlocked": "Fair RSCD delta.",
    },
    {
        "run": "baseline_single_roadsc_global_convnext",
        "phase": "Fair baseline",
        "hypothesis": "A modern global ConvNeXt baseline is the minimum fair comparator on RoadSC.",
        "addresses": "Strong baseline requirement.",
        "primary_metrics": ["same-split snow/friction F1", "risk F1", "low-friction recall"],
        "success_criteria": "Provides a clean matched comparator for paired deltas.",
        "failure_action": "If baseline dominates, use RoadSC to guide interval and ROI constraints.",
        "retention_rule": "Baseline is always retained for reviewer fairness.",
        "claim_unlocked": "Fair RoadSC delta.",
    },
    {
        "run": "v6_full_faf_fourier",
        "phase": "P1 shortcut",
        "hypothesis": "Low-frequency Fourier style jitter reduces camera/dataset style shortcut while preserving geometry and road texture.",
        "addresses": "Dataset-ID shortcut.",
        "primary_metrics": ["dataset-ID balanced accuracy", "risk F1", "low-friction recall", "worst-dataset F1"],
        "success_criteria": "Dataset-ID probe drops meaningfully without a material safety-metric regression.",
        "failure_action": "If shortcut remains high, combine with DANN/adapters or move to condition-aware alignment.",
        "retention_rule": "Keep Fourier jitter if it lowers shortcut or improves held-out RoadSaW at similar average F1.",
        "claim_unlocked": "Style-shortcut mitigation evidence.",
    },
    {
        "run": "v7_full_faf_fourier_dann",
        "phase": "P1 shortcut",
        "hypothesis": "Adversarial domain suppression after Fourier jitter removes residual dataset identity from shared features.",
        "addresses": "Dataset-ID shortcut after style augmentation.",
        "primary_metrics": ["dataset-ID balanced accuracy", "worst-dataset F1", "risk F1", "training stability"],
        "success_criteria": "Further reduces dataset-ID probe and does not damage worst-dataset or low-friction recall.",
        "failure_action": "If risk F1 or worst-dataset F1 collapses, remove DANN from final route.",
        "retention_rule": "Retain DANN only with shortcut reduction plus stable safety metrics.",
        "claim_unlocked": "Adversarial shortcut mitigation evidence.",
    },
    {
        "run": "v8_full_faf_fourier_roadprior",
        "phase": "P2 evidence",
        "hypothesis": "Road-likelihood priors keep EvidenceField attention on plausible tire-contact evidence rather than background style.",
        "addresses": "Evidence leakage outside road regions.",
        "primary_metrics": ["attention bottom/road mass", "evidence audit pass rate", "RoadSaW failures", "risk F1"],
        "success_criteria": "Improves road-focused evidence metrics without hurting low-friction recall.",
        "failure_action": "If attention improves but task metrics fall, weaken or make prior conditional.",
        "retention_rule": "Keep road prior if interpretability improves with no major safety cost.",
        "claim_unlocked": "Grounded visual evidence story.",
    },
    {
        "run": "v9_full_faf_roadsaw_hard_sampling",
        "phase": "P1 RoadSaW",
        "hypothesis": "Hard sampling damp/wet/very-wet RoadSaW states improves the most safety-relevant confusion modes.",
        "addresses": "RoadSaW wetness weakness.",
        "primary_metrics": ["RoadSaW wetness macro-F1", "ordinal MAE", "severe misorder", "risk F1"],
        "success_criteria": "Wetness F1 and severe-misorder improve without overfitting RoadSaW or harming other datasets.",
        "failure_action": "If it overfits, replace with condition-aware loss weighting rather than sampling.",
        "retention_rule": "Keep only if held-out or same-split RoadSaW wetness improves with stable pooled metrics.",
        "claim_unlocked": "Wetness-state robustness evidence.",
    },
    {
        "run": "v10_full_faf_consistency",
        "phase": "P2 evidence",
        "hypothesis": "Weak-view consistency stabilizes predictions and evidence maps under style-preserving augmentations.",
        "addresses": "Evidence instability and weak-label noise.",
        "primary_metrics": ["risk F1", "attention consistency", "evidence failures", "dataset-ID probe"],
        "success_criteria": "Reduces evidence failures or shortcut while preserving primary metrics.",
        "failure_action": "If it slows training or harms F1, keep only as supplemental analysis or remove.",
        "retention_rule": "Retain consistency if it improves interpretable evidence or generalization.",
        "claim_unlocked": "Stable evidence-field claim.",
    },
    {
        "run": "v11_full_faf_domain_adapter",
        "phase": "P1 shortcut",
        "hypothesis": "Small domain adapters allow dataset-specific visual style while the shared trunk learns friction-relevant structure.",
        "addresses": "Shortcut versus real domain difference tradeoff.",
        "primary_metrics": ["dataset-ID probe", "LODO metrics", "worst-dataset F1", "risk F1"],
        "success_criteria": "Improves LODO or worst-dataset metrics without becoming a memorized dataset classifier.",
        "failure_action": "If adapters only improve in-domain results, exclude them from OOD claims.",
        "retention_rule": "Keep adapters only if they improve held-out transfer or calibrated interval quality.",
        "claim_unlocked": "Domain-specific style accommodation evidence.",
    },
    {
        "run": "v12_full_faf_roi_interval_safety",
        "phase": "P3 interval",
        "hypothesis": "Road ROI attention plus safety-weighted coverage improves conditional friction intervals without simply widening them.",
        "addresses": "Conditional interval undercoverage.",
        "primary_metrics": ["conditional coverage", "calibrated width", "low-friction recall", "RoadSaW wetness coverage"],
        "success_criteria": "Fewer undercovered cells with bounded width and stable safety metrics.",
        "failure_action": "If coverage improves only by widening, tighten width penalty or use conditional calibration.",
        "retention_rule": "Keep interval-safety terms if they improve coverage-width tradeoff.",
        "claim_unlocked": "Useful calibrated weak-friction interval claim.",
    },
    {
        "run": "v13_lean_physics_evidence",
        "phase": "P3 pruning",
        "hypothesis": "Removing unstable FrictionSet/DG complexity while keeping PhysicsTexture and EvidenceField improves robustness per parameter.",
        "addresses": "Module pruning and over-complexity.",
        "primary_metrics": ["risk F1", "low-friction recall", "worst-dataset F1", "dataset-ID probe", "evidence audit"],
        "success_criteria": "Matches or beats full candidates on safety/generalization with simpler architecture.",
        "failure_action": "If too weak, re-add only the single mechanism that earned evidence in v6-v12.",
        "retention_rule": "Prefer lean route unless full fusion gives statistically defensible gains.",
        "claim_unlocked": "Principled module pruning.",
    },
    {
        "run": "v14_lean_road_roi_safety",
        "phase": "P3 final candidate",
        "hypothesis": "A lean PhysicsTexture + EvidenceField + road-ROI safety route is the strongest paper-method candidate.",
        "addresses": "Final top-venue story: simple, grounded, calibrated, and less shortcut-prone.",
        "primary_metrics": ["risk F1", "low-friction recall", "worst-dataset F1", "dataset-ID probe", "conditional coverage", "calibrated width"],
        "success_criteria": "Best safety/generalization score among candidates and credible evidence maps.",
        "failure_action": "If it underperforms, freeze the best evidence-supported candidate and revise final configs.",
        "retention_rule": "Promote to final method only after LODO and matched ConvNeXt evidence support it.",
        "claim_unlocked": "Final-method candidate selection.",
    },
    {
        "run": "v15_lean_bottom_square_style_safety",
        "phase": "P1 shortcut",
        "hypothesis": "Bottom-centered square road cropping removes native aspect/format shortcuts while keeping the near-road friction evidence.",
        "addresses": "Dataset-ID shortcut from native size/aspect and non-road content.",
        "primary_metrics": ["dataset-ID balanced accuracy", "held-out RoadSaW risk F1", "low-friction recall", "worst-dataset F1", "conditional coverage"],
        "success_criteria": "Dataset-ID probe drops or held-out/worst-domain metrics improve without widening intervals excessively.",
        "failure_action": "If padding/cropping hurts RoadSaW or RSCD, keep the idea as an ablation and return to Fourier/ROI-only normalization.",
        "retention_rule": "Promote only if it improves shortcut or LODO evidence relative to v14.",
        "claim_unlocked": "Input canonicalization against dataset-style shortcut.",
    },
    {
        "run": "v16_lean_bottom_square_color_constancy_safety",
        "phase": "P1 shortcut",
        "hypothesis": "Soft gray-world color constancy suppresses camera/dataset color cast on top of bottom-road input canonicalization.",
        "addresses": "Dataset-ID shortcut from color pipeline, illumination, and file-format style.",
        "primary_metrics": ["dataset-ID balanced accuracy", "held-out RoadSaW risk F1", "low-friction recall", "worst-dataset F1", "RoadSaW wetness F1"],
        "success_criteria": "Dataset-ID probe drops beyond v15 while preserving wetness/risk cues and keeping interval width bounded.",
        "failure_action": "If color constancy weakens wetness discrimination, keep v16 as a negative style-canonicalization ablation and prefer v15/v14.",
        "retention_rule": "Promote only if it improves shortcut or held-out evidence relative to v15 without hiding wet-road reflectance.",
        "claim_unlocked": "Color-canonicalized input normalization against dataset-style shortcut.",
    },
    {
        "run": "v17_lean_quality_physics_safety",
        "phase": "P1/P3 wet-road quality",
        "hypothesis": "Explicit near-white, low-texture, and wet-road regional physics cues improve RoadSaW wet/very-wet robustness and interval calibration without relying on dataset identity.",
        "addresses": "RoadSaW near-white wet patches, RoadSC low-texture snow patches, and weak visual evidence uncertainty.",
        "primary_metrics": ["RoadSaW near-white F1", "RoadSaW normal-quality F1", "wetness F1", "low-friction recall", "conditional coverage", "dataset-ID probe"],
        "success_criteria": "Improves near-white/wet slices or conditional coverage relative to v16/v14 while keeping normal-quality and worst-dataset metrics stable.",
        "failure_action": "If it only memorizes quality artifacts or hurts normal-quality samples, keep the slice diagnostics but remove the quality-cue branch from the final method.",
        "retention_rule": "Promote only if quality slices improve and dataset shortcut does not increase.",
        "claim_unlocked": "Quality-aware wet-road physics evidence.",
    },
    {
        "run": "v18_lean_mixstyle_quality_safety",
        "phase": "P1 shortcut",
        "hypothesis": "Training-only Feature MixStyle reduces feature-statistics shortcut beyond v17 without adding a heavy backbone or new data.",
        "addresses": "Dataset-ID shortcut that remains after bottom-square cropping, color constancy, Fourier jitter, and quality-aware physics cues.",
        "primary_metrics": ["dataset-ID balanced accuracy", "worst-dataset F1", "risk F1", "low-friction recall", "conditional coverage", "calibrated width"],
        "success_criteria": "Dataset-ID probe drops or worst-domain behavior improves relative to v17 while risk F1, low-friction recall, and coverage-width remain stable.",
        "failure_action": "If it only injects noise or hurts low-friction recall, drop MixStyle immediately and keep v17/v14 as the lean route.",
        "retention_rule": "Keep only as a shortcut-mitigation module; never retain it for a pooled-accuracy-only gain.",
        "claim_unlocked": "Feature-statistics shortcut mitigation evidence.",
    },
    {
        "run": "v19_lean_state_contrast_quality_safety",
        "phase": "P1 shortcut/semantic alignment",
        "hypothesis": "Cross-dataset same-state contrastive alignment pulls shared dry/wet/snow/risk semantics together while preserving separation between different road states.",
        "addresses": "Dataset-ID shortcut that remains when the same weak road condition appears with different camera styles and dataset protocols.",
        "primary_metrics": ["dataset-ID balanced accuracy", "worst-dataset F1", "risk F1", "low-friction recall", "conditional coverage", "calibrated width"],
        "success_criteria": "Dataset-ID probe drops or worst-domain behavior improves relative to v17/v18 while risk F1, low-friction recall, and coverage-width remain stable.",
        "failure_action": "If cross-domain positives are too sparse or the loss collapses useful state separation, drop the contrastive term and keep only the failure analysis.",
        "retention_rule": "Promote only if shortcut reduction is accompanied by safety/generalization stability; do not retain for pooled accuracy alone.",
        "claim_unlocked": "State-level cross-dataset semantic alignment evidence.",
    },
    {
        "run": "v20_lean_interval_order_quality_safety",
        "phase": "P1/P3 weak-interval physics",
        "hypothesis": "Non-overlapping weak friction intervals provide a dataset-agnostic physical order signal: lower interval anchors should predict lower friction means than higher anchors.",
        "addresses": "Weak-label noise and cross-dataset ontology alignment without requiring measured tire-road friction coefficients.",
        "primary_metrics": ["low-friction recall", "worst-dataset F1", "risk F1", "calibrated coverage", "calibrated width", "dataset-ID probe"],
        "success_criteria": "Improves low-friction recall or worst-dataset behavior relative to v17/v19 without widening calibrated intervals or increasing shortcut leakage.",
        "failure_action": "If the order loss over-constrains noisy intervals or widens uncertainty, drop it and keep the failure as evidence that interval anchors are too coarse for pairwise ranking.",
        "retention_rule": "Promote only if physical order consistency yields a safety/generalization gain; never retain it for pooled classification alone.",
        "claim_unlocked": "Weak interval-order physics regularization from public proxy labels.",
    },
    {
        "run": "v21_lean_quality_uncertainty_safety",
        "phase": "P1/P3 visual-quality uncertainty",
        "hypothesis": "Near-white, low-texture, and specular visual-quality scores can make weak-friction intervals more conservative exactly where camera evidence is ambiguous.",
        "addresses": "RoadSaW wet/overexposed quality slices, RoadSC low-texture snow slices, and raw interval undercoverage.",
        "primary_metrics": ["RoadSaW near-white F1", "quality-slice coverage", "low-friction recall", "calibrated coverage", "calibrated width", "risk F1"],
        "success_criteria": "Improves quality-slice coverage or low-friction recall relative to v17 without increasing calibrated width beyond the fail-fast tolerance.",
        "failure_action": "If it only widens intervals or hurts normal-quality samples, drop the visual-quality coverage weights and keep the quality diagnostics as analysis.",
        "retention_rule": "Promote only if visual-quality uncertainty improves safety/coverage on ambiguous slices with bounded width.",
        "claim_unlocked": "Visual-quality-aware conservative weak-friction interval learning.",
    },
    {
        "run": "v22_lean_quality_order_contrast_safety",
        "phase": "P1/P3 ambiguity-order alignment",
        "hypothesis": "Combining visual-quality uncertainty, weak interval-order regularization, and small cross-domain same-state contrast can improve ambiguous wet/snow slices without learning dataset identity.",
        "addresses": "RoadSaW near-white wet patches, RoadSC low-texture snow patches, weak interval ordering, and dataset-style shortcut.",
        "primary_metrics": ["quality-slice coverage", "low-friction recall", "worst-dataset F1", "dataset-ID balanced accuracy", "calibrated width", "risk F1"],
        "success_criteria": "Improves quality-slice or worst-domain behavior relative to v21/v20 while keeping calibrated width within the fail-fast tolerance and not increasing shortcut leakage.",
        "failure_action": "If the combined constraints fight each other or only widen intervals, discard v22 and keep the best single-mechanism candidate among v17/v20/v21.",
        "retention_rule": "Promote only if it gives a multi-metric gain beyond both visual-quality weighting and interval-order regularization alone.",
        "claim_unlocked": "Ambiguity-aware ordered state alignment from public proxy labels.",
    },
    {
        "run": "v23_lean_region_mixture_evidence_safety",
        "phase": "P2/P3 segmentation-transfer evidence",
        "hypothesis": "Segmentation-style local region-mixture evidence improves weak-friction interval reliability on spatially heterogeneous wet, snowy, reflective, or low-texture road appearances.",
        "addresses": "CV subfield transfer from semantic segmentation/material-region reasoning to camera-only road-friction proxy estimation without pixel-level labels.",
        "primary_metrics": ["quality-slice coverage", "RoadSaW wet/very-wet coverage", "RoadSC snow coverage", "calibrated width", "risk F1", "low-friction recall"],
        "success_criteria": "Improves quality-slice or conditional coverage relative to v21/v22 with bounded width and no safety-recall regression.",
        "failure_action": "If it only widens intervals or does not beat v21/v22 slices, discard train-time region-mixture cues and keep the post-hoc region-mixture calibration as analysis only.",
        "retention_rule": "Promote only if the segmentation-transfer cue gives measurable coverage-width or slice robustness beyond existing visual-quality weighting.",
        "claim_unlocked": "Segmentation-style region evidence for weak visual friction estimation.",
    },
    {
        "run": "v24_lean_multi_query_region_evidence_safety",
        "phase": "P2/P3 segmentation-transfer evidence",
        "hypothesis": "Mask-query-style multi-region evidence can separate heterogeneous wet/dry/snow/glare patches better than a single local evidence map, and query disagreement should widen weak friction intervals only when local regions imply different friction states.",
        "addresses": "CV transfer from semantic segmentation mask queries to weak camera-only friction-affordance interval estimation without pixel-level labels.",
        "primary_metrics": ["quality-slice coverage", "RoadSaW wet/very-wet coverage", "RoadSC snow coverage", "calibrated width", "risk F1", "low-friction recall", "query attention overlap"],
        "success_criteria": "Improves quality-slice or conditional coverage relative to v23 with bounded width, lower query overlap, and no low-friction recall regression.",
        "failure_action": "If multiple queries collapse, only widen intervals, or hurt risk/safety metrics, discard v24 and keep the simpler single-query v23/v21 route.",
        "retention_rule": "Promote only if multi-query evidence gives measurable benefit beyond v23; do not retain for visualization alone.",
        "claim_unlocked": "Mask-query local evidence for weak visual friction-affordance intervals.",
    },
    {
        "run": "v25_lean_masked_query_consistency_safety",
        "phase": "P2/P3 segmentation-transfer consistency",
        "hypothesis": "MIC-style random region masking in the weak view makes multi-query road evidence stable under local occlusion and camera-style perturbation, reducing reliance on dataset texture shortcuts.",
        "addresses": "CV transfer from semi-supervised semantic segmentation consistency to weak camera-only friction-affordance interval estimation.",
        "primary_metrics": ["quality-slice coverage", "RoadSaW wet/very-wet coverage", "calibrated width", "risk F1", "low-friction recall", "query attention overlap", "dataset-ID balanced accuracy"],
        "success_criteria": "Improves v24 stability or shortcut metrics without lowering low-friction recall and without widening intervals beyond the fail-fast tolerance.",
        "failure_action": "If masked consistency oversmooths wet/snow hard cases or hurts risk F1, discard v25 and keep v24/v23 as the segmentation-transfer branch.",
        "retention_rule": "Promote only if masked consistency improves robustness beyond v24; do not keep it as a generic augmentation.",
        "claim_unlocked": "Masked weak-view consistency for multi-region visual friction evidence.",
    },
    {
        "run": "final_lodo_roadsaw_lean_road_roi_safety",
        "phase": "Final LODO",
        "hypothesis": "The lean final method improves the hardest held-out RoadSaW test relative to full FAF.",
        "addresses": "Final OOD proof.",
        "primary_metrics": ["held-out RoadSaW risk F1", "friction F1", "low-friction recall", "conditional coverage", "width"],
        "success_criteria": "Improves or stabilizes held-out RoadSaW versus full FAF, with credible interval quality.",
        "failure_action": "Do not claim final OOD robustness; return to v9/v10/v12 evidence and redesign.",
        "retention_rule": "Required before final method claim.",
        "claim_unlocked": "Final held-out RoadSaW result.",
    },
    {
        "run": "final_lodo_rscd_lean_road_roi_safety",
        "phase": "Final LODO",
        "hypothesis": "The lean final method transfers to held-out RSCD without source-domain memorization.",
        "addresses": "Final broad public-data transfer.",
        "primary_metrics": ["held-out RSCD friction F1", "risk F1", "low-friction recall", "coverage-width"],
        "success_criteria": "Competitive held-out RSCD metrics with no evidence of shortcut reliance.",
        "failure_action": "Limit final generalization claims to datasets that pass LODO.",
        "retention_rule": "Required for broad multi-dataset generalization claim.",
        "claim_unlocked": "Final held-out RSCD result.",
    },
    {
        "run": "final_lodo_roadsc_lean_road_roi_safety",
        "phase": "Final LODO",
        "hypothesis": "The lean final method transfers to held-out snow-coverage states.",
        "addresses": "Final winter-domain transfer.",
        "primary_metrics": ["held-out RoadSC F1", "snow-state recall", "low-friction recall", "conditional coverage"],
        "success_criteria": "Preserves winter low-friction safety evidence with bounded interval width.",
        "failure_action": "Make winter transfer a limitation and future-work item.",
        "retention_rule": "Required for winter-domain robustness claim.",
        "claim_unlocked": "Final held-out RoadSC result.",
    },
    {
        "run": "final_single_roadsaw_lean_road_roi_safety",
        "phase": "Final fair baseline",
        "hypothesis": "The final lean method beats or complements ConvNeXt on RoadSaW under a matched public split.",
        "addresses": "Final same-dataset fairness.",
        "primary_metrics": ["paired delta risk F1", "paired delta wetness F1", "coverage-width delta"],
        "success_criteria": "Positive paired delta or better calibrated safety intervals at similar F1.",
        "failure_action": "If not better, present final method as OOD/interpretability-focused rather than RoadSaW in-domain SOTA.",
        "retention_rule": "Required for final RoadSaW fair-comparison claim.",
        "claim_unlocked": "Final RoadSaW vs ConvNeXt comparison.",
    },
    {
        "run": "final_single_rscd_lean_road_roi_safety",
        "phase": "Final fair baseline",
        "hypothesis": "The final lean method beats or complements ConvNeXt on RSCD under a matched public split.",
        "addresses": "Final RSCD fairness.",
        "primary_metrics": ["paired delta friction F1", "risk F1", "coverage-width delta"],
        "success_criteria": "Positive paired delta or better calibrated safety intervals at similar F1.",
        "failure_action": "If not better, use ConvNeXt as RSCD main baseline and keep FAF for uncertainty/generalization.",
        "retention_rule": "Required for final RSCD fair-comparison claim.",
        "claim_unlocked": "Final RSCD vs ConvNeXt comparison.",
    },
    {
        "run": "final_single_roadsc_lean_road_roi_safety",
        "phase": "Final fair baseline",
        "hypothesis": "The final lean method beats or complements ConvNeXt on RoadSC under a matched public split.",
        "addresses": "Final RoadSC fairness.",
        "primary_metrics": ["paired delta snow/friction F1", "low-friction recall", "coverage-width delta"],
        "success_criteria": "Positive paired delta or better calibrated safety intervals at similar F1.",
        "failure_action": "If not better, present RoadSC as stress-test/failure analysis.",
        "retention_rule": "Required for final RoadSC fair-comparison claim.",
        "claim_unlocked": "Final RoadSC vs ConvNeXt comparison.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "candidate_hypothesis_matrix.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "candidate_hypothesis_matrix.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "candidate_hypothesis_matrix.csv")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    write_csv(args.out_csv, report["rows"])
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    contract = _load_json(summary_dir / "artifact_contract_report.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    algorithm = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}

    contract_by_run = {row.get("name"): row for row in contract.get("rows", []) if isinstance(row, dict)}
    modules_by_run = {row.get("run"): row for row in algorithm.get("rows", []) if isinstance(row, dict)}
    metrics_by_run = _metrics_by_run(summary)

    rows = [
        _build_row(spec, contract_by_run, modules_by_run, metrics_by_run)
        for spec in EXPERIMENT_SPECS
    ]
    coverage = _coverage(rows)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": "complete" if coverage["missing_specs"] == [] and coverage["incomplete_fields"] == [] else "incomplete",
        "num_rows": len(rows),
        "coverage": coverage,
        "current_failure_signals": _failure_signals(shortcut, wetness, interval),
        "requirement_status": _requirement_status(completeness),
        "rows": rows,
        "decision_policy": [
            "Do not promote a candidate to the final method until P0, LODO, fair baselines, and final rows are complete.",
            "Keep a module only if it improves safety/generalization, interval quality, or interpretable evidence under the same protocol.",
            "Use matched ConvNeXt rows as the main numeric baseline when external papers use incompatible splits or labels.",
            "Use weak friction-affordance wording for RSCD/RoadSaW/RoadSC labels; never call them measured tire-road friction coefficients.",
            "Prefer lean modules when full fusion does not earn statistically defensible gains.",
        ],
    }


def _build_row(
    spec: dict[str, Any],
    contract_by_run: dict[str, dict[str, Any]],
    modules_by_run: dict[str, dict[str, Any]],
    metrics_by_run: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run = spec["run"]
    contract = contract_by_run.get(run, {})
    audit = modules_by_run.get(run, {})
    metrics = metrics_by_run.get(run, {})
    true_modules = _true_modules(audit.get("modules", {}))
    missing = contract.get("missing_required_artifacts", [])
    if not missing and contract.get("contract_status") == "complete":
        next_action = "ready_for_claim_limited_to_predeclared_metrics"
    elif contract.get("progress_status") == "running_or_partial":
        next_action = "finish_current_training_then_postprocess"
    elif contract.get("config_exists") is False:
        next_action = "fix_or_create_config_before_training"
    else:
        next_action = "run_training_and_full_eval_pipeline"
    return {
        "run": run,
        "phase": spec["phase"],
        "progress_status": contract.get("progress_status", "missing"),
        "contract_status": contract.get("contract_status", "missing"),
        "hypothesis": spec["hypothesis"],
        "addresses": spec["addresses"],
        "key_modules": true_modules,
        "primary_metrics": spec["primary_metrics"],
        "success_criteria": spec["success_criteria"],
        "failure_action": spec["failure_action"],
        "retention_rule": spec["retention_rule"],
        "claim_unlocked": spec["claim_unlocked"],
        "available_metrics": _available_metrics(metrics),
        "missing_required_artifacts": missing,
        "next_action": next_action,
    }


def _true_modules(modules: dict[str, Any]) -> list[str]:
    out = []
    for name in INTERESTING_MODULES:
        if modules.get(name):
            out.append(name)
    return out


def _available_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "friction_macro_f1",
        "risk_macro_f1",
        "low_friction_recall",
        "worst_dataset_f1",
        "calibrated_coverage",
        "calibrated_width",
        "dataset_id_balanced_accuracy",
    ]
    return {key: metrics.get(key) for key in keys if metrics.get(key) is not None}


def _metrics_by_run(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in [
        "ablation",
        "lodo",
        "single_dataset",
        "fair_baselines",
        "final_lodo",
        "final_single_dataset",
    ]:
        source_rows = summary.get(key, [])
        if not isinstance(source_rows, list):
            source_rows = []
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            run = _run_from_output(row.get("output_dir"))
            if run:
                out[run] = row
    return out


def _run_from_output(output_dir: Any) -> str | None:
    if not output_dir:
        return None
    return Path(str(output_dir)).name


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing_specs = [
        row["run"]
        for row in rows
        if not row.get("hypothesis")
        or not row.get("success_criteria")
        or not row.get("failure_action")
        or not row.get("retention_rule")
        or not row.get("primary_metrics")
    ]
    incomplete_fields = []
    for row in rows:
        for key in ["phase", "addresses", "claim_unlocked", "next_action"]:
            if not row.get(key):
                incomplete_fields.append({"run": row["run"], "field": key})
    phases: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        phases[row["phase"]] = phases.get(row["phase"], 0) + 1
        statuses[row["contract_status"]] = statuses.get(row["contract_status"], 0) + 1
    return {
        "missing_specs": missing_specs,
        "incomplete_fields": incomplete_fields,
        "phase_counts": phases,
        "contract_status_counts": statuses,
        "candidate_runs": [row["run"] for row in rows if row["phase"].startswith(("P1", "P2", "P3"))],
        "final_runs": [row["run"] for row in rows if row["phase"].startswith("Final")],
        "fair_baseline_runs": [row["run"] for row in rows if row["phase"] == "Fair baseline"],
        "lodo_runs": [row["run"] for row in rows if row["phase"] == "P0.5 LODO"],
    }


def _failure_signals(
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    interval: dict[str, Any],
) -> list[dict[str, Any]]:
    signals = []
    if shortcut.get("verdict") == "warn":
        signals.append(
            {
                "signal": "dataset_shortcut_high",
                "evidence": f"{shortcut.get('num_high_shortcut')} of {shortcut.get('num_complete')} completed rows exceed the shortcut threshold.",
                "candidate_response": "v6/v7/v11/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25 plus final lean rows must reduce dataset-ID probes.",
            }
        )
    if int(wetness.get("num_watchlist", 0) or 0) > 0:
        signals.append(
            {
                "signal": "roadsaw_wetness_weak",
                "evidence": f"{wetness.get('num_watchlist')} completed rows are on the RoadSaW wetness watchlist.",
                "candidate_response": "v9/v10/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25 and final RoadSaW rows must improve damp/wet/very_wet behavior.",
            }
        )
    if int(interval.get("num_watchlist_items", 0) or 0) > 0:
        signals.append(
            {
                "signal": "conditional_interval_undercoverage",
                "evidence": f"{interval.get('num_watchlist_items')} conditional cells are undercovered.",
                "candidate_response": "v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25/final rows must improve conditional coverage without excessive width.",
            }
        )
    return signals


def _requirement_status(completeness: dict[str, Any]) -> dict[str, str]:
    out = {}
    rows = completeness.get("requirements", [])
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        out[str(row.get("name"))] = str(row.get("status"))
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Hypothesis Matrix",
        "",
        f"Generated at: {report['generated_at']}",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Coverage",
        "",
        f"- Rows: `{report['num_rows']}`.",
        f"- Phase counts: `{json.dumps(report['coverage']['phase_counts'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Contract status counts: `{json.dumps(report['coverage']['contract_status_counts'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Missing specs: `{', '.join(report['coverage']['missing_specs']) or '-'}`.",
        "",
        "## Current Failure Signals",
        "",
    ]
    if report["current_failure_signals"]:
        for item in report["current_failure_signals"]:
            lines.append(f"- `{item['signal']}`: {item['evidence']} Response: {item['candidate_response']}")
    else:
        lines.append("- No active failure signal recorded.")
    lines.extend(["", "## Matrix", ""])
    lines.append("| Phase | Run | Status | Hypothesis | Key modules | Primary metrics | Success rule | Failure action |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in report["rows"]:
        lines.append(
            "| {phase} | `{run}` | `{status}` | {hyp} | {modules} | {metrics} | {success} | {failure} |".format(
                phase=row["phase"],
                run=row["run"],
                status=row["contract_status"],
                hyp=row["hypothesis"],
                modules=", ".join(f"`{item}`" for item in row["key_modules"]) or "-",
                metrics=", ".join(row["primary_metrics"]),
                success=row["success_criteria"],
                failure=row["failure_action"],
            )
        )
    lines.extend(["", "## Retention And Claim Rules", ""])
    lines.append("| Run | Retention rule | Claim unlocked | Next action |")
    lines.append("|---|---|---|---|")
    for row in report["rows"]:
        lines.append(
            "| `{run}` | {retain} | {claim} | `{next}` |".format(
                run=row["run"],
                retain=row["retention_rule"],
                claim=row["claim_unlocked"],
                next=row["next_action"],
            )
        )
    lines.extend(["", "## Decision Policy", ""])
    for item in report["decision_policy"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "run",
        "progress_status",
        "contract_status",
        "hypothesis",
        "addresses",
        "key_modules",
        "primary_metrics",
        "success_criteria",
        "failure_action",
        "retention_rule",
        "claim_unlocked",
        "next_action",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "; ".join(str(item) for item in row.get(key, []))
                    if isinstance(row.get(key), list)
                    else row.get(key, "")
                    for key in fields
                }
            )


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
