from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


SOURCE_ROWS = [
    {
        "name": "DomainBed",
        "venue_or_role": "ICLR 2021 domain-generalization protocol",
        "url": "https://openreview.net/forum?id=lQdXeXDoWtI",
        "official_code": "https://github.com/facebookresearch/DomainBed",
        "innovation_pattern": "Do not trust a new DG method without fixed held-out domains, strong ERM baselines, and identical evaluation.",
        "project_mapping": "LODO runs, matched ConvNeXt baselines, dataset-ID probes, and strict no-OOD-claim gates.",
        "current_status": "protocol_aligned",
        "evidence_needed": "Complete lodo_* rows, especially lodo_roadsaw_full_faf.",
    },
    {
        "name": "ConvNeXt",
        "venue_or_role": "CVPR 2022 strong vision backbone baseline",
        "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html",
        "official_code": "https://pytorch.org/vision/stable/models/convnext.html",
        "innovation_pattern": "A new task-specific model must beat or complement a clean modern backbone under the same split and metrics.",
        "project_mapping": "baseline_single_*_global_convnext configs provide the matched public-dataset comparison.",
        "current_status": "configured_pending_results",
        "evidence_needed": "Complete single_*_full_faf and baseline_single_*_global_convnext rows.",
    },
    {
        "name": "FDA",
        "venue_or_role": "CVPR 2020 Fourier style/domain adaptation",
        "url": "https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html",
        "official_code": "https://github.com/YanchaoYang/FDA",
        "innovation_pattern": "Perturb low-frequency image style while preserving geometry to reduce camera/dataset shortcut.",
        "project_mapping": "v6-v12/v14/v15/v16/v18/v19/v20/v21/v22/v23/v24/final configs use Fourier style jitter; v15 tests road-centric input canonicalization, and v16/v18/v19/v20/v21/v22/v23/v24 add soft color-constancy canonicalization.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Show lower dataset-ID balanced accuracy without losing risk F1 or low-friction recall.",
    },
    {
        "name": "MixStyle",
        "venue_or_role": "ICLR 2021 feature-statistic style randomization",
        "url": "https://openreview.net/forum?id=6xHJ37MVxxp",
        "official_code": "https://github.com/KaiyangZhou/mixstyle-release",
        "innovation_pattern": "Randomize feature statistics so the classifier relies less on domain-specific style.",
        "project_mapping": "v18_lean_mixstyle_quality_safety adds training-only Feature MixStyle on shared normalized features as a cheap shortcut-mitigation probe.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if dataset-ID balanced accuracy or worst-domain behavior improves without hurting risk F1 or low-friction recall.",
    },
    {
        "name": "Supervised Contrastive Learning",
        "venue_or_role": "NeurIPS 2020 state/label-aware representation learning",
        "url": "https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html",
        "official_code": "https://github.com/HobbitLong/SupContrast",
        "innovation_pattern": "Use labels to pull semantically similar samples together and push different states apart in representation space.",
        "project_mapping": "v19_lean_state_contrast_quality_safety applies this idea as cross-dataset same-state alignment for weak friction/risk/wetness labels.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if dataset-ID shortcut drops or worst-domain behavior improves without losing safety recall or interval coverage.",
    },
    {
        "name": "Weak Interval-Order Regularization",
        "venue_or_role": "ordinal/ranking-inspired weak-supervision pattern",
        "url": "https://papers.nips.cc/paper_files/paper/2005/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html",
        "official_code": "",
        "innovation_pattern": "Use pairwise order constraints when labels are interval-valued or ordinal rather than exact continuous measurements.",
        "project_mapping": "v20_lean_interval_order_quality_safety uses non-overlapping weak friction intervals to order predicted friction means without pretending public proxy labels are measured tire-road friction.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if low-friction recall or worst-dataset behavior improves without widening calibrated intervals or increasing shortcut leakage.",
    },
    {
        "name": "Visual Quality Aware Uncertainty",
        "venue_or_role": "uncertainty/calibration pattern for ambiguous visual evidence",
        "url": "https://arxiv.org/abs/1706.04599",
        "official_code": "",
        "innovation_pattern": "Make predictions more conservative when the image evidence itself is unreliable or ambiguous, and judge success by coverage-width tradeoff rather than accuracy alone.",
        "project_mapping": "v21_lean_quality_uncertainty_safety, v22_lean_quality_order_contrast_safety, v23_lean_region_mixture_evidence_safety, and v24_lean_multi_query_region_evidence_safety use image-derived near-white, low-texture, specular-highlight, local region-mixture, and multi-query disagreement scores to make intervals conservative on ambiguous road images.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if quality-slice coverage or low-friction recall improves without excessive calibrated-width inflation.",
    },
    {
        "name": "Ambiguity Ordered State Alignment",
        "venue_or_role": "compound weak-supervision pattern for public proxy labels",
        "url": "https://papers.nips.cc/paper_files/paper/2005/hash/afdec7005cc9f14302cd0474fd0f3c96-Abstract.html",
        "official_code": "",
        "innovation_pattern": "Combine conservative uncertainty, ordinal/interval ordering, and label-aware representation alignment only where public weak labels support the constraint.",
        "project_mapping": "v22_lean_quality_order_contrast_safety combines v21 visual-quality coverage weighting with v20 weak interval-order loss and a small v19-style same-state contrastive term.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if it beats v20 and v21 on quality-slice/worst-domain metrics with bounded calibrated width and no extra shortcut leakage.",
    },
    {
        "name": "Segmentation-style region mixture and mask-query evidence",
        "venue_or_role": "CV segmentation/material-region reasoning pattern",
        "url": "https://arxiv.org/abs/2112.01527",
        "official_code": "https://github.com/facebookresearch/Mask2Former",
        "innovation_pattern": "Reason over local regions and material mixtures instead of treating the whole image as one homogeneous class.",
        "project_mapping": "v23_lean_region_mixture_evidence_safety implements a lightweight differentiable region-mixture proxy inside EvidenceField, v24_lean_multi_query_region_evidence_safety adds mask-query-style local evidence pooling and query-disagreement interval expansion, and v25_lean_masked_query_consistency_safety adds MIC-style masked weak-view consistency without requiring pixel-level road masks.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Keep only if it improves wet/snow/low-texture coverage-width tradeoff beyond v21/v22 and single-query v23/v24, without over-smoothing hard wet/snow states.",
    },
    {
        "name": "DANN",
        "venue_or_role": "JMLR domain-adversarial representation learning",
        "url": "https://jmlr.org/papers/v17/15-239.html",
        "official_code": "https://jmlr.org/papers/v17/15-239.html",
        "innovation_pattern": "Use gradient reversal so the shared representation is predictive for the task but weak for domain ID.",
        "project_mapping": "v7_full_faf_fourier_dann tests whether adversarial domain suppression helps after style jitter.",
        "current_status": "implemented_candidate_pending_results",
        "evidence_needed": "Pass the dataset-ID probe while preserving worst-dataset F1.",
    },
    {
        "name": "GroupDRO",
        "venue_or_role": "ICLR 2020 worst-group robustness",
        "url": "https://arxiv.org/abs/1911.08731",
        "official_code": "https://github.com/kohpangwei/group_DRO",
        "innovation_pattern": "Optimize for the worst group, not only the average, when groups are known or inferable.",
        "project_mapping": "Current DG losses are provisional because v3 hurt primary metrics; use reports to decide removal or redesign.",
        "current_status": "partly_implemented_needs_rework",
        "evidence_needed": "Must improve worst-dataset F1 or RoadSaW wetness without major average-metric loss.",
    },
    {
        "name": "SAM optimizer",
        "venue_or_role": "ICLR 2021 sharpness-aware generalization",
        "url": "https://openreview.net/forum?id=6Tm1mposlrM",
        "official_code": "https://github.com/google-research/sam",
        "innovation_pattern": "Favor flat minima when models overfit source domains or unstable weak labels.",
        "project_mapping": "Future lightweight optimizer candidate if LODO shows overfitting and GPU budget remains acceptable.",
        "current_status": "future_candidate",
        "evidence_needed": "Needs matched train-time and same protocol; do not mix with final claims until ablated.",
    },
    {
        "name": "DINOv2",
        "venue_or_role": "self-supervised foundation visual features",
        "url": "https://arxiv.org/abs/2304.07193",
        "official_code": "https://github.com/facebookresearch/dinov2",
        "innovation_pattern": "Use broad self-supervised pretraining to reduce dependence on small public road-condition labels.",
        "project_mapping": "Future frozen-feature baseline or backbone if ConvNeXt/FAF is not strong enough.",
        "current_status": "future_candidate",
        "evidence_needed": "Needs same-split baseline and no foundation-model claim without actual runs.",
    },
    {
        "name": "WCamNet",
        "venue_or_role": "2024/2025 visual winter road-friction estimation reference",
        "url": "https://arxiv.org/abs/2404.16578",
        "official_code": "https://arxiv.org/abs/2404.16578",
        "innovation_pattern": "Fuse foundation visual features with local road-texture reasoning for winter friction targets.",
        "project_mapping": "Supports a future DINOv2/foundation-feature baseline if ConvNeXt and FAF remain weak on RoadSC/RoadSaW winter or wet states.",
        "current_status": "contextual_external_reference",
        "evidence_needed": "Needs local reproduction or a protocol-equivalent dataset before any numeric comparison.",
    },
    {
        "name": "Non-contact RSC detection review",
        "venue_or_role": "Sensors 2022 road-surface-condition sensing survey",
        "url": "https://www.mdpi.com/1424-8220/22/24/9583",
        "official_code": "https://www.mdpi.com/1424-8220/22/24/9583",
        "innovation_pattern": "Wet, icy, watery, and snowy states are not only semantic classes; they have different optical, thermal, and water-film mechanisms.",
        "project_mapping": "Use as the cross-discipline rationale for separating wetness ordinal learning, specular/reflection cues, and conservative friction intervals.",
        "current_status": "contextual_physics_reference",
        "evidence_needed": "Tie every wet-road module to RoadSaW wetness F1, severe wetness misorder, low-friction recall, and interval coverage-width.",
    },
    {
        "name": "Water hazard reflection attention",
        "venue_or_role": "ECCV 2018 water detection from moving cameras",
        "url": "https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xiaofeng_Han_Single_Image_Water_ECCV_2018_paper.pdf",
        "official_code": "https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Xiaofeng_Han_Single_Image_Water_ECCV_2018_paper.pdf",
        "innovation_pattern": "Water is highly refractive and view-dependent; reflection-aware attention is a physically motivated way to detect puddles and water films.",
        "project_mapping": "Motivates a wet-surface cue head: brightness/specular contrast, sky-reflection consistency, and ROI-limited evidence for damp/wet/very_wet states.",
        "current_status": "future_candidate",
        "evidence_needed": "Add only after v9/v10/v14/v15/v16 identify wet-state residual errors; ablate against ordinary wetness ordinal loss.",
    },
    {
        "name": "Polarization wet-road cue",
        "venue_or_role": "classical wet-road optical sensing reference",
        "url": "https://www.researchgate.net/publication/373220207_Detection_of_Wet-Road_Conditions_from_Images_Captured_by_a_Vehicle-Mounted_Camera",
        "official_code": "https://www.researchgate.net/publication/373220207_Detection_of_Wet-Road_Conditions_from_Images_Captured_by_a_Vehicle-Mounted_Camera",
        "innovation_pattern": "Wet asphalt can be easier to distinguish through specular/polarization behavior than through RGB color alone.",
        "project_mapping": "Use as inspiration for RGB-only proxy features: highlight fraction, local contrast loss, and reflection-like texture statistics inside road ROI.",
        "current_status": "contextual_physics_reference",
        "evidence_needed": "Do not claim polarization without a polarization sensor; use it only to motivate RGB proxy features and wetness failure analysis.",
    },
    {
        "name": "Black-ice optical ambiguity",
        "venue_or_role": "2024 transportation safety report",
        "url": "https://www.dot.state.wy.us/files/live/sites/wydot/files/shared/Planning/Research/RS04225_Black_Ice.pdf",
        "official_code": "https://www.dot.state.wy.us/files/live/sites/wydot/files/shared/Planning/Research/RS04225_Black_Ice.pdf",
        "innovation_pattern": "Black ice can look visually transparent, wet, or glossy, so image-only friction intervals must stay conservative when evidence is ambiguous.",
        "project_mapping": "Motivates uncertainty expansion for visually wet/glossy/ice-like states instead of forcing overconfident point labels.",
        "current_status": "contextual_safety_reference",
        "evidence_needed": "Final interval metrics must show high low-friction recall and subgroup coverage for ice/wet/very_wet states even if width grows modestly.",
    },
    {
        "name": "Segment Anything",
        "venue_or_role": "ICCV 2023 promptable segmentation",
        "url": "https://arxiv.org/abs/2304.02643",
        "official_code": "https://github.com/facebookresearch/segment-anything",
        "innovation_pattern": "Use external segmentation priors to constrain where evidence is allowed to come from.",
        "project_mapping": "Future offline pseudo-road-mask generator for EvidenceField supervision.",
        "current_status": "future_optional_pseudo_label_source",
        "evidence_needed": "Generated masks must be versioned, audited, and ablated against bottom-road priors.",
    },
    {
        "name": "Mask2Former",
        "venue_or_role": "CVPR 2022 universal segmentation",
        "url": "https://arxiv.org/abs/2112.01527",
        "official_code": "https://github.com/facebookresearch/Mask2Former",
        "innovation_pattern": "Use semantic segmentation to separate road pixels from vehicles/sky/background shortcuts.",
        "project_mapping": "Alternative pseudo-road-mask source if SAM masks are unstable for road scenes.",
        "current_status": "future_optional_pseudo_label_source",
        "evidence_needed": "Only add after current ROI candidates show attention still leaks outside road regions.",
    },
    {
        "name": "RAPS / conformal prediction",
        "venue_or_role": "ICLR 2021 uncertainty-set calibration",
        "url": "https://openreview.net/forum?id=eNdiU_DbM9",
        "official_code": "https://openreview.net/forum?id=eNdiU_DbM9",
        "innovation_pattern": "Report coverage together with set/interval size, and avoid hiding poor conditional coverage.",
        "project_mapping": "interval_calibration_90, conditional interval watchlists, and coverage-width selection score.",
        "current_status": "implemented_reporting_and_calibration",
        "evidence_needed": "Final rows must report pooled and conditional coverage plus width.",
    },
    {
        "name": "RoadSaW",
        "venue_or_role": "CVPRW 2022 road surface and wetness dataset",
        "url": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/papers/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.pdf",
        "official_code": "https://openaccess.thecvf.com/content/CVPR2022W/WAD/html/Cordes_RoadSaW_A_Large-Scale_Dataset_for_Camera-Based_Road_Surface_and_Wetness_CVPRW_2022_paper.html",
        "innovation_pattern": "Use wetness as the hardest visual proxy for low-friction safety risk.",
        "project_mapping": "Held-out RoadSaW LODO and wetness-state diagnostics are the main OOD stress tests.",
        "current_status": "base_lodo_complete_failed_final_rows_pending",
        "evidence_needed": "Use lodo_roadsaw_full_faf as failure evidence; need final_lodo_roadsaw_* results before any held-out RoadSaW success claim.",
    },
    {
        "name": "RoadSC",
        "venue_or_role": "ICCVW 2023 road snow coverage dataset",
        "url": "https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/papers/Cordes_Camera-Based_Road_Snow_Coverage_Estimation_ICCVW_2023_paper.pdf",
        "official_code": "https://viscoda.com/index.php/en/downloads-en/roadsc-dataset",
        "innovation_pattern": "Use snow coverage levels as a low-friction winter-domain stress test that complements RoadSaW wetness.",
        "project_mapping": "RoadSC single-dataset comparison and held-out RoadSC LODO rows test snow/ice transfer and interval coverage.",
        "current_status": "base_lodo_complete_failed_final_rows_pending",
        "evidence_needed": "Use lodo_roadsc_full_faf as failure evidence; need final_lodo_roadsc_* results and RoadSC conditional interval cells before any winter-domain success claim.",
    },
    {
        "name": "RSCD",
        "venue_or_role": "public road-surface classification dataset",
        "url": "https://thu-rsxd.com/rscd/",
        "official_code": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9343931/",
        "innovation_pattern": "Use public road-surface labels as weak friction-affordance anchors, not measured tire-road mu.",
        "project_mapping": "RSCD single-dataset comparison and RSCD held-out LODO rows.",
        "current_status": "dataset_in_protocol_pending_fair_baseline",
        "evidence_needed": "Need single_rscd_full_faf versus baseline_single_rscd_global_convnext.",
    },
    {
        "name": "RSCD per-day split",
        "venue_or_role": "public RSCD split for stronger generalization checks",
        "url": "https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split",
        "official_code": "https://github.com/MiviaLab/Road-Surface-Dataset-per-day-split",
        "innovation_pattern": "Reduce acquisition-day leakage by testing recognition across disjoint collection days.",
        "project_mapping": "Future optional protocol if single-dataset RSCD looks too easy or dataset shortcut remains high.",
        "current_status": "future_protocol_candidate",
        "evidence_needed": "Only use after manifests are regenerated and compared against the current official/matched split.",
    },
    {
        "name": "Extreme Road Image Dataset",
        "venue_or_role": "public extreme-road image dataset linked to image+dynamics friction research",
        "url": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "official_code": "https://github.com/sean-shiyuez/Extreme-Road-Image-Dataset",
        "innovation_pattern": "Use more direct extreme-road visual categories as an intermediate bridge between weak road-condition labels and friction-estimation literature.",
        "project_mapping": "Future direct-friction validation route; keep outside current benchmark until splits, labels, dynamics alignment, and license are audited.",
        "current_status": "future_dataset_candidate",
        "evidence_needed": "Create a separate protocol before using it; do not merge it into current RSCD/RoadSaW/RoadSC results.",
    },
    {
        "name": "Continual Cross-Dataset Adaptation",
        "venue_or_role": "road-surface classification cross-dataset adaptation reference",
        "url": "https://arxiv.org/abs/2309.02210",
        "official_code": "https://github.com/PCudrano/continual_road_surface_classification",
        "innovation_pattern": "Treat cross-dataset degradation as an expected failure mode and evaluate adaptation separately from ordinary supervised training.",
        "project_mapping": "Future continual/test-time adaptation route if LODO confirms that RoadSaW or RoadSC transfer fails badly.",
        "current_status": "future_protocol_candidate",
        "evidence_needed": "Needs controlled no-leakage adaptation configs and matched baselines before inclusion.",
    },
    {
        "name": "RoadFormer",
        "venue_or_role": "2025 RSCD-focused local-global feature fusion architecture",
        "url": "https://arxiv.org/html/2506.02358v1",
        "official_code": "https://arxiv.org/html/2506.02358v1",
        "innovation_pattern": "Combine local texture cues with global scene context and explicitly separate foreground/background road-relevant evidence.",
        "project_mapping": "Supports the lean road-ROI EvidenceField direction and motivates reporting RSCD single-dataset results against a protocol-matched ConvNeXt baseline before citing external RSCD numbers.",
        "current_status": "contextual_external_reference",
        "evidence_needed": "Do not claim a numeric win unless RSCD split, label space, preprocessing, and metric definition match; otherwise use it as architectural motivation only.",
    },
]


PATTERN_ROWS = [
    {
        "category": "Protocol and fair comparison",
        "topvenue_routine": "Fixed splits, strong ERM/backbone baseline, LODO before OOD claims.",
        "configured_runs": [
            "lodo_roadsaw_full_faf",
            "lodo_rscd_full_faf",
            "lodo_roadsc_full_faf",
            "baseline_single_roadsaw_global_convnext",
            "baseline_single_rscd_global_convnext",
            "baseline_single_roadsc_global_convnext",
        ],
        "decision_rule": "No top-venue claim until LODO and matched ConvNeXt rows are complete.",
    },
    {
        "category": "Shortcut mitigation",
        "topvenue_routine": "Break style shortcuts with Fourier/style perturbation, adversarial domain heads, or small adapters.",
        "configured_runs": [
            "v6_full_faf_fourier",
            "v7_full_faf_fourier_dann",
            "v11_full_faf_domain_adapter",
            "v18_lean_mixstyle_quality_safety",
            "v19_lean_state_contrast_quality_safety",
            "v21_lean_quality_uncertainty_safety",
            "v22_lean_quality_order_contrast_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "decision_rule": "Keep only if dataset-ID balanced accuracy drops and safety metrics do not collapse.",
    },
    {
        "category": "Condition-aware robustness",
        "topvenue_routine": "Align or rebalance within the same semantic/physical condition instead of forcing all domains together.",
        "configured_runs": [
            "v9_full_faf_roadsaw_hard_sampling",
            "v10_full_faf_consistency",
            "v12_full_faf_roi_interval_safety",
            "v14_lean_road_roi_safety",
            "v19_lean_state_contrast_quality_safety",
            "v21_lean_quality_uncertainty_safety",
            "v22_lean_quality_order_contrast_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
            "final_lodo_roadsaw_lean_road_roi_safety",
        ],
        "decision_rule": "RoadSaW wetness macro-F1, very_wet coverage, and dataset-ID probe should improve without reviving the unstable full DG stack.",
    },
    {
        "category": "Wet-road physics cues",
        "topvenue_routine": "Use physics to decide which visual evidence is meaningful: wet roads add specular/reflection cues, water films reduce texture contrast, and black ice requires conservative uncertainty.",
        "configured_runs": [
            "v9_full_faf_roadsaw_hard_sampling",
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
            "v25_lean_masked_query_consistency_safety",
        ],
        "decision_rule": "Keep a wet-road module only if damp/wet/very_wet F1 or ordinal MAE improves while low-friction recall and calibrated coverage do not regress.",
    },
    {
        "category": "Evidence grounding",
        "topvenue_routine": "Constrain explanations to plausible road regions; test success/failure examples quantitatively.",
        "configured_runs": [
            "v8_full_faf_fourier_roadprior",
            "v12_full_faf_roi_interval_safety",
            "v14_lean_road_roi_safety",
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
        "decision_rule": "Attention should remain in road/bottom ROI without hurting low-friction recall.",
    },
    {
        "category": "Segmentation-transfer local regions",
        "topvenue_routine": "Borrow segmentation-style region reasoning while respecting weak public labels: no pixel-level friction claims without pixel labels.",
        "configured_runs": [
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
        ],
        "decision_rule": "Keep only if region-mixture, multi-query, or masked-consistency cues improve quality-slice/conditional interval coverage at bounded width.",
    },
    {
        "category": "Interval and safety quality",
        "topvenue_routine": "Treat coverage and width as a pair; report subgroup failures instead of only pooled calibration.",
        "configured_runs": [
            "v12_full_faf_roi_interval_safety",
            "v14_lean_road_roi_safety",
            "v17_lean_quality_physics_safety",
            "v18_lean_mixstyle_quality_safety",
            "v19_lean_state_contrast_quality_safety",
            "v20_lean_interval_order_quality_safety",
            "v21_lean_quality_uncertainty_safety",
            "v22_lean_quality_order_contrast_safety",
            "v23_lean_region_mixture_evidence_safety",
            "v24_lean_multi_query_region_evidence_safety",
            "v25_lean_masked_query_consistency_safety",
            "final_lodo_roadsaw_lean_road_roi_safety",
        ],
        "decision_rule": "Final method must improve conditional coverage/width tradeoff, not only average F1.",
    },
    {
        "category": "Module pruning",
        "topvenue_routine": "Remove complexity that does not earn a metric or interpretability gain.",
        "configured_runs": [
            "v13_lean_physics_evidence",
            "v14_lean_road_roi_safety",
            "final_single_roadsaw_lean_road_roi_safety",
        ],
        "decision_rule": "FrictionSet/full DG losses stay out of the final method unless later evidence rescues them; small semantic conditional alignment is judged separately.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "topvenue_innovation_roadmap.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "topvenue_innovation_roadmap.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    live = _load_json(summary_dir / "v5_full_faf_training_diagnosis.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    lodo_generalization = _load_json(summary_dir / "lodo_generalization_report.json") or {}

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "readiness": {
            "verdict": gate.get("verdict"),
            "num_blocks": gate.get("num_blocks"),
            "num_warnings": gate.get("num_warnings"),
        },
        "progress_counts": dashboard.get("progress_counts", {}),
        "source_rows": SOURCE_ROWS,
        "pattern_rows": PATTERN_ROWS,
        "current_evidence": _current_evidence(summary, shortcut, wetness, interval, p0, live, final_selection),
        "next_decisions": _next_decisions(summary, shortcut, wetness, interval, p0, live, lodo_generalization),
        "strict_claim_rules": [
            "Use published papers and GitHub repositories as method/protocol references, not as comparable numbers unless split, labels, and metrics match.",
            "Use matched ConvNeXt rows as the primary fair visual baseline for RSCD, RoadSaW, and RoadSC.",
            "Use LODO rows, especially held-out RoadSaW, as the main cross-dataset generalization evidence.",
            "Describe the target as a visual-evidence-conditioned friction affordance interval; do not call weak public labels measured tire-road friction coefficients.",
            "Keep a module only if it improves safety/generalization or clearly improves interpretable evidence without a major shortcut or worst-group regression.",
            "Treat wet/ice ambiguity as a calibrated safety problem: a visually ambiguous wet or glossy road may justify a wider interval if it protects low-friction recall.",
        ],
    }


def _current_evidence(
    summary: dict[str, Any],
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    interval: dict[str, Any],
    p0: dict[str, Any],
    live: dict[str, Any],
    final_selection: dict[str, Any],
) -> dict[str, Any]:
    core = summary.get("core_ablation", [])
    complete_core = [row for row in core if row.get("status") == "complete"]
    best_by_risk = max(
        complete_core,
        key=lambda row: float(row.get("risk_f1") or row.get("risk_macro_f1") or -1.0),
        default={},
    )
    best_by_safety = max(complete_core, key=_p0_safety_score, default=best_by_risk)
    latest_epoch = live.get("latest_epoch") or {}
    best_epoch = live.get("best_val_loss_epoch") or live.get("best_safety_proxy_epoch") or {}
    signals = live.get("signals") or {}
    return {
        "p0_complete_rows": len(complete_core),
        "p0_total_rows": len(core),
        "p0_best_completed_by_risk_method": best_by_risk.get("method"),
        "p0_best_completed_by_risk_f1": best_by_risk.get("risk_f1") or best_by_risk.get("risk_macro_f1"),
        "p0_best_completed_by_safety_score_method": best_by_safety.get("method"),
        "p0_best_completed_by_safety_score": _p0_safety_score(best_by_safety) if best_by_safety else None,
        "p0_best_completed_by_safety_risk_f1": best_by_safety.get("risk_f1") or best_by_safety.get("risk_macro_f1"),
        "p0_best_completed_by_safety_low_friction_recall": best_by_safety.get("low_friction_recall"),
        "p0_best_completed_by_safety_worst_dataset_f1": best_by_safety.get("worst_dataset_f1"),
        "p0_core_status": p0.get("core_status"),
        "v5_best_epoch": best_epoch.get("epoch"),
        "v5_latest_epoch": latest_epoch.get("epoch"),
        "v5_coverage_degradation_flag": signals.get("coverage_degradation_flag"),
        "dataset_shortcut_verdict": shortcut.get("verdict"),
        "dataset_shortcut_high_rows": shortcut.get("num_high_shortcut"),
        "roadsaw_wetness_watchlist": wetness.get("num_watchlist"),
        "conditional_interval_watchlist": interval.get("num_watchlist_items"),
    }


def _p0_safety_score(row: dict[str, Any]) -> float:
    risk = float(row.get("risk_f1") or row.get("risk_macro_f1") or 0.0)
    low_recall = float(row.get("low_friction_recall") or 0.0)
    worst = float(row.get("worst_dataset_f1") or 0.0)
    coverage = float(row.get("calibrated_coverage") or row.get("mu_interval_coverage_calibrated") or 0.0)
    raw = float(row.get("raw_coverage") or row.get("mu_interval_coverage") or 0.0)
    return 0.25 * risk + 0.25 * low_recall + 0.25 * worst + 0.15 * coverage + 0.10 * raw


def _next_decisions(
    summary: dict[str, Any],
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    interval: dict[str, Any],
    p0: dict[str, Any],
    live: dict[str, Any],
    lodo_generalization: dict[str, Any],
) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    core_status = p0.get("core_status")
    if core_status != "complete":
        decisions.append(
            {
                "decision": "Close P0 before changing the method again.",
                "trigger": "Full model row is not postprocessed yet.",
                "action": "Let v5_full_faf finish, then postprocess test/calibration/bootstrap artifacts.",
            }
        )
    lodo_rows = lodo_generalization.get("rows", []) if isinstance(lodo_generalization.get("rows"), list) else []
    lodo_complete = bool(lodo_rows) and all(row.get("status") == "complete" for row in lodo_rows)
    if not lodo_complete and not _group_complete(summary.get("lodo", [])):
        decisions.append(
            {
                "decision": "Run LODO before claiming cross-dataset generalization.",
                "trigger": "LODO rows are missing.",
                "action": "Inspect held-out RoadSaW first; it is the current stress test.",
            }
        )
    elif "failure" in str(lodo_generalization.get("verdict") or ""):
        decisions.append(
            {
                "decision": "Use completed LODO as failure evidence before claiming OOD robustness.",
                "trigger": "LODO rows are complete but transfer collapses on held-out domains.",
                "action": "Prioritize style, wetness, quality, ROI, and interval candidates; do not claim cross-dataset success yet.",
            }
        )
    if not _group_complete(summary.get("fair_single_dataset_deltas", [])):
        decisions.append(
            {
                "decision": "Use matched ConvNeXt as the fair public baseline.",
                "trigger": "Single-dataset FAF-vs-ConvNeXt rows are pending.",
                "action": "Report same split, same labels, same metrics, paired bootstrap deltas.",
            }
        )
    if shortcut.get("num_high_shortcut"):
        decisions.append(
            {
                "decision": "Prioritize shortcut-mitigation candidates.",
                "trigger": "Dataset-ID probe is high on completed rows.",
                "action": "Rank v6/v7/v11/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 by dataset-ID drop, risk F1, low recall, quality-slice robustness, and worst-dataset F1.",
            }
        )
    if wetness.get("num_watchlist"):
        decisions.append(
            {
                "decision": "Prioritize RoadSaW wetness candidates.",
                "trigger": "RoadSaW damp/wet/very_wet confusion is on the watchlist.",
                "action": "Rank v9/v10/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 by RoadSaW wetness macro-F1, ordinal MAE, quality-slice robustness, and severe misorder.",
            }
        )
    if interval.get("num_watchlist_items"):
        decisions.append(
            {
                "decision": "Prioritize interval-quality candidates.",
                "trigger": "Conditional undercoverage remains high.",
                "action": "Rank v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/final by conditional coverage-width tradeoff.",
            }
        )
    if (live.get("signals") or {}).get("coverage_degradation_flag"):
        decisions.append(
            {
                "decision": "Do not select the latest v5 epoch blindly.",
                "trigger": "Raw interval coverage is degrading during v5 training.",
                "action": "Use selected best checkpoint and compare against lean v13/v14 routes.",
            }
        )
    return decisions


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Top-Venue Innovation Roadmap",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Current Evidence",
        "",
    ]
    evidence = report["current_evidence"]
    for key, value in evidence.items():
        lines.append(f"- `{key}`: `{value}`.")

    lines.extend(["", "## Innovation Patterns", ""])
    lines.append("| Category | Top-venue routine | Configured runs | Decision rule |")
    lines.append("|---|---|---|---|")
    for row in report["pattern_rows"]:
        lines.append(
            "| {category} | {routine} | {runs} | {rule} |".format(
                category=row["category"],
                routine=row["topvenue_routine"],
                runs=", ".join(f"`{run}`" for run in row["configured_runs"]),
                rule=row["decision_rule"],
            )
        )

    lines.extend(["", "## Source-To-Project Map", ""])
    lines.append("| Source | Role | Status | Project mapping | Evidence needed |")
    lines.append("|---|---|---|---|---|")
    for row in report["source_rows"]:
        lines.append(
            "| {name} | {role} | `{status}` | {mapping} | {needed} |".format(
                name=f"[{row['name']}]({row['url']})",
                role=row["venue_or_role"],
                status=row["current_status"],
                mapping=row["project_mapping"],
                needed=row["evidence_needed"],
            )
        )

    lines.extend(["", "## Next Decisions", ""])
    for row in report["next_decisions"]:
        lines.append(f"- `{row['decision']}` Trigger: {row['trigger']} Action: {row['action']}")

    lines.extend(["", "## Strict Claim Rules", ""])
    for item in report["strict_claim_rules"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _group_complete(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(row.get("status") == "complete" for row in rows)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
