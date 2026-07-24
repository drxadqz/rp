from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("reports/paper_protocol_summary/formal_candidate_queue_20260709")
SOTA_CSV = Path("reports/paper_protocol_summary/rscd_literature_sota_protocol_audit_20260703.csv")

PROGRESS_RE = re.compile(
    r"(?P<stage>eval|train):\s+(?P<pct>\d+)%\|.*?\|\s+"
    r"(?P<done>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>[^<\]]+)(?:<(?P<eta>[^,\]]+))?"
)


CANDIDATES: list[dict[str, str]] = [
    {
        "id": "S7",
        "name": "Formal Full-Manifest S7",
        "role": "current formal full-manifest baseline from S7 anchor route",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s7_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s7_20260709",
    },
    {
        "id": "S22",
        "name": "Formal Full-Manifest S22 RoadCortexNet self-designed backbone",
        "role": "first-principles self-designed RSCD backbone with material/film/roughness/coupling streams",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_s22_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s22_road_cortex_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_s22_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s22_road_cortex_20260709",
    },
    {
        "id": "S23",
        "name": "Formal Full-Manifest S23 RoadCortexFormer factor-attention backbone",
        "role": "self-designed Transformer route where RSCD factor tokens query local road texture and physics evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_former_s23_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s23_road_cortex_former_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_former_s23_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s23_road_cortex_former_20260709",
    },
    {
        "id": "S24",
        "name": "Formal Full-Manifest S24 RoadCortexGraph diffusion backbone",
        "role": "self-designed graph-diffusion route with evidence-guided material/film/roughness edge propagation",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_graph_s24_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s24_road_cortex_graph_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_graph_s24_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s24_road_cortex_graph_20260709",
    },
    {
        "id": "S25",
        "name": "Formal Full-Manifest S25 RoadCortexTensor coupling backbone",
        "role": "self-designed tensor-coupling route with explicit friction/material/roughness pair and triple terms",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_tensor_s25_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s25_road_cortex_tensor_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_tensor_s25_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s25_road_cortex_tensor_20260709",
    },
    {
        "id": "S26",
        "name": "Formal Full-Manifest S26 RoadCortexTensorFormer factor-token backbone",
        "role": "self-designed Transformer route where RSCD factor tokens control tensor-coupling branches",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_tensor_former_s26_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s26_road_cortex_tensor_former_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_tensor_former_s26_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s26_road_cortex_tensor_former_20260709",
    },
    {
        "id": "S27",
        "name": "Formal Full-Manifest S27 RoadCortexRetina bio-optical backbone",
        "role": "self-designed bio-optical route with center-surround, opponent, film, roughness, and lateral-inhibition mechanisms",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_retina_s27_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s27_road_cortex_retina_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_retina_s27_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s27_road_cortex_retina_20260709",
    },
    {
        "id": "S28",
        "name": "Formal Full-Manifest S28 RoadCortexMeanField factor-graph backbone",
        "role": "self-designed statistical-physics route with local mean-field inference over RSCD factors",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_mean_field_s28_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s28_road_cortex_mean_field_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_mean_field_s28_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s28_road_cortex_mean_field_20260709",
    },
    {
        "id": "S29",
        "name": "Formal Full-Manifest S29 RoadCortexScattering texture backbone",
        "role": "self-designed multi-scale directional scattering route for RSCD patch texture and roughness",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_scattering_s29_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s29_road_cortex_scattering_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_scattering_s29_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s29_road_cortex_scattering_20260709",
    },
    {
        "id": "S30",
        "name": "Formal Full-Manifest S30 RoadCortexReactionDiffusion PDE backbone",
        "role": "self-designed reaction-diffusion route for wet-film smoothing, roughness activation, and diffusion barriers",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_reaction_diffusion_s30_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s30_road_cortex_reaction_diffusion_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_reaction_diffusion_s30_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s30_road_cortex_reaction_diffusion_20260709",
    },
    {
        "id": "S31",
        "name": "Formal Full-Manifest S31 RoadCortexStateSpace control backbone",
        "role": "self-designed control-inspired 2D state-space observer route for RSCD texture fields",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_state_space_s31_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s31_road_cortex_state_space_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_state_space_s31_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s31_road_cortex_state_space_20260709",
    },
    {
        "id": "S32",
        "name": "Formal Full-Manifest S32 RoadCortexMorphology persistence backbone",
        "role": "self-designed morphology-persistence route for roughness, wet-film erasure, and marking suppression",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_morphology_s32_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s32_road_cortex_morphology_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_morphology_s32_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s32_road_cortex_morphology_20260709",
    },
    {
        "id": "S33",
        "name": "Formal Full-Manifest S33 RoadCortexFactorCascade backbone",
        "role": "self-designed causal factor-cascade route: material first, friction conditioned on material, roughness conditioned on both",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_factor_cascade_s33_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s33_road_cortex_factor_cascade_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_factor_cascade_s33_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s33_road_cortex_factor_cascade_20260709",
    },
    {
        "id": "S34",
        "name": "Formal Full-Manifest S34 RoadCortexInvariantSubspace backbone",
        "role": "self-designed evidence-guided invariant subspace route for task factors and nuisance suppression",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_invariant_subspace_s34_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s34_road_cortex_invariant_subspace_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_invariant_subspace_s34_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s34_road_cortex_invariant_subspace_20260709",
    },
    {
        "id": "S35",
        "name": "Formal Full-Manifest S35 RoadCortexPrecisionFusion backbone",
        "role": "self-designed Bayesian precision-fusion route for conflict-aware material/wet-film/roughness evidence updates",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_precision_fusion_s35_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s35_road_cortex_precision_fusion_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_precision_fusion_s35_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s35_road_cortex_precision_fusion_20260709",
    },
    {
        "id": "S36",
        "name": "Formal Full-Manifest S36 RoadCortexSpectralFactorTransformer backbone",
        "role": "self-designed Transformer route with evidence-biased factor tokens and spectral texture paths",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_spectral_factor_transformer_s36_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s36_road_cortex_spectral_factor_transformer_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_spectral_factor_transformer_s36_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s36_road_cortex_spectral_factor_transformer_20260709",
    },
    {
        "id": "S37",
        "name": "Formal Full-Manifest S37 RoadCortexWaveletPhase backbone",
        "role": "self-designed RSPNet-inspired but original wavelet-phase backbone preserving road texture bands",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_wavelet_phase_s37_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s37_road_cortex_wavelet_phase_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_wavelet_phase_s37_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s37_road_cortex_wavelet_phase_20260709",
    },
    {
        "id": "S38",
        "name": "Formal Full-Manifest S38 RoadCortexHierarchicalBelief backbone",
        "role": "self-designed progressive material-friction-roughness belief mixer with wavelet-frequency experts",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_hierarchical_belief_s38_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s38_road_cortex_hierarchical_belief_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_hierarchical_belief_s38_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s38_road_cortex_hierarchical_belief_20260709",
    },
    {
        "id": "S39",
        "name": "Formal Full-Manifest S39 RoadCortexFactorTransport backbone",
        "role": "self-designed evidence-constrained factor transport route for local RSCD factor assignment",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_factor_transport_s39_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s39_road_cortex_factor_transport_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_factor_transport_s39_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s39_road_cortex_factor_transport_20260709",
    },
    {
        "id": "S40",
        "name": "Formal Full-Manifest S40 RoadCortexCounterfactual backbone",
        "role": "self-designed causal counterfactual route for material/wet/water/roughness factor effects",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_counterfactual_s40_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s40_road_cortex_counterfactual_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_counterfactual_s40_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s40_road_cortex_counterfactual_20260709",
    },
    {
        "id": "S41",
        "name": "Formal Full-Manifest S41 RoadCortexStructureTensor backbone",
        "role": "self-designed structure-tensor geometry route for directed markings, isotropic roughness, and wet-film smoothness",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_structure_tensor_s41_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s41_road_cortex_structure_tensor_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_structure_tensor_s41_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s41_road_cortex_structure_tensor_20260709",
    },
    {
        "id": "S42",
        "name": "Formal Full-Manifest S42 RoadCortexSteerableOrientation backbone",
        "role": "self-designed steerable orientation route for rotation-stable texture and direction-sensitive nuisance",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_steerable_orientation_s42_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s42_road_cortex_steerable_orientation_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_steerable_orientation_s42_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s42_road_cortex_steerable_orientation_20260709",
    },
    {
        "id": "S43",
        "name": "Formal Full-Manifest S43 RoadCortexInformationSynergy backbone",
        "role": "self-designed information-decomposition route for unique/shared/synergistic RSCD factor evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_information_synergy_s43_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s43_road_cortex_information_synergy_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_information_synergy_s43_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s43_road_cortex_information_synergy_20260709",
    },
    {
        "id": "S44",
        "name": "Formal Full-Manifest S44 RoadCortexPredictiveCoding backbone",
        "role": "self-designed predictive-coding route for clean-road expectations and RSCD factor prediction errors",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_predictive_coding_s44_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s44_road_cortex_predictive_coding_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_predictive_coding_s44_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s44_road_cortex_predictive_coding_20260709",
    },
    {
        "id": "S45",
        "name": "Formal Full-Manifest S45 RoadCortexPhaseField backbone",
        "role": "self-designed phase-field energy route for dry/wet-film/water/roughness local phase coupling",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_phase_field_s45_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s45_road_cortex_phase_field_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_phase_field_s45_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s45_road_cortex_phase_field_20260709",
    },
    {
        "id": "S46",
        "name": "Formal Full-Manifest S46 RoadCortexFactorLatticeTransformer backbone",
        "role": "self-designed Transformer route with RSCD factor/coupling tokens and pairwise tournament comparison",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_factor_lattice_transformer_s46_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s46_road_cortex_factor_lattice_transformer_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_factor_lattice_transformer_s46_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s46_road_cortex_factor_lattice_transformer_20260709",
    },
    {
        "id": "S47",
        "name": "Formal Full-Manifest S47 RoadCortexHodgeHelmholtz backbone",
        "role": "self-designed discrete-Hodge route separating road texture, wet smoothness, and directional nuisance",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_hodge_helmholtz_s47_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s47_road_cortex_hodge_helmholtz_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_hodge_helmholtz_s47_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s47_road_cortex_hodge_helmholtz_20260709",
    },
    {
        "id": "S48",
        "name": "Formal Full-Manifest S48 RoadCortexRenormalizationFlow backbone",
        "role": "self-designed scale-renormalization route for micro/meso/macro RSCD factor evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_renormalization_flow_s48_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s48_road_cortex_renormalization_flow_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_renormalization_flow_s48_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s48_road_cortex_renormalization_flow_20260709",
    },
    {
        "id": "S49",
        "name": "Formal Full-Manifest S49 RoadCortexWettingCapillarity backbone",
        "role": "self-designed wetting-physics route for Wenzel/Cassie/pooling/material-roughness coupling",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_wetting_capillarity_s49_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s49_road_cortex_wetting_capillarity_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_wetting_capillarity_s49_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s49_road_cortex_wetting_capillarity_20260709",
    },
    {
        "id": "S50",
        "name": "Formal Full-Manifest S50 RoadCortexFisherRao backbone",
        "role": "self-designed information-geometric route over friction/material/roughness probability-simplex fields",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_fisher_rao_s50_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s50_road_cortex_fisher_rao_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_fisher_rao_s50_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s50_road_cortex_fisher_rao_20260709",
    },
    {
        "id": "S51",
        "name": "Formal Full-Manifest S51 RoadCortexMobiusCoupling backbone",
        "role": "self-designed algebraic Mobius route for singleton, pairwise, and triple RSCD factor couplings",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_mobius_coupling_s51_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s51_road_cortex_mobius_coupling_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_mobius_coupling_s51_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s51_road_cortex_mobius_coupling_20260709",
    },
    {
        "id": "S52",
        "name": "Formal Full-Manifest S52 RoadCortexEvidenceTheory backbone",
        "role": "self-designed evidence-theory route for factor agreement, conflict, uncertainty, and coupling",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_evidence_theory_s52_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s52_road_cortex_evidence_theory_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_evidence_theory_s52_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s52_road_cortex_evidence_theory_20260709",
    },
    {
        "id": "S53",
        "name": "Formal Full-Manifest S53 RoadCortexSparsePrototype backbone",
        "role": "self-designed sparse physical-prototype dictionary route for RSCD factor and coupling evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_sparse_prototype_s53_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s53_road_cortex_sparse_prototype_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_sparse_prototype_s53_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s53_road_cortex_sparse_prototype_20260709",
    },
    {
        "id": "S54",
        "name": "Formal Full-Manifest S54 RoadCortexTriFactorTransformer backbone",
        "role": "self-designed factor-token Transformer route for RSCD single-factor, pairwise, and full coupling evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_tri_factor_transformer_s54_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s54_road_cortex_tri_factor_transformer_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_tri_factor_transformer_s54_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s54_road_cortex_tri_factor_transformer_20260709",
    },
    {
        "id": "S55",
        "name": "Formal Full-Manifest S55 RoadCortexMechanismGraph backbone",
        "role": "self-designed early mechanism-graph route for dry/film/water/material/roughness and RSCD coupling experts",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_mechanism_graph_s55_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s55_road_cortex_mechanism_graph_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_mechanism_graph_s55_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s55_road_cortex_mechanism_graph_20260709",
    },
    {
        "id": "S56",
        "name": "Formal Full-Manifest S56 RoadCortexOrdinalPotential backbone",
        "role": "self-designed ordinal threshold and coupled-energy route for liquid, roughness, material and RSCD boundary classes",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_ordinal_potential_s56_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s56_road_cortex_ordinal_potential_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_ordinal_potential_s56_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s56_road_cortex_ordinal_potential_20260709",
    },
    {
        "id": "S57",
        "name": "Formal Full-Manifest S57 RoadCortexDiscriminantAxis backbone",
        "role": "self-designed local discriminant-axis route for hard adjacent RSCD factor and coupling boundaries",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_discriminant_axis_s57_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s57_road_cortex_discriminant_axis_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_discriminant_axis_s57_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s57_road_cortex_discriminant_axis_20260709",
    },
    {
        "id": "S58",
        "name": "Formal Full-Manifest S58 RoadCortexEvidenceConservation backbone",
        "role": "self-designed evidence-conservation route with material/liquid/roughness/coupling explanation and residual correction",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_evidence_conservation_s58_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s58_road_cortex_evidence_conservation_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_evidence_conservation_s58_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s58_road_cortex_evidence_conservation_20260709",
    },
    {
        "id": "S59",
        "name": "Formal Full-Manifest S59 RoadCortexDirectionalCanonical backbone",
        "role": "self-designed direction-canonical texture route for curved or oblique RSCD patches and orientation-stable material/liquid/roughness evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_directional_canonical_s59_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s59_road_cortex_directional_canonical_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_directional_canonical_s59_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s59_road_cortex_directional_canonical_20260709",
    },
    {
        "id": "S60",
        "name": "Formal Full-Manifest S60 RoadCortexFactorInteractionTransformer backbone",
        "role": "self-designed factor-region Transformer with RSCD friction/material/roughness/coupling tokens and early feature feedback",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_factor_interaction_transformer_s60_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s60_road_cortex_factor_interaction_transformer_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_factor_interaction_transformer_s60_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s60_road_cortex_factor_interaction_transformer_20260709",
    },
    {
        "id": "S61",
        "name": "Formal Full-Manifest S61 RoadCortexSequentialFactorRefinement backbone",
        "role": "self-designed progressive material-to-liquid-to-roughness factor refinement with residual coupling for RSCD composite labels",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_road_cortex_sequential_factor_refinement_s61_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s61_road_cortex_sequential_factor_refinement_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_road_cortex_sequential_factor_refinement_s61_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s61_road_cortex_sequential_factor_refinement_20260709",
    },
    {
        "id": "S12",
        "name": "Formal Full-Manifest S12 factor-graph pair sampler",
        "role": "graph-neighbor batch construction for one-axis RSCD factor contrasts, focused on wet/water concrete boundaries",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_factor_graph_metric_pair_sampler_s12_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s12_factor_graph_pair_sampler_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_factor_graph_metric_pair_sampler_s12_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s12_factor_graph_pair_sampler_20260709",
    },
    {
        "id": "S13",
        "name": "Formal Full-Manifest S13 early family-router tensor coupling",
        "role": "early ConvNeXt mechanism conditioning with family-separated tensor routes over RSCD friction/material/roughness couplings",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_family_router_tensor_coupling_s13_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s13_family_router_tensor_coupling_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_family_router_tensor_coupling_s13_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s13_family_router_tensor_coupling_20260709",
    },
    {
        "id": "S14",
        "name": "Formal Full-Manifest S14 family-router PCGrad no-harm",
        "role": "S13 early family-router mechanism with RSCD-specific no-harm PCGrad protection for stable teacher-correct classes",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_family_router_pcgrad_noharm_s14_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s14_family_router_pcgrad_noharm_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_family_router_pcgrad_noharm_s14_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s14_family_router_pcgrad_noharm_20260709",
    },
    {
        "id": "S15",
        "name": "Formal Full-Manifest S15 concrete-film split router",
        "role": "early wet/water-concrete mechanism split into smooth-film, hidden-slight, and hidden-severe routes",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_concrete_film_split_router_s15_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s15_concrete_film_split_router_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_concrete_film_split_router_s15_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s15_concrete_film_split_router_20260709",
    },
    {
        "id": "S16",
        "name": "Formal Full-Manifest S16 concrete-film split free-energy router",
        "role": "S15 split mechanisms plus entropy/margin free-energy routing to prevent diffuse expert leakage",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_concrete_film_split_free_energy_s16_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s16_concrete_film_split_free_energy_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_concrete_film_split_free_energy_s16_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s16_concrete_film_split_free_energy_20260709",
    },
    {
        "id": "S17",
        "name": "Formal Full-Manifest S17 shape-topology concrete-film router",
        "role": "S16 free-energy split with early direction/topology evidence for hidden wet-water concrete roughness",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_shape_topology_concrete_film_s17_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s17_shape_topology_concrete_film_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_shape_topology_concrete_film_s17_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s17_shape_topology_concrete_film_20260709",
    },
    {
        "id": "S18",
        "name": "Formal Full-Manifest S18 opponent-Retinex concrete-film router",
        "role": "S17 shape/topology split plus early opponent-Retinex optical evidence for wet-water material ambiguity",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_opponent_retinex_concrete_film_s18_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s18_opponent_retinex_concrete_film_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_opponent_retinex_concrete_film_s18_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s18_opponent_retinex_concrete_film_20260709",
    },
    {
        "id": "S19",
        "name": "Formal Full-Manifest S19 precision-observer concrete-film router",
        "role": "S18 optical route with Kalman-style early precision gating for contradictory RSCD evidence",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_precision_observer_concrete_film_s19_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s19_precision_observer_concrete_film_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_precision_observer_concrete_film_s19_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s19_precision_observer_concrete_film_20260709",
    },
    {
        "id": "S20",
        "name": "Formal Full-Manifest S20 morphology-scale concrete-film router",
        "role": "S19 precision route plus mathematical morphology scale-space evidence for roughness boundaries",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_morphology_scale_concrete_film_s20_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s20_morphology_scale_concrete_film_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_morphology_scale_concrete_film_s20_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s20_morphology_scale_concrete_film_20260709",
    },
    {
        "id": "S21",
        "name": "Formal Full-Manifest S21 spectral-scattering concrete-film router",
        "role": "S20 morphology route plus early high/mid/low-frequency evidence for wet-film texture suppression",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_spectral_scattering_concrete_film_s21_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s21_spectral_scattering_concrete_film_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_spectral_scattering_concrete_film_s21_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s21_spectral_scattering_concrete_film_20260709",
    },
    {
        "id": "S11",
        "name": "Formal Full-Manifest S11 factor-graph metric",
        "role": "factor-graph metric representation shaping for friction/material/roughness/coupling",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_factor_graph_metric_s11_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s11_factor_graph_metric_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_factor_graph_metric_s11_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s11_factor_graph_metric_20260709",
    },
    {
        "id": "S8",
        "name": "Formal Full-Manifest S8 WCS incoming",
        "role": "weakest-class water_concrete_slight repair",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_source_reliable_router_s8_wcs_incoming_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s8_wcs_incoming_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_source_reliable_router_s8_wcs_incoming_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s8_wcs_incoming_20260709",
    },
    {
        "id": "S9",
        "name": "Formal Full-Manifest S9 dry-concrete ordinal + WCS",
        "role": "high-volume dry-concrete slight/severe ordinal roughness repair",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_source_reliable_router_s9_dryconcrete_ordinal_wcs_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s9_dryconcrete_ordinal_wcs_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_source_reliable_router_s9_dryconcrete_ordinal_wcs_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s9_dryconcrete_ordinal_wcs_20260709",
    },
    {
        "id": "S10",
        "name": "Formal Full-Manifest S10 graph-control stable",
        "role": "graph/control-stable route set with reverse-confusion damping",
        "config": "configs/c3_farnet/c3_farnet_formal_fullmanifest_source_reliable_router_s10_graph_control_stable_20260709.yaml",
        "run_script": "scripts/run_c3_formal_fullmanifest_s10_graph_control_stable_20260709.ps1",
        "run_dir": "E:/perception_outputs/rscd_surface_classification/c3_farnet_formal_fullmanifest_source_reliable_router_s10_graph_control_stable_20260709",
        "postprocess_dir": "reports/paper_protocol_summary/postprocess_s10_graph_control_stable_20260709",
    },
]

PRIORITY_ORDER = [
    "S7",
    "S37",
    "S38",
    "S39",
    "S40",
    "S41",
    "S42",
    "S43",
    "S44",
    "S45",
    "S46",
    "S47",
    "S48",
    "S49",
    "S50",
    "S51",
    "S52",
    "S53",
    "S54",
    "S55",
    "S56",
    "S57",
    "S58",
    "S59",
    "S60",
    "S61",
    "S35",
    "S36",
    "S25",
    "S33",
    "S34",
    "S26",
    "S27",
    "S28",
    "S29",
    "S30",
    "S31",
    "S32",
    "S22",
    "S23",
    "S24",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S11",
    "S8",
    "S9",
    "S10",
]


def priority_sort(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {candidate_id: idx for idx, candidate_id in enumerate(PRIORITY_ORDER)}
    return sorted(rows, key=lambda row: rank.get(str(row.get("id", "")), 10_000))


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_sota_thresholds(path: Path) -> dict[str, dict[str, Any]]:
    best = {
        "top1": {"value": None, "method": ""},
        "macro_f1": {"value": None, "method": ""},
    }
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            method = str(row.get("method", ""))
            if "Current" in method or "Ours" in method or "FAF" in method:
                continue
            fair_note = str(row.get("use_for_fair_claim", "")).lower()
            if "not comparable" in fair_note or "extra data" in fair_note or "expanded" in fair_note:
                continue
            top1 = parse_float(row.get("top1_%"))
            macro = parse_float(row.get("mean_f1_%") or row.get("macro_f1_%"))
            if top1 is not None and (best["top1"]["value"] is None or top1 > float(best["top1"]["value"])):
                best["top1"] = {"value": top1, "method": method}
            if macro is not None and (
                best["macro_f1"]["value"] is None or macro > float(best["macro_f1"]["value"])
            ):
                best["macro_f1"] = {"value": macro, "method": method}
    return best


def last_progress(run_dir: Path) -> dict[str, Any] | None:
    stderr_files = sorted(run_dir.glob("train_stderr_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not stderr_files:
        return None
    text = stderr_files[0].read_text(encoding="utf-8", errors="ignore")
    matches = list(PROGRESS_RE.finditer(text.replace("\r", "\n")))
    if not matches:
        return None
    match = matches[-1]
    done = int(match.group("done"))
    total = int(match.group("total"))
    return {
        "stage": match.group("stage"),
        "pct": int(match.group("pct")),
        "done": done,
        "total": total,
        "fraction": done / total if total else 0.0,
        "elapsed": match.group("elapsed"),
        "eta": match.group("eta") or "",
        "stderr": str(stderr_files[0]),
    }


def log_has_failure(run_dir: Path) -> str:
    stderr_files = sorted(run_dir.glob("train_stderr_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    stdout_files = sorted(run_dir.glob("train_stdout_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in stderr_files[:2] + stdout_files[:2]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in ["Traceback", "RuntimeError", "CUDA out of memory", "nan", "NaN"]:
            if marker in text:
                return f"{marker} in {path}"
    return ""


def candidate_status(candidate: dict[str, str], sota: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_dir = Path(candidate["run_dir"])
    config = Path(candidate["config"])
    script = Path(candidate["run_script"])
    metrics = read_json(run_dir / "metrics.json")
    history = read_json(run_dir / "history.json")
    per_class_exists = (run_dir / "per_class_metrics.csv").exists()
    confusion_exists = (run_dir / "confusion_matrix.csv").exists()
    failure = log_has_failure(run_dir) if run_dir.exists() else ""
    progress = last_progress(run_dir) if run_dir.exists() else None

    status = "not_started"
    if run_dir.exists() and failure:
        status = "failed_or_needs_inspection"
    elif metrics and per_class_exists:
        status = "complete"
    elif run_dir.exists() and (history is not None or progress is not None):
        status = "running_or_incomplete"
    elif run_dir.exists():
        status = "created_no_metrics"

    summary = metrics.get("summary", {}) if metrics else {}
    top1 = float(summary["top1"]) * 100.0 if "top1" in summary else None
    macro = float(summary["macro_f1"]) * 100.0 if "macro_f1" in summary else None
    top1_gate = sota["top1"]["value"]
    macro_gate = sota["macro_f1"]["value"]
    top1_gap = None if top1 is None or top1_gate is None else top1 - float(top1_gate)
    macro_gap = None if macro is None or macro_gate is None else macro - float(macro_gate)

    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "role": candidate["role"],
        "status": status,
        "run_dir": str(run_dir),
        "config_exists": config.exists(),
        "run_script_exists": script.exists(),
        "metrics_exists": metrics is not None,
        "per_class_exists": per_class_exists,
        "confusion_exists": confusion_exists,
        "history_exists": history is not None,
        "top1_%": top1,
        "macro_f1_%": macro,
        "top1_gap_vs_external_pp": top1_gap,
        "macro_f1_gap_vs_external_pp": macro_gap,
        "num_samples": summary.get("num_samples"),
        "num_errors": summary.get("num_errors"),
        "progress": progress,
        "failure_marker": failure,
        "postprocess_command": (
            "powershell -ExecutionPolicy Bypass -File scripts\\postprocess_c3_formal_candidate_20260709.ps1 "
            f"-RunDir '{run_dir}' -RunName '{candidate['name']}' -OutDir '{candidate['postprocess_dir']}'"
        ),
        "launch_command": (
            "powershell -ExecutionPolicy Bypass -File "
            f"{candidate['run_script']}"
        ),
    }


def recommend(statuses: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    running = [s for s in statuses if s["status"] == "running_or_incomplete" and s["progress"] is not None]
    if running:
        active = running[0]
        actions.append(
            f"Wait for {active['id']} to finish; do not launch another GPU run while it is at "
            f"{active['progress']['done']}/{active['progress']['total']}."
        )
        actions.append(f"When {active['id']} finishes, run: `{active['postprocess_command']}`")
        return actions

    complete = [s for s in statuses if s["status"] == "complete"]
    if complete:
        best = max(complete, key=lambda s: (s["macro_f1_%"] or -1.0, s["top1_%"] or -1.0))
        if (best["top1_gap_vs_external_pp"] or -999.0) >= 0.0 and (
            best["macro_f1_gap_vs_external_pp"] or -999.0
        ) >= 0.0:
            actions.append(f"{best['id']} passes both external SOTA gates; freeze and run exact-pass audit.")
            return actions
        actions.append(
            f"Best complete candidate is {best['id']} but it has not passed both SOTA gates; continue queue."
        )

    for next_id in PRIORITY_ORDER:
        if next_id == "S7":
            continue
        item = next(s for s in statuses if s["id"] == next_id)
        if item["status"] in {"not_started", "created_no_metrics"}:
            actions.append(f"Next GPU candidate after current jobs: `{item['launch_command']}`")
            actions.append(f"After it finishes, run: `{item['postprocess_command']}`")
            return actions
    actions.append("All prepared candidates have run or need inspection; design a new early/mid backbone mechanism.")
    return actions


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    sota = read_sota_thresholds(SOTA_CSV)
    statuses = priority_sort([candidate_status(candidate, sota) for candidate in CANDIDATES])
    actions = recommend(statuses)

    data = {
        "sota_thresholds": sota,
        "candidates": statuses,
        "recommended_actions": actions,
    }
    (SUMMARY_DIR / "formal_candidate_queue_status.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table = [
        "| id | status | Top-1 | Macro-F1 | Top-1 gap | Macro-F1 gap | progress | role |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in statuses:
        progress = ""
        if row["progress"]:
            progress = f"{row['progress']['stage']} {row['progress']['done']}/{row['progress']['total']}"
        table.append(
            "| {id} | {status} | {top1} | {macro} | {topgap} | {macrogap} | {progress} | {role} |".format(
                id=row["id"],
                status=row["status"],
                top1=fmt(row["top1_%"]),
                macro=fmt(row["macro_f1_%"]),
                topgap=fmt(row["top1_gap_vs_external_pp"]),
                macrogap=fmt(row["macro_f1_gap_vs_external_pp"]),
                progress=progress,
                role=row["role"],
            )
        )

    md = [
        "# Formal Candidate Queue Status",
        "",
        "## External Gates",
        "",
        f"- Top-1: {fmt(sota['top1']['value'])}% ({sota['top1']['method']})",
        f"- Macro/Mean-F1: {fmt(sota['macro_f1']['value'])}% ({sota['macro_f1']['method']})",
        "",
        "## Queue",
        "",
        *table,
        "",
        "## Recommended Actions",
        "",
        *[f"- {action}" for action in actions],
        "",
        "## Notes",
        "",
        "- Do not compare candidates unless `metrics.json`, `per_class_metrics.csv`, and `confusion_matrix.csv` exist on the full test set.",
        "- Use the postprocess command for every completed run before deciding whether to promote it.",
        "- S8/S9/S10/S11/S12/S13/S14/S15/S16/S17/S18/S19/S20/S21/S22/S23/S24/S25/S26/S27/S28/S29/S30/S31/S32/S33/S34/S35/S36/S37/S38/S39/S40/S41/S42/S43/S44/S45/S46/S47/S48/S49/S50/S51/S52/S53/S54/S55/S56/S57/S58/S59/S60/S61 are queued candidates, not final claims.",
    ]
    (SUMMARY_DIR / "formal_candidate_queue_status.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(SUMMARY_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
