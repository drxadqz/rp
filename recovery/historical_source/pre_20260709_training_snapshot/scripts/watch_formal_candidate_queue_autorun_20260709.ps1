param(
    [int]$PollSeconds = 300,
    [int]$MaxHours = 96,
    [int]$GpuMemoryFreeThresholdMb = 1200,
    [switch]$DryRun,
    [switch]$CheckOnce,
    [switch]$AllowFormalCandidateLaunch
)

$ErrorActionPreference = "Stop"

$Root = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Postprocess = Join-Path $Root "scripts\postprocess_c3_formal_candidate_20260709.ps1"
$QueueAudit = Join-Path $Root "scripts\audit_formal_candidate_queue.py"
$PostprocessWatcher = Join-Path $Root "scripts\watch_and_postprocess_formal_candidate_20260709.ps1"
$LogDir = Join-Path $Root "reports\paper_protocol_summary\formal_candidate_queue_autorun_20260709"
$Log = Join-Path $LogDir "queue_autorun.log"
$Deadline = (Get-Date).AddHours($MaxHours)

$Candidates = @(
    [pscustomobject]@{
        Id = "S7"
        Name = "Formal Full-Manifest S7"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s7_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s7_20260709"
        CanLaunch = $false
    },
    [pscustomobject]@{
        Id = "S22"
        Name = "Formal Full-Manifest S22 RoadCortexNet self-designed backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_s22_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s22_road_cortex_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s22_road_cortex_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S23"
        Name = "Formal Full-Manifest S23 RoadCortexFormer factor-attention backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_former_s23_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s23_road_cortex_former_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s23_road_cortex_former_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S24"
        Name = "Formal Full-Manifest S24 RoadCortexGraph diffusion backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_graph_s24_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s24_road_cortex_graph_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s24_road_cortex_graph_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S25"
        Name = "Formal Full-Manifest S25 RoadCortexTensor coupling backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_tensor_s25_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s25_road_cortex_tensor_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s25_road_cortex_tensor_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S26"
        Name = "Formal Full-Manifest S26 RoadCortexTensorFormer factor-token backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_tensor_former_s26_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s26_road_cortex_tensor_former_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s26_road_cortex_tensor_former_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S27"
        Name = "Formal Full-Manifest S27 RoadCortexRetina bio-optical backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_retina_s27_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s27_road_cortex_retina_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s27_road_cortex_retina_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S28"
        Name = "Formal Full-Manifest S28 RoadCortexMeanField factor-graph backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_mean_field_s28_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s28_road_cortex_mean_field_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s28_road_cortex_mean_field_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S29"
        Name = "Formal Full-Manifest S29 RoadCortexScattering texture backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_scattering_s29_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s29_road_cortex_scattering_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s29_road_cortex_scattering_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S30"
        Name = "Formal Full-Manifest S30 RoadCortexReactionDiffusion PDE backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_reaction_diffusion_s30_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s30_road_cortex_reaction_diffusion_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s30_road_cortex_reaction_diffusion_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S31"
        Name = "Formal Full-Manifest S31 RoadCortexStateSpace control backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_state_space_s31_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s31_road_cortex_state_space_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s31_road_cortex_state_space_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S32"
        Name = "Formal Full-Manifest S32 RoadCortexMorphology persistence backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_morphology_s32_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s32_road_cortex_morphology_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s32_road_cortex_morphology_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S33"
        Name = "Formal Full-Manifest S33 RoadCortexFactorCascade backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_factor_cascade_s33_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s33_road_cortex_factor_cascade_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s33_road_cortex_factor_cascade_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S34"
        Name = "Formal Full-Manifest S34 RoadCortexInvariantSubspace backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_invariant_subspace_s34_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s34_road_cortex_invariant_subspace_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s34_road_cortex_invariant_subspace_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S35"
        Name = "Formal Full-Manifest S35 RoadCortexPrecisionFusion backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_precision_fusion_s35_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s35_road_cortex_precision_fusion_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s35_road_cortex_precision_fusion_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S36"
        Name = "Formal Full-Manifest S36 RoadCortexSpectralFactorTransformer backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_spectral_factor_transformer_s36_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s36_road_cortex_spectral_factor_transformer_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s36_road_cortex_spectral_factor_transformer_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S37"
        Name = "Formal Full-Manifest S37 RoadCortexWaveletPhase backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_wavelet_phase_s37_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s37_road_cortex_wavelet_phase_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s37_road_cortex_wavelet_phase_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S38"
        Name = "Formal Full-Manifest S38 RoadCortexHierarchicalBelief backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_hierarchical_belief_s38_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s38_road_cortex_hierarchical_belief_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s38_road_cortex_hierarchical_belief_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S39"
        Name = "Formal Full-Manifest S39 RoadCortexFactorTransport backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_factor_transport_s39_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s39_road_cortex_factor_transport_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s39_road_cortex_factor_transport_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S40"
        Name = "Formal Full-Manifest S40 RoadCortexCounterfactual backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_counterfactual_s40_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s40_road_cortex_counterfactual_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s40_road_cortex_counterfactual_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S41"
        Name = "Formal Full-Manifest S41 RoadCortexStructureTensor backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_structure_tensor_s41_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s41_road_cortex_structure_tensor_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s41_road_cortex_structure_tensor_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S42"
        Name = "Formal Full-Manifest S42 RoadCortexSteerableOrientation backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_steerable_orientation_s42_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s42_road_cortex_steerable_orientation_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s42_road_cortex_steerable_orientation_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S43"
        Name = "Formal Full-Manifest S43 RoadCortexInformationSynergy backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_information_synergy_s43_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s43_road_cortex_information_synergy_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s43_road_cortex_information_synergy_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S44"
        Name = "Formal Full-Manifest S44 RoadCortexPredictiveCoding backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_predictive_coding_s44_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s44_road_cortex_predictive_coding_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s44_road_cortex_predictive_coding_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S45"
        Name = "Formal Full-Manifest S45 RoadCortexPhaseField backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_phase_field_s45_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s45_road_cortex_phase_field_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s45_road_cortex_phase_field_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S46"
        Name = "Formal Full-Manifest S46 RoadCortexFactorLatticeTransformer backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_factor_lattice_transformer_s46_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s46_road_cortex_factor_lattice_transformer_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s46_road_cortex_factor_lattice_transformer_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S47"
        Name = "Formal Full-Manifest S47 RoadCortexHodgeHelmholtz backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_hodge_helmholtz_s47_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s47_road_cortex_hodge_helmholtz_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s47_road_cortex_hodge_helmholtz_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S48"
        Name = "Formal Full-Manifest S48 RoadCortexRenormalizationFlow backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_renormalization_flow_s48_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s48_road_cortex_renormalization_flow_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s48_road_cortex_renormalization_flow_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S49"
        Name = "Formal Full-Manifest S49 RoadCortexWettingCapillarity backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_wetting_capillarity_s49_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s49_road_cortex_wetting_capillarity_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s49_road_cortex_wetting_capillarity_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S50"
        Name = "Formal Full-Manifest S50 RoadCortexFisherRao backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_fisher_rao_s50_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s50_road_cortex_fisher_rao_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s50_road_cortex_fisher_rao_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S51"
        Name = "Formal Full-Manifest S51 RoadCortexMobiusCoupling backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_mobius_coupling_s51_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s51_road_cortex_mobius_coupling_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s51_road_cortex_mobius_coupling_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S52"
        Name = "Formal Full-Manifest S52 RoadCortexEvidenceTheory backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_evidence_theory_s52_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s52_road_cortex_evidence_theory_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s52_road_cortex_evidence_theory_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S53"
        Name = "Formal Full-Manifest S53 RoadCortexSparsePrototype backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_sparse_prototype_s53_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s53_road_cortex_sparse_prototype_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s53_road_cortex_sparse_prototype_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S54"
        Name = "Formal Full-Manifest S54 RoadCortexTriFactorTransformer backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_tri_factor_transformer_s54_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s54_road_cortex_tri_factor_transformer_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s54_road_cortex_tri_factor_transformer_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S55"
        Name = "Formal Full-Manifest S55 RoadCortexMechanismGraph backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_mechanism_graph_s55_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s55_road_cortex_mechanism_graph_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s55_road_cortex_mechanism_graph_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S56"
        Name = "Formal Full-Manifest S56 RoadCortexOrdinalPotential backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_ordinal_potential_s56_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s56_road_cortex_ordinal_potential_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s56_road_cortex_ordinal_potential_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S57"
        Name = "Formal Full-Manifest S57 RoadCortexDiscriminantAxis backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_discriminant_axis_s57_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s57_road_cortex_discriminant_axis_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s57_road_cortex_discriminant_axis_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S58"
        Name = "Formal Full-Manifest S58 RoadCortexEvidenceConservation backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_evidence_conservation_s58_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s58_road_cortex_evidence_conservation_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s58_road_cortex_evidence_conservation_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S59"
        Name = "Formal Full-Manifest S59 RoadCortexDirectionalCanonical backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_directional_canonical_s59_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s59_road_cortex_directional_canonical_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s59_road_cortex_directional_canonical_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S60"
        Name = "Formal Full-Manifest S60 RoadCortexFactorInteractionTransformer backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_factor_interaction_transformer_s60_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s60_road_cortex_factor_interaction_transformer_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s60_road_cortex_factor_interaction_transformer_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S61"
        Name = "Formal Full-Manifest S61 RoadCortexSequentialFactorRefinement backbone"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_road_cortex_sequential_factor_refinement_s61_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s61_road_cortex_sequential_factor_refinement_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s61_road_cortex_sequential_factor_refinement_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S12"
        Name = "Formal Full-Manifest S12 factor-graph pair sampler"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_factor_graph_metric_pair_sampler_s12_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s12_factor_graph_pair_sampler_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s12_factor_graph_pair_sampler_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S13"
        Name = "Formal Full-Manifest S13 early family-router tensor coupling"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_family_router_tensor_coupling_s13_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s13_family_router_tensor_coupling_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s13_family_router_tensor_coupling_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S14"
        Name = "Formal Full-Manifest S14 family-router PCGrad no-harm"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_family_router_pcgrad_noharm_s14_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s14_family_router_pcgrad_noharm_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s14_family_router_pcgrad_noharm_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S15"
        Name = "Formal Full-Manifest S15 concrete-film split router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_concrete_film_split_router_s15_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s15_concrete_film_split_router_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s15_concrete_film_split_router_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S16"
        Name = "Formal Full-Manifest S16 concrete-film split free-energy router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_concrete_film_split_free_energy_s16_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s16_concrete_film_split_free_energy_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s16_concrete_film_split_free_energy_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S17"
        Name = "Formal Full-Manifest S17 shape-topology concrete-film router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_shape_topology_concrete_film_s17_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s17_shape_topology_concrete_film_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s17_shape_topology_concrete_film_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S18"
        Name = "Formal Full-Manifest S18 opponent-Retinex concrete-film router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_opponent_retinex_concrete_film_s18_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s18_opponent_retinex_concrete_film_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s18_opponent_retinex_concrete_film_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S19"
        Name = "Formal Full-Manifest S19 precision-observer concrete-film router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_precision_observer_concrete_film_s19_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s19_precision_observer_concrete_film_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s19_precision_observer_concrete_film_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S20"
        Name = "Formal Full-Manifest S20 morphology-scale concrete-film router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_morphology_scale_concrete_film_s20_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s20_morphology_scale_concrete_film_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s20_morphology_scale_concrete_film_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S21"
        Name = "Formal Full-Manifest S21 spectral-scattering concrete-film router"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_spectral_scattering_concrete_film_s21_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s21_spectral_scattering_concrete_film_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s21_spectral_scattering_concrete_film_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S11"
        Name = "Formal Full-Manifest S11 factor-graph metric"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_factor_graph_metric_s11_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s11_factor_graph_metric_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s11_factor_graph_metric_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S8"
        Name = "Formal Full-Manifest S8 WCS incoming"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s8_wcs_incoming_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s8_wcs_incoming_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s8_wcs_incoming_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S9"
        Name = "Formal Full-Manifest S9 dry-concrete ordinal + WCS"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s9_dryconcrete_ordinal_wcs_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s9_dryconcrete_ordinal_wcs_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s9_dryconcrete_ordinal_wcs_20260709"
        CanLaunch = $true
    },
    [pscustomobject]@{
        Id = "S10"
        Name = "Formal Full-Manifest S10 graph-control stable"
        RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s10_graph_control_stable_20260709"
        Script = "scripts\run_c3_formal_fullmanifest_s10_graph_control_stable_20260709.ps1"
        PostprocessDir = "reports\paper_protocol_summary\postprocess_s10_graph_control_stable_20260709"
        CanLaunch = $true
    }
)

$PriorityIds = @(
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
    "S10"
)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Test-RunComplete {
    param([string]$Dir)
    return (
        (Test-Path -LiteralPath (Join-Path $Dir "metrics.json")) -and
        (Test-Path -LiteralPath (Join-Path $Dir "per_class_metrics.csv")) -and
        (Test-Path -LiteralPath (Join-Path $Dir "confusion_matrix.csv"))
    )
}

function Test-RunHasStarted {
    param([string]$Dir)
    if (-not (Test-Path -LiteralPath $Dir)) {
        return $false
    }
    return (
        (Test-Path -LiteralPath (Join-Path $Dir "history.json")) -or
        (Test-Path -LiteralPath (Join-Path $Dir "best_checkpoint.pth")) -or
        ((Get-ChildItem -LiteralPath $Dir -Filter "train_stderr_*.log" -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
    )
}

function Get-PromotionStatusPath {
    param([object]$Candidate)
    return (Join-Path (Join-Path $Root $Candidate.PostprocessDir) "promotion_gate\promotion_gate_status.json")
}

function Get-PromotionDecision {
    param([object]$Candidate)
    $statusPath = Get-PromotionStatusPath -Candidate $Candidate
    if (-not (Test-Path -LiteralPath $statusPath)) {
        return $null
    }
    $data = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    return $data.decision
}

function Get-ActiveTrainingProcessCount {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        ($_.CommandLine -like '*train.py --config*' -or $_.CommandLine -like '*train.py*--config*')
    }
    return @($procs).Count
}

function Get-GpuMemoryUsedMb {
    try {
        $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return $null
        }
        return [int](([string]$raw).Trim().Split("`n")[0].Trim())
    }
    catch {
        return $null
    }
}

function Invoke-QueueAudit {
    Push-Location $Root
    try {
        & $Python $QueueAudit | Out-Null
    }
    finally {
        Pop-Location
    }
}

function Invoke-PostprocessIfNeeded {
    param([object]$Candidate)
    $statusPath = Get-PromotionStatusPath -Candidate $Candidate
    if (Test-Path -LiteralPath $statusPath) {
        Write-QueueLog "postprocess_present id=$($Candidate.Id) status=$statusPath"
        return
    }
    if ($DryRun) {
        Write-QueueLog "dry_run_would_postprocess id=$($Candidate.Id) run_dir=$($Candidate.RunDir)"
        return
    }
    Write-QueueLog "postprocess_start id=$($Candidate.Id)"
    Push-Location $Root
    try {
        powershell -NoProfile -ExecutionPolicy Bypass -File $Postprocess `
            -RunDir $Candidate.RunDir `
            -RunName $Candidate.Name `
            -OutDir $Candidate.PostprocessDir
    }
    finally {
        Pop-Location
    }
    Write-QueueLog "postprocess_done id=$($Candidate.Id)"
}

function Start-Candidate {
    param([object]$Candidate)
    $scriptAbs = Join-Path $Root $Candidate.Script
    if ($DryRun) {
        Write-QueueLog "dry_run_would_launch id=$($Candidate.Id) script=$scriptAbs"
        return
    }
    Write-QueueLog "launch_start id=$($Candidate.Id) script=$scriptAbs"
    $proc = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $scriptAbs) `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Hidden
    Write-QueueLog "launch_pid id=$($Candidate.Id) pid=$($proc.Id)"

    $watchProc = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PostprocessWatcher,
            "-RunDir", $Candidate.RunDir,
            "-RunName", $Candidate.Name,
            "-OutDir", $Candidate.PostprocessDir,
            "-PollSeconds", "300",
            "-MaxHours", "48"
        ) `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Hidden
    Write-QueueLog "launch_postprocess_watcher id=$($Candidate.Id) pid=$($watchProc.Id)"
}

function Process-QueueOnce {
    Invoke-QueueAudit
    $active = Get-ActiveTrainingProcessCount
    $gpuUsed = Get-GpuMemoryUsedMb
    Write-QueueLog "poll active_train=$active gpu_used_mb=$gpuUsed"

    $orderedCandidates = foreach ($id in $PriorityIds) {
        $Candidates | Where-Object { $_.Id -eq $id }
    }

    foreach ($candidate in $orderedCandidates) {
        $complete = Test-RunComplete -Dir $candidate.RunDir
        $started = Test-RunHasStarted -Dir $candidate.RunDir
        if ($complete) {
            Write-QueueLog "candidate_complete id=$($candidate.Id)"
            Invoke-PostprocessIfNeeded -Candidate $candidate
            $decision = Get-PromotionDecision -Candidate $candidate
            Write-QueueLog "candidate_decision id=$($candidate.Id) decision=$decision"
            if ($decision -eq "sota_candidate_run_exact_pass_audit") {
                Write-QueueLog "sota_gate_passed_stop id=$($candidate.Id)"
                return "stop"
            }
            continue
        }

        if ($started) {
            if ($active -gt 0) {
                Write-QueueLog "candidate_running_or_incomplete_wait id=$($candidate.Id) active_train=$active"
                return "wait"
            }
            Write-QueueLog "candidate_incomplete_without_active_training_needs_inspection id=$($candidate.Id)"
            return "wait"
        }

        if (-not $candidate.CanLaunch) {
            Write-QueueLog "candidate_not_started_manual_only id=$($candidate.Id)"
            return "wait"
        }
        if (-not $AllowFormalCandidateLaunch) {
            Write-QueueLog "candidate_not_started_screening_required id=$($candidate.Id)"
            return "wait"
        }

        if ($active -gt 0) {
            Write-QueueLog "launch_deferred_active_train id=$($candidate.Id) active_train=$active"
            return "wait"
        }
        if ($gpuUsed -ne $null -and $gpuUsed -gt (4096 - $GpuMemoryFreeThresholdMb)) {
            Write-QueueLog "launch_deferred_gpu id=$($candidate.Id) gpu_used_mb=$gpuUsed free_threshold_mb=$GpuMemoryFreeThresholdMb"
            return "wait"
        }
        Start-Candidate -Candidate $candidate
        return "wait"
    }

    Write-QueueLog "queue_exhausted_design_new_mechanism"
    return "stop"
}

Write-QueueLog "queue_autorun_start dry_run=$DryRun check_once=$CheckOnce"

while ((Get-Date) -lt $Deadline) {
    $result = Process-QueueOnce
    if ($result -eq "stop") {
        Write-QueueLog "queue_autorun_stop"
        exit 0
    }
    if ($CheckOnce) {
        Write-QueueLog "queue_autorun_check_once_exit"
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-QueueLog "queue_autorun_timeout"
exit 2
