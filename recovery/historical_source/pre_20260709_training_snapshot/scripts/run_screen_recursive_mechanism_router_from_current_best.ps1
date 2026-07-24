$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$Anchor = Join-Path $Root "screen_dry_concrete_vor_residual_scale012_lr1e3_s8k_from_anchor\best.pt"
$OutDir = Join-Path $Root "screen_recursive_mechanism_router_s4k_cap120_from_current_best_20260703"

& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $OutDir `
    --backbone convnext_tiny_recursive_mechanism_router `
    --embedding-dim 768 `
    --image-size 192 `
    --eval-resize-mode letterbox `
    --train-resize-mode letterbox `
    --train-augmentation `
    --batch-size 8 `
    --grad-accum-steps 2 `
    --epochs 1 `
    --lr 0.00008 `
    --weight-decay 0.005 `
    --dropout 0.2 `
    --use-physics-branch `
    --physics-dim 96 `
    --physics-quality-cues `
    --no-physics-quality-region-cues `
    --use-semantic-physics-attention-branch `
    --semantic-physics-attention-dim 64 `
    --use-local-physics-field-branch `
    --local-physics-field-dim 64 `
    --local-physics-field-scale 0.08 `
    --use-dry-concrete-roughness-vor-residual `
    --dry-concrete-roughness-scale 0.12 `
    --dry-concrete-roughness-gate-threshold 0.10 `
    --dry-concrete-roughness-gate-temperature 14.0 `
    --train-only-module-prefix backbone.recursive_mechanism_banks `
    --train-only-module-prefix norm `
    --online-teacher-checkpoint $Anchor `
    --online-teacher-weight 0.15 `
    --online-teacher-temperature 2.0 `
    --online-teacher-beta 4.0 `
    --online-teacher-min-confidence 0.60 `
    --relation-conditional-weight 0.001 `
    --relation-conditional-focus core `
    --relation-conditional-friction-weight 1.0 `
    --relation-conditional-material-weight 0.8 `
    --relation-conditional-unevenness-weight 1.1 `
    --relation-conditional-uncertainty-margin 0.35 `
    --relation-conditional-gate-temperature 12.0 `
    --directed-confusion-weight 0.0008 `
    --directed-confusion-margin 0.08 `
    --directed-confusion-preset rscd_smooth_gap_v3 `
    --samples-per-epoch 4000 `
    --max-val-samples-per-class 120 `
    --max-test-samples-per-class 120 `
    --checkpoint-selection-metric top1_macro_hardslice_guard `
    --checkpoint-selection-hard-slice-classes water_concrete_slight,water_concrete_severe,wet_concrete_slight,wet_concrete_severe,wet_asphalt_severe,water_asphalt_severe `
    --checkpoint-selection-macro-tolerance 0.003 `
    --checkpoint-selection-hard-slice-tolerance 0.006 `
    --early-stop-patience 2 `
    --num-workers 2 `
    --prefetch-factor 2 `
    --amp `
    --log-every-steps 80 `
    --save-predictions `
    --resume-from $Anchor
