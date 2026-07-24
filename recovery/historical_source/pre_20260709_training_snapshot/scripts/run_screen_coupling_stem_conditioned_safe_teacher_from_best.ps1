$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$Anchor = Join-Path $Root "screen_local_physics_hflip_relation_cond_w0005_core_s8k_from_hflip\best.pt"
$OutDir = Join-Path $Root "screen_coupling_stem_conditioned_safe_teacher_s8k_e2_from_best_20260702"

& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $OutDir `
    --backbone convnext_tiny_coupling_stem_conditioned_safe `
    --embedding-dim 768 `
    --image-size 192 `
    --eval-resize-mode letterbox `
    --train-resize-mode letterbox `
    --train-augmentation `
    --batch-size 6 `
    --grad-accum-steps 3 `
    --epochs 2 `
    --lr 0.00012 `
    --weight-decay 0.01 `
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
    --train-only-module-prefix backbone.coupling_stem `
    --online-teacher-checkpoint $Anchor `
    --online-teacher-weight 0.8 `
    --online-teacher-temperature 2.0 `
    --online-teacher-beta 4.0 `
    --online-teacher-min-confidence 0.45 `
    --teacher-error-replay-weight 0.15 `
    --teacher-error-replay-focus concrete_wetwater_rough_v1 `
    --teacher-error-replay-beta 0.5 `
    --teacher-error-replay-min-confidence 0.35 `
    --hflip-consistency-weight 0.002 `
    --relation-conditional-weight 0.005 `
    --relation-conditional-focus core `
    --relation-conditional-friction-weight 1.0 `
    --relation-conditional-material-weight 0.6 `
    --relation-conditional-unevenness-weight 1.2 `
    --relation-conditional-uncertainty-margin 0.35 `
    --relation-conditional-gate-temperature 12.0 `
    --samples-per-epoch 8000 `
    --max-val-samples-per-class 300 `
    --max-test-samples-per-class 300 `
    --checkpoint-selection-metric top1_macro_hardslice_guard `
    --checkpoint-selection-hard-slice-classes water_concrete_slight,water_concrete_severe,wet_concrete_slight,wet_concrete_severe,wet_asphalt_severe,water_asphalt_severe `
    --checkpoint-selection-macro-tolerance 0.003 `
    --checkpoint-selection-hard-slice-tolerance 0.006 `
    --early-stop-patience 3 `
    --num-workers 2 `
    --prefetch-factor 2 `
    --amp `
    --log-every-steps 100 `
    --save-predictions `
    --resume-from $Anchor
