$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$OutDir = Join-Path $Root "screen_convnextv2_tiny_fcmae22k_physics_s8k_e2_cap120_20260703"

& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $OutDir `
    --backbone timm:convnextv2_tiny.fcmae_ft_in22k_in1k `
    --embedding-dim 768 `
    --image-size 192 `
    --eval-resize-mode letterbox `
    --train-resize-mode letterbox `
    --train-augmentation `
    --batch-size 8 `
    --grad-accum-steps 2 `
    --epochs 2 `
    --lr 0.0001 `
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
    --relation-conditional-weight 0.001 `
    --relation-conditional-focus core `
    --relation-conditional-friction-weight 1.0 `
    --relation-conditional-material-weight 0.8 `
    --relation-conditional-unevenness-weight 1.1 `
    --relation-conditional-uncertainty-margin 0.35 `
    --relation-conditional-gate-temperature 12.0 `
    --samples-per-epoch 8000 `
    --max-val-samples-per-class 120 `
    --max-test-samples-per-class 120 `
    --checkpoint-selection-metric macro_f1 `
    --early-stop-patience 2 `
    --num-workers 2 `
    --prefetch-factor 2 `
    --amp `
    --log-every-steps 120 `
    --save-predictions
