$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$Pretrain = Join-Path $Root "pretrain_road_mechanism_tiny_evidence_s2700_20260702\best.pt"
$OutDir = Join-Path $Root "screen_road_mechanism_tiny_evidence_preinit_s2700_20260702"

& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $OutDir `
    --backbone road_mechanism_tiny `
    --embedding-dim 768 `
    --image-size 192 `
    --eval-resize-mode letterbox `
    --train-resize-mode letterbox `
    --train-augmentation `
    --batch-size 8 `
    --grad-accum-steps 2 `
    --epochs 1 `
    --lr 0.0008 `
    --weight-decay 0.02 `
    --dropout 0.2 `
    --no-pretrained `
    --backbone-init-from $Pretrain `
    --use-physics-branch `
    --physics-dim 96 `
    --physics-quality-cues `
    --no-physics-quality-region-cues `
    --use-semantic-physics-attention-branch `
    --semantic-physics-attention-dim 64 `
    --use-local-physics-field-branch `
    --local-physics-field-dim 64 `
    --local-physics-field-scale 0.08 `
    --factor-aux-weight 0.10 `
    --hflip-consistency-weight 0.002 `
    --relation-conditional-weight 0.005 `
    --relation-conditional-focus core `
    --relation-conditional-friction-weight 1.0 `
    --relation-conditional-material-weight 0.6 `
    --relation-conditional-unevenness-weight 1.2 `
    --relation-conditional-uncertainty-margin 0.35 `
    --relation-conditional-gate-temperature 12.0 `
    --samples-per-epoch 2700 `
    --max-val-samples-per-class 50 `
    --max-test-samples-per-class 50 `
    --checkpoint-selection-metric macro_f1 `
    --early-stop-patience 2 `
    --num-workers 2 `
    --prefetch-factor 2 `
    --amp `
    --log-every-steps 50 `
    --save-predictions
