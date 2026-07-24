$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$OutDir = Join-Path $Root "pretrain_road_mechanism_tiny_evidence_s2700_20260702"

& $Python scripts\pretrain_rscd_physics_evidence.py `
    --output-dir $OutDir `
    --backbone road_mechanism_tiny `
    --embedding-dim 768 `
    --image-size 192 `
    --epochs 1 `
    --batch-size 8 `
    --lr 0.0006 `
    --weight-decay 0.02 `
    --mask-ratio 0.08 `
    --max-train-samples-per-class 100 `
    --max-val-samples-per-class 30 `
    --num-workers 2 `
    --seed 79 `
    --no-pretrained `
    --amp `
    --log-every-steps 50
