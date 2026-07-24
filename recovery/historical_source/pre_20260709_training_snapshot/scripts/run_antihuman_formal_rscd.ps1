param()

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$OutDir = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\formal_physics_antihuman_texture_patch_stats_b12e20"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Stdout = Join-Path $LogDir "formal_physics_antihuman_texture_patch_stats_b12e20.stdout.log"
$Stderr = Join-Path $LogDir "formal_physics_antihuman_texture_patch_stats_b12e20.stderr.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
    Write-Host "formal AntiHumanTexture result already exists; skipping."
    exit 0
}

& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $OutDir `
    --backbone convnext_tiny `
    --embedding-dim 768 `
    --pretrained `
    --epochs 20 `
    --image-size 192 `
    --batch-size 12 `
    --grad-accum-steps 2 `
    --samples-per-epoch 36000 `
    --num-workers 0 `
    --prefetch-factor 2 `
    --early-stop-patience 5 `
    --log-every-steps 100 `
    --use-physics-branch `
    --physics-quality-cues `
    --no-physics-quality-region-cues `
    --physics-dim 96 `
    --use-anti-human-texture-branch `
    --anti-human-texture-dim 64 `
    > $Stdout 2> $Stderr

$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    "formal AntiHumanTexture run failed with exit code $ExitCode" | Out-File -FilePath $Stderr -Encoding utf8 -Append
    exit $ExitCode
}

& $Python scripts\compare_rscd_surface_candidates.py *> (Join-Path $LogDir "formal_physics_antihuman_compare.log")
& $Python scripts\compare_rscd_class_slices.py *>> (Join-Path $LogDir "formal_physics_antihuman_compare.log")
& $Python scripts\select_final_rscd_method.py *>> (Join-Path $LogDir "formal_physics_antihuman_compare.log")
