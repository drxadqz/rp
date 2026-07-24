$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "formal_promoted_candidate_after_fast.log"
$Needed = @(
    "fast_physics_directional_texture_quality",
    "fast_physics_texture_hier_smoothing",
    "fast_physics_directional_hier_smoothing",
    "fast_physics_directional_gated_hier_smoothing"
)

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

Write-QueueLog "started formal promotion watcher"
while ($true) {
    $missing = @()
    foreach ($name in $Needed) {
        $path = Join-Path (Join-Path $Root $name) "evaluate_test.json"
        if (-not (Test-Path $path)) {
            $missing += $name
        }
    }
    if ($missing.Count -eq 0) {
        break
    }
    Write-QueueLog "waiting for fast candidates: $($missing -join ', ')"
    Start-Sleep -Seconds 180
}

Write-QueueLog "all fast candidates present; selecting promotion"
$decisionText = & $Python scripts\select_rscd_promotion_candidate.py
Write-QueueLog "decision: $decisionText"
$decision = $decisionText | ConvertFrom-Json
if ($null -eq $decision.promoted) {
    Write-QueueLog "no candidate promoted"
    exit 0
}

$outDir = [string]$decision.promoted.formal_output_dir
if (Test-Path (Join-Path $outDir "evaluate_test.json")) {
    Write-QueueLog "formal result already exists for $outDir; skipping"
    exit 0
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$extra = @()
foreach ($arg in $decision.promoted.formal_args) {
    $extra += [string]$arg
}

Write-QueueLog "launching formal promoted candidate $($decision.promoted.name)"
& $Python scripts\run_rscd_surface_classification.py `
    --output-dir $outDir `
    --epochs 20 `
    --image-size 192 `
    --batch-size 12 `
    --grad-accum-steps 2 `
    --samples-per-epoch 36000 `
    --num-workers 2 `
    --prefetch-factor 2 `
    --early-stop-patience 5 `
    --log-every-steps 150 `
    @extra *>> $Log

Write-QueueLog "formal promoted candidate complete"
& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_formal_result_summary.py *>> $Log
& $Python scripts\write_rscd_training_trend_report.py *>> $Log
