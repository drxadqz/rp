param(
    [int[]]$WaitPids = @(26608, 10532, 11312, 16888, 18688, 31568, 31476)
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\direct_visual_friction"
$LogDir = Join-Path $Repo "outputs\direct_visual_friction_queue"
$Log = Join-Path $LogDir "direct_visual_friction_after_rscd_queue.log"
$GlobalCfg = "configs\experiments\direct_visual_friction\extreme_road_global_convnext_fast.yaml"
$FafCfg = "configs\experiments\direct_visual_friction\extreme_road_quality_physics_fast.yaml"
$GlobalDir = Join-Path $Root "extreme_road_global_convnext_fast"
$FafDir = Join-Path $Root "extreme_road_quality_physics_fast"
$PairDir = Join-Path $Root "fair_pairwise"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $PairDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

function Wait-ProcessIds([int[]]$Ids, [string]$Name) {
    while ($true) {
        $alive = @()
        foreach ($pidValue in $Ids) {
            $p = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($null -ne $p) {
                $alive += $pidValue
            }
        }
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for ${Name}: $($alive -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-ClassificationJobsGone() {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for remaining RSCD classification jobs: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Select-Checkpoint([string]$OutDir) {
    $bestSafety = Join-Path $OutDir "best_safety.pt"
    $best = Join-Path $OutDir "best.pt"
    if (Test-Path $bestSafety) {
        return $bestSafety
    }
    if (Test-Path $best) {
        return $best
    }
    throw "No checkpoint found in $OutDir"
}

function Run-TrainIfNeeded([string]$ConfigPath, [string]$OutDir, [string]$Name) {
    if ((Test-Path (Join-Path $OutDir "best.pt")) -or (Test-Path (Join-Path $OutDir "best_safety.pt"))) {
        Write-QueueLog "$Name checkpoint exists; skipping train."
        return
    }
    Write-QueueLog "training $Name"
    & $Python scripts\train.py --config $ConfigPath *>> $Log
    Write-QueueLog "finished training $Name"
}

function Run-EvalBundle([string]$ConfigPath, [string]$OutDir, [string]$Name) {
    $ckpt = Select-Checkpoint $OutDir
    Write-QueueLog "$Name checkpoint: $ckpt"
    if (-not (Test-Path (Join-Path $OutDir "evaluate_test.json"))) {
        & $Python scripts\evaluate.py --config $ConfigPath --checkpoint $ckpt --split test *>> $Log
    }
    if (-not (Test-Path (Join-Path $OutDir "interval_calibration_90.json"))) {
        & $Python scripts\calibrate_intervals.py --config $ConfigPath --checkpoint $ckpt --target-coverage 0.90 --num-workers 0 --out (Join-Path $OutDir "interval_calibration_90.json") *>> $Log
    }
    if (-not (Test-Path (Join-Path $OutDir "bootstrap_metrics_best_safety.json"))) {
        & $Python scripts\bootstrap_metrics.py --config $ConfigPath --checkpoint $ckpt --num-bootstrap 300 --num-workers 0 --out-json (Join-Path $OutDir "bootstrap_metrics_best_safety.json") --out-md (Join-Path $OutDir "bootstrap_metrics_best_safety.md") *>> $Log
    }
}

Write-QueueLog "started direct visual friction watcher"
Wait-ProcessIds -Ids $WaitPids -Name "RSCD queue pids"
Wait-ClassificationJobsGone

Run-TrainIfNeeded -ConfigPath $GlobalCfg -OutDir $GlobalDir -Name "extreme_road_global_convnext_fast"
Run-EvalBundle -ConfigPath $GlobalCfg -OutDir $GlobalDir -Name "extreme_road_global_convnext_fast"

Run-TrainIfNeeded -ConfigPath $FafCfg -OutDir $FafDir -Name "extreme_road_quality_physics_fast"
Run-EvalBundle -ConfigPath $FafCfg -OutDir $FafDir -Name "extreme_road_quality_physics_fast"

$globalCkpt = Select-Checkpoint $GlobalDir
$fafCkpt = Select-Checkpoint $FafDir
$pairJson = Join-Path $PairDir "extreme_road_faf_vs_global_convnext_paired_bootstrap.json"
$pairMd = Join-Path $PairDir "extreme_road_faf_vs_global_convnext_paired_bootstrap.md"
if (-not (Test-Path $pairJson)) {
    & $Python scripts\paired_model_bootstrap_compare.py `
        --config-a $GlobalCfg `
        --checkpoint-a $globalCkpt `
        --name-a global_convnext `
        --config-b $FafCfg `
        --checkpoint-b $fafCkpt `
        --name-b quality_physics_faf `
        --num-bootstrap 300 `
        --out-json $pairJson `
        --out-md $pairMd *>> $Log
}

& $Python scripts\write_direct_visual_friction_report.py *>> $Log
Write-QueueLog "direct visual friction queue complete"
