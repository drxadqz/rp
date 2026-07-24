param(
    [int[]]$FormalWaitPids = @(26608, 10532)
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "high_priority_texture_candidates_after_formal.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

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
        Start-Sleep -Seconds 180
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
        Write-QueueLog "waiting for classification jobs before priority launch: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 180
    }
}

function Run-FastCandidate([string]$Name, [string[]]$ExtraArgs) {
    $OutDir = Join-Path $Root $Name
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
        Write-QueueLog "$Name already has evaluate_test.json; skipping."
        return
    }
    Wait-ClassificationJobsGone
    Write-QueueLog "launching $Name"
    $args = @(
        "scripts\run_rscd_surface_classification.py",
        "--output-dir", $OutDir,
        "--epochs", "4",
        "--image-size", "192",
        "--batch-size", "12",
        "--grad-accum-steps", "2",
        "--samples-per-epoch", "10800",
        "--max-train-samples-per-class", "600",
        "--max-val-samples-per-class", "200",
        "--max-test-samples-per-class", "300",
        "--num-workers", "2",
        "--prefetch-factor", "2",
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim", "96",
        "--early-stop-patience", "4",
        "--log-every-steps", "150"
    ) + $ExtraArgs
    & $Python @args *>> $Log
    Write-QueueLog "finished $Name"
}

Write-QueueLog "started high-priority residual/FiLM/Wavelet candidate queue"
Wait-ProcessIds -Ids $FormalWaitPids -Name "formal RSCD seed jobs"

Run-FastCandidate `
    -Name "fast_physics_texture_residual_adapter" `
    -ExtraArgs @("--use-texture-residual-adapter", "--texture-residual-scale", "0.25")

Run-FastCandidate `
    -Name "fast_physics_directional_residual_adapter" `
    -ExtraArgs @("--use-directional-texture-branch", "--directional-texture-dim", "64", "--use-texture-gate", "--use-texture-residual-adapter", "--texture-residual-scale", "0.25", "--hierarchical-smoothing", "0.08")

Run-FastCandidate `
    -Name "fast_physics_texture_film" `
    -ExtraArgs @("--use-texture-film", "--texture-film-scale", "0.20")

Run-FastCandidate `
    -Name "fast_physics_directional_film_gate_hier" `
    -ExtraArgs @("--use-directional-texture-branch", "--directional-texture-dim", "64", "--use-texture-gate", "--use-texture-film", "--texture-film-scale", "0.20", "--hierarchical-smoothing", "0.08")

Run-FastCandidate `
    -Name "fast_physics_wavelet_film" `
    -ExtraArgs @("--use-wavelet-texture-branch", "--wavelet-texture-dim", "48", "--use-texture-film", "--texture-film-scale", "0.20")

Run-FastCandidate `
    -Name "fast_physics_wavelet_directional_film_gate_hier" `
    -ExtraArgs @("--use-wavelet-texture-branch", "--wavelet-texture-dim", "48", "--use-directional-texture-branch", "--directional-texture-dim", "48", "--use-texture-gate", "--use-texture-film", "--texture-film-scale", "0.20", "--hierarchical-smoothing", "0.08")

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\select_rscd_residual_adapter_promotion.py *>> $Log
& $Python scripts\select_rscd_texture_film_promotion.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log

Write-QueueLog "high-priority texture candidate queue complete"
