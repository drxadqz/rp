param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_retinex_texture_after_queue.log"
$FormalRun = Join-Path $Root "formal_physics_wavelet_directional_film_gate_hier"
$TtaRun = Join-Path $Root "tta_ensemble_physics_texture_formal_hflip"
$MaterialRun = Join-Path $Root "fast_physics_material_gate_patch_quality"
$MaterialLog = Join-Path $LogDir "fast_material_conditioned_gate_after_formal.log"

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
        Start-Sleep -Seconds 300
    }
}

function Wait-ClassificationJobsGone([string]$Reason) {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs before ${Reason}: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-File([string]$Path, [string]$Name) {
    while (-not (Test-Path $Path)) {
        Write-QueueLog "waiting for ${Name}: $Path"
        Start-Sleep -Seconds 300
    }
}

function Wait-MaterialGateDone() {
    while ($true) {
        if (Test-Path (Join-Path $MaterialRun "evaluate_test.json")) {
            Write-QueueLog "material-conditioned gate result found"
            break
        }
        if ((Test-Path $MaterialLog) -and ((Get-Content -LiteralPath $MaterialLog -Tail 40) -match "python exit code")) {
            Write-QueueLog "material-conditioned gate watcher reached an exit-code line; continuing after queue clears"
            break
        }
        Write-QueueLog "waiting for material-conditioned gate stage"
        Start-Sleep -Seconds 300
    }
}

function Invoke-FastCandidate([string]$Name, [string[]]$ExtraArgs) {
    Wait-ClassificationJobsGone -Reason $Name
    $OutDir = Join-Path $Root $Name
    $Stdout = Join-Path $LogDir "$Name.stdout.log"
    $Stderr = Join-Path $LogDir "$Name.stderr.log"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
        Write-QueueLog "$Name already has evaluate_test.json; skipping."
        return
    }

    Write-QueueLog "launching $Name"
    $baseArgs = @(
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
        "--num-workers", "0",
        "--prefetch-factor", "2",
        "--early-stop-patience", "4",
        "--log-every-steps", "150"
    )
    $args = $baseArgs + $ExtraArgs
    Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Write-QueueLog "$Name python exit code $($proc.ExitCode)"
    if (Test-Path $Stdout) { Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $Stderr) { Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { throw "$Name failed with exit code $($proc.ExitCode)" }
}

Write-QueueLog "started Retinex texture fast watcher"
Wait-ProcessIds -Ids $WaitPids -Name "upstream process"
Wait-File -Path (Join-Path $FormalRun "evaluate_test.json") -Name "promoted formal result"
Wait-File -Path (Join-Path $TtaRun "evaluate_test.json") -Name "TTA ensemble result"
Wait-MaterialGateDone

Invoke-FastCandidate -Name "fast_physics_retinex_texture_quality" -ExtraArgs @(
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96",
    "--use-retinex-texture-branch",
    "--no-retinex-region-cues",
    "--retinex-texture-dim", "48"
)

Invoke-FastCandidate -Name "fast_physics_retinex_film_gate_hier" -ExtraArgs @(
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96",
    "--use-retinex-texture-branch",
    "--no-retinex-region-cues",
    "--retinex-texture-dim", "48",
    "--use-texture-gate",
    "--use-texture-film",
    "--texture-film-scale", "0.15",
    "--hierarchical-smoothing", "0.08"
)

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log
Write-QueueLog "Retinex texture fast candidates complete"
