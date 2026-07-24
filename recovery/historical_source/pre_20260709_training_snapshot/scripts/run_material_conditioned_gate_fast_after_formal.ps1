param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_material_conditioned_gate_after_formal.log"
$FormalRun = Join-Path $Root "formal_physics_wavelet_directional_film_gate_hier"
$TtaRun = Join-Path $Root "tta_ensemble_physics_texture_formal_hflip"

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

function Wait-ClassificationJobsGone() {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs before material-conditioned gate: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-File([string]$Path, [string]$Name) {
    while (-not (Test-Path $Path)) {
        Write-QueueLog "waiting for ${Name}: $Path"
        Start-Sleep -Seconds 300
    }
}

Write-QueueLog "started material-conditioned gate fast watcher"
Wait-ProcessIds -Ids $WaitPids -Name "upstream process"
Wait-File -Path (Join-Path $FormalRun "evaluate_test.json") -Name "promoted formal result"
Wait-File -Path (Join-Path $TtaRun "evaluate_test.json") -Name "TTA ensemble result"
Wait-ClassificationJobsGone

$Name = "fast_physics_material_gate_patch_quality"
$OutDir = Join-Path $Root $Name
$Stdout = Join-Path $LogDir "$Name.stdout.log"
$Stderr = Join-Path $LogDir "$Name.stderr.log"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
    Write-QueueLog "$Name already has evaluate_test.json; skipping."
    exit 0
}

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
    "--num-workers", "0",
    "--prefetch-factor", "2",
    "--use-physics-branch",
    "--physics-quality-cues",
    "--no-physics-quality-region-cues",
    "--physics-dim", "96",
    "--use-material-conditioned-texture-gate",
    "--material-conditioned-gate-scale", "0.25",
    "--hierarchical-smoothing", "0.08",
    "--early-stop-patience", "4",
    "--log-every-steps", "150"
)
Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
Write-QueueLog "$Name python exit code $($proc.ExitCode)"
if (Test-Path $Stdout) { Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if (Test-Path $Stderr) { Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log
Write-QueueLog "material-conditioned gate fast candidate complete"
