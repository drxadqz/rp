param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_residual_adapter_after_queue.log"

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
        Write-QueueLog "waiting for classification jobs: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Run-FastCandidate([string]$Name, [string[]]$ExtraArgs) {
    $OutDir = Join-Path $Root $Name
    $Stdout = Join-Path $LogDir "$Name.stdout.log"
    $Stderr = Join-Path $LogDir "$Name.stderr.log"
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
        Write-QueueLog "$Name already has evaluate_test.json; skipping."
        return
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
        "--use-texture-residual-adapter",
        "--texture-residual-scale", "0.25",
        "--early-stop-patience", "4",
        "--log-every-steps", "150"
    ) + $ExtraArgs
    Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Write-QueueLog "$Name python exit code $($proc.ExitCode)"
    if (Test-Path $Stdout) { Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if (Test-Path $Stderr) { Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
    if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }
    Write-QueueLog "finished $Name"
}

Write-QueueLog "started residual-adapter RSCD fast queue"
Wait-ProcessIds -Ids $WaitPids -Name "existing RSCD queue"
Wait-ClassificationJobsGone

Run-FastCandidate `
    -Name "fast_physics_texture_residual_adapter" `
    -ExtraArgs @()

Run-FastCandidate `
    -Name "fast_physics_directional_residual_adapter" `
    -ExtraArgs @("--use-directional-texture-branch", "--directional-texture-dim", "64", "--use-texture-gate", "--hierarchical-smoothing", "0.08")

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
Write-QueueLog "residual-adapter candidates complete"
