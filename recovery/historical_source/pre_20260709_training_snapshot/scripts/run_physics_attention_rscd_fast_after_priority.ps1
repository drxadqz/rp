param(
    [int[]]$WaitPids = @(9000, 30484)
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "fast_physics_attention_after_priority.log"

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
        Write-QueueLog "waiting for classification jobs: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 180
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
    Wait-ClassificationJobsGone
    Write-QueueLog "launching $Name"
    $args = @(
        "scripts\run_rscd_surface_classification.py",
        "--output-dir", $OutDir,
        "--epochs", "3",
        "--image-size", "160",
        "--batch-size", "8",
        "--grad-accum-steps", "3",
        "--samples-per-epoch", "5400",
        "--max-train-samples-per-class", "300",
        "--max-val-samples-per-class", "120",
        "--max-test-samples-per-class", "180",
        "--num-workers", "0",
        "--prefetch-factor", "2",
        "--use-physics-branch",
        "--physics-quality-cues",
        "--no-physics-quality-region-cues",
        "--physics-dim", "96",
        "--use-physics-attention-branch",
        "--physics-attention-dim", "32",
        "--use-texture-film",
        "--texture-film-scale", "0.20",
        "--early-stop-patience", "4",
        "--log-every-steps", "80"
    ) + $ExtraArgs
    Remove-Item -LiteralPath $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    $proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Write-QueueLog "$Name python exit code $($proc.ExitCode)"
    if (Test-Path $Stdout) {
        Get-Content -LiteralPath $Stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append
    }
    if (Test-Path $Stderr) {
        Get-Content -LiteralPath $Stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append
    }
    if ($proc.ExitCode -ne 0) {
        Write-QueueLog "$Name failed with exit code $($proc.ExitCode)"
        exit $proc.ExitCode
    }
    Write-QueueLog "finished $Name"
}

Write-QueueLog "started PhysicsAttention RSCD fast queue"
Wait-ProcessIds -Ids $WaitPids -Name "high-priority texture queue"

Run-FastCandidate `
    -Name "fast_physics_attention_film" `
    -ExtraArgs @()

Run-FastCandidate `
    -Name "fast_physics_attention_wavelet_film_gate_hier" `
    -ExtraArgs @("--use-wavelet-texture-branch", "--wavelet-texture-dim", "48", "--use-texture-gate", "--hierarchical-smoothing", "0.08")

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\select_rscd_texture_film_promotion.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log
Write-QueueLog "PhysicsAttention candidates complete"
