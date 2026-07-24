param(
    [int]$HardQueuePid = 31568
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "formal_hard_condition_candidate_after_fast.log"
$Needed = @(
    "fast_physics_texture_hard_condition_boost035",
    "fast_physics_texture_hard_condition_hier_boost035"
)

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-QueueLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

function Wait-ProcessGone([int]$PidValue, [string]$Name) {
    while ($true) {
        $p = Get-Process -Id $PidValue -ErrorAction SilentlyContinue
        if ($null -eq $p) {
            break
        }
        Write-QueueLog "waiting for ${Name}: $PidValue"
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
        Write-QueueLog "waiting for classification jobs before formal hard-condition launch: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 180
    }
}

Write-QueueLog "started hard-condition formal promotion watcher"
Wait-ProcessGone -PidValue $HardQueuePid -Name "hard-condition fast queue"

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
    Write-QueueLog "waiting for hard-condition fast candidates: $($missing -join ', ')"
    Start-Sleep -Seconds 180
}

Write-QueueLog "all hard-condition fast candidates present; selecting promotion"
$decisionText = & $Python scripts\select_rscd_hard_condition_promotion.py
Write-QueueLog "decision: $decisionText"
$decision = $decisionText | ConvertFrom-Json
if ($null -eq $decision.promoted) {
    Write-QueueLog "no hard-condition candidate promoted"
    exit 0
}

Wait-ClassificationJobsGone

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

Write-QueueLog "launching formal hard-condition candidate $($decision.promoted.name)"
$stdout = Join-Path $LogDir "formal_hard_condition_candidate.stdout.log"
$stderr = Join-Path $LogDir "formal_hard_condition_candidate.stderr.log"
$args = @(
    "scripts\run_rscd_surface_classification.py",
    "--output-dir", $outDir,
    "--epochs", "20",
    "--image-size", "192",
    "--batch-size", "12",
    "--grad-accum-steps", "2",
    "--samples-per-epoch", "36000",
    "--num-workers", "0",
    "--prefetch-factor", "2",
    "--early-stop-patience", "5",
    "--log-every-steps", "150"
) + $extra
Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Write-QueueLog "formal hard-condition python exit code $($proc.ExitCode)"
if (Test-Path $stdout) { Get-Content -LiteralPath $stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

Write-QueueLog "formal hard-condition candidate complete"
& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_formal_result_summary.py *>> $Log
& $Python scripts\write_rscd_training_trend_report.py *>> $Log
