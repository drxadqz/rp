param(
    [int[]]$WaitPids = @()
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$RunRoot = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "tta_ensemble_after_texture_wavelet.log"
$FormalTextureLog = Join-Path $LogDir "formal_texture_film_candidate_after_fast.log"

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
        Write-QueueLog "waiting for classification jobs before TTA ensemble: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Wait-TextureWaveletWatcherDone() {
    while ($true) {
        if (Test-Path $FormalTextureLog) {
            $text = Get-Content -LiteralPath $FormalTextureLog -Raw
            if ($text -match "no texture-FiLM/Wavelet candidate promoted" -or
                $text -match "formal texture-FiLM/Wavelet candidate complete" -or
                $text -match "formal result already exists") {
                break
            }
        }
        Write-QueueLog "waiting for texture-FiLM/Wavelet promotion watcher to finish"
        Start-Sleep -Seconds 300
    }
}

Write-QueueLog "started TTA/ensemble watcher"
Wait-ProcessIds -Ids $WaitPids -Name "upstream queue/watcher"
Wait-TextureWaveletWatcherDone
Wait-ClassificationJobsGone

$OutDir = Join-Path $RunRoot "tta_ensemble_physics_texture_formal_hflip"
if (Test-Path (Join-Path $OutDir "evaluate_test.json")) {
    Write-QueueLog "TTA ensemble result already exists; skipping"
    exit 0
}

$stdout = Join-Path $LogDir "tta_ensemble_physics_texture_formal_hflip.stdout.log"
$stderr = Join-Path $LogDir "tta_ensemble_physics_texture_formal_hflip.stderr.log"
$args = @(
    "scripts\evaluate_rscd_tta_ensemble.py",
    "--run-dirs",
    (Join-Path $RunRoot "formal_physics_texture_quality_b12e20_resume"),
    (Join-Path $RunRoot "formal_physics_texture_quality_b12e20_parallel"),
    "--output-dir", $OutDir,
    "--batch-size", "16",
    "--num-workers", "0",
    "--tta", "hflip"
)

Write-QueueLog "launching TTA ensemble evaluation"
Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $Python -ArgumentList $args -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Write-QueueLog "TTA ensemble python exit code $($proc.ExitCode)"
if (Test-Path $stdout) { Get-Content -LiteralPath $stdout -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if (Test-Path $stderr) { Get-Content -LiteralPath $stderr -Tail 80 | Out-File -FilePath $Log -Encoding utf8 -Append }
if ($proc.ExitCode -ne 0) { exit $proc.ExitCode }

& $Python scripts\compare_rscd_surface_candidates.py *>> $Log
& $Python scripts\compare_rscd_class_slices.py *>> $Log
& $Python scripts\write_rscd_formal_result_summary.py *>> $Log
& $Python scripts\write_rscd_decision_dashboard.py *>> $Log
& $Python scripts\write_goal_completion_audit.py *>> $Log
Write-QueueLog "TTA ensemble evaluation complete"
