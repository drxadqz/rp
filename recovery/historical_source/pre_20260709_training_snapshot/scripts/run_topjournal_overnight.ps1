param(
    [string] $Config = "configs\experiments\topjournal_overnight.yaml",
    [string] $SmokeConfig = "configs\experiments\topjournal_smoke.yaml",
    [switch] $SkipSmoke
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPrefix = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper"
$CondaExe = "E:\Anaconda\Scripts\conda.exe"
$TorchHome = "D:\NMI_SPWFM_datasets\torch_cache"

if (!(Test-Path $EnvPrefix)) {
    throw "Expected conda env not found: $EnvPrefix"
}
if (!(Test-Path $CondaExe)) {
    throw "Expected conda executable not found: $CondaExe"
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $TorchHome | Out-Null
$env:TORCH_HOME = $TorchHome

$OutputDirLine = Select-String -Path $Config -Pattern "^\s*output_dir:\s*(.+)\s*$" | Select-Object -First 1
if ($OutputDirLine) {
    $OutputDir = $OutputDirLine.Matches[0].Groups[1].Value.Trim()
} else {
    $OutputDir = "outputs\topjournal_overnight"
}
$Checkpoint = Join-Path $OutputDir "best.pt"
$SmokeOutputDirLine = Select-String -Path $SmokeConfig -Pattern "^\s*output_dir:\s*(.+)\s*$" | Select-Object -First 1
if ($SmokeOutputDirLine) {
    $SmokeOutputDir = $SmokeOutputDirLine.Matches[0].Groups[1].Value.Trim()
} else {
    $SmokeOutputDir = "outputs\topjournal_smoke"
}
$SmokeCheckpoint = Join-Path $SmokeOutputDir "best.pt"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Transcript = Join-Path $OutputDir ("pipeline_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $Transcript | Out-Null

function Run-Step {
    param(
        [string] $Name,
        [string[]] $CommandArgs
    )
    Write-Host ""
    Write-Host "==== $Name ====" -ForegroundColor Cyan
    Write-Host "$CondaExe run -p $EnvPrefix $($CommandArgs -join ' ')" -ForegroundColor DarkCyan
    & $CondaExe run -p $EnvPrefix @CommandArgs
    $Code = $LASTEXITCODE
    $global:LASTEXITCODE = $Code
    if ($Code -ne 0) {
        throw "Step failed: $Name (exit code $Code)"
    }
}

try {
    Run-Step "verify gpu" @("python", "scripts\verify_gpu.py")
    Run-Step "audit data" @("python", "scripts\audit_data.py", "--write", "configs\data\local_paths.yaml")
    Run-Step "build smoke manifests" @("python", "scripts\build_manifests.py", "--config", "configs\data\local_paths.yaml", "--out-dir", "data\manifests", "--max-per-class", "200")
    Run-Step "build full manifests" @("python", "scripts\build_manifests.py", "--config", "configs\data\local_paths.yaml", "--out-dir", "data\manifests_full")
    Run-Step "manifest stats" @("python", "scripts\manifest_stats.py", "data\manifests_full\rscd_prepared_train.csv", "data\manifests_full\roadsaw_train.csv", "data\manifests_full\roadsc_train.csv", "--out", (Join-Path $OutputDir "manifest_stats_train.json"))

    if (!$SkipSmoke) {
        Run-Step "inspect smoke batch" @("python", "scripts\inspect_batch.py", "--config", $SmokeConfig, "--split", "train")
        Run-Step "smoke train" @("python", "scripts\train.py", "--config", $SmokeConfig)
        Run-Step "smoke eval" @("python", "scripts\evaluate.py", "--config", $SmokeConfig, "--checkpoint", $SmokeCheckpoint, "--split", "val")
    }

    Run-Step "runtime estimate" @("python", "scripts\estimate_runtime.py", "--config", $Config)
    Run-Step "train topjournal" @("python", "scripts\train.py", "--config", $Config)
    Run-Step "test topjournal" @("python", "scripts\evaluate.py", "--config", $Config, "--checkpoint", $Checkpoint, "--split", "test")
    Run-Step "calibrate intervals" @("python", "scripts\calibrate_intervals.py", "--config", $Config, "--checkpoint", $Checkpoint, "--target-coverage", "0.90", "--out", (Join-Path $OutputDir "interval_calibration_90.json"))
    Run-Step "dataset shortcut diagnostic" @("python", "scripts\dataset_id_diagnostic.py", "--config", $Config, "--checkpoint", $Checkpoint, "--max-samples", "5000")
}
finally {
    Stop-Transcript | Out-Null
    Write-Host ""
    Write-Host "Pipeline log: $Transcript" -ForegroundColor Green
}
