param(
    [Parameter(Mandatory = $true)]
    [string] $Config,
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $OutputDir = "",
    [switch] $SkipTrain,
    [switch] $ForceTrain,
    [switch] $SkipDatasetDiagnostic,
    [switch] $SkipAudit,
    [switch] $KeepLastCheckpoint
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (!(Test-Path $Python)) {
    throw "Python not found: $Python"
}
if (!(Test-Path $Config)) {
    throw "Config not found: $Config"
}

if (!$OutputDir) {
    $OutputDirLine = Select-String -Path $Config -Pattern "^\s*output_dir:\s*(.+)\s*$" | Select-Object -First 1
    if ($OutputDirLine) {
        $OutputDir = $OutputDirLine.Matches[0].Groups[1].Value.Trim()
    } else {
        throw "Could not infer output_dir from config: $Config"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:TORCH_HOME = "D:\NMI_SPWFM_datasets\torch_cache"
$TempRoot = "D:\NMI_SPWFM_datasets\tmp"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot

function Run-Step {
    param(
        [string] $Name,
        [string[]] $StepArgs
    )
    Write-Host ""
    Write-Host "==== $Name ====" -ForegroundColor Cyan
    Write-Host "$Python $($StepArgs -join ' ')" -ForegroundColor DarkCyan
    & $Python @StepArgs
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "Step failed: $Name (exit code $Code)"
    }
}

function Get-FreeGB {
    param([string] $Path)
    $Root = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Path -ErrorAction SilentlyContinue))
    if (!$Root) {
        $Root = [System.IO.Path]::GetPathRoot($Path)
    }
    $DriveName = $Root.Substring(0, 1)
    $Drive = Get-PSDrive -Name $DriveName -ErrorAction SilentlyContinue
    if (!$Drive) {
        return $null
    }
    return [math]::Round($Drive.Free / 1GB, 2)
}

function Invoke-SafeOutputCleanup {
    param([string] $CurrentOutputDir)
    $Root = Split-Path -Parent $CurrentOutputDir
    if (!(Test-Path $Root)) {
        return
    }
    $Removed = 0
    foreach ($RunDir in Get-ChildItem -LiteralPath $Root -Directory -ErrorAction SilentlyContinue) {
        $IsComplete = $true
        foreach ($Name in @("best.pt", "detailed_test.json", "interval_calibration_90.json", "bootstrap_metrics.json")) {
            if (!(Test-Path (Join-Path $RunDir.FullName $Name))) {
                $IsComplete = $false
                break
            }
        }
        if (!$IsComplete) {
            continue
        }
        $Last = Join-Path $RunDir.FullName "last.pt"
        if (Test-Path $Last) {
            Remove-Item -LiteralPath $Last -Force
            $Removed += 1
        }
    }
    $Free = Get-FreeGB $CurrentOutputDir
    Write-Host "Safe output cleanup: removed $Removed completed-run last.pt files; free space=$Free GB" -ForegroundColor DarkCyan
}

$Checkpoint = Join-Path $OutputDir "best.pt"
$LastCheckpoint = Join-Path $OutputDir "last.pt"
$TrainingState = Join-Path $OutputDir "training_state.json"

Invoke-SafeOutputCleanup $OutputDir

if ($ForceTrain) {
    $StaleArtifacts = @(
        "best.pt",
        "best_safety.pt",
        "last.pt",
        "training_state.json",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "bootstrap_metrics.md",
        "dataset_id_diagnostic.json",
        "topvenue_result_audit.json",
        "topvenue_result_audit.md",
        "manifest_stats_train.json",
        "confusion_friction_overall.csv",
        "confusion_friction_overall.md",
        "confusion_risk_overall.csv",
        "confusion_risk_overall.md",
        "confusion_friction_roadsaw.csv",
        "confusion_friction_roadsaw.md",
        "confusion_risk_roadsaw.csv",
        "confusion_risk_roadsaw.md"
    )
    foreach ($Artifact in $StaleArtifacts) {
        $Path = Join-Path $OutputDir $Artifact
        if (Test-Path $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
    }
    foreach ($DirName in @("tb", "evidence_maps")) {
        $DirPath = Join-Path $OutputDir $DirName
        if (Test-Path $DirPath) {
            Remove-Item -LiteralPath $DirPath -Recurse -Force
        }
    }
}

if (!$ForceTrain -and !$SkipTrain -and (Test-Path $TrainingState) -and (Test-Path $Checkpoint)) {
    $StateJson = Get-Content -LiteralPath $TrainingState -Raw | ConvertFrom-Json
    $CfgYaml = Get-Content -LiteralPath $Config -Raw
    $PatienceMatch = [regex]::Match($CfgYaml, "(?m)^\s*early_stop_patience:\s*(\d+)\s*$")
    $EpochsMatch = [regex]::Match($CfgYaml, "(?m)^\s*epochs:\s*(\d+)\s*$")
    $Patience = if ($PatienceMatch.Success) { [int]$PatienceMatch.Groups[1].Value } else { $null }
    $Epochs = if ($EpochsMatch.Success) { [int]$EpochsMatch.Groups[1].Value } else { [int]$StateJson.epochs }
    $ReachedPatience = ($Patience -ne $null) -and ([int]$StateJson.stale_epochs -ge $Patience)
    $ReachedEpochs = ([int]$StateJson.epoch -ge $Epochs)
    if ($ReachedPatience -or $ReachedEpochs) {
        Write-Host "Skipping train: existing training_state indicates completion for $OutputDir" -ForegroundColor DarkGreen
        $SkipTrain = $true
    }
}

if (!$SkipTrain) {
    $TrainArgs = @("-u", "scripts\train.py", "--config", $Config)
    if (!$ForceTrain -and (Test-Path $LastCheckpoint)) {
        $TrainArgs += "--resume"
        $TrainArgs += $LastCheckpoint
    }
    Run-Step "train" $TrainArgs
}

if (!(Test-Path $Checkpoint)) {
    throw "Checkpoint not found after training: $Checkpoint"
}

$ConfigJson = Join-Path $OutputDir "config.json"
if (!(Test-Path $ConfigJson)) {
    throw "Run config not found after training: $ConfigJson"
}

$TrainManifests = & $Python -c "import json; cfg=json.load(open(r'$ConfigJson', encoding='utf-8')); print('\n'.join(cfg['data']['train_manifests']))"
$ManifestArgs = @()
foreach ($Manifest in ($TrainManifests -split "`n")) {
    $Trimmed = $Manifest.Trim()
    if ($Trimmed) {
        $ManifestArgs += "--manifest"
        $ManifestArgs += $Trimmed
    }
}
$ManifestStatsArgs = @(
    "-u", "scripts\manifest_stats.py"
) + $ManifestArgs + @(
    "--out", (Join-Path $OutputDir "manifest_stats_train.json")
)
Run-Step "training manifest stats" $ManifestStatsArgs

Run-Step "test eval" @(
    "-u", "scripts\evaluate.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", "test",
    "--out", (Join-Path $OutputDir "evaluate_test.json")
)
$DetailedTestOut = Join-Path $OutputDir "detailed_test.json"
Run-Step "detailed test eval" @(
    "-u", "scripts\evaluate_detailed.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", "test",
    "--out", $DetailedTestOut
)

Run-Step "friction confusion summary" @(
    "-u", "scripts\summarize_confusions.py",
    "--detailed", $DetailedTestOut,
    "--task", "friction",
    "--out-csv", (Join-Path $OutputDir "confusion_friction_overall.csv"),
    "--out-md", (Join-Path $OutputDir "confusion_friction_overall.md")
)
Run-Step "risk confusion summary" @(
    "-u", "scripts\summarize_confusions.py",
    "--detailed", $DetailedTestOut,
    "--task", "risk",
    "--out-csv", (Join-Path $OutputDir "confusion_risk_overall.csv"),
    "--out-md", (Join-Path $OutputDir "confusion_risk_overall.md")
)
$DetailedJsonText = Get-Content -LiteralPath $DetailedTestOut -Raw
if ($DetailedJsonText -match '"roadsaw"') {
    Run-Step "RoadSaW friction confusion summary" @(
        "-u", "scripts\summarize_confusions.py",
        "--detailed", $DetailedTestOut,
        "--task", "friction",
        "--dataset", "roadsaw",
        "--out-csv", (Join-Path $OutputDir "confusion_friction_roadsaw.csv"),
        "--out-md", (Join-Path $OutputDir "confusion_friction_roadsaw.md")
    )
    Run-Step "RoadSaW risk confusion summary" @(
        "-u", "scripts\summarize_confusions.py",
        "--detailed", $DetailedTestOut,
        "--task", "risk",
        "--dataset", "roadsaw",
        "--out-csv", (Join-Path $OutputDir "confusion_risk_roadsaw.csv"),
        "--out-md", (Join-Path $OutputDir "confusion_risk_roadsaw.md")
    )
}
Run-Step "calibrate intervals" @(
    "-u", "scripts\calibrate_intervals.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--target-coverage", "0.90",
    "--out", (Join-Path $OutputDir "interval_calibration_90.json")
)

Run-Step "bootstrap metric confidence intervals" @(
    "-u", "scripts\bootstrap_metrics.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", "test",
    "--target-coverage", "0.90",
    "--num-bootstrap", "500",
    "--out-json", (Join-Path $OutputDir "bootstrap_metrics.json"),
    "--out-md", (Join-Path $OutputDir "bootstrap_metrics.md")
)

if (!$SkipDatasetDiagnostic) {
    Run-Step "dataset shortcut diagnostic" @(
        "-u", "scripts\dataset_id_diagnostic.py",
        "--config", $Config,
        "--checkpoint", $Checkpoint,
        "--max-samples", "5000",
        "--out", (Join-Path $OutputDir "dataset_id_diagnostic.json")
    )
}

$UsesEvidence = & $Python -c "import json; cfg=json.load(open(r'$ConfigJson', encoding='utf-8')); print('1' if cfg.get('model',{}).get('use_evidence_field') else '0')"
if ($UsesEvidence.Trim() -eq "1") {
    Run-Step "evidence maps" @(
        "-u", "scripts\export_evidence_maps.py",
        "--config", $Config,
        "--checkpoint", $Checkpoint,
        "--split", "test",
        "--out-dir", (Join-Path $OutputDir "evidence_maps"),
        "--max-samples", "24",
        "--selection", "mixed",
        "--clean"
    )
    Run-Step "evidence field audit" @(
        "-u", "scripts\analyze_evidence_field.py",
        "--config", $Config,
        "--checkpoint", $Checkpoint,
        "--split", "test",
        "--max-samples", "3000",
        "--out-json", (Join-Path $OutputDir "evidence_field_audit.json"),
        "--out-md", (Join-Path $OutputDir "evidence_field_audit.md")
    )
}

if (!$SkipAudit) {
    Run-Step "audit" @(
        "-u", "scripts\audit_topvenue_results.py",
        "--output-dir", $OutputDir,
        "--out-md", (Join-Path $OutputDir "topvenue_result_audit.md"),
        "--out-json", (Join-Path $OutputDir "topvenue_result_audit.json")
    )
}

Run-Step "slim best checkpoint" @(
    "-u", "scripts\slim_best_checkpoints.py",
    "--root", (Split-Path -Parent $OutputDir),
    "--apply"
)

$CompleteArtifacts = @(
    (Join-Path $OutputDir "best.pt"),
    (Join-Path $OutputDir "detailed_test.json"),
    (Join-Path $OutputDir "interval_calibration_90.json"),
    (Join-Path $OutputDir "bootstrap_metrics.json")
)
if (!$KeepLastCheckpoint -and ($CompleteArtifacts | ForEach-Object { Test-Path $_ } | Where-Object { -not $_ } | Measure-Object).Count -eq 0) {
    $Last = Join-Path $OutputDir "last.pt"
    if (Test-Path $Last) {
        Write-Host ""
        Write-Host "==== cleanup completed run ====" -ForegroundColor Cyan
        Write-Host "Removing completed-run resume checkpoint: $Last" -ForegroundColor DarkCyan
        Remove-Item -LiteralPath $Last -Force
    }
}
