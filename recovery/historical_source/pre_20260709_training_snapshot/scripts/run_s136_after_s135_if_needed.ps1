param(
    [switch]$DryRun,
    [int]$PollSeconds = 300,
    [double]$ScreenMinFreeGB = 2.0,
    [double]$FullMinFreeGB = 4.0
)

$ErrorActionPreference = "Stop"

$ProjectDir = "E:\perception\friction_affordance_field"
$PythonExe = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$ComparisonDir = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715"
$S96Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s96_wc_pair_relative_boundary_20260712"
$S7Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
$S133Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715"
$S133StrictAuditDir = Join-Path $S133Dir "strict_promotion_audit_vs_s7_full"
$S135Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
$S135FullDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
$S135StrictAuditDir = Join-Path $S135FullDir "strict_promotion_audit_vs_s7_full"
$S136Dir = "E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_screen_20260715"
$S136FullDir = "E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_full_20260715"
$S136ControlDir = "E:\perception_outputs\rscd_surface_classification\s136_control_fixed_uniform_gate_screen_20260715"
$S136Config = "configs\c3_farnet\c3_farnet_s136_coupled_factor_backbone_screen_20260715.yaml"
$S136FullConfig = "configs\c3_farnet\c3_farnet_s136_coupled_factor_backbone_full_20260715.yaml"
$S136ControlConfig = "configs\c3_farnet\c3_farnet_s136_control_fixed_uniform_gate_screen_20260715.yaml"
$TrainScript = "scripts\train_coupled_factor_backbone.py"
$PromotionAuditScript = "scripts\audit_rscd_candidate_promotion.py"
$ReadinessScript = "scripts\audit_s136_queue_readiness.py"
$CompareScript = "scripts\compare_rscd_runs.py"
$SotaGapScript = "scripts\analyze_sota_gap_budget.py"
$NextMechanismScript = "scripts\decide_rscd_next_mechanism.py"
$FeatureDiagnosis = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\s7_high_error_feature_values_quick_20260716\feature_classifier_results.json"
$HandoffLog = Join-Path $S136Dir "handoff_after_s135c.log"

New-Item -ItemType Directory -Force -Path $S136Dir | Out-Null

function Write-Handoff([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HandoffLog -Value "[$stamp] $Message"
}

function Get-ActiveRscdTraining {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            ($_.CommandLine -like "*train.py*" -and $_.CommandLine -like "*c3_farnet*") -or
            ($_.CommandLine -like "*train_coupled_factor_backbone.py*")
        }
}

function Test-FreeSpaceGB([string]$PathText, [double]$RequiredGB) {
    $root = [System.IO.Path]::GetPathRoot($PathText)
    $driveName = $root.Substring(0, 1)
    $drive = Get-PSDrive -Name $driveName -ErrorAction Stop
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Handoff "Disk check for $PathText on drive $driveName`: free=${freeGB}GB required=${RequiredGB}GB"
    return ($drive.Free -ge ($RequiredGB * 1GB))
}

function Read-SotaPassFromMetrics([string]$RunDir) {
    $metricsPath = Join-Path $RunDir "test_metrics.json"
    if (!(Test-Path $metricsPath)) {
        return $null
    }
    $payload = Get-Content $metricsPath -Raw | ConvertFrom-Json
    if ($payload.summary) {
        $summary = $payload.summary
    } else {
        $summary = $payload
    }
    $top1 = [double]$summary.top1
    $macroF1 = [double]$summary.macro_f1
    $samples = [int]$summary.num_samples
    return (($samples -eq 49500) -and ($top1 -ge 0.9286) -and ($macroF1 -ge 0.8949))
}

function Read-PromotionPassed([string]$AuditDir) {
    $auditPath = Join-Path $AuditDir "promotion_audit.json"
    if (!(Test-Path $auditPath)) {
        return $null
    }
    $payload = Get-Content $auditPath -Raw | ConvertFrom-Json
    if ($payload.ok -eq $false) {
        return $false
    }
    return [bool]$payload.decision.passed
}

function Invoke-Run([string]$Name, [string[]]$RunArgs, [string]$WorkingDir, [string]$LogDir) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $LogDir "$($Name)_stdout_$stamp.log"
    $stderr = Join-Path $LogDir "$($Name)_stderr_$stamp.log"
    Write-Handoff "Starting $Name`: $($RunArgs -join ' ')"
    if ($DryRun) {
        Write-Handoff "DryRun: would start $Name stdout=$stdout stderr=$stderr"
        return 0
    }
    $process = Start-Process -FilePath $PythonExe `
        -ArgumentList $RunArgs `
        -WorkingDirectory $WorkingDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Write-Handoff "$Name started. pid=$($process.Id) stdout=$stdout stderr=$stderr"
    Wait-Process -Id $process.Id
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    Write-Handoff "$Name exited. pid=$($process.Id) exit=$exitCode"
    return $exitCode
}

function Invoke-Analysis([string]$Name, [string[]]$AnalysisArgs, [string]$OutputDir) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Handoff "Running analysis $Name`: $($AnalysisArgs -join ' ')"
    if ($DryRun) {
        Write-Handoff "DryRun: would run analysis $Name"
        return 0
    }
    & $PythonExe $AnalysisArgs | Add-Content -Path $HandoffLog
    $exit = $LASTEXITCODE
    Write-Handoff "Analysis $Name exited with code $exit."
    return $exit
}

function Invoke-NextMechanismDecision([string]$Name, [string]$RunDir, [string]$RunName, [string]$BaselineDir, [string]$BaselineName, [string]$Protocol) {
    $outputDir = Join-Path $RunDir "next_mechanism_decision"
    $args = @(
        $NextMechanismScript,
        "--candidate-dir", $RunDir,
        "--candidate-name", $RunName,
        "--baseline-dir", $BaselineDir,
        "--baseline-name", $BaselineName,
        "--protocol", $Protocol,
        "--output-dir", $outputDir
    )
    if (Test-Path $FeatureDiagnosis) {
        $args += @("--feature-diagnosis", $FeatureDiagnosis)
    }
    Invoke-Analysis $Name $args $outputDir | Out-Null
}

function Invoke-S136Readiness {
    $readinessDir = Join-Path $ComparisonDir "S136_queue_readiness_20260715"
    New-Item -ItemType Directory -Force -Path $readinessDir | Out-Null
    Write-Handoff "Running S136 readiness audit at $readinessDir."
    if ($DryRun) {
        Write-Handoff "DryRun: would run S136 readiness audit."
        return 0
    }
    & $PythonExe $ReadinessScript `
        --screen-config $S136Config `
        --full-config $S136FullConfig `
        --control-config $S136ControlConfig `
        --s96-dir $S96Dir `
        --s7-dir $S7Dir `
        --smoke-dir "E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_smoke_20260715" `
        --watcher-script "scripts\run_s136_after_s135_if_needed.ps1" `
        --output-dir $readinessDir |
        Add-Content -Path $HandoffLog
    $exit = $LASTEXITCODE
    Write-Handoff "S136 readiness audit exited with code $exit."
    return $exit
}

Write-Handoff "S136 after-S135 watcher started. DryRun=$DryRun PollSeconds=$PollSeconds"

while ($true) {
    $s133StrictSota = Read-PromotionPassed $S133StrictAuditDir
    if ($s133StrictSota -eq $true) {
        Write-Handoff "S133c full already passed strict SOTA audit. S136 fallback not needed. Exiting."
        exit 0
    }
    $decisionPath = Join-Path $S135Dir "screen_promotion_decision.json"
    if (Test-Path (Join-Path $S135FullDir "test_metrics.json")) {
        $s135FullStrict = Read-PromotionPassed $S135StrictAuditDir
        if ($s135FullStrict -eq $true) {
            Write-Handoff "S135 full passed strict SOTA audit. S136 fallback not needed. Exiting."
            exit 0
        }
        if ($s135FullStrict -eq $false) {
            Write-Handoff "S135 full strict SOTA audit failed. S136 fallback may start."
            break
        }
        Write-Handoff "S135 full metrics exist; waiting for strict SOTA audit before deciding on S136 fallback."
        if ($DryRun) {
            Write-Handoff "DryRun: stopping after S135 full audit wait-state inspection."
            exit 0
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    if (Test-Path $decisionPath) {
        $decision = Get-Content $decisionPath -Raw | ConvertFrom-Json
        if ([bool]$decision.promote_to_full) {
            Write-Handoff "S135 screen promoted to full. Waiting for S135 full metrics before deciding on S136 fallback."
            if ($DryRun) {
                Write-Handoff "DryRun: stopping after promoted-to-full wait-state inspection."
                exit 0
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        Write-Handoff "S135 screen decision exists and did not promote. S136 fallback may start."
        break
    }
    $active = Get-ActiveRscdTraining
    if ($active) {
        $activeText = ($active | Select-Object ProcessId,CommandLine | Out-String).Trim()
        Write-Handoff "Waiting for S135 decision; active training detected: $activeText"
    } else {
        Write-Handoff "Waiting for S135 screen promotion decision at $decisionPath."
    }
    if ($DryRun) {
        Write-Handoff "DryRun: stopping after one wait-state inspection."
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

if (Test-Path (Join-Path $S136Dir "test_metrics.json")) {
    Write-Handoff "S136 screen metrics already exist. Not starting duplicate screen."
} else {
    $active = Get-ActiveRscdTraining
    if ($active) {
        Write-Handoff "Another RSCD training process is active. S136 screen not started."
        $active | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    $readinessExit = Invoke-S136Readiness
    if ($readinessExit -ne 0) {
        Write-Handoff "S136 readiness audit failed. Screen not started."
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S136Dir $ScreenMinFreeGB)) {
        Write-Handoff "Not enough free disk for S136 screen. Exiting."
        exit 0
    }
    $screenExit = Invoke-Run "s136_screen" @("-u", $TrainScript, "--config", $S136Config, "--device", "cuda") $ProjectDir $S136Dir
    if ($screenExit -ne 0) {
        Write-Handoff "S136 screen exited non-zero. Full promotion not considered."
        exit 0
    }
}

if (!(Test-Path (Join-Path $S136Dir "test_metrics.json"))) {
    Write-Handoff "S136 screen has no test_metrics.json. Full promotion not considered."
    exit 0
}

$screenCompareDir = Join-Path $S136Dir "compare_to_s96"
Invoke-Analysis "s136_screen_compare_s96" @(
    $CompareScript,
    "--candidate-dir", $S136Dir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S136_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenCompareDir
) $screenCompareDir | Out-Null

$screenGapDir = Join-Path $S136Dir "sota_gap_budget"
Invoke-Analysis "s136_screen_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S136Dir,
    "--run-name", "S136_screen",
    "--output-dir", $screenGapDir
) $screenGapDir | Out-Null

Invoke-NextMechanismDecision "s136_screen_next_mechanism" $S136Dir "S136_screen" $S96Dir "S96_cap250" "screen"

$screenAuditDir = Join-Path $ComparisonDir "S136_screen_promotion_audit_vs_S96"
$auditExit = Invoke-Analysis "s136_screen_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S136Dir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S136_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenAuditDir
) $screenAuditDir

$auditPath = Join-Path $screenAuditDir "promotion_audit.json"
if (!(Test-Path $auditPath)) {
    Write-Handoff "S136 promotion audit JSON missing. Full run not started."
    exit 0
}
$audit = Get-Content $auditPath -Raw | ConvertFrom-Json
if (-not [bool]$audit.decision.passed) {
    Write-Handoff "S136 screen did not pass promotion audit. Full run not started."
    exit 0
}

if (Test-Path (Join-Path $S136ControlDir "test_metrics.json")) {
    Write-Handoff "S136 fixed-uniform control screen metrics already exist. Not starting duplicate control."
} else {
    $activeBeforeControl = Get-ActiveRscdTraining
    if ($activeBeforeControl) {
        Write-Handoff "Another RSCD training process is active before S136 control. Control screen and full not started."
        $activeBeforeControl | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S136ControlDir $ScreenMinFreeGB)) {
        Write-Handoff "Not enough free disk for S136 fixed-uniform control. Full run not started."
        exit 0
    }
    $controlExit = Invoke-Run "s136_fixed_uniform_control_screen" @("-u", $TrainScript, "--config", $S136ControlConfig, "--device", "cuda") $ProjectDir $S136ControlDir
    if ($controlExit -ne 0) {
        Write-Handoff "S136 fixed-uniform control exited non-zero. Full run not started because same-budget control is missing."
        exit 0
    }
}

if (!(Test-Path (Join-Path $S136ControlDir "test_metrics.json"))) {
    Write-Handoff "S136 fixed-uniform control has no test_metrics.json. Full run not started because same-budget control is missing."
    exit 0
}

$controlCompareDir = Join-Path $S136Dir "compare_to_fixed_uniform_control"
Invoke-Analysis "s136_learned_gate_compare_control" @(
    $CompareScript,
    "--candidate-dir", $S136Dir,
    "--baseline-dir", $S136ControlDir,
    "--candidate-name", "S136_learned_gate",
    "--baseline-name", "S136_fixed_uniform_control",
    "--output-dir", $controlCompareDir
) $controlCompareDir | Out-Null

$controlAuditDir = Join-Path $ComparisonDir "S136_learned_gate_vs_fixed_uniform_control"
Invoke-Analysis "s136_learned_gate_control_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S136Dir,
    "--baseline-dir", $S136ControlDir,
    "--candidate-name", "S136_learned_gate",
    "--baseline-name", "S136_fixed_uniform_control",
    "--output-dir", $controlAuditDir
) $controlAuditDir | Out-Null

if (Test-Path (Join-Path $S136FullDir "test_metrics.json")) {
    Write-Handoff "S136 full metrics already exist. Not starting duplicate full."
    exit 0
}
$activeAfterScreen = Get-ActiveRscdTraining
if ($activeAfterScreen) {
    Write-Handoff "Another RSCD training process is active after S136 screen. Full not started."
    $activeAfterScreen | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
    exit 0
}
if (-not (Test-FreeSpaceGB $S136FullDir $FullMinFreeGB)) {
    Write-Handoff "Not enough free disk for S136 full. Full not started."
    exit 0
}

$fullExit = Invoke-Run "s136_full" @("-u", $TrainScript, "--config", $S136FullConfig, "--device", "cuda") $ProjectDir $S136FullDir
if ($fullExit -ne 0) {
    Write-Handoff "S136 full exited non-zero. Final SOTA audit skipped."
    exit 0
}
if (!(Test-Path (Join-Path $S136FullDir "test_metrics.json"))) {
    Write-Handoff "S136 full finished without test_metrics.json. Final SOTA audit skipped."
    exit 0
}

$fullCompareDir = Join-Path $S136FullDir "compare_to_s7_full"
Invoke-Analysis "s136_full_compare_s7" @(
    $CompareScript,
    "--candidate-dir", $S136FullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S136_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullCompareDir
) $fullCompareDir | Out-Null

$fullGapDir = Join-Path $S136FullDir "sota_gap_budget"
Invoke-Analysis "s136_full_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S136FullDir,
    "--run-name", "S136_full",
    "--output-dir", $fullGapDir
) $fullGapDir | Out-Null

Invoke-NextMechanismDecision "s136_full_next_mechanism" $S136FullDir "S136_full" $S7Dir "S7_full" "full"

$fullAuditDir = Join-Path $ComparisonDir "S136_full_promotion_audit_vs_S7"
Invoke-Analysis "s136_full_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S136FullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S136_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullAuditDir,
    "--require-sota"
) $fullAuditDir | Out-Null

Write-Handoff "S136 fallback pipeline finished."
