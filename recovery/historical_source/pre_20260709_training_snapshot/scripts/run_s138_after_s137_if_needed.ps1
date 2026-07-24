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
$S135FullDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
$S135StrictAuditDir = Join-Path $S135FullDir "strict_promotion_audit_vs_s7_full"
$S136FullDir = "E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_full_20260715"
$S136StrictAuditDir = Join-Path $ComparisonDir "S136_full_promotion_audit_vs_S7"
$S136dFullDir = "E:\perception_outputs\rscd_surface_classification\s136d_coupled_factor_backbone_safe_distill_full_20260715"
$S136dStrictAuditDir = Join-Path $ComparisonDir "S136d_full_promotion_audit_vs_S7"
$S137Dir = "E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_screen_20260715"
$S137FullDir = "E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_full_20260715"
$S137StrictAuditDir = Join-Path $ComparisonDir "S137_full_promotion_audit_vs_S7"
$S138Dir = "E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_screen_20260716"
$S138ControlDir = "E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_control_20260716"
$S138FullDir = "E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_full_20260716"
$S138StrictAuditDir = Join-Path $ComparisonDir "S138_full_promotion_audit_vs_S7"
$S138Config = "configs\c3_farnet\c3_farnet_s138_dual_film_texture_roughness_screen_20260716.yaml"
$S138ControlConfig = "configs\c3_farnet\c3_farnet_s138_dual_film_texture_roughness_control_20260716.yaml"
$S138FullConfig = "configs\c3_farnet\c3_farnet_s138_dual_film_texture_roughness_full_20260716.yaml"
$TrainScript = "scripts\train_coupled_factor_backbone.py"
$PromotionAuditScript = "scripts\audit_rscd_candidate_promotion.py"
$ReadinessScript = "scripts\audit_s138_queue_readiness.py"
$CompareScript = "scripts\compare_rscd_runs.py"
$SotaGapScript = "scripts\analyze_sota_gap_budget.py"
$NextMechanismScript = "scripts\decide_rscd_next_mechanism.py"
$FeatureDiagnosis = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\s7_high_error_feature_values_quick_20260716\feature_classifier_results.json"
$WatcherLogDir = Join-Path $ComparisonDir "S138_watcher_20260716"
$HandoffLog = Join-Path $WatcherLogDir "handoff_after_s137.log"

New-Item -ItemType Directory -Force -Path $WatcherLogDir | Out-Null

function Write-Handoff([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HandoffLog -Value "[$stamp] $Message"
}

function Get-ActiveRscdTraining {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            ($_.CommandLine -like "*train.py*" -and $_.CommandLine -like "*c3_farnet*") -or
            ($_.CommandLine -like "*train_coupled_factor_backbone.py*") -or
            ($_.CommandLine -like "*cache_teacher_logits.py*")
        }
}

function Get-ActiveUpstreamWatchers {
    Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
        Where-Object {
            ($_.CommandLine -like "*run_s135_after_s133c.ps1*") -or
            ($_.CommandLine -like "*run_s136_after_s135_if_needed.ps1*") -or
            ($_.CommandLine -like "*run_s136d_after_s136_if_needed.ps1*") -or
            ($_.CommandLine -like "*run_s137_after_current_queue_if_needed.ps1*")
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

function Any-FullSotaPass {
    foreach ($dir in @($S133StrictAuditDir, $S135StrictAuditDir, $S136StrictAuditDir, $S136dStrictAuditDir, $S137StrictAuditDir, $S138StrictAuditDir)) {
        $pass = Read-PromotionPassed $dir
        if ($pass -eq $true) {
            return $true
        }
    }
    return $false
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

function Invoke-NextMechanismDecision([string]$Name, [string]$RunDir, [string]$RunName, [string]$BaselineDir, [string]$BaselineName, [string]$Protocol, [string]$OutputDir) {
    $args = @(
        $NextMechanismScript,
        "--candidate-dir", $RunDir,
        "--candidate-name", $RunName,
        "--baseline-dir", $BaselineDir,
        "--baseline-name", $BaselineName,
        "--protocol", $Protocol,
        "--output-dir", $OutputDir
    )
    if (Test-Path $FeatureDiagnosis) {
        $args += @("--feature-diagnosis", $FeatureDiagnosis)
    }
    Invoke-Analysis $Name $args $OutputDir | Out-Null
}

function Invoke-S138Readiness {
    $readinessDir = Join-Path $ComparisonDir "S138_queue_readiness_20260716"
    New-Item -ItemType Directory -Force -Path $readinessDir | Out-Null
    Write-Handoff "Running S138 readiness audit at $readinessDir."
    if ($DryRun) {
        Write-Handoff "DryRun: would run S138 readiness audit."
        return 0
    }
    & $PythonExe $ReadinessScript `
        --screen-config $S138Config `
        --control-config $S138ControlConfig `
        --full-config $S138FullConfig `
        --output-dir $readinessDir |
        Add-Content -Path $HandoffLog
    $exit = $LASTEXITCODE
    Write-Handoff "S138 readiness audit exited with code $exit."
    return $exit
}

Write-Handoff "S138 watcher started. DryRun=$DryRun PollSeconds=$PollSeconds"

$trigger = $null
while ($null -eq $trigger) {
    if (Any-FullSotaPass) {
        Write-Handoff "An upstream full run already clears public SOTA. S138 not needed. Exiting."
        exit 0
    }

    $s137FullPass = Read-PromotionPassed (Join-Path $ComparisonDir "S137_full_promotion_audit_vs_S7")
    if ($s137FullPass -eq $true) {
        Write-Handoff "S137 full promotion audit passed. S138 not needed. Exiting."
        exit 0
    }
    if ($s137FullPass -eq $false) {
        $trigger = "S137_full_failed_final_audit"
        break
    }

    $s137ControlPass = Read-PromotionPassed (Join-Path $ComparisonDir "S137_learned_scale_space_vs_off_control")
    if ($s137ControlPass -eq $false) {
        $trigger = "S137_failed_same_budget_control"
        break
    }

    $s137ScreenPass = Read-PromotionPassed (Join-Path $ComparisonDir "S137_screen_promotion_audit_vs_S96")
    if ($s137ScreenPass -eq $false) {
        $trigger = "S137_screen_failed_promotion"
        break
    }
    if ($s137ScreenPass -eq $true) {
        Write-Handoff "S137 screen passed; waiting for S137 full audit or SOTA pass."
    } else {
        $upstreamWatchers = Get-ActiveUpstreamWatchers
        if (-not $upstreamWatchers) {
            $s137FullSota = Read-SotaPassFromMetrics $S137FullDir
            if ($s137FullSota -eq $false) {
                $trigger = "upstream_queue_idle_without_full_sota"
                break
            }
            $s137ScreenMetrics = Test-Path (Join-Path $S137Dir "test_metrics.json")
            if ($s137ScreenMetrics -and ($s137ScreenPass -ne $true)) {
                $trigger = "upstream_queue_idle_s137_unpromoted"
                break
            }
        }
        Write-Handoff "Waiting for S133/S135/S136/S136d/S137 queue to produce a final decision."
    }

    $active = Get-ActiveRscdTraining
    if ($active) {
        $activeText = ($active | Select-Object ProcessId,CommandLine | Out-String).Trim()
        Write-Handoff "Active RSCD process detected while waiting: $activeText"
    }
    if ($DryRun) {
        Write-Handoff "DryRun: stopping after one wait-state inspection."
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Handoff "S138 trigger: $trigger"

if (Test-Path (Join-Path $S138Dir "test_metrics.json")) {
    Write-Handoff "S138 screen metrics already exist. Not starting duplicate screen."
} else {
    $active = Get-ActiveRscdTraining
    if ($active) {
        Write-Handoff "Another RSCD process is active. S138 screen not started."
        $active | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    $readinessExit = Invoke-S138Readiness
    if ($readinessExit -ne 0) {
        Write-Handoff "S138 readiness audit failed. Screen not started."
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S138Dir $ScreenMinFreeGB)) {
        Write-Handoff "Not enough free disk for S138 screen. Exiting."
        exit 0
    }
    $screenExit = Invoke-Run "s138_screen" @("-u", $TrainScript, "--config", $S138Config, "--device", "cuda") $ProjectDir $S138Dir
    if ($screenExit -ne 0) {
        Write-Handoff "S138 screen exited non-zero. Full promotion not considered."
        exit 0
    }
}

if (!(Test-Path (Join-Path $S138Dir "test_metrics.json"))) {
    Write-Handoff "S138 screen has no test_metrics.json. Full promotion not considered."
    exit 0
}

$screenCompareDir = Join-Path $S138Dir "compare_to_s96"
Invoke-Analysis "s138_screen_compare_s96" @(
    $CompareScript,
    "--candidate-dir", $S138Dir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S138_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenCompareDir
) $screenCompareDir | Out-Null

$screenGapDir = Join-Path $S138Dir "sota_gap_budget"
Invoke-Analysis "s138_screen_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S138Dir,
    "--run-name", "S138_screen",
    "--output-dir", $screenGapDir
) $screenGapDir | Out-Null

Invoke-NextMechanismDecision "s138_screen_next_mechanism_vs_s96" $S138Dir "S138_screen" $S96Dir "S96_cap250" "screen" (Join-Path $S138Dir "next_mechanism_decision")

if (Test-Path (Join-Path $S138ControlDir "test_metrics.json")) {
    Write-Handoff "S138 off-control metrics already exist. Not starting duplicate control."
} else {
    $activeBeforeControl = Get-ActiveRscdTraining
    if ($activeBeforeControl) {
        Write-Handoff "Another RSCD process is active before S138 control. Control and full not started."
        $activeBeforeControl | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S138ControlDir $ScreenMinFreeGB)) {
        Write-Handoff "Not enough free disk for S138 off-control. Full run not started."
        exit 0
    }
    $controlExit = Invoke-Run "s138_off_control_screen" @("-u", $TrainScript, "--config", $S138ControlConfig, "--device", "cuda") $ProjectDir $S138ControlDir
    if ($controlExit -ne 0) {
        Write-Handoff "S138 off-control exited non-zero. Full run not started because same-budget control is missing."
        exit 0
    }
}

if (!(Test-Path (Join-Path $S138ControlDir "test_metrics.json"))) {
    Write-Handoff "S138 off-control has no test_metrics.json. Full run not started because same-budget control is missing."
    exit 0
}

$controlCompareDir = Join-Path $S138Dir "compare_to_off_control"
Invoke-Analysis "s138_screen_compare_off_control" @(
    $CompareScript,
    "--candidate-dir", $S138Dir,
    "--baseline-dir", $S138ControlDir,
    "--candidate-name", "S138_dual_film_texture",
    "--baseline-name", "S138_off_control",
    "--output-dir", $controlCompareDir
) $controlCompareDir | Out-Null

$controlAuditDir = Join-Path $ComparisonDir "S138_dual_film_texture_vs_off_control"
Invoke-Analysis "s138_control_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S138Dir,
    "--baseline-dir", $S138ControlDir,
    "--candidate-name", "S138_dual_film_texture",
    "--baseline-name", "S138_off_control",
    "--output-dir", $controlAuditDir
) $controlAuditDir | Out-Null

$controlPass = Read-PromotionPassed $controlAuditDir
if ($controlPass -ne $true) {
    Write-Handoff "S138 learned route did not beat its same-budget off-control. Full run not started."
    exit 0
}

$screenAuditDir = Join-Path $ComparisonDir "S138_screen_promotion_audit_vs_S96"
Invoke-Analysis "s138_screen_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S138Dir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S138_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenAuditDir
) $screenAuditDir | Out-Null

$screenPass = Read-PromotionPassed $screenAuditDir
if ($screenPass -ne $true) {
    Write-Handoff "S138 screen did not pass S96 promotion audit. Full run not started."
    exit 0
}

if (Test-Path (Join-Path $S138FullDir "test_metrics.json")) {
    Write-Handoff "S138 full metrics already exist. Not starting duplicate full."
    exit 0
}
$activeAfterScreen = Get-ActiveRscdTraining
if ($activeAfterScreen) {
    Write-Handoff "Another RSCD process is active after S138 screen. Full not started."
    $activeAfterScreen | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
    exit 0
}
if (-not (Test-FreeSpaceGB $S138FullDir $FullMinFreeGB)) {
    Write-Handoff "Not enough free disk for S138 full. Full not started."
    exit 0
}

$fullExit = Invoke-Run "s138_full" @("-u", $TrainScript, "--config", $S138FullConfig, "--device", "cuda") $ProjectDir $S138FullDir
if ($fullExit -ne 0) {
    Write-Handoff "S138 full exited non-zero. Final SOTA audit skipped."
    exit 0
}
if (!(Test-Path (Join-Path $S138FullDir "test_metrics.json"))) {
    Write-Handoff "S138 full finished without test_metrics.json. Final SOTA audit skipped."
    exit 0
}

$fullCompareDir = Join-Path $S138FullDir "compare_to_s7_full"
Invoke-Analysis "s138_full_compare_s7" @(
    $CompareScript,
    "--candidate-dir", $S138FullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S138_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullCompareDir
) $fullCompareDir | Out-Null

$fullGapDir = Join-Path $S138FullDir "sota_gap_budget"
Invoke-Analysis "s138_full_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S138FullDir,
    "--run-name", "S138_full",
    "--output-dir", $fullGapDir
) $fullGapDir | Out-Null

Invoke-NextMechanismDecision "s138_full_next_mechanism" $S138FullDir "S138_full" $S7Dir "S7_full" "full" (Join-Path $S138FullDir "next_mechanism_decision")

$fullAuditDir = Join-Path $ComparisonDir "S138_full_promotion_audit_vs_S7"
Invoke-Analysis "s138_full_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S138FullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S138_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullAuditDir,
    "--require-sota"
) $fullAuditDir | Out-Null

Write-Handoff "S138 pipeline finished."
