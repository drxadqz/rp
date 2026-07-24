param(
    [switch]$DryRun,
    [int]$PollSeconds = 300,
    [double]$CacheMinFreeGB = 1.0,
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
$S136dDir = "E:\perception_outputs\rscd_surface_classification\s136d_coupled_factor_backbone_safe_distill_screen_20260715"
$S136dFullDir = "E:\perception_outputs\rscd_surface_classification\s136d_coupled_factor_backbone_safe_distill_full_20260715"
$S136Config = "configs\c3_farnet\c3_farnet_s136_coupled_factor_backbone_screen_20260715.yaml"
$S136dConfig = "configs\c3_farnet\c3_farnet_s136d_coupled_factor_backbone_safe_distill_screen_20260715.yaml"
$S136dFullConfig = "configs\c3_farnet\c3_farnet_s136d_coupled_factor_backbone_safe_distill_full_20260715.yaml"
$S136dTeacherScreenConfig = "configs\c3_farnet\c3_farnet_s7_teacher_cache_for_s136d_screen_20260715.yaml"
$S136dTeacherFullConfig = "configs\c3_farnet\c3_farnet_s7_teacher_cache_for_s136d_full_20260715.yaml"
$S136dScreenCache = Join-Path $S136dDir "s7_teacher_logits_train_cap1000.pt"
$S136dFullCache = Join-Path $S136dFullDir "s7_teacher_logits_train_full.pt"
$TrainScript = "scripts\train_coupled_factor_backbone.py"
$CacheTeacherScript = "scripts\cache_teacher_logits.py"
$PromotionAuditScript = "scripts\audit_rscd_candidate_promotion.py"
$CompareScript = "scripts\compare_rscd_runs.py"
$SotaGapScript = "scripts\analyze_sota_gap_budget.py"
$ReadinessScript = "scripts\audit_s136d_queue_readiness.py"
$DiagnosisScript = "scripts\diagnose_s136d_mechanism_route.py"
$NextMechanismScript = "scripts\decide_rscd_next_mechanism.py"
$FeatureDiagnosis = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\s7_high_error_feature_values_quick_20260716\feature_classifier_results.json"
$HandoffLog = Join-Path $S136dDir "handoff_after_s136.log"
$WatcherScriptPath = $PSCommandPath

New-Item -ItemType Directory -Force -Path $S136dDir | Out-Null

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

function Ensure-TeacherCache([string]$Name, [string]$Config, [string]$CachePath, [double]$MinFreeGB) {
    if (Test-Path $CachePath) {
        Write-Handoff "$Name teacher cache already exists: $CachePath"
        return 0
    }
    if (-not (Test-FreeSpaceGB (Split-Path $CachePath -Parent) $MinFreeGB)) {
        Write-Handoff "Not enough disk for $Name teacher cache."
        return 1
    }
    return Invoke-Run $Name @(
        "-u", $CacheTeacherScript,
        "--config", $Config,
        "--role", "anchor",
        "--split", "train",
        "--output", $CachePath,
        "--batch-size", "32",
        "--num-workers", "2",
        "--device", "cuda"
    ) $ProjectDir (Split-Path $CachePath -Parent)
}

function Invoke-S136dReadiness {
    $readinessDir = Join-Path $ComparisonDir "S136d_queue_readiness_20260715"
    New-Item -ItemType Directory -Force -Path $readinessDir | Out-Null
    Write-Handoff "Running S136d readiness audit at $readinessDir."
    if ($DryRun) {
        Write-Handoff "DryRun: would run S136d readiness audit."
        return 0
    }
    & $PythonExe $ReadinessScript `
        --screen-config $S136dConfig `
        --full-config $S136dFullConfig `
        --teacher-screen-config $S136dTeacherScreenConfig `
        --teacher-full-config $S136dTeacherFullConfig `
        --s96-dir $S96Dir `
        --s7-dir $S7Dir `
        --distill-smoke-dir "E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_distill_smoke_20260715" `
        --watcher-script $WatcherScriptPath `
        --nodistill-screen-config $S136Config `
        --nodistill-screen-dir $S136Dir `
        --output-dir $readinessDir |
        Add-Content -Path $HandoffLog
    $exit = $LASTEXITCODE
    Write-Handoff "S136d readiness audit exited with code $exit."
    return $exit
}

function Invoke-S136dMechanismDiagnosis([string]$Name) {
    $diagnosisDir = Join-Path $ComparisonDir "S136d_mechanism_diagnosis_latest"
    New-Item -ItemType Directory -Force -Path $diagnosisDir | Out-Null
    Write-Handoff "Running S136d mechanism diagnosis $Name at $diagnosisDir."
    if ($DryRun) {
        Write-Handoff "DryRun: would run S136d mechanism diagnosis $Name."
        return 0
    }
    & $PythonExe $DiagnosisScript `
        --s96-dir $S96Dir `
        --s7-dir $S7Dir `
        --s136-dir $S136Dir `
        --s136d-dir $S136dDir `
        --s136d-full-dir $S136dFullDir `
        --output-dir $diagnosisDir |
        Add-Content -Path $HandoffLog
    $exit = $LASTEXITCODE
    Write-Handoff "S136d mechanism diagnosis $Name exited with code $exit."
    return $exit
}

Write-Handoff "S136d after-S136 watcher started. DryRun=$DryRun PollSeconds=$PollSeconds"

$trigger = $null
while ($null -eq $trigger) {
    $s133StrictSota = Read-PromotionPassed $S133StrictAuditDir
    if ($s133StrictSota -eq $true) {
        Write-Handoff "S133c full already passed strict SOTA audit. S136d fallback not needed. Exiting."
        exit 0
    }
    $s136FullPass = Read-PromotionPassed (Join-Path $ComparisonDir "S136_full_promotion_audit_vs_S7")
    if ($s136FullPass -eq $true) {
        Write-Handoff "S136 full passed final promotion audit. S136d fallback not needed. Exiting."
        exit 0
    }
    if ($s136FullPass -eq $false) {
        $trigger = "S136_full_failed_final_audit"
        break
    }

    $s136ScreenPass = Read-PromotionPassed (Join-Path $ComparisonDir "S136_screen_promotion_audit_vs_S96")
    if ($s136ScreenPass -eq $false) {
        $trigger = "S136_screen_failed_promotion"
        break
    }
    if ($s136ScreenPass -eq $true) {
        Write-Handoff "S136 screen passed; waiting for S136 full audit before deciding on S136d."
    } else {
        $s135FullStrict = Read-PromotionPassed $S135StrictAuditDir
        if ($s135FullStrict -eq $true) {
            Write-Handoff "S135 full already passed strict SOTA audit. S136d fallback not needed. Exiting."
            exit 0
        }
        if ($s135FullStrict -eq $false) {
            $activeWatchers = Get-CimInstance Win32_Process -Filter "name='powershell.exe'" |
                Where-Object { $_.CommandLine -like "*run_s136_after_s135_if_needed.ps1*" }
            if ($activeWatchers) {
                Write-Handoff "S135 full strict audit failed but S136 watcher is still active; waiting for S136 decision first."
            } else {
                $trigger = "S135_full_failed_strict_audit_no_active_s136"
                break
            }
        } elseif (Test-Path (Join-Path $S135FullDir "test_metrics.json")) {
            Write-Handoff "S135 full metrics exist; waiting for strict SOTA audit before deciding on S136d fallback."
        } elseif (Test-Path (Join-Path $S135Dir "screen_promotion_decision.json")) {
            Write-Handoff "S135 decision exists; waiting for S136 screen audit or S135 full result."
        } else {
            Write-Handoff "Waiting for S135/S136 upstream result."
        }
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

Write-Handoff "S136d fallback trigger: $trigger"

if (Test-Path (Join-Path $S136dDir "test_metrics.json")) {
    Write-Handoff "S136d screen metrics already exist. Not starting duplicate screen."
} else {
    $active = Get-ActiveRscdTraining
    if ($active) {
        Write-Handoff "Another RSCD process is active. S136d screen not started."
        $active | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    $readinessExit = Invoke-S136dReadiness
    if ($readinessExit -ne 0) {
        Write-Handoff "S136d readiness audit failed. Screen not started."
        exit 0
    }
    $cacheExit = Ensure-TeacherCache "s136d_screen_s7_teacher_cache" $S136dTeacherScreenConfig $S136dScreenCache $CacheMinFreeGB
    if ($cacheExit -ne 0) {
        Write-Handoff "S136d screen teacher cache failed. Screen not started."
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S136dDir $ScreenMinFreeGB)) {
        Write-Handoff "Not enough free disk for S136d screen. Exiting."
        exit 0
    }
    $screenExit = Invoke-Run "s136d_screen" @("-u", $TrainScript, "--config", $S136dConfig, "--device", "cuda") $ProjectDir $S136dDir
    if ($screenExit -ne 0) {
        Write-Handoff "S136d screen exited non-zero. Full promotion not considered."
        exit 0
    }
}

if (!(Test-Path (Join-Path $S136dDir "test_metrics.json"))) {
    Write-Handoff "S136d screen has no test_metrics.json. Full promotion not considered."
    exit 0
}

$screenCompareDir = Join-Path $S136dDir "compare_to_s96"
Invoke-Analysis "s136d_screen_compare_s96" @(
    $CompareScript,
    "--candidate-dir", $S136dDir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S136d_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenCompareDir
) $screenCompareDir | Out-Null

$screenGapDir = Join-Path $S136dDir "sota_gap_budget"
Invoke-Analysis "s136d_screen_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S136dDir,
    "--run-name", "S136d_screen",
    "--output-dir", $screenGapDir
) $screenGapDir | Out-Null

Invoke-NextMechanismDecision "s136d_screen_next_mechanism_vs_s96" $S136dDir "S136d_screen" $S96Dir "S96_cap250" "screen" (Join-Path $S136dDir "next_mechanism_decision")

if (Test-Path (Join-Path $S136Dir "test_metrics.json")) {
    $distillCompareDir = Join-Path $S136dDir "compare_to_s136_nodistill"
    Invoke-Analysis "s136d_screen_compare_s136_nodistill" @(
        $CompareScript,
        "--candidate-dir", $S136dDir,
        "--baseline-dir", $S136Dir,
        "--candidate-name", "S136d_safe_distill_screen",
        "--baseline-name", "S136_no_distill_screen",
        "--output-dir", $distillCompareDir
    ) $distillCompareDir | Out-Null

    $distillAuditDir = Join-Path $ComparisonDir "S136d_safe_distill_vs_S136_no_distill_screen"
    Invoke-Analysis "s136d_screen_safe_distill_audit" @(
        $PromotionAuditScript,
        "--candidate-dir", $S136dDir,
        "--baseline-dir", $S136Dir,
        "--candidate-name", "S136d_safe_distill_screen",
        "--baseline-name", "S136_no_distill_screen",
        "--output-dir", $distillAuditDir
    ) $distillAuditDir | Out-Null

    Invoke-NextMechanismDecision "s136d_screen_next_mechanism_vs_s136_nodistill" $S136dDir "S136d_safe_distill_screen" $S136Dir "S136_no_distill_screen" "screen" (Join-Path $S136dDir "next_mechanism_vs_s136_nodistill")
} else {
    Write-Handoff "S136 no-distill screen metrics not available; safe-distill mechanism comparison skipped for now."
}

Invoke-S136dMechanismDiagnosis "post_screen" | Out-Null

$screenAuditDir = Join-Path $ComparisonDir "S136d_screen_promotion_audit_vs_S96"
Invoke-Analysis "s136d_screen_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S136dDir,
    "--baseline-dir", $S96Dir,
    "--candidate-name", "S136d_screen",
    "--baseline-name", "S96_cap250",
    "--output-dir", $screenAuditDir
) $screenAuditDir | Out-Null

$s136dScreenPass = Read-PromotionPassed $screenAuditDir
if ($s136dScreenPass -ne $true) {
    Write-Handoff "S136d screen did not pass promotion audit. Full run not started."
    exit 0
}

if (Test-Path (Join-Path $S136dFullDir "test_metrics.json")) {
    Write-Handoff "S136d full metrics already exist. Not starting duplicate full."
    exit 0
}
$activeAfterScreen = Get-ActiveRscdTraining
if ($activeAfterScreen) {
    Write-Handoff "Another RSCD process is active after S136d screen. Full not started."
    $activeAfterScreen | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
    exit 0
}
$fullCacheExit = Ensure-TeacherCache "s136d_full_s7_teacher_cache" $S136dTeacherFullConfig $S136dFullCache $CacheMinFreeGB
if ($fullCacheExit -ne 0) {
    Write-Handoff "S136d full teacher cache failed. Full not started."
    exit 0
}
if (-not (Test-FreeSpaceGB $S136dFullDir $FullMinFreeGB)) {
    Write-Handoff "Not enough free disk for S136d full. Full not started."
    exit 0
}

$fullExit = Invoke-Run "s136d_full" @("-u", $TrainScript, "--config", $S136dFullConfig, "--device", "cuda") $ProjectDir $S136dFullDir
if ($fullExit -ne 0) {
    Write-Handoff "S136d full exited non-zero. Final SOTA audit skipped."
    exit 0
}
if (!(Test-Path (Join-Path $S136dFullDir "test_metrics.json"))) {
    Write-Handoff "S136d full finished without test_metrics.json. Final SOTA audit skipped."
    exit 0
}

$fullCompareDir = Join-Path $S136dFullDir "compare_to_s7_full"
Invoke-Analysis "s136d_full_compare_s7" @(
    $CompareScript,
    "--candidate-dir", $S136dFullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S136d_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullCompareDir
) $fullCompareDir | Out-Null

$fullGapDir = Join-Path $S136dFullDir "sota_gap_budget"
Invoke-Analysis "s136d_full_sota_gap" @(
    $SotaGapScript,
    "--run-dir", $S136dFullDir,
    "--run-name", "S136d_full",
    "--output-dir", $fullGapDir
) $fullGapDir | Out-Null

Invoke-NextMechanismDecision "s136d_full_next_mechanism" $S136dFullDir "S136d_full" $S7Dir "S7_full" "full" (Join-Path $S136dFullDir "next_mechanism_decision")

$fullAuditDir = Join-Path $ComparisonDir "S136d_full_promotion_audit_vs_S7"
Invoke-Analysis "s136d_full_promotion_audit" @(
    $PromotionAuditScript,
    "--candidate-dir", $S136dFullDir,
    "--baseline-dir", $S7Dir,
    "--candidate-name", "S136d_full",
    "--baseline-name", "S7_full",
    "--output-dir", $fullAuditDir,
    "--require-sota"
) $fullAuditDir | Out-Null

Invoke-S136dMechanismDiagnosis "post_full" | Out-Null

Write-Handoff "S136d fallback pipeline finished."
