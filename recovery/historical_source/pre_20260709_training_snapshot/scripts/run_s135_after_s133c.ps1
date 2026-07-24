$ErrorActionPreference = "Stop"

$ProjectDir = "E:\perception\friction_affordance_field"
$PythonExe = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$S133Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715"
$FullBaselineDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
$BaselineDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s96_wc_pair_relative_boundary_20260712"
$S135Dir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
$S135Config = "configs\c3_farnet\c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715.yaml"
$S135FullDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
$S135FullConfig = "configs\c3_farnet\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715.yaml"
$DecisionScript = Join-Path $ProjectDir "scripts\decide_s135c_screen_promotion.py"
$SummaryScript = Join-Path $ProjectDir "scripts\summarize_rscd_results.py"
$CompareScript = Join-Path $ProjectDir "scripts\compare_rscd_runs.py"
$StatusScript = Join-Path $ProjectDir "scripts\rscd_pipeline_status.py"
$LiveStatusScript = Join-Path $ProjectDir "scripts\write_rscd_live_route_status.py"
$AuditScript = Join-Path $ProjectDir "scripts\audit_rscd_run.py"
$PhysicsCueScript = Join-Path $ProjectDir "scripts\analyze_rscd_physics_cues.py"
$StableCueScript = Join-Path $ProjectDir "scripts\synthesize_physics_cue_evidence.py"
$ReadinessScript = Join-Path $ProjectDir "scripts\audit_s135c_queue_readiness.py"
$StemActivationScript = Join-Path $ProjectDir "scripts\audit_s135c_stem_activation.py"
$StemActivationRiskScript = Join-Path $ProjectDir "scripts\summarize_s135c_activation_risk.py"
$SnapshotScript = Join-Path $ProjectDir "scripts\snapshot_rscd_candidate.py"
$RouteDiagnosisScript = Join-Path $ProjectDir "scripts\diagnose_candidate_route.py"
$IntegrityScript = Join-Path $ProjectDir "scripts\verify_candidate_integrity.py"
$SotaGapScript = Join-Path $ProjectDir "scripts\analyze_sota_gap_budget.py"
$NextMechanismScript = Join-Path $ProjectDir "scripts\decide_rscd_next_mechanism.py"
$PromotionAuditScript = Join-Path $ProjectDir "scripts\audit_rscd_candidate_promotion.py"
$ComparisonDir = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715"
$IntegrityManifest = Join-Path $ComparisonDir "S135c_critical_integrity_freeze_20260715.json"
$HandoffLog = Join-Path $S135Dir "handoff_after_s133c.log"

New-Item -ItemType Directory -Force -Path $S135Dir | Out-Null

function Write-Handoff([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $HandoffLog -Value "[$stamp] $Message"
}

function Get-ActiveC3Training {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*train.py*" -and $_.CommandLine -like "*c3_farnet*" }
}

function Get-ActiveS133cTraining {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object {
            $_.CommandLine -like "*train.py*" -and
            $_.CommandLine -like "*c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715.yaml*"
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

function Update-ComparisonSummary {
    Write-Handoff "Updating live RSCD comparison summary."
    & $PythonExe $SummaryScript `
        --include-default-sota `
        --output-dir $ComparisonDir `
        --run "S7_full=E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709" `
        --run "S96_cap250=E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s96_wc_pair_relative_boundary_20260712" `
        --run "S133c_full=E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715" `
        --run "S135c_screen=E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715" `
        --run "S135c_full=E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715" |
        Add-Content -Path $HandoffLog
    Write-Handoff "Live comparison summary updated at $ComparisonDir."
}

function Update-S135ScreenBaselineComparison {
    $screenMetricsPath = Join-Path $S135Dir "test_metrics.json"
    if (!(Test-Path $screenMetricsPath)) {
        Write-Handoff "S135 screen metrics not available; screen-vs-baseline comparison skipped."
        return
    }
    $compareDir = Join-Path $S135Dir "compare_to_s96"
    Write-Handoff "Updating S135 screen vs S96 comparison at $compareDir."
    & $PythonExe $CompareScript `
        --candidate-dir $S135Dir `
        --baseline-dir $BaselineDir `
        --candidate-name "S135c_screen" `
        --baseline-name "S96_screen" `
        --output-dir $compareDir |
        Add-Content -Path $HandoffLog
    Write-Handoff "S135 screen vs S96 comparison updated."
}

function Update-FullRunBaselineComparison([string]$RunDir, [string]$RunName) {
    $metricsPath = Join-Path $RunDir "test_metrics.json"
    if (!(Test-Path $metricsPath)) {
        Write-Handoff "$RunName full metrics not available; full-vs-S7 comparison skipped."
        return
    }
    $compareDir = Join-Path $RunDir "compare_to_s7_full"
    Write-Handoff "Updating $RunName full vs S7 comparison at $compareDir."
    & $PythonExe $CompareScript `
        --candidate-dir $RunDir `
        --baseline-dir $FullBaselineDir `
        --candidate-name $RunName `
        --baseline-name "S7_full" `
        --output-dir $compareDir |
        Add-Content -Path $HandoffLog
    Write-Handoff "$RunName full vs S7 comparison updated."
}

function Update-FairSotaAudit([string]$RunDir, [string]$RunName) {
    $metricsPath = Join-Path $RunDir "test_metrics.json"
    if (!(Test-Path $metricsPath)) {
        Write-Handoff "$RunName metrics not available; fair SOTA audit skipped."
        return
    }
    $auditDir = Join-Path $RunDir "fair_sota_audit"
    Write-Handoff "Updating $RunName fair SOTA audit at $auditDir."
    & $PythonExe $AuditScript `
        --run-dir $RunDir `
        --run-name $RunName `
        --output-dir $auditDir `
        --top-k 15 |
        Add-Content -Path $HandoffLog
    Write-Handoff "$RunName fair SOTA audit updated."
}

function Update-PhysicsCueAnalysis([string]$RunDir, [string]$RunName) {
    $predictionsPath = Join-Path $RunDir "predictions_test.csv"
    if (!(Test-Path $predictionsPath)) {
        Write-Handoff "$RunName predictions_test.csv not available; physics cue analysis skipped."
        return
    }
    $cueDir = Join-Path $RunDir "physics_cue_analysis"
    Write-Handoff "Updating $RunName physics cue analysis at $cueDir."
    & $PythonExe $PhysicsCueScript `
        --predictions $predictionsPath `
        --output-dir $cueDir `
        --max-per-class 300 `
        --image-size 128 `
        --top-confusion-pairs 20 |
        Add-Content -Path $HandoffLog
    Write-Handoff "$RunName physics cue analysis updated."
}

function Update-StableCueEvidence([string]$RunDir, [string]$RunName) {
    $cueDir = Join-Path $RunDir "physics_cue_analysis"
    if (!(Test-Path (Join-Path $cueDir "pair_physics_separability.csv"))) {
        Write-Handoff "$RunName physics cue pair CSV not available; stable cue synthesis skipped."
        return
    }
    $stableDir = Join-Path $ComparisonDir ("stable_physics_cue_evidence_" + $RunName)
    Write-Handoff "Updating stable physics cue evidence including $RunName at $stableDir."
    & $PythonExe $StableCueScript `
        --analysis "S7_full=E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\S7_full_physics_cue_analysis" `
        --analysis "S96_cap250=E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\S96_cap250_physics_cue_analysis" `
        --analysis "$RunName=$cueDir" `
        --output-dir $stableDir `
        --top-k 8 |
        Add-Content -Path $HandoffLog
    Write-Handoff "Stable physics cue evidence including $RunName updated."
}

function Update-SotaGapBudget([string]$RunDir, [string]$RunName) {
    $metricsPath = Join-Path $RunDir "test_metrics.json"
    $classPath = Join-Path $RunDir "per_class_metrics.csv"
    if (!(Test-Path $metricsPath) -or !(Test-Path $classPath)) {
        Write-Handoff "$RunName metrics/per-class files not available; SOTA gap budget skipped."
        return
    }
    $gapDir = Join-Path $RunDir "sota_gap_budget"
    Write-Handoff "Updating $RunName SOTA gap budget at $gapDir."
    & $PythonExe $SotaGapScript `
        --run-dir $RunDir `
        --run-name $RunName `
        --output-dir $gapDir |
        Add-Content -Path $HandoffLog
    Write-Handoff "$RunName SOTA gap budget updated."
}

function Update-NextMechanismDecision([string]$RunDir, [string]$RunName, [string]$BaselineDirText, [string]$BaselineName, [string]$Protocol, [string]$OutputDir) {
    $metricsPath = Join-Path $RunDir "test_metrics.json"
    if (!(Test-Path $metricsPath)) {
        Write-Handoff "$RunName metrics not available; next-mechanism decision skipped."
        return
    }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Handoff "Updating $RunName next-mechanism decision at $OutputDir."
    & $PythonExe $NextMechanismScript `
        --candidate-dir $RunDir `
        --candidate-name $RunName `
        --baseline-dir $BaselineDirText `
        --baseline-name $BaselineName `
        --protocol $Protocol `
        --output-dir $OutputDir |
        Add-Content -Path $HandoffLog
    $nextExit = $LASTEXITCODE
    Write-Handoff "$RunName next-mechanism decision exited with code $nextExit."
}

function Update-StrictPromotionAudit([string]$RunDir, [string]$RunName, [string]$BaselineDirText, [string]$BaselineName, [string]$OutputName, [bool]$RequireSota) {
    $requiredFiles = @(
        (Join-Path $RunDir "test_metrics.json"),
        (Join-Path $RunDir "per_class_metrics.csv"),
        (Join-Path $RunDir "predictions_test.csv"),
        (Join-Path $BaselineDirText "test_metrics.json"),
        (Join-Path $BaselineDirText "per_class_metrics.csv"),
        (Join-Path $BaselineDirText "predictions_test.csv")
    )
    foreach ($path in $requiredFiles) {
        if (!(Test-Path $path)) {
            Write-Handoff "$RunName strict promotion audit skipped; missing required file: $path"
            return
        }
    }
    $auditDir = Join-Path $RunDir $OutputName
    New-Item -ItemType Directory -Force -Path $auditDir | Out-Null
    $auditArgs = @(
        $PromotionAuditScript,
        "--candidate-dir", $RunDir,
        "--baseline-dir", $BaselineDirText,
        "--candidate-name", $RunName,
        "--baseline-name", $BaselineName,
        "--output-dir", $auditDir
    )
    if ($RequireSota) {
        $auditArgs += "--require-sota"
    }
    Write-Handoff "Updating $RunName strict promotion audit at $auditDir require_sota=$RequireSota."
    & $PythonExe $auditArgs |
        Add-Content -Path $HandoffLog
    $auditExit = $LASTEXITCODE
    Write-Handoff "$RunName strict promotion audit exited with code $auditExit."
}

function Invoke-S135Readiness([string]$Mode, [string]$OutputDir, [double]$MinFreeGB) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Handoff "Running S135c readiness audit mode=$Mode output=$OutputDir."
    & $PythonExe $ReadinessScript `
        --screen-config (Join-Path $ProjectDir $S135Config) `
        --full-config (Join-Path $ProjectDir $S135FullConfig) `
        --output-dir $OutputDir `
        --mode $Mode `
        --min-free-gb $MinFreeGB |
        Add-Content -Path $HandoffLog
    $readinessExit = $LASTEXITCODE
    Write-Handoff "S135c readiness audit mode=$Mode exited with code $readinessExit."
    return ($readinessExit -eq 0)
}

function Invoke-S135ActivationPrelaunchAudit([string]$OutputDir) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $predictionsPath = Join-Path $FullBaselineDir "predictions_test.csv"
    if (!(Test-Path $predictionsPath)) {
        Write-Handoff "S135 activation prelaunch audit skipped: baseline predictions not found at $predictionsPath."
        return $false
    }
    Write-Handoff "Running S135c stem activation prelaunch audit at $OutputDir."
    & $PythonExe $StemActivationScript `
        --config (Join-Path $ProjectDir $S135Config) `
        --predictions $predictionsPath `
        --output-dir $OutputDir `
        --max-per-class 80 `
        --batch-size 16 |
        Add-Content -Path $HandoffLog
    $activationExit = $LASTEXITCODE
    Write-Handoff "S135c stem activation prelaunch audit exited with code $activationExit."
    if ($activationExit -ne 0) {
        return $false
    }
    Write-Handoff "Running S135c activation risk summary at $OutputDir."
    & $PythonExe $StemActivationRiskScript `
        --class-summary (Join-Path $OutputDir "s135c_stem_class_activation_summary.csv") `
        --pair-delta (Join-Path $OutputDir "s135c_stem_pair_activation_delta.csv") `
        --output-dir $OutputDir |
        Add-Content -Path $HandoffLog
    $riskExit = $LASTEXITCODE
    Write-Handoff "S135c activation risk summary exited with code $riskExit."
    return ($riskExit -eq 0)
}

function Update-PipelineStatusSnapshot {
    Write-Handoff "Updating live pipeline status snapshot."
    & $PythonExe $StatusScript `
        --output-dir $ComparisonDir |
        Add-Content -Path $HandoffLog
    Write-Handoff "Live pipeline status snapshot updated at $ComparisonDir."
}

function Update-LiveRouteStatusSnapshot {
    $liveDir = Join-Path $ComparisonDir "live_route_status_20260715"
    Write-Handoff "Updating live route status snapshot at $liveDir."
    & $PythonExe $LiveStatusScript `
        --output-dir $liveDir |
        Add-Content -Path $HandoffLog
    Write-Handoff "Live route status snapshot updated at $liveDir."
}

function Invoke-CandidateSnapshot([string]$OutputDir, [string]$StageName) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Write-Handoff "Writing S135c candidate reproducibility snapshot stage=$StageName output=$OutputDir."
    & $PythonExe $SnapshotScript `
        --candidate-name "S135c_water_concrete_contrast_visibility_stem_$StageName" `
        --screen-config (Join-Path $ProjectDir $S135Config) `
        --full-config (Join-Path $ProjectDir $S135FullConfig) `
        --output-dir $OutputDir `
        --run "S7_full=$FullBaselineDir" `
        --run "S96_cap250=$BaselineDir" `
        --run "S133c_full=$S133Dir" `
        --run "S135c_screen=$S135Dir" `
        --run "S135c_full=$S135FullDir" |
        Add-Content -Path $HandoffLog
    $snapshotExit = $LASTEXITCODE
    Write-Handoff "S135c candidate reproducibility snapshot stage=$StageName exited with code $snapshotExit."
    return ($snapshotExit -eq 0)
}

function Update-S135RouteDiagnosis([string]$OutputDir, [string]$StageName) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $decisionJson = Join-Path $S135Dir "screen_promotion_decision.json"
    $comparisonJson = Join-Path (Join-Path $S135Dir "compare_to_s96") "run_comparison.json"
    $sotaGapJson = Join-Path (Join-Path $S135Dir "sota_gap_budget") "sota_gap_budget.json"
    $stableCueJson = "E:\perception_outputs\rscd_surface_classification\comparison_live_20260715\stable_physics_cue_evidence\stable_physics_cue_evidence.json"
    $diagArgs = @(
        "scripts\diagnose_candidate_route.py",
        "--candidate-name", "S135c_$StageName",
        "--candidate-dir", $S135Dir,
        "--baseline-dir", $BaselineDir,
        "--output-dir", $OutputDir,
        "--stable-cue-json", $stableCueJson
    )
    if (Test-Path $decisionJson) {
        $diagArgs += @("--decision-json", $decisionJson)
    }
    if (Test-Path $comparisonJson) {
        $diagArgs += @("--comparison-json", $comparisonJson)
    }
    if (Test-Path $sotaGapJson) {
        $diagArgs += @("--sota-gap-json", $sotaGapJson)
    }
    Write-Handoff "Updating S135c route diagnosis stage=$StageName at $OutputDir."
    & $PythonExe $diagArgs |
        Add-Content -Path $HandoffLog
    $diagnosisExit = $LASTEXITCODE
    Write-Handoff "S135c route diagnosis stage=$StageName exited with code $diagnosisExit."
    return ($diagnosisExit -eq 0)
}

function Invoke-S135IntegrityCheck([string]$OutputDir, [string]$StageName) {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    if (!(Test-Path $IntegrityManifest)) {
        Write-Handoff "S135c integrity manifest missing at $IntegrityManifest. $StageName launch blocked."
        return $false
    }
    Write-Handoff "Running S135c critical integrity check stage=$StageName output=$OutputDir."
    & $PythonExe $IntegrityScript `
        --mode "check" `
        --candidate-name "S135c_water_concrete_contrast_visibility_stem_$StageName" `
        --manifest $IntegrityManifest `
        --output-dir $OutputDir |
        Add-Content -Path $HandoffLog
    $integrityExit = $LASTEXITCODE
    Write-Handoff "S135c critical integrity check stage=$StageName exited with code $integrityExit."
    return ($integrityExit -eq 0)
}

Write-Handoff "Watcher started. Waiting for S133c metrics or active S133c training."

$waitCycles = 0
$missingS133cCycles = 0
$metricsPath = Join-Path $S133Dir "test_metrics.json"
while (!(Test-Path $metricsPath)) {
    $activeS133c = Get-ActiveS133cTraining
    if (!$activeS133c) {
        $missingS133cCycles += 1
        Write-Handoff "S133c has no active training process and no test_metrics.json. Waiting for recovery watcher. missing_cycles=$missingS133cCycles"
        if (($missingS133cCycles % 10) -eq 0) {
            Update-ComparisonSummary
        }
        Start-Sleep -Seconds 60
        continue
    }
    $missingS133cCycles = 0
    Start-Sleep -Seconds 60
    $waitCycles += 1
    if (($waitCycles % 10) -eq 0) {
        Update-PipelineStatusSnapshot
    }
}

Write-Handoff "S133c test_metrics.json detected."

if (Test-Path $metricsPath) {
    Write-Handoff "S133c test_metrics.json found:"
    Get-Content $metricsPath -Raw | Add-Content -Path $HandoffLog
} else {
    Write-Handoff "S133c test_metrics.json not found after process exit."
    Update-ComparisonSummary
    Write-Handoff "Stopping handoff because S133c formal full run did not produce test_metrics.json."
    exit 0
}
Update-ComparisonSummary
Update-FullRunBaselineComparison $S133Dir "S133c_full"
Update-FairSotaAudit $S133Dir "S133c_full"
Update-PhysicsCueAnalysis $S133Dir "S133c_full"
Update-StableCueEvidence $S133Dir "S133c_full"
Update-SotaGapBudget $S133Dir "S133c_full"
Update-NextMechanismDecision $S133Dir "S133c_full" $FullBaselineDir "S7_full" "full" (Join-Path $S133Dir "next_mechanism_decision")
Update-StrictPromotionAudit $S133Dir "S133c_full" $FullBaselineDir "S7_full" "strict_promotion_audit_vs_s7_full" $true
Update-LiveRouteStatusSnapshot
Invoke-CandidateSnapshot (Join-Path $ComparisonDir "S135c_reproducibility_snapshot_after_s133c") "after_s133c" | Out-Null

$s133cStrictAuditPath = Join-Path (Join-Path $S133Dir "strict_promotion_audit_vs_s7_full") "promotion_audit.json"
if (Test-Path $s133cStrictAuditPath) {
    $s133cStrictAudit = Get-Content $s133cStrictAuditPath -Raw | ConvertFrom-Json
    if ([bool]$s133cStrictAudit.decision.passed) {
        Write-Handoff "S133c full passed strict SOTA audit. Stopping downstream S135 launch."
        exit 0
    }
    Write-Handoff "S133c full did not pass strict SOTA audit. Continuing to S135 screen gate."
} else {
    Write-Handoff "S133c strict promotion audit JSON missing after audit; continuing conservatively to S135 screen gate."
}

$alreadyDone = Test-Path (Join-Path $S135Dir "test_metrics.json")
if ($alreadyDone) {
    Write-Handoff "S135 screen already has test_metrics.json. Not starting a duplicate screen run."
} else {
    $activeTraining = Get-ActiveC3Training
    if ($activeTraining) {
        Write-Handoff "Another C3 training process is active. Not starting S135 screen."
        $activeTraining | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
        exit 0
    }
    if (-not (Test-FreeSpaceGB $S135Dir 2.0)) {
        Write-Handoff "Not enough free disk space for S135 screen. Screen run not started."
        exit 0
    }
    if (-not (Invoke-S135Readiness "queue" (Join-Path $S135Dir "s135c_queue_readiness") 2.0)) {
        Write-Handoff "S135 queue readiness failed. Screen run not started."
        exit 0
    }
    if (-not (Invoke-S135ActivationPrelaunchAudit (Join-Path $S135Dir "s135c_prelaunch_activation_audit"))) {
        Write-Handoff "S135 activation prelaunch audit failed. Screen run not started."
        exit 0
    }
    if (-not (Invoke-S135IntegrityCheck (Join-Path $S135Dir "critical_integrity_pre_screen") "pre_screen")) {
        Write-Handoff "S135 critical integrity check failed. Screen run not started."
        exit 0
    }
    Invoke-CandidateSnapshot (Join-Path $S135Dir "reproducibility_snapshot_pre_screen") "pre_screen" | Out-Null

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $S135Dir "train_stdout_$stamp.log"
    $stderr = Join-Path $S135Dir "train_stderr_$stamp.log"
    Write-Handoff "Starting S135 screen run: $S135Config"
    $screenProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList @("-u", "train.py", "--config", $S135Config) `
        -WorkingDirectory $ProjectDir `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Write-Handoff "S135 screen started. pid=$($screenProcess.Id) stdout=$stdout stderr=$stderr"
    Wait-Process -Id $screenProcess.Id
    Write-Handoff "S135 screen process exited. pid=$($screenProcess.Id)"
}

$screenMetricsPath = Join-Path $S135Dir "test_metrics.json"
if (!(Test-Path $screenMetricsPath)) {
    Write-Handoff "S135 screen finished without test_metrics.json. No promotion decision."
    exit 0
}

Update-S135ScreenBaselineComparison
Update-FairSotaAudit $S135Dir "S135c_screen"
Update-PhysicsCueAnalysis $S135Dir "S135c_screen"
Update-StableCueEvidence $S135Dir "S135c_screen"
Update-SotaGapBudget $S135Dir "S135c_screen"
Update-NextMechanismDecision $S135Dir "S135c_screen" $BaselineDir "S96_screen" "screen" (Join-Path $S135Dir "next_mechanism_decision")
Update-StrictPromotionAudit $S135Dir "S135c_screen" $BaselineDir "S96_screen" "strict_screen_promotion_audit_vs_s96" $false
Update-LiveRouteStatusSnapshot
Invoke-CandidateSnapshot (Join-Path $S135Dir "reproducibility_snapshot_post_screen") "post_screen" | Out-Null
Update-S135RouteDiagnosis (Join-Path $S135Dir "route_diagnosis_pre_decision") "pre_decision" | Out-Null
Write-Handoff "Running S135 screen promotion decision."
& $PythonExe $DecisionScript `
    --candidate-dir $S135Dir `
    --baseline-dir $BaselineDir `
    --output-dir $S135Dir `
    --candidate-name "S135c" `
    --baseline-name "S96"
$decisionExit = $LASTEXITCODE
Write-Handoff "Promotion decision script exited with code $decisionExit."
Update-ComparisonSummary
Update-S135RouteDiagnosis (Join-Path $S135Dir "route_diagnosis_post_decision") "post_decision" | Out-Null
$decisionPath = Join-Path $S135Dir "screen_promotion_decision.json"
if (!(Test-Path $decisionPath)) {
    Write-Handoff "Promotion decision JSON not found. Not starting full run."
    exit 0
}
$decision = Get-Content $decisionPath -Raw | ConvertFrom-Json
if (-not [bool]$decision.promote_to_full) {
    Update-S135RouteDiagnosis (Join-Path $S135Dir "route_diagnosis_failed_screen") "failed_screen" | Out-Null
    Write-Handoff "S135 screen did not pass promotion checks. Full run not started."
    exit 0
}

New-Item -ItemType Directory -Force -Path $S135FullDir | Out-Null
if (Test-Path (Join-Path $S135FullDir "test_metrics.json")) {
    Write-Handoff "S135 full already has test_metrics.json. Not starting duplicate full run."
    exit 0
}
$activeTraining = Get-ActiveC3Training
if ($activeTraining) {
    Write-Handoff "Another C3 training process is active after screen. Not starting S135 full."
    $activeTraining | Select-Object ProcessId,CommandLine | Out-String | Add-Content -Path $HandoffLog
    exit 0
}
if (-not (Test-FreeSpaceGB $S135FullDir 4.0)) {
    Write-Handoff "Not enough free disk space for S135 full run. Full run not started."
    exit 0
}
if (-not (Invoke-S135Readiness "full" (Join-Path $S135FullDir "s135c_full_readiness") 4.0)) {
    Write-Handoff "S135 full readiness failed. Full run not started."
    exit 0
}
if (-not (Invoke-S135IntegrityCheck (Join-Path $S135FullDir "critical_integrity_pre_full") "pre_full")) {
    Write-Handoff "S135 critical integrity check failed before full. Full run not started."
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$fullStdout = Join-Path $S135FullDir "train_stdout_$stamp.log"
$fullStderr = Join-Path $S135FullDir "train_stderr_$stamp.log"
Write-Handoff "S135 screen passed. Starting full run: $S135FullConfig"
$fullProcess = Start-Process -FilePath $PythonExe `
    -ArgumentList @("-u", "train.py", "--config", $S135FullConfig) `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $fullStdout `
    -RedirectStandardError $fullStderr `
    -WindowStyle Hidden `
    -PassThru
Write-Handoff "S135 full started. pid=$($fullProcess.Id) stdout=$fullStdout stderr=$fullStderr"
Update-ComparisonSummary
Wait-Process -Id $fullProcess.Id
Write-Handoff "S135 full process exited. pid=$($fullProcess.Id)"
if (Test-Path (Join-Path $S135FullDir "test_metrics.json")) {
    Write-Handoff "S135 full test_metrics.json found after full process exit."
} else {
    Write-Handoff "S135 full exited without test_metrics.json."
}
Update-ComparisonSummary
Update-FullRunBaselineComparison $S135FullDir "S135c_full"
Update-FairSotaAudit $S135FullDir "S135c_full"
Update-PhysicsCueAnalysis $S135FullDir "S135c_full"
Update-StableCueEvidence $S135FullDir "S135c_full"
Update-SotaGapBudget $S135FullDir "S135c_full"
Update-NextMechanismDecision $S135FullDir "S135c_full" $FullBaselineDir "S7_full" "full" (Join-Path $S135FullDir "next_mechanism_decision")
Update-StrictPromotionAudit $S135FullDir "S135c_full" $FullBaselineDir "S7_full" "strict_promotion_audit_vs_s7_full" $true
Update-LiveRouteStatusSnapshot
Invoke-CandidateSnapshot (Join-Path $S135FullDir "reproducibility_snapshot_post_full") "post_full" | Out-Null
