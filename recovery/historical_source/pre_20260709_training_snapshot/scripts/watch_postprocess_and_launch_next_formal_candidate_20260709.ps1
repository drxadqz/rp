param(
    [string]$RunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709",
    [string]$RunName = "Formal Full-Manifest S7",
    [string]$OutDir = "reports\paper_protocol_summary\postprocess_s7_20260709",
    [string]$NextId = "S12",
    [string]$NextRunName = "Formal Full-Manifest S12 factor-graph pair sampler",
    [string]$NextRunDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_factor_graph_metric_pair_sampler_s12_20260709",
    [string]$NextScript = "scripts\run_c3_formal_fullmanifest_s12_factor_graph_pair_sampler_20260709.ps1",
    [string]$NextPostprocessDir = "reports\paper_protocol_summary\postprocess_s12_factor_graph_pair_sampler_20260709",
    [int]$PollSeconds = 300,
    [int]$MaxHours = 48,
    [int]$GpuMemoryFreeThresholdMb = 1200,
    [switch]$DryRun,
    [switch]$CheckOnce
)

$ErrorActionPreference = "Stop"

$Root = "E:\perception\friction_affordance_field"
$Postprocess = Join-Path $Root "scripts\postprocess_c3_formal_candidate_20260709.ps1"
$NextScriptAbs = Join-Path $Root $NextScript
$Watcher = Join-Path $Root "scripts\watch_and_postprocess_formal_candidate_20260709.ps1"
$LogDir = Join-Path $Root "reports\paper_protocol_summary\auto_chain_s7_to_s12_20260709"
$Log = Join-Path $LogDir "auto_chain.log"
$Deadline = (Get-Date).AddHours($MaxHours)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-ChainLog {
    param([string]$Message)
    $line = "$(Get-Date -Format o) $Message"
    Write-Output $line
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Test-RunComplete {
    param([string]$Dir)
    return (
        (Test-Path -LiteralPath (Join-Path $Dir "metrics.json")) -and
        (Test-Path -LiteralPath (Join-Path $Dir "per_class_metrics.csv")) -and
        (Test-Path -LiteralPath (Join-Path $Dir "confusion_matrix.csv"))
    )
}

function Get-PromotionStatusPath {
    param([string]$PostDir)
    return (Join-Path (Join-Path $Root $PostDir) "promotion_gate\promotion_gate_status.json")
}

function Get-PromotionDecision {
    param([string]$StatusPath)
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        return $null
    }
    $data = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
    return $data.decision
}

function Get-ActiveTrainingProcessCount {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        ($_.CommandLine -like '*train.py --config*' -or $_.CommandLine -like '*train.py*--config*')
    }
    return @($procs).Count
}

function Get-GpuMemoryUsedMb {
    try {
        $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return $null
        }
        return [int](([string]$raw).Trim().Split("`n")[0].Trim())
    }
    catch {
        return $null
    }
}

function Test-NextAlreadyStarted {
    return (
        (Test-Path -LiteralPath (Join-Path $NextRunDir "history.json")) -or
        (Test-Path -LiteralPath (Join-Path $NextRunDir "metrics.json")) -or
        ((Get-ChildItem -LiteralPath $NextRunDir -Filter "train_stderr_*.log" -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
    )
}

function Invoke-PostprocessIfNeeded {
    $statusPath = Get-PromotionStatusPath -PostDir $OutDir
    if (Test-Path -LiteralPath $statusPath) {
        Write-ChainLog "postprocess_already_present status=$statusPath"
        return
    }
    Write-ChainLog "run_complete_detected_waiting_for_existing_watcher"
    Start-Sleep -Seconds 180
    if (Test-Path -LiteralPath $statusPath) {
        Write-ChainLog "postprocess_created_by_existing_watcher status=$statusPath"
        return
    }
    if ($DryRun) {
        Write-ChainLog "dry_run_would_postprocess run_dir=$RunDir out_dir=$OutDir"
        return
    }
    Write-ChainLog "postprocess_start run_dir=$RunDir out_dir=$OutDir"
    Push-Location $Root
    try {
        powershell -NoProfile -ExecutionPolicy Bypass -File $Postprocess -RunDir $RunDir -RunName $RunName -OutDir $OutDir
    }
    finally {
        Pop-Location
    }
    Write-ChainLog "postprocess_done"
}

function Invoke-NextCandidateIfNeeded {
    $statusPath = Get-PromotionStatusPath -PostDir $OutDir
    $decision = Get-PromotionDecision -StatusPath $statusPath
    if (-not $decision) {
        Write-ChainLog "promotion_decision_missing status=$statusPath"
        return
    }
    Write-ChainLog "promotion_decision=$decision"
    if ($decision -eq "sota_candidate_run_exact_pass_audit") {
        Write-ChainLog "external_sota_gate_passed_no_next_launch"
        return
    }
    if (Test-NextAlreadyStarted) {
        Write-ChainLog "next_already_started_or_complete next_id=$NextId dir=$NextRunDir"
        return
    }
    $active = Get-ActiveTrainingProcessCount
    $gpuUsed = Get-GpuMemoryUsedMb
    if ($active -gt 0) {
        Write-ChainLog "next_launch_deferred_active_training_processes=$active"
        return
    }
    if ($gpuUsed -ne $null -and $gpuUsed -gt (4096 - $GpuMemoryFreeThresholdMb)) {
        Write-ChainLog "next_launch_deferred_gpu_used_mb=$gpuUsed threshold_free_mb=$GpuMemoryFreeThresholdMb"
        return
    }
    if ($DryRun) {
        Write-ChainLog "dry_run_would_launch_next next_id=$NextId script=$NextScriptAbs"
        return
    }

    Write-ChainLog "launch_next_start next_id=$NextId script=$NextScriptAbs"
    $proc = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $NextScriptAbs) `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Hidden
    Write-ChainLog "launch_next_pid=$($proc.Id)"

    $watchProc = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Watcher,
            "-RunDir", $NextRunDir,
            "-RunName", $NextRunName,
            "-OutDir", $NextPostprocessDir,
            "-PollSeconds", "300",
            "-MaxHours", "36"
        ) `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Hidden
    Write-ChainLog "launch_next_watcher_pid=$($watchProc.Id)"
}

Write-ChainLog "auto_chain_start run=$RunName next=$NextId dry_run=$DryRun check_once=$CheckOnce"

while ((Get-Date) -lt $Deadline) {
    $complete = Test-RunComplete -Dir $RunDir
    $active = Get-ActiveTrainingProcessCount
    $gpuUsed = Get-GpuMemoryUsedMb
    Write-ChainLog "poll complete=$complete active_train=$active gpu_used_mb=$gpuUsed"
    if ($complete) {
        Invoke-PostprocessIfNeeded
        Invoke-NextCandidateIfNeeded
        Write-ChainLog "auto_chain_done"
        exit 0
    }
    if ($CheckOnce) {
        Write-ChainLog "check_once_exit"
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-ChainLog "auto_chain_timeout"
exit 2
