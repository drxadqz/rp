param(
    [string]$FormalRun = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification\formal_physics_wavelet_directional_film_gate_hier",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "post_formal_report_refresh.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-RefreshLog([string]$Text) {
    "[$(Get-Date -Format s)] $Text" | Out-File -FilePath $Log -Encoding utf8 -Append
}

$Result = Join-Path $FormalRun "evaluate_test.json"
Write-RefreshLog "started post-formal report refresh watcher"
while (-not (Test-Path $Result)) {
    Write-RefreshLog "waiting for formal result: $Result"
    Start-Sleep -Seconds $PollSeconds
}

Write-RefreshLog "formal result found; refreshing reports"
$commands = @(
    "scripts\compare_rscd_surface_candidates.py",
    "scripts\compare_rscd_class_slices.py",
    "scripts\write_rscd_formal_result_summary.py",
    "scripts\write_rscd_training_trend_report.py",
    "scripts\write_experiment_queue_health_report.py",
    "scripts\write_rscd_decision_dashboard.py",
    "scripts\select_final_rscd_method.py",
    "scripts\write_rscd_pretraining_protocol_audit.py",
    "scripts\write_goal_completion_audit.py"
)

foreach ($cmd in $commands) {
    Write-RefreshLog "running $cmd"
    & $Python $cmd *>> $Log
}
Write-RefreshLog "post-formal report refresh complete"
