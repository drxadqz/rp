param(
    [string]$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string]$Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol",
    [string]$SummaryDir = "reports\paper_protocol_summary",
    [string]$LogDir = "outputs\paper_protocol_queue",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $Root "baseline_single_roadsc_global_convnext"
$audit = Join-Path $runDir "topvenue_result_audit.json"
$log = Join-Path $LogDir ("roadsc_word_refresh_watcher_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
"started $(Get-Date -Format s)" | Out-File -LiteralPath $log -Encoding utf8

while ($true) {
    if (Test-Path -LiteralPath $audit) {
        "detected RoadSC audit artifact $(Get-Date -Format s)" | Out-File -LiteralPath $log -Encoding utf8 -Append
        & $Python scripts\write_queue_recovery_report.py `
            --root $Root `
            --summary-dir $SummaryDir `
            --log-dir $LogDir `
            --out-md (Join-Path $SummaryDir "queue_recovery_report.md") `
            --out-json (Join-Path $SummaryDir "queue_recovery_report.json") *>> $log
        & $Python scripts\make_weekly_progress_word.py *>> $log
        "refreshed Word $(Get-Date -Format s)" | Out-File -LiteralPath $log -Encoding utf8 -Append
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
