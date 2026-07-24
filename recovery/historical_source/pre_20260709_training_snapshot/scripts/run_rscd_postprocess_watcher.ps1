param(
    [int[]]$WaitPids = @(26608, 10532, 11312, 16888, 18688, 31568, 31476)
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "rscd_postprocess_watcher.log"

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

function Run-Report([string]$Name, [string[]]$ArgsList) {
    Write-QueueLog "running $Name"
    & $Python @ArgsList *>> $Log
    Write-QueueLog "finished $Name"
}

Write-QueueLog "started RSCD postprocess watcher"
Wait-ProcessIds -Ids $WaitPids -Name "RSCD training/fast/promotion queue"

Run-Report "write_rscd_training_trend_report" @("scripts\write_rscd_training_trend_report.py")
Run-Report "write_rscd_formal_validation_diagnosis" @("scripts\write_rscd_formal_validation_diagnosis.py")
Run-Report "compare_rscd_surface_candidates" @("scripts\compare_rscd_surface_candidates.py")
Run-Report "compare_rscd_class_slices" @("scripts\compare_rscd_class_slices.py")
Run-Report "select_rscd_promotion_candidate" @("scripts\select_rscd_promotion_candidate.py")
Run-Report "select_rscd_hard_condition_promotion" @("scripts\select_rscd_hard_condition_promotion.py")
Run-Report "write_rscd_formal_result_summary" @("scripts\write_rscd_formal_result_summary.py")
Run-Report "write_rscd_external_sota_gap" @("scripts\write_rscd_external_sota_gap.py")
Run-Report "write_rscd_decision_dashboard" @("scripts\write_rscd_decision_dashboard.py")
Run-Report "write_goal_completion_audit" @("scripts\write_goal_completion_audit.py")

Write-QueueLog "RSCD postprocess complete"
