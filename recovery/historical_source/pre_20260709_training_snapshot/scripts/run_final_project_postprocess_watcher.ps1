param(
    [int[]]$WaitPids = @(26608, 10532, 11312, 16888, 18688, 31568, 31476, 32208, 1164, 21644, 33592, 31884, 10564, 34332, 9000, 30484, 12532, 9688, 23104, 26308, 25196, 5928, 36248, 29868, 26796, 34236, 4260, 14144, 8304, 27364, 32116, 9664, 27304, 25472)
)

$ErrorActionPreference = "Stop"
$Repo = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$LogDir = Join-Path $Repo "outputs\rscd_surface_formal_queue"
$Log = Join-Path $LogDir "final_project_postprocess_watcher.log"

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

function Wait-ClassificationJobsGone() {
    while ($true) {
        $alive = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -match "run_rscd_surface_classification.py" -and
            $_.CommandLine -notmatch "smoke_"
        })
        if ($alive.Count -eq 0) {
            break
        }
        Write-QueueLog "waiting for classification jobs: $($alive.ProcessId -join ', ')"
        Start-Sleep -Seconds 300
    }
}

function Run-Report([string[]]$Args) {
    Write-QueueLog "running $($Args -join ' ')"
    & $Python @Args *>> $Log
}

Write-QueueLog "started final project postprocess watcher"
Wait-ProcessIds -Ids $WaitPids -Name "all queued RSCD/direct experiments"
Wait-ClassificationJobsGone

Run-Report @("scripts\write_rscd_training_trend_report.py")
Run-Report @("scripts\compare_rscd_surface_candidates.py")
Run-Report @("scripts\compare_rscd_class_slices.py")
Run-Report @("scripts\select_rscd_promotion_candidate.py")
Run-Report @("scripts\select_rscd_hard_condition_promotion.py")
Run-Report @("scripts\select_rscd_residual_adapter_promotion.py")
Run-Report @("scripts\select_rscd_texture_film_promotion.py")
Run-Report @("scripts\write_rscd_formal_validation_diagnosis.py")
Run-Report @("scripts\write_rscd_formal_result_summary.py")
Run-Report @("scripts\write_rscd_external_sota_gap.py")

if (Test-Path "scripts\write_direct_visual_friction_report.py") {
    Run-Report @("scripts\write_direct_visual_friction_report.py")
}

Run-Report @("scripts\write_rscd_decision_dashboard.py")
Run-Report @("scripts\write_goal_completion_audit.py")

Write-QueueLog "final project postprocess complete"
