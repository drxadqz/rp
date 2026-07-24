param(
    [string] $CurrentRun = "lodo_rscd_full_faf",
    [string] $PriorityRun = "lodo_roadsaw_full_faf",
    [string] $Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol",
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $LogDir = "outputs\paper_protocol_queue",
    [int] $PollSeconds = 60,
    [int] $TimeoutHours = 48,
    [int] $PostCurrentGraceSeconds = 180,
    [ValidateSet("p0", "ablation", "lodo", "single", "baselines", "candidates", "final_lodo", "final_single", "final", "all")]
    [string] $FollowUpPhase = "all",
    [int] $FollowUpPriorityWatcherGraceSeconds = 120,
    [switch] $RunFastScreenBeforeFollowUp,
    [switch] $LeanFirstWaveFastScreen
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$ResolvedLogDir = if ([System.IO.Path]::IsPathRooted($LogDir)) { $LogDir } else { Join-Path $ProjectRoot $LogDir }
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WatchLog = Join-Path $ResolvedLogDir "roadsaw_priority_after_$CurrentRun`_$Stamp.log"
$QueueOut = Join-Path $ResolvedLogDir "roadsaw_priority_after_$CurrentRun`_$Stamp.out.log"
$QueueErr = Join-Path $ResolvedLogDir "roadsaw_priority_after_$CurrentRun`_$Stamp.err.log"
$CurrentCompleteAt = $null

function Write-WatchLog {
    param([string] $Message)
    "$(Get-Date -Format s) $Message" | Out-File -FilePath $WatchLog -Append -Encoding utf8
}

function Test-RunComplete {
    param([string] $RunName)
    $RunDir = Join-Path $Root $RunName
    $Required = @(
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json"
    )
    foreach ($Name in $Required) {
        if (-not (Test-Path (Join-Path $RunDir $Name))) {
            return $false
        }
    }
    return $true
}

function Get-TrackedProcesses {
    return Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like '*run_paper_protocol_direct.py*' -or
            $_.CommandLine -like '*scripts\train.py*' -or
            $_.CommandLine -like '*scripts/train.py*'
        }
}

function Get-QueueProcesses {
    return @(Get-TrackedProcesses | Where-Object { $_.CommandLine -like '*run_paper_protocol_direct.py*' })
}

function Get-TrainProcesses {
    return @(Get-TrackedProcesses | Where-Object { $_.CommandLine -like '*scripts\train.py*' -or $_.CommandLine -like '*scripts/train.py*' })
}

function Get-RunNameFromCommand {
    param([string] $CommandLine)
    if ($CommandLine -match 'configs[\\/]+experiments[\\/]+paper_protocol[\\/]+([^\\/"''\s]+)\.yaml') {
        return $Matches[1]
    }
    return ""
}

function Stop-ProcessTree {
    param([int] $RootPid)
    $Children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootPid })
    foreach ($Child in $Children) {
        Stop-ProcessTree -RootPid ([int] $Child.ProcessId)
    }
    try {
        Write-WatchLog "stopping pid=$RootPid"
        Stop-Process -Id $RootPid -Force -ErrorAction Stop
    } catch {
        Write-WatchLog "stop skipped for pid=${RootPid}: $($_.Exception.Message)"
    }
}

function Stop-QueueAndTrain {
    $QueueProcesses = Get-QueueProcesses
    $TrainProcesses = Get-TrainProcesses
    foreach ($Proc in $QueueProcesses) {
        Stop-ProcessTree -RootPid ([int] $Proc.ProcessId)
    }
    foreach ($Proc in $TrainProcesses) {
        try {
            Write-WatchLog "stopping orphan train pid=$($Proc.ProcessId)"
            Stop-Process -Id ([int] $Proc.ProcessId) -Force -ErrorAction Stop
        } catch {
            Write-WatchLog "orphan train stop skipped pid=$($Proc.ProcessId): $($_.Exception.Message)"
        }
    }
}

function Start-PriorityLodoQueue {
    Write-WatchLog "starting priority LODO queue for $PriorityRun"
    $Args = @(
        (Join-Path $ProjectRoot "scripts\run_paper_protocol_direct.py"),
        "--phase",
        "lodo",
        "--python",
        $Python,
        "--root",
        $Root,
        "--log-dir",
        $ResolvedLogDir,
        "--postprocess-each"
    )
    $Proc = Start-Process -FilePath $Python `
        -ArgumentList $Args `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $QueueOut `
        -RedirectStandardError $QueueErr `
        -PassThru
    Write-WatchLog "priority queue launched pid=$($Proc.Id) out=$QueueOut err=$QueueErr"
    Start-FollowUpWatcher -WaitPid ([int] $Proc.Id)
}

function Start-FollowUpWatcher {
    param([int] $WaitPid)
    $FollowArgs = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (Join-Path $ProjectRoot "scripts\run_paper_protocol_after_pid.ps1"),
        "-WaitPid",
        $WaitPid,
        "-Phase",
        $FollowUpPhase,
        "-Python",
        $Python,
        "-Root",
        $Root,
        "-LogDir",
        $ResolvedLogDir,
        "-PriorityWatcherGraceSeconds",
        $FollowUpPriorityWatcherGraceSeconds,
        "-PostprocessEach"
    )
    if ($RunFastScreenBeforeFollowUp) {
        $FollowArgs += "-RunFastScreenBeforeFollowUp"
    }
    if ($LeanFirstWaveFastScreen) {
        $FollowArgs += "-LeanFirstWaveFastScreen"
    }
    $Follow = Start-Process -FilePath "powershell.exe" `
        -ArgumentList $FollowArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -PassThru
    Write-WatchLog "follow-up watcher launched pid=$($Follow.Id) wait_pid=$WaitPid phase=$FollowUpPhase fast_screen_before_followup=$RunFastScreenBeforeFollowUp"
}

$Deadline = (Get-Date).AddHours($TimeoutHours)
Write-WatchLog "watching current=$CurrentRun priority=$PriorityRun root=$Root"

while ((Get-Date) -lt $Deadline) {
    if (Test-RunComplete -RunName $PriorityRun) {
        Write-WatchLog "$PriorityRun complete; watcher exiting"
        exit 0
    }

    $CurrentComplete = Test-RunComplete -RunName $CurrentRun
    $TrainProcesses = Get-TrainProcesses
    $TrainRuns = @()
    foreach ($Proc in $TrainProcesses) {
        $RunName = Get-RunNameFromCommand -CommandLine $Proc.CommandLine
        if ($RunName) {
            $TrainRuns += [PSCustomObject]@{
                Pid = [int] $Proc.ProcessId
                Run = $RunName
            }
        }
    }
    $TrainText = ($TrainRuns | ForEach-Object { "$($_.Run):$($_.Pid)" }) -join ","

    if (-not $CurrentComplete) {
        Write-WatchLog "$CurrentRun incomplete; active_train=[$TrainText]"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($null -eq $CurrentCompleteAt) {
        $CurrentCompleteAt = Get-Date
        Write-WatchLog "$CurrentRun complete; entering grace period before enforcing $PriorityRun"
    }

    if ($TrainRuns | Where-Object { $_.Run -eq $PriorityRun }) {
        Write-WatchLog "$PriorityRun is already running; watcher exiting"
        exit 0
    }

    $NonPriorityRuns = @($TrainRuns | Where-Object { $_.Run -ne $PriorityRun })
    if ($NonPriorityRuns.Count -gt 0) {
        $RunList = ($NonPriorityRuns | ForEach-Object { "$($_.Run):$($_.Pid)" }) -join ","
        Write-WatchLog "non-priority train detected after $CurrentRun completion: [$RunList]; taking over for $PriorityRun"
        Stop-QueueAndTrain
        Start-Sleep -Seconds 10
        Start-PriorityLodoQueue
        exit 0
    }

    $Elapsed = ((Get-Date) - $CurrentCompleteAt).TotalSeconds
    if ($Elapsed -ge $PostCurrentGraceSeconds) {
        Write-WatchLog "no train active for ${Elapsed}s after $CurrentRun completion; taking over for $PriorityRun"
        Stop-QueueAndTrain
        Start-Sleep -Seconds 10
        Start-PriorityLodoQueue
        exit 0
    }

    Write-WatchLog "$CurrentRun complete; waiting grace ${Elapsed}s active_train=[$TrainText]"
    Start-Sleep -Seconds $PollSeconds
}

Write-WatchLog "timeout after $TimeoutHours hours"
exit 2
