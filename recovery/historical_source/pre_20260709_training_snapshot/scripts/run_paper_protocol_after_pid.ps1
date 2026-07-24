param(
    [Parameter(Mandatory = $true)]
    [int] $WaitPid,
    [ValidateSet("p0", "ablation", "lodo", "single", "baselines", "candidates", "final_lodo", "final_single", "final", "all")]
    [string] $Phase = "all",
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol",
    [switch] $PostprocessEach,
    [string] $LogDir = "outputs\paper_protocol_queue",
    [int] $PriorityWatcherGraceSeconds = 90,
    [switch] $RunFastScreenBeforeFollowUp,
    [switch] $LeanFirstWaveFastScreen,
    [ValidateSet("candidates", "roadsaw", "all")]
    [string] $FastScreenScope = "candidates"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$ResolvedLogDir = if ([System.IO.Path]::IsPathRooted($LogDir)) { $LogDir } else { Join-Path $ProjectRoot $LogDir }
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$WatcherLog = Join-Path $ResolvedLogDir "paper_protocol_after_pid_$WaitPid`_$Stamp.log"
$QueueOut = Join-Path $ResolvedLogDir "paper_protocol_after_pid_$WaitPid`_$Stamp.out.log"
$QueueErr = Join-Path $ResolvedLogDir "paper_protocol_after_pid_$WaitPid`_$Stamp.err.log"

"$(Get-Date -Format s) waiting for pid=$WaitPid" | Out-File -FilePath $WatcherLog -Encoding utf8

function Write-WatcherLog {
    param([string] $Message)
    "$(Get-Date -Format s) $Message" | Out-File -FilePath $WatcherLog -Append -Encoding utf8
}

function Get-ActiveQueueCount {
    $Processes = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like '*run_paper_protocol_direct.py*' -and
            $_.ProcessId -ne $PID
        }
    return @($Processes).Count
}

function Get-PriorityWatcherCount {
    $Processes = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like '*watch_roadsaw_priority_after_current_lodo.ps1*' -and
            $_.ProcessId -ne $PID
        }
    return @($Processes).Count
}

function Get-QueueRecovery {
    $SummaryDir = Join-Path $ProjectRoot "reports\paper_protocol_summary"
    New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null
    $RecoveryJson = Join-Path $SummaryDir "queue_recovery_report.json"
    $RecoveryMd = Join-Path $SummaryDir "queue_recovery_report.md"
    $Args = @(
        (Join-Path $ProjectRoot "scripts\write_queue_recovery_report.py"),
        "--root",
        $Root,
        "--summary-dir",
        $SummaryDir,
        "--log-dir",
        $ResolvedLogDir,
        "--out-md",
        $RecoveryMd,
        "--out-json",
        $RecoveryJson
    )
    $Proc = Start-Process -FilePath $Python `
        -ArgumentList $Args `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($Proc.ExitCode -ne 0 -or -not (Test-Path $RecoveryJson)) {
        Write-WatcherLog "queue recovery refresh failed exit_code=$($Proc.ExitCode); follow-up will run defensively"
        return $null
    }
    return Get-Content -LiteralPath $RecoveryJson -Raw | ConvertFrom-Json
}

function Test-HasIncompleteWork {
    param($Report)
    if ($null -eq $Report) {
        return $true
    }
    return ([int] $Report.num_missing -gt 0 -or [int] $Report.num_partial -gt 0)
}

function Invoke-FastScreenBeforeFollowUp {
    if (-not $RunFastScreenBeforeFollowUp) {
        return
    }

    $FastLogDir = Join-Path $ProjectRoot "outputs\fast_screen_queue"
    New-Item -ItemType Directory -Force -Path $FastLogDir | Out-Null
    $FastOut = Join-Path $FastLogDir "fast_screen_before_followup_$Stamp.out.log"
    $FastErr = Join-Path $FastLogDir "fast_screen_before_followup_$Stamp.err.log"
    Write-WatcherLog "running fast-screen before follow-up scope=$FastScreenScope"

    $Args = @(
        (Join-Path $ProjectRoot "scripts\run_fast_screen_protocol.py"),
        "--scope",
        $FastScreenScope,
        "--python",
        $Python,
        "--log-dir",
        $FastLogDir
    )
    if ($LeanFirstWaveFastScreen) {
        $Args += "--lean-first-wave"
    }

    $Proc = Start-Process -FilePath $Python `
        -ArgumentList $Args `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $FastOut `
        -RedirectStandardError $FastErr `
        -Wait `
        -PassThru
    Write-WatcherLog "fast-screen before follow-up finished exit_code=$($Proc.ExitCode) out=$FastOut err=$FastErr"
    if ($Proc.ExitCode -ne 0) {
        Write-WatcherLog "fast-screen failed; continuing formal follow-up so hard evidence is not blocked"
        return
    }

    Invoke-FastScreenPromotionFormalRun -FastLogDir $FastLogDir
}

function Invoke-FastScreenPromotionFormalRun {
    param([string] $FastLogDir)
    if ($Phase -notin @("all", "candidates")) {
        Write-WatcherLog "fast-screen promotion formal run skipped for phase=$Phase"
        return
    }

    $SummaryDir = Join-Path $ProjectRoot "reports\paper_protocol_summary"
    New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null
    $PromotionJson = Join-Path $SummaryDir "fast_to_formal_promotion_report.json"
    $PromotionMd = Join-Path $SummaryDir "fast_to_formal_promotion_report.md"
    $PromotionOut = Join-Path $FastLogDir "fast_to_formal_promotion_$Stamp.out.log"
    $PromotionErr = Join-Path $FastLogDir "fast_to_formal_promotion_$Stamp.err.log"

    $PromotionArgs = @(
        (Join-Path $ProjectRoot "scripts\write_fast_to_formal_promotion_report.py"),
        "--summary-dir",
        $SummaryDir,
        "--python",
        $Python,
        "--root",
        $Root,
        "--log-dir",
        $ResolvedLogDir,
        "--out-md",
        $PromotionMd,
        "--out-json",
        $PromotionJson
    )
    $PromotionProc = Start-Process -FilePath $Python `
        -ArgumentList $PromotionArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $PromotionOut `
        -RedirectStandardError $PromotionErr `
        -Wait `
        -PassThru
    Write-WatcherLog "fast-to-formal promotion refresh finished exit_code=$($PromotionProc.ExitCode) out=$PromotionOut err=$PromotionErr"
    if ($PromotionProc.ExitCode -ne 0 -or -not (Test-Path $PromotionJson)) {
        Write-WatcherLog "promotion report unavailable; continuing main follow-up"
        return
    }

    $Promotion = Get-Content -LiteralPath $PromotionJson -Raw | ConvertFrom-Json
    $Sources = @()
    if ($null -ne $Promotion.promoted) {
        foreach ($Row in @($Promotion.promoted)) {
            if ($Row.source_run) {
                $Sources += [string] $Row.source_run
            }
        }
    }
    $Sources = @($Sources | Select-Object -Unique)
    if ($Sources.Count -eq 0) {
        Write-WatcherLog "no fast-screen formal candidates selected; continuing main follow-up under fail_fast candidate policy"
        return
    }

    $PromotedOut = Join-Path $ResolvedLogDir "fast_screen_promoted_candidates_$Stamp.out.log"
    $PromotedErr = Join-Path $ResolvedLogDir "fast_screen_promoted_candidates_$Stamp.err.log"
    Write-WatcherLog "running formal promoted candidates before main follow-up: $($Sources -join ',')"
    $PromotedArgs = @(
        (Join-Path $ProjectRoot "scripts\run_paper_protocol_direct.py"),
        "--phase",
        "candidates",
        "--only"
    )
    foreach ($Source in $Sources) {
        $PromotedArgs += $Source
    }
    $PromotedArgs += @(
        "--python",
        $Python,
        "--root",
        $Root,
        "--log-dir",
        $ResolvedLogDir
    )
    if ($PostprocessEach) {
        $PromotedArgs += "--postprocess-each"
    }

    $PromotedProc = Start-Process -FilePath $Python `
        -ArgumentList $PromotedArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $PromotedOut `
        -RedirectStandardError $PromotedErr `
        -Wait `
        -PassThru
    Write-WatcherLog "formal promoted candidates finished exit_code=$($PromotedProc.ExitCode) out=$PromotedOut err=$PromotedErr"
    if ($PromotedProc.ExitCode -ne 0) {
        Write-WatcherLog "promoted formal run failed; continuing main follow-up for recovery/complete evidence"
    }
}

try {
    Wait-Process -Id $WaitPid -ErrorAction Stop
} catch {
    Write-WatcherLog "wait skipped or pid already gone: $($_.Exception.Message)"
}

$ActiveQueues = Get-ActiveQueueCount
if ($ActiveQueues -gt 0) {
    Write-WatcherLog "another queue process is already active count=$ActiveQueues; no follow-up launched"
    exit 0
}

$PriorityWatchers = Get-PriorityWatcherCount
if ($PriorityWatchers -gt 0) {
    Write-WatcherLog "priority watcher active count=$PriorityWatchers; waiting up to ${PriorityWatcherGraceSeconds}s before follow-up decision"
    $Deadline = (Get-Date).AddSeconds($PriorityWatcherGraceSeconds)
    while ((Get-Date) -lt $Deadline) {
        Start-Sleep -Seconds 5
        $ActiveQueues = Get-ActiveQueueCount
        if ($ActiveQueues -gt 0) {
            Write-WatcherLog "priority watcher or another runner launched queue count=$ActiveQueues; no follow-up launched"
            exit 0
        }
        $PriorityWatchers = Get-PriorityWatcherCount
        if ($PriorityWatchers -le 0) {
            break
        }
    }
    $PriorityWatchers = Get-PriorityWatcherCount
    if ($PriorityWatchers -gt 0) {
        Write-WatcherLog "priority watcher still active count=$PriorityWatchers; no follow-up launched to avoid duplicate queues"
        exit 0
    }
}

$Recovery = Get-QueueRecovery
if (-not (Test-HasIncompleteWork -Report $Recovery)) {
    Write-WatcherLog "queue recovery reports all queued runs complete; no follow-up launched"
    exit 0
}

Invoke-FastScreenBeforeFollowUp

Write-WatcherLog "launching follow-up phase=$Phase because incomplete work remains"
$Args = @(
    (Join-Path $ProjectRoot "scripts\run_paper_protocol_direct.py"),
    "--phase",
    $Phase,
    "--python",
    $Python,
    "--root",
    $Root,
    "--log-dir",
    $ResolvedLogDir,
    "--candidate-policy",
    "fail_fast",
    "--final-policy",
    "defer_until_candidate_complete"
)
if ($PostprocessEach) {
    $Args += "--postprocess-each"
}

$Proc = Start-Process -FilePath $Python `
    -ArgumentList $Args `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $QueueOut `
    -RedirectStandardError $QueueErr `
    -Wait `
    -PassThru
$Code = $Proc.ExitCode
"$(Get-Date -Format s) follow-up finished exit_code=$Code out=$QueueOut err=$QueueErr" | Out-File -FilePath $WatcherLog -Append -Encoding utf8
exit $Code
