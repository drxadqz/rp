param(
    [Parameter(Mandatory = $true)]
    [int] $QueuePid,
    [string] $Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol",
    [string] $LogDir = "outputs\paper_protocol_queue",
    [int] $PollSeconds = 30,
    [int] $TimeoutHours = 12,
    [switch] $RestartQueueAfterV5
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$ResolvedLogDir = if ([System.IO.Path]::IsPathRooted($LogDir)) { $LogDir } else { Join-Path $ProjectRoot $LogDir }
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ResolvedLogDir "roadsaw_priority_handoff_$QueuePid`_$Stamp.log"

function Write-HandoffLog {
    param([string] $Message)
    "$(Get-Date -Format s) $Message" | Out-File -FilePath $LogPath -Append -Encoding utf8
}

function Test-V5Complete {
    $V5 = Join-Path $Root "v5_full_faf"
    $Required = @(
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json"
    )
    foreach ($Name in $Required) {
        if (-not (Test-Path (Join-Path $V5 $Name))) {
            return $false
        }
    }
    return $true
}

function Stop-QueueTree {
    param([int] $PidToStop)
    $Children = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $PidToStop } |
        Select-Object -ExpandProperty ProcessId
    foreach ($ChildPid in $Children) {
        try {
            Write-HandoffLog "stopping child pid=$ChildPid"
            Stop-Process -Id $ChildPid -Force -ErrorAction Stop
        } catch {
            Write-HandoffLog "child pid=$ChildPid stop skipped: $($_.Exception.Message)"
        }
    }
    try {
        Write-HandoffLog "stopping queue pid=$PidToStop"
        Stop-Process -Id $PidToStop -Force -ErrorAction Stop
    } catch {
        Write-HandoffLog "queue pid=$PidToStop stop skipped: $($_.Exception.Message)"
    }
}

$Deadline = (Get-Date).AddHours($TimeoutHours)
Write-HandoffLog "watching queue pid=$QueuePid until v5_full_faf is fully postprocessed; log=$LogPath"
while ((Get-Date) -lt $Deadline) {
    $Queue = Get-Process -Id $QueuePid -ErrorAction SilentlyContinue
    if (-not $Queue) {
        Write-HandoffLog "queue pid=$QueuePid already exited; no handoff needed"
        exit 0
    }
    if (Test-V5Complete) {
        if (Test-Path (Join-Path $Root "lodo_roadsaw_full_faf\topvenue_result_audit.json")) {
            Write-HandoffLog "lodo_roadsaw_full_faf already complete; no handoff needed"
            exit 0
        }
        if ($RestartQueueAfterV5) {
            Write-HandoffLog "v5_full_faf artifacts complete; terminating old queue because RestartQueueAfterV5 was requested"
            Stop-QueueTree -PidToStop $QueuePid
        } else {
            Write-HandoffLog "v5_full_faf artifacts complete; queue is left running because RoadSaW-priority order is already configured"
        }
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-HandoffLog "timeout after $TimeoutHours hours without complete v5_full_faf artifacts"
exit 2
