param(
    [Parameter(Mandatory = $true)]
    [int] $TargetPid,
    [string] $LogPath = ""
)

$ErrorActionPreference = "Stop"

Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$ES_AWAYMODE_REQUIRED = [uint32]64

function Write-KeepAwakeLog {
    param([string] $Message)
    if ($LogPath) {
        $dir = Split-Path -Parent $LogPath
        if ($dir) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
        Add-Content -Path $LogPath -Value ("{0} {1}" -f (Get-Date -Format "s"), $Message)
    }
}

try {
    Write-KeepAwakeLog "start target_pid=$TargetPid"
    while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
        [Win32.Power]::SetThreadExecutionState(
            $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
        ) | Out-Null
        Start-Sleep -Seconds 30
    }
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-KeepAwakeLog "stop target_pid=$TargetPid"
}
