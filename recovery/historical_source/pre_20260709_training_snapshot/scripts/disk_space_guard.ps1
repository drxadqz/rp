param(
    [int64] $MinCFreeBytes = 750MB,
    [int64] $MinDFreeBytes = 2GB,
    [int] $PollSeconds = 60,
    [string] $LogPath = "outputs\paper_protocol_queue\disk_space_guard.log",
    [string] $ProjectMarker = "friction_affordance_field",
    [string] $PythonPath = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Write-GuardLog {
    param([string] $Message)
    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogPath -Value $Line
    Write-Host $Line
}

Write-GuardLog "disk guard started MinCFreeBytes=$MinCFreeBytes MinDFreeBytes=$MinDFreeBytes PollSeconds=$PollSeconds"

while ($true) {
    $C = Get-PSDrive -Name C -PSProvider FileSystem
    $D = Get-PSDrive -Name D -PSProvider FileSystem
    if ($C.Free -lt $MinCFreeBytes -or $D.Free -lt $MinDFreeBytes) {
        Write-GuardLog "free space below threshold: C=$($C.Free) D=$($D.Free). Stopping experiment processes."
        Get-Process python -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -eq $PythonPath } |
            ForEach-Object {
                Write-GuardLog "Stop python pid=$($_.Id) path=$($_.Path)"
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            }
        Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -like "*$ProjectMarker*" -and (
                    $_.CommandLine -like "*run_paper_protocol_queue.ps1*" -or
                    $_.CommandLine -like "*run_config_pipeline.ps1*" -or
                    $_.CommandLine -like "*run_paper_protocol_after_pid.ps1*"
                )
            } |
            ForEach-Object {
                Write-GuardLog "Stop powershell pid=$($_.ProcessId)"
                Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            }
        Write-GuardLog "disk guard exiting after emergency stop"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
