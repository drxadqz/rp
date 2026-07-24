param(
    [string[]] $Configs,
    [int] $WaitForPid = 0,
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $LogDir = "outputs\queued_experiments"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if ($WaitForPid -gt 0) {
    Write-Host "Waiting for PID $WaitForPid before starting queue..."
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
}

if (!$Configs -or $Configs.Count -eq 0) {
    throw "No configs supplied."
}

foreach ($Config in $Configs) {
    if (!(Test-Path $Config)) {
        throw "Config not found: $Config"
    }
    $Name = [IO.Path]::GetFileNameWithoutExtension($Config)
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutLog = Join-Path $LogDir "$Name`_$Stamp.out.log"
    $ErrLog = Join-Path $LogDir "$Name`_$Stamp.err.log"
    Write-Host ""
    Write-Host "==== queued config: $Config ====" -ForegroundColor Cyan
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\run_config_pipeline.ps1 `
        -Config $Config `
        -Python $Python `
        > $OutLog 2> $ErrLog
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "Queued config failed: $Config (exit code $Code). Logs: $OutLog $ErrLog"
    }
}
