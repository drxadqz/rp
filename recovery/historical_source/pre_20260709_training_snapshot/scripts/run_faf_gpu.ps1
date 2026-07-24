param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Command
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPrefix = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper"
$CondaExe = "E:\Anaconda\Scripts\conda.exe"
$TorchHome = "D:\NMI_SPWFM_datasets\torch_cache"

if (!(Test-Path $EnvPrefix)) {
    throw "Expected conda env not found: $EnvPrefix"
}
if (!(Test-Path $CondaExe)) {
    throw "Expected conda executable not found: $CondaExe"
}

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $TorchHome | Out-Null
$env:TORCH_HOME = $TorchHome

if ($Command.Count -eq 0) {
    $Command = @("python", "scripts\verify_gpu.py")
}

Write-Host "Running in fixed GPU env: $EnvPrefix" -ForegroundColor Green
Write-Host "Command: $($Command -join ' ')" -ForegroundColor Cyan
& $CondaExe run -p $EnvPrefix @Command
$Code = $LASTEXITCODE
$global:LASTEXITCODE = $Code
if ($Code -ne 0) {
    throw "Command failed with exit code $Code"
}
