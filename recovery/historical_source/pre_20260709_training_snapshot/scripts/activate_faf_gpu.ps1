$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPrefix = "D:\NMI_SPWFM_datasets\conda_envs\faf_gpu"
$CondaExe = "E:\Anaconda\Scripts\conda.exe"

if (!(Test-Path $EnvPrefix)) {
    throw "Expected conda env not found: $EnvPrefix"
}
if (!(Test-Path $CondaExe)) {
    throw "Expected conda executable not found: $CondaExe"
}

Set-Location $ProjectRoot
(& $CondaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $EnvPrefix

Write-Host ""
Write-Host "Activated Friction-Affordance GPU environment:" -ForegroundColor Green
Write-Host "  Project : $ProjectRoot"
Write-Host "  Python  : $EnvPrefix\python.exe"
Write-Host "  GPU     : RTX 5070 Ti Laptop GPU / CUDA PyTorch cu128"
Write-Host ""
Write-Host "Quick check: python scripts\verify_gpu.py" -ForegroundColor Cyan

