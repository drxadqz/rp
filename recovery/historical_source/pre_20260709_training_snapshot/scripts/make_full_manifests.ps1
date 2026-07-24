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

& $CondaExe run -p $EnvPrefix python scripts\build_manifests.py `
    --config configs\data\local_paths.yaml `
    --out-dir data\manifests_full

$Code = $LASTEXITCODE
$global:LASTEXITCODE = $Code
if ($Code -ne 0) {
    throw "Command failed with exit code $Code"
}
