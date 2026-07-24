param(
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

$Root = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$Config = "configs\c3_farnet\c3_farnet_formal_fullmanifest_factor_graph_metric_s11_20260709.yaml"
$OutDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_factor_graph_metric_s11_20260709"
$TmpDir = "E:\perception_tmp"
$CacheDir = "E:\perception_cache\torch"

New-Item -ItemType Directory -Force -Path $OutDir, $TmpDir, $CacheDir | Out-Null

$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUNBUFFERED = "1"
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:TORCH_HOME = $CacheDir

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $OutDir "train_stdout_$Stamp.log"
$Stderr = Join-Path $OutDir "train_stderr_$Stamp.log"

if ($Foreground) {
    Push-Location $Root
    try {
        & $Python -u train.py --config $Config 1> $Stdout 2> $Stderr
    }
    finally {
        Pop-Location
    }
    Write-Output "foreground_complete"
    Write-Output "stdout=$Stdout"
    Write-Output "stderr=$Stderr"
}
else {
    $proc = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-u", "train.py", "--config", $Config) `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru `
        -WindowStyle Hidden
    Write-Output "pid=$($proc.Id)"
    Write-Output "stdout=$Stdout"
    Write-Output "stderr=$Stderr"
}
