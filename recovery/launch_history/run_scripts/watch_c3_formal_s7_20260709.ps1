$ErrorActionPreference = "Stop"

$PidToCheck = 13808
$OutDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"

Write-Output "== Process =="
Get-Process -Id $PidToCheck -ErrorAction SilentlyContinue |
    Select-Object Id,CPU,WorkingSet64,StartTime,Path |
    Format-List

Write-Output "== GPU =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>$null

Write-Output "== Output files =="
Get-ChildItem $OutDir -Force -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name,Length,LastWriteTime |
    Format-Table -AutoSize

$History = Join-Path $OutDir "history.json"
if (Test-Path $History) {
    Write-Output "== history.json =="
    Get-Content $History -Tail 80
}

$Stdout = Get-ChildItem $OutDir -Filter "train_stdout_*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$Stderr = Get-ChildItem $OutDir -Filter "train_stderr_*.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($Stdout) {
    Write-Output "== stdout tail: $($Stdout.Name) =="
    Get-Content $Stdout.FullName -Tail 80
}
if ($Stderr) {
    Write-Output "== stderr tail: $($Stderr.Name) =="
    Get-Content $Stderr.FullName -Tail 80
}
