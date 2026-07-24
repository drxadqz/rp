param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$RunName,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [int]$PollSeconds = 300,

    [int]$MaxHours = 36
)

$ErrorActionPreference = "Stop"

$Root = "E:\perception\friction_affordance_field"
$Postprocess = Join-Path $Root "scripts\postprocess_c3_formal_candidate_20260709.ps1"
$Deadline = (Get-Date).AddHours($MaxHours)
$Metrics = Join-Path $RunDir "metrics.json"
$PerClass = Join-Path $RunDir "per_class_metrics.csv"
$Confusion = Join-Path $RunDir "confusion_matrix.csv"

Write-Output "watch_start=$(Get-Date -Format o)"
Write-Output "run_dir=$RunDir"
Write-Output "run_name=$RunName"
Write-Output "out_dir=$OutDir"

while ((Get-Date) -lt $Deadline) {
    $hasMetrics = Test-Path -LiteralPath $Metrics
    $hasPerClass = Test-Path -LiteralPath $PerClass
    $hasConfusion = Test-Path -LiteralPath $Confusion
    if ($hasMetrics -and $hasPerClass -and $hasConfusion) {
        Write-Output "detected_complete=$(Get-Date -Format o)"
        Push-Location $Root
        try {
            powershell -ExecutionPolicy Bypass -File $Postprocess -RunDir $RunDir -RunName $RunName -OutDir $OutDir
        }
        finally {
            Pop-Location
        }
        Write-Output "watch_done=$(Get-Date -Format o)"
        exit 0
    }
    Write-Output "pending=$(Get-Date -Format o) metrics=$hasMetrics per_class=$hasPerClass confusion=$hasConfusion"
    Start-Sleep -Seconds $PollSeconds
}

Write-Output "watch_timeout=$(Get-Date -Format o)"
exit 2
