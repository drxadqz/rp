param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$RunName,

    [Parameter(Mandatory = $true)]
    [string]$OutDir
)

$ErrorActionPreference = "Stop"

$Root = "E:\perception\friction_affordance_field"
$Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe"
$AnchorDir = "E:\perception_outputs\rscd_surface_classification\c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708\fast_test"
$CandidateCsv = "reports\paper_protocol_summary\srbr_route_candidates_20260709\srbr_route_candidates.csv"

$env:PYTHONPATH = Join-Path $Root "src"

Push-Location $Root
try {
    $SummaryDir = Join-Path $OutDir "summary"
    $BoundaryDir = Join-Path $OutDir "boundary_delta"
    New-Item -ItemType Directory -Force -Path $SummaryDir, $BoundaryDir | Out-Null

    & $Python scripts\summarize_formal_fullmanifest_result.py `
        --run-dir $RunDir `
        --anchor-dir $AnchorDir `
        --out-dir $SummaryDir `
        --run-name $RunName
    $SummaryExit = $LASTEXITCODE
    if (($SummaryExit -ne 0) -and ($SummaryExit -ne 2)) {
        throw "summarize_formal_fullmanifest_result.py failed with exit code $SummaryExit"
    }

    & $Python scripts\compare_boundary_confusion_deltas.py `
        --run-dir $RunDir `
        --anchor-dir $AnchorDir `
        --candidate-csv $CandidateCsv `
        --out-dir $BoundaryDir
    $BoundaryExit = $LASTEXITCODE
    if (($BoundaryExit -ne 0) -and ($BoundaryExit -ne 2)) {
        throw "compare_boundary_confusion_deltas.py failed with exit code $BoundaryExit"
    }

    $GateDir = Join-Path $OutDir "promotion_gate"
    New-Item -ItemType Directory -Force -Path $GateDir | Out-Null
    & $Python scripts\formal_candidate_promotion_gate.py `
        --run-dir $RunDir `
        --run-name $RunName `
        --out-dir $GateDir `
        --anchor-dir $AnchorDir `
        --boundary-csv $CandidateCsv
    $GateExit = $LASTEXITCODE
    if (($GateExit -ne 0) -and ($GateExit -ne 2)) {
        throw "formal_candidate_promotion_gate.py failed with exit code $GateExit"
    }

    Write-Output "postprocess_done"
    Write-Output "summary_dir=$SummaryDir"
    Write-Output "boundary_dir=$BoundaryDir"
    Write-Output "promotion_gate_dir=$GateDir"
    Write-Output "summary_exit=$SummaryExit"
    Write-Output "boundary_exit=$BoundaryExit"
    Write-Output "promotion_gate_exit=$GateExit"
}
finally {
    Pop-Location
}
