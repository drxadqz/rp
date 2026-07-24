param(
    [string] $Python = "D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe",
    [string] $Root = "D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol",
    [string] $SummaryDir = "reports\paper_protocol_summary",
    [string] $LogDir = "outputs\paper_protocol_queue",
    [int] $PollSeconds = 120,
    [int] $MaxHours = 10
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$ResolvedSummaryDir = if ([System.IO.Path]::IsPathRooted($SummaryDir)) { $SummaryDir } else { Join-Path $ProjectRoot $SummaryDir }
$ResolvedLogDir = if ([System.IO.Path]::IsPathRooted($LogDir)) { $LogDir } else { Join-Path $ProjectRoot $LogDir }
New-Item -ItemType Directory -Force -Path $ResolvedSummaryDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResolvedLogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ResolvedLogDir "fair_report_refresh_watcher_$Stamp.log"
$OutPath = Join-Path $ResolvedLogDir "fair_report_refresh_watcher_$Stamp.out.log"
$ErrPath = Join-Path $ResolvedLogDir "fair_report_refresh_watcher_$Stamp.err.log"

function Write-RefreshLog {
    param([string] $Message)
    "$(Get-Date -Format s) $Message" | Out-File -FilePath $LogPath -Append -Encoding utf8
}

function Get-ReadyBaselineArtifacts {
    $Runs = @(
        "baseline_single_roadsaw_global_convnext",
        "baseline_single_rscd_global_convnext",
        "baseline_single_roadsc_global_convnext"
    )
    $Ready = @()
    foreach ($Run in $Runs) {
        $Bootstrap = Join-Path $Root "$Run\bootstrap_metrics.json"
        $Audit = Join-Path $Root "$Run\topvenue_result_audit.json"
        if ((Test-Path $Bootstrap) -and (Test-Path $Audit)) {
            $Ready += $Run
        }
    }
    return $Ready
}

function Test-BaselineArtifactsReady {
    return @((Get-ReadyBaselineArtifacts)).Count -eq 3
}

function Wait-ForNoCompetingProtocolWork {
    while ($true) {
        $Active = Get-CimInstance Win32_Process |
            Where-Object {
                $_.ProcessId -ne $PID -and
                $_.CommandLine -notlike '*Get-CimInstance*' -and
                (
                    $_.CommandLine -match 'scripts[\\/]train\.py' -or
                    $_.CommandLine -like '*postprocess_protocol_outputs.py*' -or
                    $_.CommandLine -like '*evaluate.py*' -or
                    $_.CommandLine -like '*evaluate_detailed.py*' -or
                    $_.CommandLine -like '*calibrate_intervals.py*' -or
                    $_.CommandLine -like '*bootstrap_metrics.py*'
                )
            }
        if (@($Active).Count -eq 0) {
            return
        }
        $Names = (@($Active) | ForEach-Object { "$($_.ProcessId)" }) -join ","
        Write-RefreshLog "protocol work active (pids=$Names); waiting before report refresh"
        Start-Sleep -Seconds 30
    }
}

function Invoke-ReportCommand {
    param([string[]] $CommandArgs, [string] $Name)
    Write-RefreshLog "RUN $Name $($CommandArgs -join ' ')"
    $Proc = Start-Process -FilePath $Python `
        -ArgumentList $CommandArgs `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $OutPath `
        -RedirectStandardError $ErrPath `
        -Wait `
        -PassThru
    Write-RefreshLog "DONE $Name exit_code=$($Proc.ExitCode)"
    if ($Proc.ExitCode -ne 0) {
        throw "$Name failed with exit code $($Proc.ExitCode)"
    }
}

Write-RefreshLog "watching for completed single-dataset baseline artifacts root=$Root"
$LastRefreshSignature = ""
$Deadline = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $Deadline) {
    $ReadyRuns = @(Get-ReadyBaselineArtifacts)
    $ReadySignature = ($ReadyRuns | Sort-Object) -join ","
    if ($ReadySignature -and $ReadySignature -ne $LastRefreshSignature) {
        if (Test-BaselineArtifactsReady) {
            Write-RefreshLog "all baseline artifacts ready; refreshing final fair reports"
        } else {
            Write-RefreshLog "new baseline artifacts ready ($ReadySignature); refreshing interim fair reports"
        }
        Wait-ForNoCompetingProtocolWork
        Invoke-ReportCommand -CommandArgs @(
            "scripts\postprocess_protocol_outputs.py",
            "--root", $Root,
            "--summary-dir", $ResolvedSummaryDir
        ) -Name "postprocess_protocol_outputs"
        Invoke-ReportCommand -CommandArgs @("scripts\write_ten_hour_closure_report.py") -Name "write_ten_hour_closure_report"
        Invoke-ReportCommand -CommandArgs @("scripts\write_topvenue_innovation_roadmap.py") -Name "write_topvenue_innovation_roadmap"
        Invoke-ReportCommand -CommandArgs @("scripts\write_next_experiment_decision_report.py") -Name "write_next_experiment_decision_report"
        Invoke-ReportCommand -CommandArgs @("scripts\write_module_retention_report.py") -Name "write_module_retention_report"
        Invoke-ReportCommand -CommandArgs @("scripts\write_reviewer_action_matrix.py") -Name "write_reviewer_action_matrix"
        $LastRefreshSignature = $ReadySignature
        if (Test-BaselineArtifactsReady) {
            Write-RefreshLog "final fair report refresh complete"
            exit 0
        }
        Write-RefreshLog "interim fair report refresh complete; continuing to watch for remaining baselines"
    }
    Write-RefreshLog "ready baselines: $ReadySignature; sleeping ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}

Write-RefreshLog "timeout after $MaxHours hours without complete baseline artifacts"
exit 2
